# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.


import os
import io
import fnmatch
import json
import signal
import time
import requests
import shutil
import logging
import tarfile
import tempfile
import torch
from filelock import FileLock
from contextlib import contextmanager
from torch.multiprocessing import spawn
from functools import partial, wraps
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile, is_zipfile
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from transformers.utils.hub import is_remote_url, urlparse, http_get, http_user_agent
from huggingface_hub import HfFolder
from huggingface_hub.utils import hf_raise_for_status

from unitorch import is_offline_mode, get_cache_home
from unitorch.utils.decorators import replace
from unitorch.utils.functional import (
    pop_value,
    rpartial,
    truncate_sequence_pair,
    nested_dict_value,
)
from unitorch.utils.image_utils import (
    make_grid,
    resize_shortest_edge,
    image_list_to_tensor,
    numpy_to_pil,
)
from unitorch.utils.video_utils import tensor2vid
from unitorch.utils.import_utils import (
    is_deepspeed_available,
    is_accelerate_available,
    is_megatron_available,
    is_fastapi_available,
    is_diffusers_available,
    is_safetensors_available,
    is_xformers_available,
    is_opencv_available,
    is_torch_available,
    is_torch2_available,
    is_bitsandbytes_available,
    is_auto_gptq_available,
    is_onnxruntime_available,
)
from unitorch.utils.io import GenericWriter, IOProcess, PostProcess, GENERATE_FINISHED
from unitorch.utils.torch_utils import get_local_rank
from unitorch.utils.torch_utils import (
    DistributedSkipSampler,
    RandomSkipSampler,
    SequentialSkipSampler,
)


def url_to_filename(url: str, etag: Optional[str] = None) -> str:
    url_bytes = url.encode("utf-8")
    filename = sha256(url_bytes).hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        filename += "." + sha256(etag_bytes).hexdigest()

    if url.endswith(".h5"):
        filename += ".h5"

    return filename


def get_from_cache(
    url: str,
    cache_dir=None,
    force_download=False,
    proxies=None,
    etag_timeout=10,
    resume_download=False,
    user_agent: Union[Dict, str, None] = None,
    use_auth_token: Union[bool, str, None] = None,
    local_files_only=False,
) -> Optional[str]:
    if cache_dir is None:
        cache_dir = get_cache_home()
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)

    headers = {"user-agent": http_user_agent(user_agent)}
    if isinstance(use_auth_token, str):
        headers["authorization"] = f"Bearer {use_auth_token}"
    elif use_auth_token:
        token = HfFolder.get_token()
        if token is None:
            raise EnvironmentError(
                "You specified use_auth_token=True, but a huggingface token was not found."
            )
        headers["authorization"] = f"Bearer {token}"

    url_to_download = url
    etag = None
    if not local_files_only:
        try:
            r = requests.head(
                url,
                headers=headers,
                allow_redirects=False,
                proxies=proxies,
                timeout=etag_timeout,
            )
            hf_raise_for_status(r)
            etag = r.headers.get("X-Linked-Etag") or r.headers.get("ETag")
            # We favor a custom header indicating the etag of the linked resource, and
            # we fallback to the regular etag header.
            # If we don't have any of those, raise an error.
            if etag is None:
                raise OSError(
                    "Distant resource does not have an ETag, we won't be able to reliably ensure reproducibility."
                )
            # In case of a redirect,
            # save an extra redirect on the request.get call,
            # and ensure we download the exact atomic version even if it changed
            # between the HEAD and the GET (unlikely, but hey).
            if 300 <= r.status_code <= 399:
                url_to_download = r.headers["Location"]
        except (requests.exceptions.SSLError, requests.exceptions.ProxyError):
            # Actually raise for those subclasses of ConnectionError
            raise
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            # Otherwise, our Internet connection is down.
            # etag is None
            pass

    filename = url_to_filename(url, etag)

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)

    # etag is None == we don't have a connection or we passed local_files_only.
    # try to get the last downloaded one
    if etag is None:
        if os.path.exists(cache_path):
            return cache_path
        else:
            matching_files = [
                file
                for file in fnmatch.filter(
                    os.listdir(cache_dir), filename.split(".")[0] + ".*"
                )
                if not file.endswith(".json") and not file.endswith(".lock")
            ]
            if len(matching_files) > 0:
                return os.path.join(cache_dir, matching_files[-1])
            else:
                # If files cannot be found and local_files_only=True,
                # the models might've been found if local_files_only=False
                # Notify the user about that
                if local_files_only:
                    raise FileNotFoundError(
                        "Cannot find the requested files in the cached path and outgoing traffic has been"
                        " disabled. To enable model look-ups and downloads online, set 'local_files_only'"
                        " to False."
                    )
                else:
                    raise ValueError(
                        "Connection error, and we cannot find the requested files in the cached path."
                        " Please try again or make sure your Internet connection is on."
                    )

    # From now on, etag is not None.
    if os.path.exists(cache_path) and not force_download:
        return cache_path

    # Prevent parallel downloads of the same file with a lock.
    lock_path = cache_path + ".lock"
    with FileLock(lock_path):
        # If the download just completed while the lock was activated.
        if os.path.exists(cache_path) and not force_download:
            # Even if returning early like here, the lock will be released.
            return cache_path

        if resume_download:
            incomplete_path = cache_path + ".incomplete"

            @contextmanager
            def _resumable_file_manager() -> "io.BufferedWriter":
                with open(incomplete_path, "ab") as f:
                    yield f

            temp_file_manager = _resumable_file_manager
            if os.path.exists(incomplete_path):
                resume_size = os.stat(incomplete_path).st_size
            else:
                resume_size = 0
        else:
            temp_file_manager = partial(
                tempfile.NamedTemporaryFile, mode="wb", dir=cache_dir, delete=False
            )
            resume_size = 0

        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with temp_file_manager() as temp_file:
            logging.debug(
                f"{url} not found in cache or force_download set to True, downloading to {temp_file.name}"
            )

            http_get(
                url_to_download,
                temp_file,
                proxies=proxies,
                resume_size=resume_size,
                headers=headers,
            )

        logging.debug(f"storing {url} in cache at {cache_path}")
        os.replace(temp_file.name, cache_path)

        # NamedTemporaryFile creates a file with hardwired 0600 perms (ignoring umask), so fixing it.
        umask = os.umask(0o666)
        os.umask(umask)
        os.chmod(cache_path, 0o666 & ~umask)

        logging.debug(f"creating metadata file for {cache_path}")
        meta = {"url": url, "etag": etag}
        meta_path = cache_path + ".json"
        with open(meta_path, "w") as meta_file:
            json.dump(meta, meta_file)

    return cache_path


def cached_path(
    url_or_filename: str,
    cache_dir: Optional[str] = None,
    force_download: Optional[bool] = False,
    proxies: Optional[Dict] = None,
    resume_download: Optional[bool] = False,
    user_agent: Union[Dict, str, None] = None,
    extract_compressed_file: Optional[bool] = False,
    force_extract: Optional[bool] = False,
    use_auth_token: Union[bool, str, None] = None,
    local_files_only: Optional[bool] = False,
) -> Optional[str]:
    """
    Retrieves a file from a given URL or local path and caches it for future use.

    Args:
        url_or_filename (str): URL or local path of the file.
        cache_dir (str, optional): Directory to store cached files. If not provided, the default cache directory will be used.
        force_download (bool, optional): Whether to force download the file even if it already exists in the cache. Defaults to False.
        proxies (dict, optional): Proxy configuration for downloading the file. Defaults to None.
        resume_download (bool, optional): Whether to resume a partially downloaded file. Defaults to False.
        user_agent (dict, str, None, optional): User agent configuration for the download request. Defaults to None.
        extract_compressed_file (bool, optional): Whether to extract compressed files (e.g., zip, tar) after downloading. Defaults to False.
        force_extract (bool, optional): Whether to force extract the file even if it has been extracted before. Defaults to False.
        use_auth_token (bool, str, None, optional): Authentication token to be used for downloading. Defaults to None.
        local_files_only (bool, optional): Whether to only use locally available files in offline mode. Defaults to False.

    Returns:
        Optional[str]: Path to the cached file or extracted directory, or None if the file is not found.

    Raises:
        EnvironmentError: If the file is not found or the URL or local path cannot be parsed.
        ValueError: If the URL or local path is in an unknown format.
    """
    if cache_dir is None:
        cache_dir = get_cache_home()

    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)

    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if is_offline_mode() and not local_files_only:
        logging.debug("Offline mode: forcing local_files_only=True")
        local_files_only = True

    if is_remote_url(url_or_filename):
        # URL, so get it from the cache (downloading if necessary)
        output_path = get_from_cache(
            url_or_filename,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            user_agent=user_agent,
            use_auth_token=use_auth_token,
            local_files_only=local_files_only,
        )
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        output_path = url_or_filename
    elif urlparse(url_or_filename).scheme == "":
        # File, but it doesn't exist.
        raise EnvironmentError(f"file {url_or_filename} not found")
    else:
        # Something unknown
        raise ValueError(
            f"unable to parse {url_or_filename} as a URL or as a local path"
        )

    if extract_compressed_file:
        if not is_zipfile(output_path) and not tarfile.is_tarfile(output_path):
            return output_path

        # Path where we extract compressed archives
        # We avoid '.' in dir name and add "-extracted" at the end: "./model.zip" => "./model-zip-extracted/"
        output_dir, output_file = os.path.split(output_path)
        output_extract_dir_name = output_file.replace(".", "-") + "-extracted"
        output_path_extracted = os.path.join(output_dir, output_extract_dir_name)

        if (
            os.path.isdir(output_path_extracted)
            and os.listdir(output_path_extracted)
            and not force_extract
        ):
            return output_path_extracted

        # Prevent parallel extractions
        lock_path = output_path + ".lock"
        with FileLock(lock_path):
            shutil.rmtree(output_path_extracted, ignore_errors=True)
            os.makedirs(output_path_extracted)
            if is_zipfile(output_path):
                with ZipFile(output_path, "r") as zip_file:
                    zip_file.extractall(output_path_extracted)
                    zip_file.close()
            elif tarfile.is_tarfile(output_path):
                tar_file = tarfile.open(output_path)
                tar_file.extractall(output_path_extracted)
                tar_file.close()
            else:
                raise EnvironmentError(
                    f"Archive format of {output_path} could not be identified"
                )

        return output_path_extracted

    return output_path


def read_file(file, lines=False):
    result = open(file, "r").read()
    if lines:
        result = result.split("\n")
        if result[-1] == "":
            result = result[:-1]
    return result


def read_json_file(file):
    with open(file, "r") as f:
        return json.load(f)
