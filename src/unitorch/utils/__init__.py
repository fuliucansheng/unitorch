# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import fnmatch
import json
import logging
import os
import re
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin, urlparse

import requests
import safetensors
import torch
from filelock import FileLock
from transformers import AddedToken

from unitorch import get_cache_dir
from unitorch.utils.decorators import replace, retry
from unitorch.utils.functional import (
    nested_dict_value,
    pop_value,
    rpartial,
    truncate_sequence_pair,
    update_nested_dict,
)
from unitorch.utils.image_utils import (
    image_list_to_tensor,
    make_grid,
    numpy_to_pil,
    resize_shortest_edge,
)
from unitorch.utils.import_utils import (
    is_bfloat16_available,
    is_cuda_available,
    is_deepspeed_available,
    is_diffusers_available,
    is_fastapi_available,
    is_gradio_available,
    is_megatron_available,
    is_onnxruntime_available,
    is_opencv_available,
    is_wandb_available,
    is_vllm_available,
    reload_module,
)
from unitorch.utils.io import GENERATE_FINISHED, GenericWriter, IOProcess, PostProcess
from unitorch.utils.torch_utils import (
    DistributedSkipSampler,
    RandomSkipSampler,
    SequentialSkipSampler,
    get_local_rank,
)
from unitorch.utils.video_utils import tensor2vid


def is_remote_url(url_or_filename: str) -> bool:
    """Return ``True`` if *url_or_filename* is an HTTP(S) URL."""
    return urlparse(url_or_filename).scheme in ("http", "https")


def url_to_filename(url: str, etag: Optional[str] = None) -> str:
    """Derive a deterministic cache filename from a URL and optional ETag."""
    filename = sha256(url.encode("utf-8")).hexdigest()
    if etag:
        filename += "." + sha256(etag.encode("utf-8")).hexdigest()
    if url.endswith(".h5"):
        filename += ".h5"
    return filename


def get_from_cache(
    url: str,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    proxies: Optional[Dict] = None,
    etag_timeout: int = 10,
    resume_download: bool = False,
    use_auth_token: Union[bool, str, None] = None,
) -> Optional[str]:
    """Download *url* to *cache_dir* (or the default cache) and return the local path."""
    if cache_dir is None:
        cache_dir = get_cache_dir()
    cache_dir = str(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    headers = {"user-agent": "unitorch"}
    if isinstance(use_auth_token, str):
        headers["authorization"] = f"Bearer {use_auth_token}"
    elif use_auth_token:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token is None:
            raise EnvironmentError(
                "use_auth_token=True but no HuggingFace token was found."
            )
        headers["authorization"] = f"Bearer {token}"

    url_to_download = url
    etag = None

    try:
        response = requests.head(
            url,
            headers=headers,
            allow_redirects=False,
            proxies=proxies,
            timeout=etag_timeout,
        )
        response.raise_for_status()
        etag = response.headers.get("X-Linked-Etag") or response.headers.get("ETag")
        if etag is None:
            raise OSError(
                "Remote resource has no ETag; reproducibility cannot be guaranteed."
            )
        if 300 <= response.status_code <= 399:
            url_to_download = urljoin(response.url, response.headers["Location"])
    except (requests.exceptions.SSLError, requests.exceptions.ProxyError):
        raise
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        pass  # etag remains None; fall through to cached-file lookup

    filename = url_to_filename(url, etag)
    cache_path = os.path.join(cache_dir, filename)

    if etag is None:
        if os.path.exists(cache_path):
            return cache_path
        stem = filename.split(".")[0]
        candidates = [
            f
            for f in fnmatch.filter(os.listdir(cache_dir), stem + ".*")
            if not f.endswith((".json", ".lock"))
        ]
        if candidates:
            return os.path.join(cache_dir, candidates[-1])
        raise ValueError(
            "No internet connection and the requested files were not found in cache."
        )

    if os.path.exists(cache_path) and not force_download:
        return cache_path

    lock_path = cache_path + ".lock"
    with FileLock(lock_path):
        if os.path.exists(cache_path) and not force_download:
            return cache_path

        if resume_download:
            incomplete_path = cache_path + ".incomplete"
            resume_size = (
                os.stat(incomplete_path).st_size
                if os.path.exists(incomplete_path)
                else 0
            )
            dest = open(incomplete_path, "ab")
        else:
            tmp_fd, incomplete_path = tempfile.mkstemp(dir=cache_dir)
            dest = os.fdopen(tmp_fd, "wb")
            resume_size = 0

        try:
            dl_headers = dict(headers)
            if resume_size:
                dl_headers["Range"] = f"bytes={resume_size}-"
            logging.debug(f"Downloading {url} to {incomplete_path}")
            with requests.get(
                url_to_download, headers=dl_headers, proxies=proxies, stream=True
            ) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        dest.write(chunk)
        finally:
            dest.close()

        logging.debug(f"Storing {url} in cache at {cache_path}")
        os.replace(incomplete_path, cache_path)

        umask = os.umask(0o666)
        os.umask(umask)
        os.chmod(cache_path, 0o666 & ~umask)

        meta_path = cache_path + ".json"
        with open(meta_path, "w") as meta_file:
            json.dump({"url": url, "etag": etag}, meta_file)

    return cache_path


def cached_path(
    url_or_filename: str,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    proxies: Optional[Dict] = None,
    resume_download: bool = False,
    use_auth_token: Union[bool, str, None] = None,
) -> Optional[str]:
    """Resolve a URL or local path to a cached local file.

    Downloads remote files on first access and returns the local cache path.

    Args:
        url_or_filename: URL or local filesystem path.
        cache_dir: Override the default cache directory.
        force_download: Re-download even if already cached.
        proxies: Proxy settings forwarded to ``requests``.
        resume_download: Continue an interrupted download.
        use_auth_token: HuggingFace auth token (``True`` reads from the environment).

    Returns:
        Absolute path to the cached file.

    Raises:
        EnvironmentError: File not found locally or remotely.
        ValueError: Unrecognised URL / path format.
    """
    if cache_dir is None:
        cache_dir = get_cache_dir()
    url_or_filename = str(url_or_filename)
    cache_dir = str(cache_dir)

    if is_remote_url(url_or_filename):
        return get_from_cache(
            url_or_filename,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            use_auth_token=use_auth_token,
        )
    elif os.path.exists(url_or_filename):
        return url_or_filename
    elif urlparse(url_or_filename).scheme == "":
        raise EnvironmentError(f"File not found: {url_or_filename}")
    else:
        raise ValueError(f"Unable to parse as a URL or local path: {url_or_filename}")


def read_file(file: str, lines: bool = False) -> Union[str, List[str]]:
    """Read a text file and return its contents as a string or list of lines."""
    with open(file, "r") as f:
        content = f.read()
    return content.splitlines() if lines else content


def read_json_file(file: str) -> Any:
    """Parse and return the contents of a JSON file."""
    with open(file, "r") as f:
        return json.load(f)


def load_weight(
    path: Optional[Union[str, List[str]]],
    replace_keys: Optional[Dict[str, str]] = None,
    prefix_keys: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Load model weights from one or more checkpoint files.

    Args:
        path: Single path or list of paths to ``.safetensors`` or PyTorch checkpoint files.
        replace_keys: Mapping of regex patterns to replacement strings applied to state-dict keys.
        prefix_keys: Mapping of regex patterns to prefix strings prepended to matching keys.

    Returns:
        Merged state dictionary with keys transformed according to *replace_keys* and *prefix_keys*.
    """
    if path is None:
        return {}

    replace_keys = replace_keys or {}
    prefix_keys = prefix_keys or {}

    if isinstance(path, str):
        path = [path]

    state_dict: Dict[str, Any] = {}
    for p in path:
        local = cached_path(p)
        if p.endswith(".safetensors"):
            state_dict.update(safetensors.torch.load_file(local))
        else:
            state_dict.update(torch.load(local, map_location="cpu"))

    results: Dict[str, Any] = {}
    for key, value in state_dict.items():
        for pattern, prefix in prefix_keys.items():
            if re.match(pattern, key):
                key = prefix + key
                break
        for pattern, replacement in replace_keys.items():
            key = re.sub(pattern, replacement, key)
        results[key] = value

    return results


def get_added_token(spec: Union[str, Dict[str, Any]]) -> AddedToken:
    """Construct an ``AddedToken`` from a string or attribute dictionary.

    Args:
        spec: Token string or dict with ``content`` and optional fields
              ``lstrip``, ``rstrip``, ``normalized``, ``special``, ``single_word``.

    Returns:
        Configured ``AddedToken`` instance.

    Raises:
        ValueError: If *spec* is not a ``str`` or ``dict``.
    """
    if isinstance(spec, str):
        return AddedToken(
            spec,
            lstrip=False,
            rstrip=False,
            normalized=False,
            single_word=False,
        )
    if isinstance(spec, dict):
        return AddedToken(
            spec["content"],
            lstrip=spec.get("lstrip", False),
            rstrip=spec.get("rstrip", False),
            normalized=spec.get("normalized", False),
            special=spec.get("special", False),
            single_word=spec.get("single_word", False),
        )
    raise ValueError(f"Unsupported spec type: {type(spec)}")
