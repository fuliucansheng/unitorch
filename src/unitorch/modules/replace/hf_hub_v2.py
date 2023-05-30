# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import requests
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from datasets import config
from huggingface_hub.hf_api import (
    HfApi,
    DatasetInfo,
    hf_raise_for_status,
    quote,
    validate_hf_hub_args,
)
from unitorch.utils.decorators import replace


@replace(HfApi)
class HfApiV2(HfApi):
    @validate_hf_hub_args
    def dataset_info(
        self,
        repo_id: str,
        *,
        revision: Optional[str] = None,
        timeout: Optional[float] = None,
        files_metadata: Optional[bool] = False,
        token: Optional[Union[bool, str]] = None,
    ) -> DatasetInfo:
        """
        Get info on one specific dataset on huggingface.co.
        Dataset can be private if you pass an acceptable token.
        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            revision (`str`, *optional*):
                The revision of the dataset repository from which to get the
                information.
            token (`str`, *optional*):
                Deprecated in favor of `use_auth_token`. Will be removed in 0.12.0.
                An authentication token (See https://huggingface.co/settings/token)
            timeout (`float`, *optional*):
                Whether to set a timeout for the request to the Hub.
            files_metadata (`bool`, *optional*):
                Whether or not to retrieve metadata for files in the repository
                (size, LFS metadata, etc). Defaults to `False`.
            use_auth_token (`bool` or `str`, *optional*):
                Whether to use the `auth_token` provided from the
                `huggingface_hub` cli. If not logged in, a valid `auth_token`
                can be passed in as a string.
        Returns:
            [`hf_api.DatasetInfo`]: The dataset repository information.
        <Tip>
        Raises the following errors:
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.
            - [`~utils.RevisionNotFoundError`]
              If the revision to download from cannot be found.
        </Tip>
        """
        headers = self._build_hf_headers(token=token)
        params = {}
        if files_metadata:
            params["blobs"] = True
        hub_endpoint = os.environ.get("UNITORCH_HUB_ENDPOINT", None)
        if hub_endpoint is not None:
            try:
                path = (
                    f"{hub_endpoint}/api/datasets/{repo_id}"
                    if revision is None
                    else (
                        f"{hub_endpoint}/api/datasets/{repo_id}/revision/{quote(revision, safe='')}"
                    )
                )
                r = requests.get(path, headers=headers, params=params, timeout=timeout)
                config.HF_ENDPOINT = hub_endpoint
                config.HUB_DATASETS_URL = (
                    config.HF_ENDPOINT + "/datasets/{path}/resolve/{revision}/{name}"
                )
                config.HUB_DEFAULT_VERSION = "master"
            except:
                path = (
                    f"{self.endpoint}/api/datasets/{repo_id}"
                    if revision is None
                    else (
                        f"{self.endpoint}/api/datasets/{repo_id}/revision/{quote(revision, safe='')}"
                    )
                )
                r = requests.get(path, headers=headers, params=params, timeout=timeout)
        else:
            path = (
                f"{self.endpoint}/api/datasets/{repo_id}"
                if revision is None
                else (
                    f"{self.endpoint}/api/datasets/{repo_id}/revision/{quote(revision, safe='')}"
                )
            )
            r = requests.get(path, headers=headers, params=params, timeout=timeout)

        hf_raise_for_status(r)
        d = r.json()
        return DatasetInfo(**d)
