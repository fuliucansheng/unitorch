# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import re
import sys
import platform
import logging
import torch
from itertools import chain
from setuptools import find_packages, setup

enabled_extensions = os.environ.get("UNITORCH_EXTENSIONS", None)

if enabled_extensions is not None:
    enabled_extensions = re.split(";|,| ", enabled_extensions)
    enabled_extensions = [ext.upper() for ext in enabled_extensions]
else:
    enabled_extensions = []

if torch.cuda.is_available():
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    ngram_cuda_extension = CUDAExtension(
        "unitorch.clib.ngram_repeat_block_cuda",
        [
            "unitorch/clib/cuda/ngram_repeat_block/module.cpp",
            "unitorch/clib/cuda/ngram_repeat_block/kernel.cu",
        ],
    )

    all_extensions = {
        "NGRAM": ngram_cuda_extension,
    }

else:
    all_extensions = {}


setup(
    ext_modules=list(
        chain(
            [all_extensions[ext] for ext in enabled_extensions if ext in all_extensions]
        )
    ),
)
