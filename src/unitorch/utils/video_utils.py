# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import numpy as np
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union


def tensor2vid(video: torch.Tensor) -> List[np.ndarray]:
    # prepare the final outputs
    i, f, c, h, w = video.shape
    images = video.permute(1, 3, 0, 4, 2).reshape(
        f, h, i * w, c
    )  # 1st (frames, h, batch_size, w, c) 2nd (frames, h, batch_size * w, c)
    images = images.unbind(dim=0)  # prepare a list of indvidual (consecutive frames)
    images = [
        (image.cpu().numpy() * 255).astype("uint8") for image in images
    ]  # f h w c
    return images
