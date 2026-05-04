# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import List

import numpy as np
import torch


def tensor2vid(video: torch.Tensor) -> List[np.ndarray]:
    """Convert a batched video tensor to a list of per-frame uint8 NumPy arrays.

    Args:
        video: Float tensor of shape ``(batch, frames, channels, height, width)``
               with values in ``[0, 1]``.

    Returns:
        A list of ``frames`` arrays, each of shape
        ``(height, batch * width, channels)`` and dtype ``uint8``.
    """
    batch, frames, channels, height, width = video.shape
    # Rearrange to (frames, height, batch * width, channels)
    images = video.permute(1, 3, 0, 4, 2).reshape(frames, height, batch * width, channels)
    return [(frame.cpu().numpy() * 255).astype("uint8") for frame in images.unbind(dim=0)]
