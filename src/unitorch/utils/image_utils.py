# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image


def make_grid(
    images: List[Image.Image],
    rows: int,
    cols: int,
    resize: Optional[List[int]] = None,
) -> Image.Image:
    """Arrange a list of images into a grid.

    Args:
        images: PIL images to arrange; must contain exactly ``rows * cols`` items.
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
        resize: ``[width, height]`` to resize each cell. Defaults to the size of
                the first image.

    Returns:
        A single PIL image containing all inputs arranged in the grid.

    Raises:
        AssertionError: If ``len(images) != rows * cols``.
    """
    assert len(images) == rows * cols, (
        "Number of images must equal rows * cols."
    )
    w, h = resize if resize is not None else images[0].size
    grid = Image.new("RGB", (w * cols, h * rows))
    for i, image in enumerate(images):
        grid.paste(image, box=(w * (i % cols), h * (i // cols)))
    return grid


def resize_shortest_edge(
    image: Image.Image,
    short_size: List[int],
    max_size: int,
) -> Image.Image:
    """Resize *image* so its shortest edge fits within *short_size* and no edge exceeds *max_size*.

    Args:
        image: Input PIL image.
        short_size: ``[min, max]`` range for the shortest-edge length.
        max_size: Hard upper bound for any edge length after scaling.

    Returns:
        Resized PIL image.
    """
    w, h = image.size
    shortest = min(w, h)
    scale = min(short_size[0] / shortest, short_size[1] / shortest)
    if scale * shortest > max_size:
        scale = max_size / shortest
    return image.resize((int(w * scale), int(h * scale)), resample=Image.LANCZOS)


def image_list_to_tensor(
    images: List[torch.Tensor],
    size_divisibility: int = 0,
    pad_value: float = 0.0,
    padding_constraints: Optional[Dict[str, int]] = None,
) -> torch.Tensor:
    """Batch a list of image tensors into a single padded tensor.

    All tensors are right/bottom-padded to the maximum observed spatial size.
    Optionally rounds dimensions up to the nearest multiple of *size_divisibility*.

    Args:
        images: List of ``(C, H, W)`` tensors to batch.
        size_divisibility: When > 0, pad spatial dimensions to a multiple of this value.
        pad_value: Fill value used for padding.
        padding_constraints: Unused minimum padding per side; defaults to all zeros.

    Returns:
        A ``(N, C, H, W)`` tensor containing all images.
    """
    if padding_constraints is None:
        padding_constraints = {"top": 0, "bottom": 0, "left": 0, "right": 0}

    max_size = list(max(s) for s in zip(*[img.shape for img in images]))
    if size_divisibility > 0:
        max_size = [
            (s + size_divisibility - 1) // size_divisibility * size_divisibility
            for s in max_size
        ]

    batched = images[0].new_full((len(images), *max_size), pad_value)
    for img, slot in zip(images, batched):
        slot[: img.shape[0], : img.shape[1], : img.shape[2]] = img
    return batched


def numpy_to_pil(
    images: "np.ndarray",
) -> Union[Image.Image, List[Image.Image]]:
    """Convert a NumPy array (single image or batch) to PIL image(s).

    Args:
        images: Float array with values in ``[0, 1]``. Shape ``(H, W, C)`` for a
                single image or ``(N, H, W, C)`` for a batch.

    Returns:
        A single :class:`PIL.Image.Image` for a single input, or a list of them
        for a batch.
    """
    images = (images * 255).round().astype("uint8")

    if images.ndim <= 3:
        if images.shape[-1] == 1:
            return Image.fromarray(images.squeeze(), mode="L")
        return Image.fromarray(images)

    if images.shape[-1] == 1:
        return [Image.fromarray(img.squeeze(), mode="L") for img in images]
    return [Image.fromarray(img) for img in images]
