# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import io
import requests
import torch
from PIL import Image
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import unitorch.utils.palette


def make_grid(
    images: List[Image.Image], rows: int, cols: int, resize: Optional[List[int]] = None
) -> Image.Image:
    """
    Combines a list of images into a grid layout.

    Args:
        images (List[Image.Image]): List of PIL Images to combine into a grid.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
        resize (Optional[List[int]], optional): Size to resize the images. Defaults to None.

    Returns:
        Image.Image: PIL Image object representing the grid layout.

    Raises:
        AssertionError: If the number of images is not equal to the product of rows and cols.
    """
    assert (
        len(images) == rows * cols
    ), "Number of images must be equal to the product of rows and cols."

    if resize is not None:
        w, h = resize
    else:
        w, h = images[0].size

    grid = Image.new("RGB", (w * cols, h * rows))
    for i, image in enumerate(images):
        grid.paste(image, box=(w * (i % cols), h * (i // cols)))

    return grid


def resize_shortest_edge(
    image: Image.Image,
    short_size: List[int],
    max_size: int,
):
    """
    Resize the image to the given short size and maximum size.

    Args:
        image (Image.Image): Input image.
        short_size (List[int]): Shortest edge size.
        max_size (int): Maximum size.

    Returns:
        Image.Image: Resized image.
    """
    w, h = image.size
    size = min(w, h)
    scale = min(short_size[0] / size, short_size[1] / size)
    if scale * size > max_size:
        scale = max_size / size
    new_w, new_h = int(w * scale), int(h * scale)
    return image.resize((new_w, new_h), Image.BILINEAR)


def image_list_to_tensor(
    images: List[torch.Tensor],
    size_divisibility: int = 0,
    pad_value: float = 0.0,
    padding_constraints: Optional[Dict[str, int]] = None,
) -> torch.Tensor:
    """
    Convert a list of images to a tensor.

    Args:
        images (List[torch.Tensor]): List of images to convert to a tensor.
        size_divisibility (int, optional): Size divisibility. Defaults to 0.
        pad_value (float, optional): Padding value. Defaults to 0.0.
        padding_constraints (Optional[Dict[str, int]], optional): Padding constraints. Defaults to None.

    Returns:
        torch.Tensor: Tensor representing the images.
    """
    if padding_constraints is None:
        padding_constraints = {"top": 0, "bottom": 0, "left": 0, "right": 0}
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    if size_divisibility > 0:
        stride = size_divisibility
        max_size = [(s + stride - 1) // stride * stride for s in max_size]
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new_full(batch_shape, pad_value)
    for img, pad in zip(images, batched_imgs):
        pad[
            : img.shape[0],
            : img.shape[1],
            : img.shape[2],
        ] = img
    return batched_imgs


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim <= 3:
        # single image
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            return Image.fromarray(images.squeeze(), mode="L")
        else:
            return Image.fromarray(images)

    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images
