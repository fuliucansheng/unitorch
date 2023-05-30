# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

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
