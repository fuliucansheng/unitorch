# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import io
import requests
import time
import base64
import logging
import numpy as np
from typing import Optional, Tuple
from random import random
from PIL import Image, ImageOps, ImageFile, ImageFilter
from unitorch.utils import is_opencv_available
from unitorch.cli import (
    config_defaults_init,
    register_process,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageProcessor:
    """Processor for image-related operations."""

    def __init__(
        self,
        image_type: Optional[str] = None,
        image_size: tuple = (256, 256),
        http_url: Optional[str] = None,
    ):
        self.image_type = image_type
        self.image_size = image_size
        self.http_url = http_url

    @classmethod
    @config_defaults_init("core/process/image")
    def from_config(cls, config, **kwargs):
        pass

    def _request_url(self, url):
        """Retry-loop GET request returning the response."""
        while True:
            try:
                doc = requests.get(url, timeout=600)
                return doc
            except Exception:
                time.sleep(random() * 2)

    @register_process("core/process/image/read")
    def _read(
        self,
        image,
        image_type=None,
    ):
        """Read an image from a file path, base64/hex string, or HTTP URL."""
        image_type = image_type if image_type is not None else self.image_type
        try:
            if image_type == "base64":
                image = io.BytesIO(base64.b64decode(image))
                return Image.open(image).convert("RGB")

            if image_type == "hex":
                image = io.BytesIO(bytes.fromhex(image))
                return Image.open(image).convert("RGB")

            if self.http_url is None:
                return Image.open(image).convert("RGB")

            url = self.http_url.format(image)
            doc = self._request_url(url)
            if doc.status_code != 200 or doc.content == b"":
                raise ValueError(f"can't find the image {image}")

            return Image.open(io.BytesIO(doc.content)).convert("RGB")
        except Exception:
            logging.debug(f"core/process/image/read use fake image for {image}")
            return Image.new("RGB", self.image_size, (255, 255, 255))

    @register_process("core/process/image/translate")
    def _translate(
        self,
        image: Image.Image,
        up: Optional[int] = 0,
        left: Optional[int] = 0,
    ):
        """Translate the image by (left, up) pixels using an affine transform."""
        return image.transform(image.size, Image.AFFINE, (1, 0, left, 0, 1, up))

    @register_process("core/process/image/scale")
    def _scale(
        self,
        image: Image.Image,
        scale: Optional[float] = 0.0,
    ):
        """Scale the image by the given factor."""
        return ImageOps.scale(image, scale)

    @register_process("core/process/image/rotate")
    def _rotate(
        self,
        image: Image.Image,
        degree: Optional[int] = 0.0,
    ):
        """Rotate the image by the given degree with expansion."""
        return image.rotate(degree, Image.NEAREST, expand=1)

    @register_process("core/process/image/flip")
    def _flip(
        self,
        image: Image.Image,
        horizontal: Optional[bool] = False,
        vertical: Optional[bool] = False,
    ):
        """Flip the image horizontally and/or vertically."""
        if horizontal:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if vertical:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        return image

    @register_process("core/process/image/resize")
    def _resize(
        self,
        image: Image.Image,
        size: Optional[Tuple[int, int]] = (256, 256),
    ):
        """Resize the image using Lanczos resampling."""
        return image.resize(size, resample=Image.LANCZOS)

    @register_process("core/process/image/canny")
    def _canny(
        self,
        image: Image.Image,
    ):
        """Detect edges using Canny (OpenCV if available, otherwise PIL FIND_EDGES)."""
        if is_opencv_available():
            import cv2

            image = np.array(image, np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.Canny(image, 100, 200)
            image = Image.fromarray(image)
        else:
            image = image.convert("L")
            image = image.filter(ImageFilter.FIND_EDGES)
        return image

    @register_process("core/process/image/mask")
    def _mask(
        self,
        image: Image.Image,
        mask: Image.Image,
    ):
        result = Image.new("RGBA", image.size, (0, 0, 0, 0))
        mask = mask.convert("L").resize(image.size, resample=Image.LANCZOS)
        result.paste(image, (0, 0), mask)
        return result

    @register_process("core/process/image/dilate")
    def _dilate(
        self,
        image: Image.Image,
        kernel_size: Optional[int] = 3,
        iterations: Optional[int] = 1,
    ):
        """Dilate the image using an ellipse kernel (requires OpenCV)."""
        if is_opencv_available():
            import cv2

            image = np.array(image, np.uint8)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
            image = cv2.dilate(image, kernel, iterations=iterations)
            return Image.fromarray(image)
        else:
            raise NotImplementedError("Dilate operation requires OpenCV.")

    @register_process("core/process/image/crop")
    def _crop(
        self,
        image: Image.Image,
        box: Tuple[int, int, int, int],
    ):
        """Crop the image to the given (left, upper, right, lower) box."""
        return image.crop(box)

    @register_process("core/process/image/center_crop")
    def _center_crop(
        self,
        image: Image.Image,
        size: Optional[Tuple[int, int]] = (224, 224),
    ):
        """Crop the image to the given size from the center."""
        width, height = image.size
        left = (width - size[0]) // 2
        top = (height - size[1]) // 2
        left, top = max(0, left), max(0, top)
        right = left + size[0]
        bottom = top + size[1]
        right, bottom = min(width, right), min(height, bottom)
        return image.crop((left, top, right, bottom))
