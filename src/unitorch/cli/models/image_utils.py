# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import io
import requests
import time
import base64
import json
import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from random import random
from PIL import Image, ImageOps, ImageFile, ImageFilter

from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageProcessor:
    """
    Processor for image-related operations.
    """

    def __init__(
        self,
        image_type: Optional[str] = None,
        image_size: tuple = (256, 256),
        http_url: Optional[str] = None,
    ):
        """
        Initializes a new instance of the ImageProcessor.

        Args:
            image_type (Optional[str]): The type of the image. Defaults to None.
            image_size (tuple): The size of the image. Defaults to (256, 256).
            http_url (Optional[str]): The URL for fetching images. Defaults to None.
        """
        self.image_type = image_type
        self.image_size = image_size
        self.http_url = http_url

    @classmethod
    @add_default_section_for_init("core/process/image")
    def from_core_configure(cls, config, **kwargs):
        """
        Creates a new instance of the ImageProcessor using the configuration from the core.

        Args:
            config: The configuration object.
            **kwargs: Additional keyword arguments.

        Returns:
            An instance of the ImageProcessor.
        """
        pass

    def _request_url(self, url):
        """
        Sends an HTTP request to the specified URL and returns the response.

        Args:
            url (str): The URL to request.

        Returns:
            The response received from the URL.
        """
        while True:
            try:
                doc = requests.get(url, timeout=600)
                return doc
            except:
                time.sleep(random() * 2)

    @register_process("core/process/image/read")
    def _read(
        self,
        image,
        image_type=None,
    ):
        """
        Reads and processes an image.

        Args:
            image: The image to read and process.
            image_type (Optional[str]): The type of the image. Defaults to None.

        Returns:
            The processed image as a PIL Image object.
        """
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
        except:
            logging.debug(f"core/process/image/read use fake image for {image}")
            return Image.new("RGB", self.image_size, (255, 255, 255))

    @register_process("core/process/image/translate")
    def _translate(
        self,
        image: Image.Image,
        up: Optional[int] = 0,
        left: Optional[int] = 0,
    ):
        """
        Translates the image by the specified amount.

        Args:
            image (Image.Image): The image to translate.
            up (Optional[int]): The amount to move the image upwards. Defaults to 0.
            left (Optional[int]): The amount to move the image to the left. Defaults to 0.

        Returns:
            The translated image as a PIL Image object.
        """
        return image.transform(image.size, Image.AFFINE, (1, 0, left, 0, 1, up))

    @register_process("core/process/image/scale")
    def _scale(
        self,
        image: Image.Image,
        scale: Optional[float] = 0.0,
    ):
        """
        Scales the image by the specified factor.

        Args:
            image (Image.Image): The image to scale.
            scale (Optional[float]): The scale factor. Defaults to 0.0.

        Returns:
            The scaled image as a PIL Image object.
        """
        return ImageOps.scale(image, scale)

    @register_process("core/process/image/rotate")
    def _rotate(
        self,
        image: Image.Image,
        degree: Optional[int] = 0.0,
    ):
        """
        Rotates the image by the specified degree.

        Args:
            image (Image.Image): The image to rotate.
            degree (Optional[int]): The degree of rotation. Defaults to 0.0.

        Returns:
            The rotated image as a PIL Image object.
        """
        return image.rotate(degree, Image.NEAREST, expand=1)

    @register_process("core/process/image/flip")
    def _flip(
        self,
        image: Image.Image,
        horizontal: Optional[bool] = False,
        vertical: Optional[bool] = False,
    ):
        """
        Flips the image horizontally or vertically.

        Args:
            image (Image.Image): The image to flip.
            horizontal (Optional[bool]): Whether to flip the image horizontally. Defaults to False.
            vertical (Optional[bool]): Whether to flip the image vertically. Defaults to False.

        Returns:
            The flipped image as a PIL Image object.
        """
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
        """
        Resizes the image to the specified size.

        Args:
            image (Image.Image): The image to resize.
            size (Optional[Tuple[int, int]]): The size to resize to. Defaults to (256, 256).

        Returns:
            The resized image as a PIL Image object.
        """
        return image.resize(size, Image.BICUBIC)

    @register_process("core/process/image/canny")
    def _canny(
        self,
        image: Image.Image,
    ):
        """
        Detects edges in the image using the Canny algorithm.

        Args:
            image (Image.Image): The image to detect edges in.

        Returns:
            The image with detected edges as a PIL Image object.
        """
        return image.filter(ImageFilter.FIND_EDGES)
