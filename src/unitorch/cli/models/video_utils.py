# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import io
import cv2
import imageio
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
from unitorch.utils import is_opencv_available
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


class VideoProcessor:
    """
    Processor for image-related operations.
    """

    def __init__(
        self,
        video_type: Optional[str] = None,
        video_size: tuple = (256, 256),
        http_url: Optional[str] = None,
    ):
        """
        Initializes a new instance of the ImageProcessor.

        Args:
            image_type (Optional[str]): The type of the image. Defaults to None.
            image_size (tuple): The size of the image. Defaults to (256, 256).
            http_url (Optional[str]): The URL for fetching images. Defaults to None.
        """
        self.video_type = video_type
        self.video_size = video_size
        self.http_url = http_url

    @classmethod
    @add_default_section_for_init("core/process/video")
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

    @register_process("core/process/video/read")
    def _read(
        self,
        video,
        video_type=None,
    ):
        """
        Reads and processes an image.

        Args:
            image: The image to read and process.
            image_type (Optional[str]): The type of the image. Defaults to None.

        Returns:
            The processed image as a PIL Image object.
        """
        video_type = video_type if video_type is not None else self.video_type
        try:
            if video_type == "base64":
                video = io.BytesIO(base64.b64decode(video))

                frames = []
                with imageio.v3.imopen(video, plugin="ffmpeg", format="mp4") as reader:
                    for frame in reader:
                        img = Image.fromarray(frame)
                        frames.append(img)
                return frames

            if video_type == "hex":
                video = io.BytesIO(bytes.fromhex(video))

                frames = []
                with imageio.v3.imopen(video, plugin="ffmpeg", format="mp4") as reader:
                    for frame in reader:
                        img = Image.fromarray(frame)
                        frames.append(img)
                return frames

            if self.http_url is None:
                video = cv2.VideoCapture(video)
                return video

            url = self.http_url.format(video)
            doc = self._request_url(url)
            if doc.status_code != 200 or doc.content == b"":
                raise ValueError(f"can't find the video {video}")

            video = io.BytesIO(doc.content)
            frames = []
            with imageio.v3.imopen(video, plugin="ffmpeg", format="mp4") as reader:
                for frame in reader:
                    img = Image.fromarray(frame)
                    frames.append(img)
            return frames

        except:
            logging.debug(f"core/process/video/read use fake video for {video}")
            return []
