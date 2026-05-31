# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import io
import cv2
import imageio
import requests
import time
import base64
import logging
from typing import List, Optional, Union
from random import random
from PIL import Image, ImageFile
from unitorch.cli import (
    config_defaults_init,
    register_process,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


class VideoProcessor:
    """Processor for video-related operations."""

    def __init__(
        self,
        video_type: Optional[str] = None,
        video_size: tuple = (256, 256),
        http_url: Optional[str] = None,
    ):
        self.video_type = video_type
        self.video_size = video_size
        self.http_url = http_url

    @classmethod
    @config_defaults_init("core/process/video")
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

    @register_process("core/process/video/read")
    def _read(
        self,
        video,
        video_type=None,
    ):
        """Read a video from a file path, base64/hex string, or HTTP URL."""
        video_type = video_type if video_type is not None else self.video_type
        try:
            if video_type == "base64":
                video = io.BytesIO(base64.b64decode(video))

                frames = []
                with imageio.v3.imopen(
                    video, "r", plugin="pyav", format="mp4"
                ) as reader:
                    for frame in reader.iter():
                        img = Image.fromarray(frame)
                        frames.append(img)
                return frames

            if video_type == "hex":
                video = io.BytesIO(bytes.fromhex(video))

                frames = []
                with imageio.v3.imopen(
                    video, "r", plugin="pyav", format="mp4"
                ) as reader:
                    for frame in reader.iter():
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
            with imageio.v3.imopen(video, "r", plugin="pyav", format="mp4") as reader:
                for frame in reader.iter():
                    img = Image.fromarray(frame)
                    frames.append(img)
            return frames

        except Exception:
            logging.debug(f"core/process/video/read use fake video for {video}")
            return []

    @register_process("core/process/video/sample")
    def _sample(
        self,
        video: Union[cv2.VideoCapture, str, List[Image.Image]],
        freq: Optional[int] = 0,
        num: Optional[int] = None,
        mode: Optional[str] = "random",
    ):
        """Sample up to num frames from a video, with optional stride (freq) and selection mode."""
        if isinstance(video, str):
            video = cv2.VideoCapture(video)

        if isinstance(video, cv2.VideoCapture):
            frames = []
            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                frames.append(pil_img)
        else:
            frames = video

        if freq > 0:
            frames = frames[:: (freq + 1)]

        if num is not None and len(frames) > num:
            if mode == "random":
                start = int(random() * (len(frames) - num))
                end = start + num
                frames = frames[start:end]
            elif mode == "first":
                frames = frames[:num]
            elif mode == "last":
                frames = frames[-num:]
            elif mode == "middle":
                start = (len(frames) - num) // 2
                end = start + num
                frames = frames[start:end]
            else:
                raise ValueError(f"Unknown sampling mode: {mode}")

        return frames
