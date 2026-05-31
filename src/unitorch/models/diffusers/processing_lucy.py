# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import cv2
from PIL import Image
from typing import List, Optional, Union

from unitorch.models import GenericOutputs
from unitorch.models.diffusers.processing_wan import WanProcessor


class LucyProcessor(WanProcessor):
    def _video_to_pixel_values(
        self,
        video: Union[cv2.VideoCapture, str, List[Image.Image]],
    ):
        frames = self.get_video_frames(video)

        pixel_values = []
        for frame in frames:
            if self.frame_processor is None:
                raise ValueError(
                    "frame_processor is None, please set video_size to process video"
                )
            if self.video_size is not None:
                width, height = frame.size
                scale = max(self.video_size[0] / width, self.video_size[1] / height)
                frame = frame.resize(
                    (round(width * scale), round(height * scale)),
                    resample=Image.LANCZOS,
                )
                frame = self.center_crop_processor(frame)
            else:
                width, height = frame.size
                new_width = width // self.divisor * self.divisor
                new_height = height // self.divisor * self.divisor
                frame = frame.resize((new_width, new_height), resample=Image.LANCZOS)
            pixel_values.append(self.frame_processor(frame))

        pixel_values = torch.stack(pixel_values, dim=0)
        return pixel_values.permute(1, 0, 2, 3)

    def video_editing(
        self,
        prompt: str,
        refer_video: Union[cv2.VideoCapture, str, List[Image.Image]],
        video: Union[cv2.VideoCapture, str, List[Image.Image]],
        max_seq_length: Optional[int] = None,
    ):
        prompt_outputs = self.classification(prompt, max_seq_length=max_seq_length)

        return GenericOutputs(
            pixel_values=self._video_to_pixel_values(video),
            refer_pixel_values=self._video_to_pixel_values(refer_video),
            input_ids=prompt_outputs.input_ids,
            attention_mask=prompt_outputs.attention_mask,
        )

    def video_editing_inputs(
        self,
        prompt: str,
        refer_video: Union[cv2.VideoCapture, str, List[Image.Image]],
        negative_prompt: Optional[str] = "",
        max_seq_length: Optional[int] = None,
    ):
        text_outputs = self.text2video_inputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            max_seq_length=max_seq_length,
        )

        return GenericOutputs(
            refer_pixel_values=self._video_to_pixel_values(refer_video),
            input_ids=text_outputs.input_ids,
            attention_mask=text_outputs.attention_mask,
            negative_input_ids=text_outputs.negative_input_ids,
            negative_attention_mask=text_outputs.negative_attention_mask,
        )
