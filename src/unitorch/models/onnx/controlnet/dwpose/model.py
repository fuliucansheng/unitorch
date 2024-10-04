# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import cv2
import numpy as np
import onnxruntime
from unitorch.models import GenericOutputs
from unitorch.models.onnx import GenericOnnxModel
from unitorch.models.onnx.controlnet.dwpose.utils import nms, multiclass_nms


class DWPoseOnnxModel(GenericOnnxModel):
    def __init__(self):
        super().__init__()
        onnx_det_path = "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx"
        onnx_pose_path = (
            "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx"
        )
        self.det = onnxruntime.InferenceSession(onnx_det_path, providers=self.providers)
        self.pose = onnxruntime.InferenceSession(
            onnx_pose_path, providers=self.providers
        )

    def process(image, input_size=(640, 640), swap=(2, 0, 1)):
        if len(image.shape) == 3:
            padded_image = (
                np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
            )
        else:
            padded_image = np.ones(input_size, dtype=np.uint8) * 114

        r = min(input_size[0] / image.shape[0], input_size[1] / image.shape[1])
        resized_image = cv2.resize(
            image,
            (int(image.shape[1] * r), int(image.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_image[
            : int(image.shape[0] * r), : int(image.shape[1] * r)
        ] = resized_image

        padded_image = padded_image.transpose(swap)
        padded_image = np.ascontiguousarray(padded_image, dtype=np.float32)
        return GenericOutputs(pixel_values=padded_image, ratios=r)

    def forward(self, pixel_values, ratios):
        for pixel_value, ratio in zip(pixel_values, ratios):
            output = self.det.run(None, {"images": pixel_value.unsqueeze(0).numpy()})[0]

            grids = []
            expanded_strides = []
            strides = [8, 16, 32]

            hsizes = [640 // stride for stride in strides]
            wsizes = [640 // stride for stride in strides]

            for hsize, wsize, stride in zip(hsizes, wsizes, strides):
                xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
                grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
                grids.append(grid)
                shape = grid.shape[:2]
                expanded_strides.append(np.full((*shape, 1), stride))

            grids = np.concatenate(grids, 1)
            expanded_strides = np.concatenate(expanded_strides, 1)
            output[..., :2] = (output[..., :2] + grids) * expanded_strides
            output[..., 2:4] = np.exp(output[..., 2:4]) * expanded_strides

            predictions = output[0]

            boxes = predictions[:, :4]
            scores = predictions[:, 4:5] * predictions[:, 5:]

            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
            boxes_xyxy /= ratio
            dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
            if dets is not None:
                final_boxes, final_scores, final_cls_inds = (
                    dets[:, :4],
                    dets[:, 4],
                    dets[:, 5],
                )
                isscore = final_scores > 0.3
                iscat = final_cls_inds == 0
                isbbox = [i and j for (i, j) in zip(isscore, iscat)]
                final_boxes = final_boxes[isbbox]
            else:
                final_boxes = np.array([])

            h, w = 384, 288
            model_input_size = (w, h)
            resized_img, center, scale = preprocess(oriImg, out_bbox, model_input_size)
            outputs = self.pose.run(None, {"input": resized_img})
            keypoints, scores = postprocess(outputs, model_input_size, center, scale)
