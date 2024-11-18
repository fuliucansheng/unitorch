# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import cv2
import torch
import numpy as np
from PIL import Image
import onnxruntime as ort
from unitorch.cli import cached_path, hf_endpoint_url
import unitorch.cli.pipelines.tools.controlnet.dwpose.utils as utils
from unitorch.cli.pipelines.tools.controlnet.dwpose.onnxdet import inference_detector
from unitorch.cli.pipelines.tools.controlnet.dwpose.onnxpose import inference_pose


class Wholebody:
    def __init__(self):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if torch.cuda.is_available()
            else ["CPUExecutionProvider"]
        )
        onnx_det_path = hf_endpoint_url("/yzd-v/DWPose/resolve/main/yolox_l.onnx")
        onnx_pose_path = hf_endpoint_url(
            "/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx"
        )

        self.session_det = ort.InferenceSession(
            path_or_bytes=cached_path(onnx_det_path), providers=providers
        )
        self.session_pose = ort.InferenceSession(
            path_or_bytes=cached_path(onnx_pose_path), providers=providers
        )

    def __call__(self, oriImg):
        det_result = inference_detector(self.session_det, oriImg)
        keypoints, scores = inference_pose(self.session_pose, det_result, oriImg)

        keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)
        # compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        # neck score when visualizing pred
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.3, keypoints_info[:, 6, 2:4] > 0.3
        ).astype(int)
        new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)
        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        keypoints, scores = keypoints_info[..., :2], keypoints_info[..., 2]

        return keypoints, scores


def draw_pose(pose, H, W):
    bodies = pose["bodies"]
    faces = pose["faces"]
    hands = pose["hands"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    canvas = utils.draw_bodypose(canvas, candidate, subset)
    canvas = utils.draw_handpose(canvas, hands)
    canvas = utils.draw_facepose(canvas, faces)
    return canvas


class DWposeDetector:
    def __init__(self):
        self.pose_estimation = Wholebody()

    @torch.no_grad()
    def __call__(self, oriImg):
        oriImg = cv2.cvtColor(np.array(oriImg), cv2.COLOR_RGB2BGR)
        H, W, C = oriImg.shape
        candidate, subset = self.pose_estimation(oriImg)
        nums, keys, locs = candidate.shape
        candidate[..., 0] /= float(W)
        candidate[..., 1] /= float(H)
        body = candidate[:, :18].copy()
        body = body.reshape(nums * 18, locs)
        score = subset[:, :18]
        for i in range(len(score)):
            for j in range(len(score[i])):
                if score[i][j] > 0.3:
                    score[i][j] = int(18 * i + j)
                else:
                    score[i][j] = -1

        un_visible = subset < 0.3
        candidate[un_visible] = -1

        foot = candidate[:, 18:24]

        faces = candidate[:, 24:92]

        hands = candidate[:, 92:113]
        hands = np.vstack([hands, candidate[:, 113:]])

        bodies = dict(candidate=body, subset=score)
        pose = dict(bodies=bodies, hands=hands, faces=faces)

        results = draw_pose(pose, H, W)
        return Image.fromarray(cv2.cvtColor(results, cv2.COLOR_BGR2RGB))


dwpose_pipe = None


def dwpose(image: Image.Image):
    global dwpose_pipe
    if dwpose_pipe is None:
        dwpose_pipe = DWposeDetector()
    return dwpose_pipe(image)
