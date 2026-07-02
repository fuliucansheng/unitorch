# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import io
from typing import Any, Dict, List, Optional, Union

import imageio
import torch
from fastapi import APIRouter, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from torch import autocast

from unitorch.models.diffusers import LucyForVideoEditingGeneration, LucyProcessor
from unitorch.utils import is_bfloat16_available, nested_dict_value, pop_value
from unitorch.utils import tensor2vid
from unitorch.cli import (
    Config,
    GenericFastAPI,
    cached_path,
    config_defaults_init,
    config_defaults_method,
    register_fastapi,
)
from unitorch.cli.models.diffusion_utils import export_to_video
from unitorch.cli.models.diffusers import (
    load_weight,
    pretrained_stable_extensions_infos,
    pretrained_stable_infos,
)
from unitorch.cli.models.diffusers.modeling_lucy import (
    _lucy_model_kwargs,
    _lucy_state_dict,
)
from unitorch.cli import PipelineReplicaPool


class LucyForVideoEditingFastAPIPipeline(LucyForVideoEditingGeneration):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        vocab_path: str,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        seed: Optional[int] = 1123,
        gradient_checkpointing: Optional[bool] = False,
        expand_timesteps: Optional[bool] = True,
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        lora_checkpoints: Optional[Union[str, List[str]]] = None,
        lora_weights: Optional[Union[float, List[float]]] = 1.0,
        lora_alphas: Optional[Union[float, List[float]]] = 32,
        device: Optional[Union[str, int]] = "cpu",
        enable_cpu_offload: Optional[bool] = False,
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            seed=seed,
            gradient_checkpointing=gradient_checkpointing,
            expand_timesteps=expand_timesteps,
        )
        self.processor = LucyProcessor(
            vocab_path=vocab_path,
            vae_config_path=vae_config_path,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        if state_dict is not None:
            self.from_pretrained(state_dict=state_dict)
        else:
            self.from_pretrained(weight_path)
        self.eval()

        if lora_checkpoints is not None:
            self.load_lora_weights(
                lora_checkpoints,
                lora_weights=lora_weights,
                lora_alphas=lora_alphas,
                save_base_state=False,
            )

        self._enable_cpu_offload = enable_cpu_offload
        if self._enable_cpu_offload and self._device != "cpu":
            self.pipeline.enable_model_cpu_offload(self._device)
        else:
            self.to(device=self._device)

    @classmethod
    @config_defaults_init("core/fastapi/pipeline/lucy/video_editing")
    def from_config(
        cls,
        config,
        pretrained_name: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = None,
        pretrained_lora_names: Optional[Union[str, List[str]]] = None,
        pretrained_lora_weights_path: Optional[Union[str, List[str]]] = None,
        pretrained_lora_weights: Optional[Union[float, List[float]]] = None,
        pretrained_lora_alphas: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ):
        section = "core/fastapi/pipeline/lucy/video_editing"
        model_kwargs = _lucy_model_kwargs(
            config,
            section,
            pretrained_name=pretrained_name,
        )
        pretrained_infos = model_kwargs.pop("pretrained_infos")
        use_auth_token = model_kwargs.pop("use_auth_token")

        vocab_path = config.getoption("vocab_path", None)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrained_infos, "text", "vocab"),
        )
        vocab_path = cached_path(vocab_path, use_auth_token=use_auth_token)

        weight_path = pretrained_weight_path or config.getoption(
            "pretrained_weight_path", None
        )
        device = config.getoption("device", "cpu") if device is None else device
        enable_cpu_offload = config.getoption("enable_cpu_offload", True)

        state_dict = None
        if weight_path is None:
            state_dict = _lucy_state_dict(pretrained_infos, use_auth_token)
        else:
            state_dict = load_weight(weight_path, use_auth_token=use_auth_token)

        pretrained_lora_names = pretrained_lora_names or config.getoption(
            "pretrained_lora_names", None
        )
        pretrained_lora_weights = pretrained_lora_weights or config.getoption(
            "pretrained_lora_weights", 1.0
        )
        pretrained_lora_alphas = pretrained_lora_alphas or config.getoption(
            "pretrained_lora_alphas", 32.0
        )

        if (
            isinstance(pretrained_lora_names, str)
            and pretrained_lora_weights_path is None
        ):
            pretrained_lora_weights_path = nested_dict_value(
                pretrained_stable_extensions_infos,
                pretrained_lora_names,
                "lora",
                "weight",
            )
        elif (
            isinstance(pretrained_lora_names, list)
            and pretrained_lora_weights_path is None
        ):
            pretrained_lora_weights_path = [
                nested_dict_value(
                    pretrained_stable_extensions_infos,
                    name,
                    "lora",
                    "weight",
                )
                for name in pretrained_lora_names
            ]

        lora_weights_path = pretrained_lora_weights_path or config.getoption(
            "pretrained_lora_weights_path", None
        )

        return cls(
            config_path=model_kwargs["config_path"],
            text_config_path=model_kwargs["text_config_path"],
            vae_config_path=model_kwargs["vae_config_path"],
            scheduler_config_path=model_kwargs["scheduler_config_path"],
            vocab_path=vocab_path,
            num_train_timesteps=model_kwargs["num_train_timesteps"],
            num_infer_timesteps=model_kwargs["num_infer_timesteps"],
            seed=model_kwargs["seed"],
            gradient_checkpointing=model_kwargs["gradient_checkpointing"],
            expand_timesteps=model_kwargs["expand_timesteps"],
            weight_path=weight_path,
            state_dict=state_dict,
            lora_checkpoints=lora_weights_path,
            lora_weights=pretrained_lora_weights,
            lora_alphas=pretrained_lora_alphas,
            device=device,
            enable_cpu_offload=enable_cpu_offload,
        )

    @torch.no_grad()
    @autocast(
        device_type=("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=(torch.bfloat16 if is_bfloat16_available() else torch.float32),
    )
    @config_defaults_method("core/fastapi/pipeline/lucy/video_editing")
    def __call__(
        self,
        text: str,
        video: List[Image.Image],
        neg_text: Optional[str] = "",
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames: Optional[int] = 81,
        num_fps: Optional[int] = 16,
        guidance_scale: Optional[float] = 5.0,
        num_timesteps: Optional[int] = 50,
        seed: Optional[int] = 1123,
    ):
        inputs = self.processor.video_editing_inputs(
            text,
            video,
            negative_prompt=neg_text,
            max_seq_length=512,
        )
        self.seed = seed

        inputs = {k: v.unsqueeze(0) if v is not None else v for k, v in inputs.items()}
        inputs = {
            k: v.to(device=self.device) if v is not None else v
            for k, v in inputs.items()
        }

        prompt_outputs = self.get_prompt_outputs(
            input_ids=inputs["input_ids"],
            negative_input_ids=inputs["negative_input_ids"],
            attention_mask=inputs["attention_mask"],
            negative_attention_mask=inputs["negative_attention_mask"],
            enable_cpu_offload=self._enable_cpu_offload,
            cpu_offload_device=self._device,
        )

        outputs = self.pipeline(
            video=inputs["refer_pixel_values"].permute(0, 2, 1, 3, 4),
            prompt_embeds=prompt_outputs.prompt_embeds,
            negative_prompt_embeds=prompt_outputs.negative_prompt_embeds,
            generator=torch.Generator(device=self.pipeline.device).manual_seed(seed),
            num_inference_steps=num_timesteps,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            output_type="pt",
        )

        frames = tensor2vid(outputs.frames.float())
        return export_to_video(frames, fps=num_fps)


@register_fastapi("core/fastapi/lucy/video_editing")
class LucyForVideoEditingFastAPI(GenericFastAPI):
    def __init__(self, config: Config):
        self.config = config
        config.set_default_section("core/fastapi/lucy/video_editing")
        self._section = "core/fastapi/lucy/video_editing"
        router = config.getoption("router", "/core/fastapi/lucy/video_editing")
        self._pipes = None
        self._router = APIRouter(prefix=router)
        self._router.add_api_route("/generate", self.generate, methods=["POST"])
        self._router.add_api_route("/status", self.status, methods=["GET"])
        self._router.add_api_route("/start", self.start, methods=["POST"])
        self._router.add_api_route("/stop", self.stop, methods=["GET"])

    @property
    def router(self):
        return self._router

    def start(
        self,
        pretrained_name: Optional[str] = "lucy-edit-v1.1-dev",
        pretrained_lora_names: Optional[Union[str, List[str]]] = None,
        pretrained_lora_weights: Optional[Union[float, List[float]]] = 1.0,
        pretrained_lora_alphas: Optional[Union[float, List[float]]] = 32.0,
    ):
        num_replicas = int(
            self.config.getdefault(self._section, "pipeline_num_replicas", 1)
        )
        devices = self.config.getdefault(
            self._section, "pipeline_replica_devices", "cpu"
        )
        lock = self.config.getdefault(
            self._section, "pipeline_replica_lock", True
        )
        if devices is None:
            devices = []
        if isinstance(devices, str):
            devices = [devices] * num_replicas
        pipelines = [
            LucyForVideoEditingFastAPIPipeline.from_config(
                self.config,
                pretrained_name=pretrained_name,
                pretrained_lora_names=pretrained_lora_names,
                pretrained_lora_weights=pretrained_lora_weights,
                pretrained_lora_alphas=pretrained_lora_alphas,
            )
            for _ in range(num_replicas)
        ]
        for pipe, device in zip(pipelines, devices):
            if device is not None and hasattr(pipe, "to"):
                pipe.to(device)
        self._pipes = PipelineReplicaPool(pipelines, lock=lock)
        return "start success"

    def stop(self):
        if self._pipes is not None:
            self._pipes.close()
        self._pipes = None
        return "stop success"

    def status(self):
        return "running" if self._pipes is not None else "stopped"

    async def generate(
        self,
        text: str,
        video: UploadFile,
        neg_text: Optional[str] = "",
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames: Optional[int] = 81,
        num_fps: Optional[int] = 16,
        guidance_scale: Optional[float] = 5.0,
        num_timesteps: Optional[int] = 50,
        seed: Optional[int] = 1123,
    ):
        assert self._pipes is not None
        video_bytes = await video.read()
        frames = []
        with imageio.v3.imopen(io.BytesIO(video_bytes), "r", plugin="pyav") as reader:
            for frame in reader.iter():
                frames.append(Image.fromarray(frame).convert("RGB"))

        pipe = self._pipes.acquire()
        try:
            output_video = pipe(
                text,
                frames,
                neg_text=neg_text,
                height=height,
                width=width,
                num_frames=num_frames,
                num_fps=num_fps,
                guidance_scale=guidance_scale,
                num_timesteps=num_timesteps,
                seed=seed,
            )
        finally:
            pipe.release()
        buffer = io.BytesIO()
        with open(output_video, "rb") as f:
            buffer.write(f.read())
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="video/mp4",
            headers={"Content-Disposition": "attachment; filename=output.mp4"},
        )
