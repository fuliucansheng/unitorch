# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import json
import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import diffusers.schedulers as schedulers
from peft import LoraConfig
from transformers import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from diffusers.schedulers import SchedulerMixin, FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import (
    compute_loss_weighting_for_sd3,
    compute_density_for_timestep_sampling,
)
from diffusers.models import (
    QwenImageTransformer2DModel,
    AutoencoderKLQwenImage,
)
from diffusers.pipelines import (
    QwenImagePipeline,
    QwenImageEditPipeline,
)
from unitorch.models import (
    GenericModel,
    GenericOutputs,
    QuantizationConfig,
    QuantizationMixin,
)
from unitorch.models.peft import GenericPeftModel
from unitorch.models.diffusers import compute_snr
from unitorch.models.diffusers.modeling_qwen_image import (
    _pack_latents,
    _unpack_latents,
)


class GenericQWenImageLoraModel(GenericPeftModel, QuantizationMixin):
    prefix_keys_in_state_dict = {
        # vae weights
        "^encoder.*": "vae.",
        "^decoder.*": "vae.",
        "^post_quant_conv.*": "vae.",
        "^quant_conv.*": "vae.",
    }

    replace_keys_in_state_dict = {
        "\.query\.": ".to_q.",
        "\.key\.": ".to_k.",
        "\.value\.": ".to_v.",
        "\.proj_attn\.": ".to_out.0.",
    }

    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        image_config_path: Optional[str] = None,
        quant_config_path: Optional[str] = None,
        image_size: Optional[int] = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        snr_gamma: Optional[float] = 5.0,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = True,
        target_modules: Optional[Union[List[str], str]] = [
            "to_q",
            "to_k",
            "to_v",
            "q_proj",
            "k_proj",
            "v_proj",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
        ],
        enable_text_adapter: Optional[bool] = True,
        enable_transformer_adapter: Optional[bool] = True,
        seed: Optional[int] = 1123,
    ):
        super().__init__()
        self.seed = seed
        self.num_train_timesteps = num_train_timesteps
        self.num_infer_timesteps = num_infer_timesteps
        self.snr_gamma = snr_gamma

        config_dict = json.load(open(config_path))
        if image_size is not None:
            config_dict.update({"sample_size": image_size})
        if in_channels is not None:
            config_dict.update({"in_channels": in_channels})
        if out_channels is not None:
            config_dict.update({"out_channels": out_channels})
        self.transformer = QwenImageTransformer2DModel.from_config(config_dict).to(
            torch.bfloat16
        )

        text_config = Qwen2_5_VLConfig.from_json_file(text_config_path)
        self.text = Qwen2_5_VLForConditionalGeneration(text_config).to(torch.bfloat16)

        vae_config_dict = json.load(open(vae_config_path))
        self.vae = AutoencoderKLQwenImage.from_config(vae_config_dict).to(
            torch.bfloat16
        )

        scheduler_config_dict = json.load(open(scheduler_config_path))
        scheduler_class_name = scheduler_config_dict.get(
            "_class_name", "FlowMatchEulerDiscreteScheduler"
        )
        assert hasattr(schedulers, scheduler_class_name)
        scheduler_class = getattr(schedulers, scheduler_class_name)
        assert issubclass(scheduler_class, SchedulerMixin)
        scheduler_config_dict["num_train_timesteps"] = num_train_timesteps
        self.scheduler = scheduler_class.from_config(scheduler_config_dict)

        for param in self.vae.parameters():
            param.requires_grad = False

        for param in self.text.parameters():
            param.requires_grad = False

        for param in self.transformer.parameters():
            param.requires_grad = False

        if quant_config_path is not None:
            self.quant_config = QuantizationConfig.from_json_file(quant_config_path)
            self.quantize(
                self.quant_config, ignore_modules=["lm_head", "transformer", "vae"]
            )

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            init_lora_weights="gaussian",
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
        )
        if enable_text_adapter:
            self.text.add_adapter(lora_config)
        if enable_transformer_adapter:
            self.transformer.add_adapter(lora_config)

    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.scheduler.sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self.scheduler.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

        return split_result

    def get_prompt_outputs(
        self,
        input_ids: torch.Tensor,
        negative_input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        prompt_start_index: Optional[int] = None,
        enable_cpu_offload: Optional[bool] = False,
        cpu_offload_device: Optional[str] = "cpu",
    ):
        if enable_cpu_offload:
            self.text = self.text.to(cpu_offload_device)
            input_ids = input_ids.to(cpu_offload_device)
            negative_input_ids = negative_input_ids.to(cpu_offload_device)
            if pixel_values is not None:
                pixel_values = pixel_values.to(cpu_offload_device)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(cpu_offload_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(cpu_offload_device)
            if negative_attention_mask is not None:
                negative_attention_mask = negative_attention_mask.to(cpu_offload_device)

        assert prompt_start_index is not None, "prompt_start_index must be provided"

        if pixel_values is not None and image_grid_thw is not None:
            image_grid_thw = image_grid_thw.view(-1, image_grid_thw.size(-1))
            pixel_values = pixel_values.view(-1, pixel_values.size(-1))

        prompt_outputs = self.text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
        )

        hidden_states = prompt_outputs.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, attention_mask)
        split_hidden_states = [e[prompt_start_index:] for e in split_hidden_states]
        attn_mask_list = [
            torch.ones(e.size(0), dtype=torch.long, device=e.device)
            for e in split_hidden_states
        ]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))])
                for u in split_hidden_states
            ]
        )
        prompt_embeds_mask = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_seq_len - u.size(0))])
                for u in attn_mask_list
            ]
        )

        negative_prompt_outputs = self.text(
            negative_input_ids,
            negative_attention_mask,
            output_hidden_states=True,
        )
        negative_hidden_states = negative_prompt_outputs.hidden_states[-1]
        negative_split_hidden_states = self._extract_masked_hidden(
            negative_hidden_states, negative_attention_mask
        )
        negative_split_hidden_states = [
            e[prompt_start_index:] for e in negative_split_hidden_states
        ]
        negative_attn_mask_list = [
            torch.ones(e.size(0), dtype=torch.long, device=e.device)
            for e in negative_split_hidden_states
        ]
        negative_max_seq_len = max([e.size(0) for e in negative_split_hidden_states])
        negative_prompt_embeds = torch.stack(
            [
                torch.cat(
                    [
                        u,
                        u.new_zeros(negative_max_seq_len - u.size(0), u.size(1)),
                    ]
                )
                for u in negative_split_hidden_states
            ]
        )
        negative_prompt_embeds_mask = torch.stack(
            [
                torch.cat(
                    [
                        u,
                        u.new_zeros(negative_max_seq_len - u.size(0)),
                    ]
                )
                for u in negative_attn_mask_list
            ]
        )

        if enable_cpu_offload:
            self.text = self.text.to("cpu")

        return GenericOutputs(
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
        )


class QWenImageLoraForText2ImageGeneration(GenericQWenImageLoraModel):
    modules_to_save_checkpoints = ["lora"]

    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        quant_config_path: Optional[str] = None,
        image_size: Optional[int] = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        snr_gamma: Optional[float] = 5.0,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = True,
        target_modules: Optional[Union[List[str], str]] = [
            "to_q",
            "to_k",
            "to_v",
            "q_proj",
            "k_proj",
            "v_proj",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
        ],
        enable_text_adapter: Optional[bool] = True,
        enable_transformer_adapter: Optional[bool] = True,
        seed: Optional[int] = 1123,
        gradient_checkpointing: Optional[bool] = True,
        guidance_scale: Optional[float] = 1.0,
        prompt_start_index: Optional[int] = 34,
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            quant_config_path=quant_config_path,
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            snr_gamma=snr_gamma,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
            enable_text_adapter=enable_text_adapter,
            enable_transformer_adapter=enable_transformer_adapter,
            seed=seed,
        )

        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

        self.pipeline = QwenImagePipeline(
            vae=self.vae,
            text_encoder=self.text,
            transformer=self.transformer,
            scheduler=self.scheduler,
            tokenizer=None,
        )
        self.pipeline.set_progress_bar_config(disable=True)
        self.guidance_scale = guidance_scale
        self.prompt_start_index = prompt_start_index

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        prompt_outputs = self.text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        hidden_states = prompt_outputs.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, attention_mask)
        split_hidden_states = [
            e[self.prompt_start_index :] for e in split_hidden_states
        ]
        attn_mask_list = [
            torch.ones(e.size(0), dtype=torch.long, device=e.device)
            for e in split_hidden_states
        ]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))])
                for u in split_hidden_states
            ]
        )
        prompt_embeds_mask = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_seq_len - u.size(0))])
                for u in attn_mask_list
            ]
        )

        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean).view(
                1, self.vae.config.z_dim, 1, 1, 1
            )
        ).to(pixel_values.device)
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(pixel_values.device)
        latents = (latents - latents_mean) * latents_std

        noise = torch.randn(latents.shape).to(latents.device)

        u = compute_density_for_timestep_sampling(
            weighting_scheme="none",
            batch_size=batch_size,
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
        )
        indices = (u * self.scheduler.config.num_train_timesteps).long()
        timesteps = self.scheduler.timesteps[indices].to(device=self.device)

        sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
        noise_latents = (1.0 - sigmas) * latents + sigmas * noise

        noise_latents = _pack_latents(
            noise_latents,
            batch_size=latents.shape[0],
            num_channels_latents=latents.shape[1],
            height=latents.shape[2],
            width=latents.shape[3],
        )

        batch_size, width, height = (
            pixel_values.size(0),
            pixel_values.size(-1),
            pixel_values.size(-2),
        )
        img_shapes = [
            [
                (
                    1,
                    height // self.vae_scale_factor // 2,
                    width // self.vae_scale_factor // 2,
                )
            ]
        ] * batch_size

        if self.transformer.config.guidance_embeds and self.guidance_scale is not None:
            guidance = torch.full(
                [1], self.guidance_scale, device=self.device, dtype=torch.float32
            )
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        outputs = self.transformer(
            hidden_states=noise_latents,
            timestep=timesteps / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_embeds_mask,
            img_shapes=img_shapes,
            txt_seq_lens=prompt_embeds_mask.sum(dim=1).tolist(),
            return_dict=False,
        )[0]

        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        outputs = _unpack_latents(
            outputs,
            height=latents.shape[2] * vae_scale_factor,
            width=latents.shape[3] * vae_scale_factor,
            vae_scale_factor=vae_scale_factor,
        )

        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme="none", sigmas=sigmas
        )
        target = noise - latents
        loss = torch.mean(
            (weighting.float() * (outputs.float() - target.float()) ** 2).reshape(
                target.shape[0], -1
            ),
            1,
        )
        loss = loss.mean()
        return loss

    def generate(
        self,
        input_ids: torch.Tensor,
        negative_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        guidance_scale: Optional[float] = 1.0,
        true_cfg_scale: Optional[float] = 4.0,
    ):
        outputs = self.get_prompt_outputs(
            input_ids=input_ids,
            negative_input_ids=negative_input_ids,
            attention_mask=attention_mask,
            negative_attention_mask=negative_attention_mask,
            prompt_start_index=self.prompt_start_index,
        )

        images = self.pipeline(
            prompt_embeds=outputs.prompt_embeds,
            negative_prompt_embeds=outputs.negative_prompt_embeds,
            prompt_embeds_mask=outputs.prompt_embeds_mask,
            negative_prompt_embeds_mask=outputs.negative_prompt_embeds_mask,
            generator=torch.Generator(device=self.pipeline.device).manual_seed(
                self.seed
            ),
            num_inference_steps=self.num_infer_timesteps,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            true_cfg_scale=true_cfg_scale,
            output_type="np.array",
        ).images

        return GenericOutputs(images=torch.from_numpy(images))


class QWenImageLoraForImageEditing(GenericQWenImageLoraModel):
    modules_to_save_checkpoints = ["lora"]

    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        quant_config_path: Optional[str] = None,
        image_size: Optional[int] = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        snr_gamma: Optional[float] = 5.0,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = True,
        target_modules: Optional[Union[List[str], str]] = [
            "to_q",
            "to_k",
            "to_v",
            "q_proj",
            "k_proj",
            "v_proj",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
        ],
        enable_text_adapter: Optional[bool] = True,
        enable_transformer_adapter: Optional[bool] = True,
        seed: Optional[int] = 1123,
        gradient_checkpointing: Optional[bool] = True,
        guidance_scale: Optional[float] = 1.0,
        prompt_start_index: Optional[int] = 64,
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            quant_config_path=quant_config_path,
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            snr_gamma=snr_gamma,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
            enable_text_adapter=enable_text_adapter,
            enable_transformer_adapter=enable_transformer_adapter,
            seed=seed,
        )
        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

        self.pipeline = QwenImageEditPipeline(
            vae=self.vae,
            text_encoder=self.text,
            transformer=self.transformer,
            scheduler=self.scheduler,
            tokenizer=None,
            processor=None,
        )
        self.pipeline.set_progress_bar_config(disable=True)
        self.guidance_scale = guidance_scale
        self.prompt_start_index = prompt_start_index

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        refer_pixel_values: torch.Tensor,
        refer_image_grid_thw: torch.Tensor,
        refer_vae_pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        refer_image_grid_thw = refer_image_grid_thw.view(
            -1, refer_image_grid_thw.size(-1)
        )
        refer_pixel_values = refer_pixel_values.view(-1, refer_pixel_values.size(-1))
        prompt_outputs = self.text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=refer_pixel_values,
            image_grid_thw=refer_image_grid_thw,
            output_hidden_states=True,
        )

        hidden_states = prompt_outputs.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, attention_mask)
        split_hidden_states = [
            e[self.prompt_start_index :] for e in split_hidden_states
        ]
        attn_mask_list = [
            torch.ones(e.size(0), dtype=torch.long, device=e.device)
            for e in split_hidden_states
        ]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))])
                for u in split_hidden_states
            ]
        )
        prompt_embeds_mask = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_seq_len - u.size(0))])
                for u in attn_mask_list
            ]
        )

        latents = self.vae.encode(pixel_values).latent_dist.sample()
        refer_latents = self.vae.encode(refer_vae_pixel_values).latent_dist.mode()
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean).view(
                1, self.vae.config.z_dim, 1, 1, 1
            )
        ).to(pixel_values.device)
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(pixel_values.device)
        latents = (latents - latents_mean) * latents_std
        refer_latents = (refer_latents - latents_mean) * latents_std

        noise = torch.randn(latents.shape).to(latents.device)

        u = compute_density_for_timestep_sampling(
            weighting_scheme="none",
            batch_size=batch_size,
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
        )
        indices = (u * self.scheduler.config.num_train_timesteps).long()
        timesteps = self.scheduler.timesteps[indices].to(device=self.device)

        sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
        noise_latents = (1.0 - sigmas) * latents + sigmas * noise

        noise_latents = _pack_latents(
            noise_latents,
            batch_size=latents.shape[0],
            num_channels_latents=latents.shape[1],
            height=latents.shape[2],
            width=latents.shape[3],
        )
        refer_latents = _pack_latents(
            refer_latents,
            batch_size=refer_latents.shape[0],
            num_channels_latents=refer_latents.shape[1],
            height=refer_latents.shape[2],
            width=refer_latents.shape[3],
        )
        latent_model_input = torch.cat([noise_latents, refer_latents], dim=1)

        batch_size, width, height = (
            pixel_values.size(0),
            pixel_values.size(-1),
            pixel_values.size(-2),
        )
        img_shapes = [
            [
                (
                    1,
                    height // self.vae_scale_factor // 2,
                    width // self.vae_scale_factor // 2,
                )
            ]
        ] * batch_size

        if self.transformer.config.guidance_embeds and self.guidance_scale is not None:
            guidance = torch.full(
                [1], self.guidance_scale, device=self.device, dtype=torch.float32
            )
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        outputs = self.transformer(
            hidden_states=latent_model_input,
            timestep=timesteps / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_embeds_mask,
            img_shapes=img_shapes,
            txt_seq_lens=prompt_embeds_mask.sum(dim=1).tolist(),
            return_dict=False,
        )[0]
        outputs = outputs[:, : latents.shape[1]]

        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        outputs = _unpack_latents(
            outputs,
            height=latents.shape[2] * vae_scale_factor,
            width=latents.shape[3] * vae_scale_factor,
            vae_scale_factor=vae_scale_factor,
        )

        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme="none", sigmas=sigmas
        )
        target = noise - latents
        loss = torch.mean(
            (weighting.float() * (outputs.float() - target.float()) ** 2).reshape(
                target.shape[0], -1
            ),
            1,
        )
        loss = loss.mean()
        return loss

    def generate(
        self,
        input_ids: torch.Tensor,
        negative_input_ids: torch.Tensor,
        refer_pixel_values: torch.Tensor,
        refer_image_grid_thw: torch.Tensor,
        refer_vae_pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        guidance_scale: Optional[float] = 1.0,
        true_cfg_scale: Optional[float] = 4.0,
    ):
        outputs = self.get_prompt_outputs(
            input_ids=input_ids,
            negative_input_ids=negative_input_ids,
            pixel_values=refer_pixel_values,
            image_grid_thw=refer_image_grid_thw,
            attention_mask=attention_mask,
            negative_attention_mask=negative_attention_mask,
            prompt_start_index=self.prompt_start_index,
        )

        images = self.pipeline(
            images=refer_vae_pixel_values,
            prompt_embeds=outputs.prompt_embeds,
            negative_prompt_embeds=outputs.negative_prompt_embeds,
            prompt_embeds_mask=outputs.prompt_embeds_mask,
            negative_prompt_embeds_mask=outputs.negative_prompt_embeds_mask,
            generator=torch.Generator(device=self.pipeline.device).manual_seed(
                self.seed
            ),
            num_inference_steps=self.num_infer_timesteps,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            true_cfg_scale=true_cfg_scale,
            output_type="np.array",
        ).images

        return GenericOutputs(images=torch.from_numpy(images))
