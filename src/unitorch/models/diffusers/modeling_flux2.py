# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import json
import logging
import os
import re
import sys
from contextlib import contextmanager
from copy import deepcopy
from typing import Optional, Tuple, Union

import numpy as np
import safetensors.torch
import torch
import diffusers.schedulers as schedulers
from diffusers import Flux2Pipeline, Flux2KleinPipeline
from diffusers.models import AutoencoderKLFlux2, Flux2Transformer2DModel
from diffusers.pipelines.flux2.pipeline_flux2 import (
    compute_empirical_mu,
    retrieve_timesteps,
)
from diffusers.schedulers import SchedulerMixin
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils.torch_utils import randn_tensor
from transformers import (
    Mistral3Config,
    Mistral3ForConditionalGeneration,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from unitorch.models import GenericModel, GenericOutputs
from unitorch.models.peft import PeftWeightLoaderMixin
from unitorch.utils import load_weight


@contextmanager
def _initialization_dtype(dtype):
    """Construct weights directly in their storage dtype, avoiding an fp32 copy."""
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(previous)


def _zero3_initializing():
    partition = sys.modules.get("deepspeed.runtime.zero.partition_parameters")
    return partition is not None and partition.zero_init_context > 0


class GenericFlux2Model(GenericModel, PeftWeightLoaderMixin):
    prefix_keys_in_state_dict = {
        "^encoder.*": "vae.",
        "^decoder.*": "vae.",
        "^post_quant_conv.*": "vae.",
        "^quant_conv.*": "vae.",
        "^bn.*": "vae.",
    }

    @staticmethod
    def _build_text_encoder(text_config_path: str, pretrained_model_path=None, dtype=None):
        with open(text_config_path) as f:
            text_config_dict = json.load(f)
        model_type = text_config_dict.get("model_type")

        if _zero3_initializing():
            # Transformers 5 initializes composite text encoders through
            # smart_apply, outside nn.Module.apply patched by ZeRO. Initialize
            # each real module under a gather instead of indexing empty shards.
            from transformers.initialization import no_init_weights
            from transformers.integrations.deepspeed import initialize_weights_zero3

            with no_init_weights():
                if model_type == "mistral3":
                    model = Mistral3ForConditionalGeneration(Mistral3Config.from_dict(text_config_dict))
                elif model_type == "qwen3":
                    model = Qwen3ForCausalLM(Qwen3Config.from_dict(text_config_dict))
                else:
                    raise ValueError(f"Unsupported FLUX.2 text encoder model_type: {model_type}")
            initialize_weights_zero3(model)
            model.tie_weights()
            return model

        if model_type == "mistral3":
            text_config = Mistral3Config.from_dict(text_config_dict)
            if pretrained_model_path is not None:
                return Mistral3ForConditionalGeneration.from_pretrained(
                    os.path.join(pretrained_model_path, "text_encoder"),
                    config=text_config,
                    torch_dtype=dtype,
                    local_files_only=True,
                )
            return Mistral3ForConditionalGeneration(text_config)

        if model_type == "qwen3":
            text_config = Qwen3Config.from_dict(text_config_dict)
            if pretrained_model_path is not None:
                return Qwen3ForCausalLM.from_pretrained(
                    os.path.join(pretrained_model_path, "text_encoder"),
                    config=text_config,
                    torch_dtype=dtype,
                    local_files_only=True,
                )
            return Qwen3ForCausalLM(text_config)

        raise ValueError(f"Unsupported FLUX.2 text encoder model_type: {model_type}")

    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        num_train_timesteps: int = 1000,
        num_infer_timesteps: Optional[int] = None,
        freeze_vae_encoder: bool = True,
        freeze_text_encoder: bool = True,
        freeze_transformer_encoder: bool = False,
        snr_gamma: float = 5.0,
        seed: int = 1123,
        text_encoder_out_layers: Optional[Tuple[int, ...]] = None,
        pretrained_model_path: Optional[str] = None,
        torch_dtype: Union[str, torch.dtype] = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.seed = seed
        self.num_train_timesteps = num_train_timesteps
        self.snr_gamma = snr_gamma
        if isinstance(torch_dtype, str):
            torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                           "float16": torch.float16}[torch_dtype]

        # Local Hugging Face snapshots use the concrete classes' streaming loaders.
        # This also loads the immutable base before PEFT inserts its base_layer keys.
        if pretrained_model_path is not None:
            self.transformer = Flux2Transformer2DModel.from_pretrained(
                os.path.join(pretrained_model_path, "transformer"),
                torch_dtype=torch_dtype, local_files_only=True, low_cpu_mem_usage=True,
            )
            self.text = self._build_text_encoder(
                text_config_path, pretrained_model_path, torch_dtype
            )
            self.vae = AutoencoderKLFlux2.from_pretrained(
                os.path.join(pretrained_model_path, "vae"),
                torch_dtype=torch_dtype, local_files_only=True, low_cpu_mem_usage=True,
            )
        else:
            with _initialization_dtype(torch_dtype):
                with open(config_path) as f:
                    self.transformer = Flux2Transformer2DModel.from_config(json.load(f))
                self.text = self._build_text_encoder(text_config_path)
                with open(vae_config_path) as f:
                    self.vae = AutoencoderKLFlux2.from_config(json.load(f))

        self.is_klein = self.text.config.model_type == "qwen3"
        self.text_encoder_out_layers = tuple(
            text_encoder_out_layers
            if text_encoder_out_layers is not None
            else ((9, 18, 27) if self.is_klein else (10, 20, 30))
        )
        self.num_infer_timesteps = (
            num_infer_timesteps if num_infer_timesteps is not None
            else (4 if self.is_klein else 50)
        )

        with open(scheduler_config_path) as f:
            scheduler_config_dict = json.load(f)
        scheduler_class_name = scheduler_config_dict.get(
            "_class_name", "FlowMatchEulerDiscreteScheduler"
        )
        assert hasattr(schedulers, scheduler_class_name)
        scheduler_class = getattr(schedulers, scheduler_class_name)
        assert issubclass(scheduler_class, SchedulerMixin)
        scheduler_config_dict["num_train_timesteps"] = num_train_timesteps
        self.scheduler = scheduler_class.from_config(scheduler_config_dict)
        # set_timesteps()/step() mutate the inference scheduler. Never sample
        # training noise from it after validation or checkpoint inference.
        self.training_scheduler = deepcopy(self.scheduler)

        if self.is_klein:
            self.pipeline = Flux2KleinPipeline(
                scheduler=self.scheduler, vae=self.vae, text_encoder=self.text,
                tokenizer=None, transformer=self.transformer, is_distilled=True,
            )
        else:
            self.pipeline = Flux2Pipeline(
                scheduler=self.scheduler, vae=self.vae, text_encoder=self.text,
                tokenizer=None, transformer=self.transformer,
            )
        self.pipeline.set_progress_bar_config(disable=True)
        self.vae_scale_factor = self.pipeline.vae_scale_factor

        # The decoder and BN calibration are not optimized by a latent flow loss.
        self.vae.requires_grad_(False)
        if not freeze_vae_encoder:
            self.vae.encoder.requires_grad_(True)
            if self.vae.quant_conv is not None:
                self.vae.quant_conv.requires_grad_(True)
        self.text.requires_grad_(not freeze_text_encoder)
        self.transformer.requires_grad_(not freeze_transformer_encoder)
        self.train(self.training)

    def train(self, mode: bool = True):
        """Keep frozen conditioners deterministic while training the transformer."""
        super().train(mode)
        if not any(p.requires_grad for p in self.text.parameters()):
            self.text.eval()
        if not any(p.requires_grad for p in self.vae.parameters()):
            self.vae.eval()
        self.vae.decoder.eval()
        self.vae.bn.eval()
        return self

    def _map_pretrained_key(self, key, own, prefix_keys, replace_keys):
        """Resolve base and legacy keys against the concrete model's layout."""
        for pattern, prefix in prefix_keys.items():
            if re.match(pattern, key):
                key = prefix + key
                break
        for pattern, replacement in replace_keys.items():
            key = re.sub(pattern, replacement, key)
        if key not in own and not self.is_klein:
            # Mistral3 checkpoints retain the Transformers 4 names;
            # Transformers 5 nests the backbone under model instead.
            # Accept both layouts, including old UniTorch checkpoints.
            for source, target in (
                ("text.language_model.model.", "text.model.language_model."),
                ("text.language_model.lm_head.", "text.lm_head."),
                ("text.vision_tower.", "text.model.vision_tower."),
                ("text.multi_modal_projector.", "text.model.multi_modal_projector."),
            ):
                if key.startswith(source):
                    candidate = target + key[len(source):]
                    base, name = candidate.rsplit(".", 1)
                    if candidate in own or f"{base}.base_layer.{name}" in own:
                        key = candidate
                    break
        if key not in own and "." in key:
            module, parameter = key.rsplit(".", 1)
            candidate = f"{module}.base_layer.{parameter}"
            if candidate in own:
                key = candidate
        return key

    def from_pretrained(self, weight_path=None, state_dict=None,
                        replace_keys=None, prefix_keys=None):
        """Load base or adapter keys, including arbitrary PEFT target modules.

        Files are consumed one at a time. A base key is redirected only when the
        corresponding injected ``base_layer`` actually exists in this model.
        """
        prefix_keys = {**self.prefix_keys_in_state_dict, **(prefix_keys or {})}
        replace_keys = {**self.replace_keys_in_state_dict, **(replace_keys or {})}
        paths = [weight_path] if isinstance(weight_path, str) else (weight_path or [])
        states = [state_dict] if isinstance(state_dict, dict) else (state_dict or [])

        def chunks():
            for path in paths:
                yield load_weight(path)
            yield from states

        own = self.state_dict()
        parameters = dict(self.named_parameters(remove_duplicate=False))
        for chunk in chunks():
            mapped = {}
            loaded = 0
            for key, value in chunk.items():
                key = self._map_pretrained_key(key, own, prefix_keys, replace_keys)
                parameter = parameters.get(key)
                shape = getattr(parameter, "ds_shape", own[key].shape if key in own else None)
                if shape == value.shape:
                    if parameter is not None and hasattr(parameter, "ds_id"):
                        import deepspeed

                        # Gather one tensor, not the 32B transformer or the
                        # complete text encoder. Rank zero writes and broadcasts.
                        with deepspeed.zero.GatheredParameters([parameter], modifier_rank=0):
                            if deepspeed.comm.get_rank() == 0:
                                with torch.no_grad():
                                    parameter.copy_(value)
                        loaded += 1
                    else:
                        mapped[key] = value
            self.load_state_dict(mapped, strict=False)
            logging.info("%s loaded %d pretrained tensors", type(self).__name__, len(mapped) + loaded)

    def checkpoint_state_dict(self, state_dict=None):
        """Select trained components from a native or consolidated ZeRO state."""
        prefixes = ["transformer."]
        if any(p.requires_grad for p in self.text.parameters()):
            prefixes.append("text.")
        if any(p.requires_grad for p in self.vae.encoder.parameters()):
            prefixes.extend(["vae.encoder.", "vae.quant_conv."])
        state_dict = self.state_dict() if state_dict is None else state_dict
        return {key: value for key, value in state_dict.items()
                if key.startswith(tuple(prefixes))}

    def save_checkpoint(self, ckpt_dir, weight_name=None, **kwargs):
        """Save the full transformer and any explicitly unfrozen encoders.

        Frozen text/VAE weights are reused from the same pretrained base. LoRA
        subclasses retain PeftCheckpointMixin's adapter-only checkpoint format.
        """
        state = self.checkpoint_state_dict()
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, weight_name or self.checkpoint_name)
        if path.endswith(".safetensors"):
            safetensors.torch.save_file(state, path)
        else:
            torch.save(state, path)
        logging.info("%s saved checkpoint to %s", type(self).__name__, path)

    def from_checkpoint(self, ckpt_dir, weight_name=None, **kwargs):
        """Validate and restore trained components, including partitioned adapters."""
        path = os.path.join(ckpt_dir, weight_name or self.checkpoint_name)
        if not os.path.exists(path):
            return
        if path.endswith(".safetensors"):
            state = safetensors.torch.load_file(path)
        else:
            # mmap avoids allocating a second anonymous 60+ GiB weight copy.
            try:
                state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            except RuntimeError as error:
                if "mmap can only be used" not in str(error):
                    raise
                # Older torch.save files predate the mmap-capable zip format.
                state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or any(
            not isinstance(key, str) or not isinstance(value, torch.Tensor)
            for key, value in state.items()
        ):
            raise RuntimeError("FLUX.2 checkpoint must contain a tensor state dictionary")
        own = self.state_dict()
        state = {
            self._map_pretrained_key(key, own, self.prefix_keys_in_state_dict,
                                     self.replace_keys_in_state_dict): value
            for key, value in state.items()
        }
        self._validate_checkpoint_state(state)
        self.from_pretrained(state_dict=state)
        logging.info("%s loaded weights from %s", type(self).__name__, path)

    def _validate_checkpoint_state(self, state):
        """Validate the entire restore before any parameter (or ZeRO shard) changes."""
        own = self.state_dict()
        required = self.checkpoint_state_dict(own)
        missing = required.keys() - state.keys()
        if missing:
            raise RuntimeError(
                f"FLUX.2 checkpoint missing {len(missing)} required tensors: "
                + ", ".join(sorted(missing)[:5])
            )
        parameters = dict(self.named_parameters(remove_duplicate=False))
        for key, value in state.items():
            if key not in own:
                continue  # Preserve legacy full-model/extra-key compatibility.
            expected = getattr(parameters.get(key), "ds_shape", own[key].shape)
            if not isinstance(value, torch.Tensor) or value.shape != expected:
                actual = getattr(value, "shape", type(value).__name__)
                raise RuntimeError(
                    f"FLUX.2 checkpoint shape mismatch for {key}: "
                    f"expected {tuple(expected)}, got {actual}"
                )

    @property
    def device(self):
        """Use the execution device when offload stores parameters on meta."""
        return getattr(self, "_inference_device", None) or super().device

    def _prepare_inference_offload(self, enabled, device):
        if enabled and getattr(self, "_inference_device", None) is None:
            if torch.device(device).type == "cpu":
                # Diffusers turns "cpu" into "cpu:0", for which Tensor.to()
                # loses Parameter identity in Accelerate's meta hooks. Use
                # the same real hooks with an unindexed CPU execution device.
                from accelerate import cpu_offload

                self._inference_device = torch.device("cpu")
                for component in (self.text, self.transformer, self.vae):
                    cpu_offload(component, self._inference_device)
            else:
                self.pipeline.enable_sequential_cpu_offload(device=device)
                self._inference_device = self.pipeline._offload_device

    def _guidance(self, batch_size, guidance_scale=None):
        if not self.transformer.config.guidance_embeds:
            return None
        scale = self.guidance_scale if guidance_scale is None else guidance_scale
        return torch.full((batch_size,), scale, device=self.device, dtype=torch.float32)

    def get_sigmas(
        self, timesteps: torch.Tensor, n_dim: int = 4, dtype=torch.float32
    ) -> torch.Tensor:
        sigmas = self.training_scheduler.sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self.training_scheduler.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while sigma.dim() < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def _encode_prompt(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        text_encoder_out_layers: Optional[Tuple[int, ...]] = None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        text_encoder_out_layers = tuple(
            text_encoder_out_layers or self.text_encoder_out_layers
        )
        # Hidden states are identical to the generation wrapper's; skip its
        # enormous, unused vocabulary projection (especially Mistral3 24B).
        outputs = self.text.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = torch.stack(
            [outputs.hidden_states[k] for k in text_encoder_out_layers], dim=1
        )
        hidden_states = hidden_states.to(dtype=self.transformer.dtype, device=self.device)
        batch_size, num_channels, seq_len, hidden_dim = hidden_states.shape
        prompt_embeds = hidden_states.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, num_channels * hidden_dim
        )
        text_ids = Flux2Pipeline._prepare_text_ids(prompt_embeds).to(self.device)
        return prompt_embeds, text_ids

    def _normalize_patched_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents_bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        return (latents - latents_bn_mean) / latents_bn_std

    def _denormalize_patched_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents_bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        return latents * latents_bn_std + latents_bn_mean

    def _encode_vae_image(
        self,
        image: torch.Tensor,
        sample_mode: str = "argmax",
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(device=self.device, dtype=self.vae.dtype)
        encoded = self.vae.encode(image)
        if sample_mode == "sample":
            latents = encoded.latent_dist.sample(generator)
        else:
            latents = encoded.latent_dist.mode()
        latents = Flux2Pipeline._patchify_latents(latents)
        return self._normalize_patched_latents(latents)

    def _sample_timesteps_and_sigmas(self, latents: torch.Tensor):
        u = compute_density_for_timestep_sampling(
            weighting_scheme="none",
            batch_size=latents.shape[0],
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
        )
        indices = (u * self.training_scheduler.config.num_train_timesteps).long()
        timesteps = self.training_scheduler.timesteps[indices].to(device=self.device)
        sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
        return timesteps, sigmas

    def _compute_flow_loss(
        self,
        pred: torch.Tensor,
        noise: torch.Tensor,
        latents: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> torch.Tensor:
        while sigmas.dim() > pred.dim():
            sigmas = sigmas.squeeze(-1)
        while sigmas.dim() < pred.dim():
            sigmas = sigmas.unsqueeze(-1)
        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme="none", sigmas=sigmas
        )
        target = noise - latents
        loss = torch.mean(
            (weighting.float() * (pred.float() - target.float()) ** 2).reshape(
                target.shape[0], -1
            ),
            dim=1,
        )
        return loss.mean()

    def _prepare_condition_latents(
        self,
        refer_pixel_values: torch.Tensor,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ):
        if refer_pixel_values.ndim == 3:
            refer_pixel_values = refer_pixel_values.unsqueeze(0)
        if refer_pixel_values.ndim == 4:
            refer_pixel_values = refer_pixel_values.unsqueeze(1)
        if refer_pixel_values.ndim != 5 or refer_pixel_values.shape[0] != batch_size:
            raise ValueError("Reference batch must be [B,C,H,W] or [B,R,C,H,W] with the prompt batch size.")
        if refer_pixel_values.shape[1] == 0:
            raise ValueError("The reference batch must contain at least one image.")
        # R is a reference axis, never a batch axis. Position t=10,20,... is
        # shared across samples, but each sample keeps its own reference tokens.
        latents = [self._encode_vae_image(refer_pixel_values[:, index], generator=generator)
                   for index in range(refer_pixel_values.shape[1])]
        image_latent_ids = Flux2Pipeline._prepare_image_ids([x[:1] for x in latents])
        image_latent_ids = image_latent_ids.to(self.device).expand(batch_size, -1, -1)
        image_latents = torch.cat([Flux2Pipeline._pack_latents(x) for x in latents], dim=1)
        return image_latents, image_latent_ids

    def _prepare_noise_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        generator: torch.Generator,
        latents: Optional[torch.Tensor] = None,
    ):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        num_latents_channels = self.transformer.config.in_channels // 4
        shape = (batch_size, num_latents_channels * 4, height // 2, width // 2)
        if latents is None:
            latents = randn_tensor(
                shape,
                generator=generator,
                device=self.device,
                dtype=dtype,
            )
        else:
            latents = latents.to(device=self.device, dtype=dtype)

        latent_ids = Flux2Pipeline._prepare_latent_ids(latents).to(self.device)
        latents = Flux2Pipeline._pack_latents(latents)
        return latents, latent_ids

    def _decode_latents(self, latents: torch.Tensor, latent_ids: torch.Tensor):
        latents = Flux2Pipeline._unpack_latents_with_ids(latents, latent_ids)
        latents = self._denormalize_patched_latents(latents)
        latents = Flux2Pipeline._unpatchify_latents(latents)
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.pipeline.image_processor.postprocess(image, output_type="np")
        return torch.from_numpy(image)

    @torch.no_grad()
    def _generate_from_embeds(
        self,
        prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        refer_pixel_values: Optional[torch.Tensor] = None,
        height: int = 1024,
        width: int = 1024,
        guidance_scale: Optional[float] = None,
        num_infer_timesteps: Optional[int] = None,
    ) -> GenericOutputs:
        batch_size = prompt_embeds.shape[0]
        num_infer_timesteps = num_infer_timesteps or self.num_infer_timesteps
        generator = torch.Generator(device=self.device).manual_seed(self.seed)

        latents, latent_ids = self._prepare_noise_latents(
            batch_size=batch_size,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            generator=generator,
        )

        image_latents = None
        image_latent_ids = None
        if refer_pixel_values is not None:
            image_latents, image_latent_ids = self._prepare_condition_latents(
                refer_pixel_values=refer_pixel_values,
                batch_size=batch_size,
                generator=generator,
            )

        sigmas = np.linspace(1.0, 1 / num_infer_timesteps, num_infer_timesteps)
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None
        mu = compute_empirical_mu(
            image_seq_len=latents.shape[1],
            num_steps=num_infer_timesteps,
        )
        timesteps, num_infer_timesteps = retrieve_timesteps(
            self.scheduler,
            num_infer_timesteps,
            self.device,
            sigmas=sigmas,
            mu=mu,
        )

        guidance = self._guidance(batch_size, guidance_scale)

        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(0)

        for t in timesteps:
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            latent_model_input = latents.to(self.transformer.dtype)
            latent_image_ids = latent_ids

            if image_latents is not None:
                latent_model_input = torch.cat([latents, image_latents], dim=1).to(
                    self.transformer.dtype
                )
                latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]
            noise_pred = noise_pred[:, : latents.size(1)]
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        images = self._decode_latents(latents, latent_ids)
        return GenericOutputs(images=images)


class Flux2ForText2ImageGeneration(GenericFlux2Model):
    """Flow matching for FLUX.2 dev or distilled klein-4B text-to-image."""
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        num_train_timesteps: int = 1000,
        num_infer_timesteps: Optional[int] = None,
        freeze_vae_encoder: bool = True,
        freeze_text_encoder: bool = True,
        snr_gamma: float = 5.0,
        seed: int = 1123,
        gradient_checkpointing: bool = True,
        guidance_scale: Optional[float] = None,
        text_encoder_out_layers: Optional[Tuple[int, ...]] = None,
        pretrained_model_path: Optional[str] = None,
        torch_dtype: Union[str, torch.dtype] = torch.bfloat16,
    ) -> None:
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            freeze_vae_encoder=freeze_vae_encoder,
            freeze_text_encoder=freeze_text_encoder,
            snr_gamma=snr_gamma,
            seed=seed,
            text_encoder_out_layers=text_encoder_out_layers,
            pretrained_model_path=pretrained_model_path,
            torch_dtype=torch_dtype,
        )
        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
        self.guidance_scale = guidance_scale if guidance_scale is not None else (1.0 if self.is_klein else 4.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prompt_embeds, text_ids = self._encode_prompt(input_ids, attention_mask)
        latents = self._encode_vae_image(pixel_values)
        noise = torch.randn_like(latents)
        timesteps, sigmas = self._sample_timesteps_and_sigmas(latents)
        noise_latents = (1.0 - sigmas) * latents + sigmas * noise

        latent_ids = Flux2Pipeline._prepare_latent_ids(noise_latents).to(self.device)
        noise_latents = Flux2Pipeline._pack_latents(noise_latents)
        noise = Flux2Pipeline._pack_latents(noise)
        latents = Flux2Pipeline._pack_latents(latents)

        batch_size = latents.shape[0]
        guidance = self._guidance(batch_size)
        pred = self.transformer(
            hidden_states=noise_latents,
            timestep=timesteps / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_ids,
            return_dict=False,
        )[0]
        return self._compute_flow_loss(pred, noise, latents, sigmas)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        height: int = 1024,
        width: int = 1024,
        guidance_scale: Optional[float] = None,
        enable_cpu_offload: bool = False,
        offload_device: str = "cuda",
    ) -> GenericOutputs:
        self._prepare_inference_offload(enable_cpu_offload, offload_device)
        prompt_embeds, text_ids = self._encode_prompt(input_ids, attention_mask)
        return self._generate_from_embeds(
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
        )


class Flux2ForImageEditingGeneration(GenericFlux2Model):
    """Flow matching with per-sample, optionally multiple reference images."""
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        num_train_timesteps: int = 1000,
        num_infer_timesteps: Optional[int] = None,
        freeze_vae_encoder: bool = True,
        freeze_text_encoder: bool = True,
        snr_gamma: float = 5.0,
        seed: int = 1123,
        gradient_checkpointing: bool = True,
        guidance_scale: Optional[float] = None,
        text_encoder_out_layers: Optional[Tuple[int, ...]] = None,
        pretrained_model_path: Optional[str] = None,
        torch_dtype: Union[str, torch.dtype] = torch.bfloat16,
    ) -> None:
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            freeze_vae_encoder=freeze_vae_encoder,
            freeze_text_encoder=freeze_text_encoder,
            snr_gamma=snr_gamma,
            seed=seed,
            text_encoder_out_layers=text_encoder_out_layers,
            pretrained_model_path=pretrained_model_path,
            torch_dtype=torch_dtype,
        )
        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
        self.guidance_scale = guidance_scale if guidance_scale is not None else (1.0 if self.is_klein else 4.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        refer_pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prompt_embeds, text_ids = self._encode_prompt(input_ids, attention_mask)
        latents = self._encode_vae_image(pixel_values)
        noise = torch.randn_like(latents)
        timesteps, sigmas = self._sample_timesteps_and_sigmas(latents)
        noise_latents = (1.0 - sigmas) * latents + sigmas * noise

        latent_ids = Flux2Pipeline._prepare_latent_ids(noise_latents).to(self.device)
        noise_latents = Flux2Pipeline._pack_latents(noise_latents)
        noise = Flux2Pipeline._pack_latents(noise)
        latents = Flux2Pipeline._pack_latents(latents)

        image_latents, image_latent_ids = self._prepare_condition_latents(
            refer_pixel_values=refer_pixel_values,
            batch_size=latents.shape[0],
        )
        latent_model_input = torch.cat([noise_latents, image_latents], dim=1)
        latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

        batch_size = latents.shape[0]
        guidance = self._guidance(batch_size)
        pred = self.transformer(
            hidden_states=latent_model_input,
            timestep=timesteps / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0][:, : noise_latents.shape[1]]
        return self._compute_flow_loss(pred, noise, latents, sigmas)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        refer_pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        height: int = 1024,
        width: int = 1024,
        guidance_scale: Optional[float] = None,
        enable_cpu_offload: bool = False,
        offload_device: str = "cuda",
    ) -> GenericOutputs:
        self._prepare_inference_offload(enable_cpu_offload, offload_device)
        prompt_embeds, text_ids = self._encode_prompt(input_ids, attention_mask)
        return self._generate_from_embeds(
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            refer_pixel_values=refer_pixel_values,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
        )
