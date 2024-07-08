# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import warnings
import logging
import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from peft import (
    PeftConfig,
    PeftType,
    PromptLearningConfig,
    PeftModelForSequenceClassification,
)
from unitorch.utils import replace, is_diffusers_available
from unitorch.models import CheckpointMixin


@replace(PeftModelForSequenceClassification)
class PeftModelForSequenceClassification(PeftModelForSequenceClassification):
    def __init__(self, model, peft_config: PeftConfig, adapter_name="default"):
        super().__init__(model, peft_config, adapter_name)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        peft_config = self.active_peft_config
        if not isinstance(peft_config, PromptLearningConfig):
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        labels = kwargs.pop("labels", None)

        batch_size = input_ids.shape[0]
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(
                batch_size, peft_config.num_virtual_tokens
            ).to(self.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        if kwargs.get("position_ids", None) is not None:
            warnings.warn(
                "Position ids are not supported for parameter efficient tuning. Ignoring position ids."
            )
            kwargs["position_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            return self._prefix_tuning_forward(input_ids=input_ids, **kwargs)
        else:
            if kwargs.get("token_type_ids", None) is not None:
                kwargs["token_type_ids"] = torch.cat(
                    (
                        torch.zeros(batch_size, peft_config.num_virtual_tokens).to(
                            self.device
                        ),
                        kwargs["token_type_ids"],
                    ),
                    dim=1,
                ).long()
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)


class PeftCheckpointMixin(CheckpointMixin):
    checkpoint_name = "pytorch_model.bin"

    modules_to_save_checkpoints = ["lora"]

    def save_checkpoint(
        self,
        ckpt_dir: str,
        weight_name: str = None,
        **kwargs,
    ):
        """
        Save the model's current state as a checkpoint.

        Args:
            ckpt_dir (str): Directory path to save the checkpoint.
            weight_name (str): Name of the weight file.

        Returns:
            None
        """
        if weight_name is None:
            weight_name = self.checkpoint_name
        state_dict = self.state_dict()
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if any(m in k for m in self.modules_to_save_checkpoints)
        }
        weight_path = os.path.join(ckpt_dir, weight_name)
        torch.save(state_dict, weight_path)
        logging.info(f"{type(self).__name__} model save checkpoint to {weight_path}")

    def from_checkpoint(
        self,
        ckpt_dir: str,
        weight_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Load model weights from a checkpoint.

        Args:
            ckpt_dir (str): Directory path of the checkpoint.
            weight_name (str): Name of the weight file.

        Returns:
            None
        """
        if weight_name is None:
            weight_name = self.checkpoint_name
        weight_path = os.path.join(ckpt_dir, weight_name)
        if not os.path.exists(weight_path):
            return
        _state_dict = self.state_dict()
        _state_dict = {
            k: v
            for k, v in _state_dict.items()
            if any(m in k for m in self.modules_to_save_checkpoints)
        }
        state_dict = torch.load(weight_path, map_location="cpu")
        assert all(
            k in state_dict.keys() and state_dict[k].shape == v.shape
            for k, v in _state_dict.items()
        )
        self.load_state_dict(state_dict, strict=False)
        logging.info(
            f"{type(self).__name__} model load weight from checkpoint {weight_path}"
        )


class GenericPeftModel(nn.Module, PeftCheckpointMixin):
    def __init__(self):
        super().__init__()
        pass

    def _init_weights(self, module):
        """
        Initialize the weights of the given module.

        Args:
            module (nn.Module): The module to initialize weights for.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)

        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_weights(self):
        """
        Initialize the weights of the model.
        """
        self.apply(self._init_weights)

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the data type of the model's parameters.

        Returns:
            torch.dtype: The data type of the model's parameters.
        """
        return next(self.parameters()).dtype

    @property
    def device(self):
        """
        Returns the device of the model's parameters.

        Returns:
            torch.device: The device of the model's parameters.
        """
        return next(self.parameters()).device


from unitorch.models.peft.modeling_bloom import (
    BloomLoraForClassification,
    BloomLoraForGeneration,
)
from unitorch.models.peft.modeling_llama import (
    LlamaLoraForClassification,
    LlamaLoraForGeneration,
)
from unitorch.models.peft.modeling_mistral import MistralLoraForGeneration

if is_diffusers_available():
    from unitorch.models.peft.diffusers import (
        StableLoraForText2ImageGeneration,
        StableLoraForImage2ImageGeneration,
        StableLoraForImageInpainting,
        StableLoraForImageResolution,
        StableXLLoraForText2ImageGeneration,
        StableXLLoraForImage2ImageGeneration,
        StableXLLoraForImageInpainting,
        ControlNetLoraForText2ImageGeneration,
        ControlNetLoraForImage2ImageGeneration,
        ControlNetLoraForImageInpainting,
        ControlNetXLLoraForText2ImageGeneration,
        ControlNetXLLoraForImage2ImageGeneration,
        ControlNetXLLoraForImageInpainting,
        MultiControlNetLoraForText2ImageGeneration,
        MultiControlNetLoraForImage2ImageGeneration,
        MultiControlNetLoraForImageInpainting,
    )
