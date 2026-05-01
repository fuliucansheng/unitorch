# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import List, Optional, Union
from torch import autocast
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.blip import (
    BlipForPretrain as _BlipForPretrain,
    BlipForClassification as _BlipForClassification,
    BlipForTextClassification as _BlipForTextClassification,
    BlipForImageClassification as _BlipForImageClassification,
    BlipForImageCaption as _BlipForImageCaption,
)
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import generation_model_decorator
from unitorch.cli.models import ClassificationOutputs, GenerationOutputs, LossOutputs
from unitorch.cli.models.blip import pretrained_blip_infos


@register_model("core/model/pretrain/blip")
class BlipForPretrain(_BlipForPretrain):
    """BLIP model for pretraining."""

    def __init__(
        self,
        config_path: str,
        projection_dim: Optional[int] = 512,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
        use_all_gather: Optional[bool] = True,
    ):
        super().__init__(
            config_path=config_path,
            projection_dim=projection_dim,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
            use_all_gather=use_all_gather,
        )

    @classmethod
    @add_default_section_for_init("core/model/pretrain/blip")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/pretrain/blip")
        pretrained_name = config.getoption(
            "pretrained_name", "blip-image-captioning-base"
        )

        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_blip_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)

        projection_dim = config.getoption("projection_dim", 512)
        freeze_base_model = config.getoption("freeze_base_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        use_all_gather = config.getoption("use_all_gather", True)

        inst = cls(
            config_path=config_path,
            projection_dim=projection_dim,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
            use_all_gather=use_all_gather,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_blip_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return LossOutputs(loss=outputs)


@register_model("core/model/classification/blip")
class BlipForClassification(_BlipForClassification):
    """BLIP model for classification."""

    def __init__(
        self,
        config_path: str,
        projection_dim: Optional[int] = 512,
        num_classes: Optional[int] = 1,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__(
            config_path=config_path,
            projection_dim=projection_dim,
            num_classes=num_classes,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @add_default_section_for_init("core/model/classification/blip")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/classification/blip")
        pretrained_name = config.getoption(
            "pretrained_name", "blip-image-captioning-base"
        )
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_blip_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)

        projection_dim = config.getoption("projection_dim", 512)
        num_classes = config.getoption("num_classes", 1)
        freeze_base_model = config.getoption("freeze_base_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_path=config_path,
            projection_dim=projection_dim,
            num_classes=num_classes,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_blip_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return ClassificationOutputs(outputs=outputs)


@register_model("core/model/classification/blip/text")
class BlipForTextClassification(_BlipForTextClassification):
    """BLIP model for text classification."""

    def __init__(
        self,
        config_path: str,
        projection_dim: Optional[int] = 512,
        num_classes: Optional[int] = 1,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__(
            config_path=config_path,
            projection_dim=projection_dim,
            num_classes=num_classes,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @add_default_section_for_init("core/model/classification/blip/text")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/classification/blip/text")
        pretrained_name = config.getoption(
            "pretrained_name", "blip-image-captioning-base"
        )
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_blip_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)

        projection_dim = config.getoption("projection_dim", 512)
        num_classes = config.getoption("num_classes", 1)
        freeze_base_model = config.getoption("freeze_base_Truemodel", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_path=config_path,
            projection_dim=projection_dim,
            num_classes=num_classes,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_blip_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        input_ids=None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return ClassificationOutputs(outputs=outputs)


@register_model("core/model/classification/blip/image")
class BlipForImageClassification(_BlipForImageClassification):
    """BLIP model for image classification."""

    def __init__(
        self,
        config_path: str,
        projection_dim: Optional[int] = 512,
        num_classes: Optional[int] = 1,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__(
            config_path=config_path,
            projection_dim=projection_dim,
            num_classes=num_classes,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @add_default_section_for_init("core/model/classification/blip/image")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/classification/blip/image")
        pretrained_name = config.getoption(
            "pretrained_name", "blip-image-captioning-base"
        )
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_blip_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)

        projection_dim = config.getoption("projection_dim", 512)
        num_classes = config.getoption("num_classes", 1)
        freeze_base_model = config.getoption("freeze_base_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_path=config_path,
            projection_dim=projection_dim,
            num_classes=num_classes,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_blip_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        pixel_values: torch.Tensor,
    ):
        outputs = super().forward(pixel_values=pixel_values)
        return ClassificationOutputs(outputs=outputs)


@register_model("core/model/caption/blip", generation_model_decorator)
class BlipForImageCaption(_BlipForImageCaption):
    """BLIP model for image captioning."""

    def __init__(
        self,
        config_path: str,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__(
            config_path=config_path,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @add_default_section_for_init("core/model/caption/blip")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/caption/blip")
        pretrained_name = config.getoption(
            "pretrained_name", "blip-image-captioning-base"
        )
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_blip_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_path=config_path,
            gradient_checkpointing=gradient_checkpointing,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_blip_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        outputs = super().forward(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return GenerationOutputs(sequences=outputs)

    @torch.no_grad()
    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def generate(
        self,
        pixel_values: torch.Tensor,
        num_beams: Optional[int] = 5,
        decoder_start_token_id: Optional[int] = 101,
        decoder_end_token_id: Optional[Union[int, List[int]]] = 102,
        num_return_sequences: Optional[int] = 1,
        min_gen_seq_length: Optional[int] = 0,
        max_gen_seq_length: Optional[int] = 48,
        repetition_penalty: Optional[float] = 1.0,
        no_repeat_ngram_size: Optional[int] = 0,
        early_stopping: Optional[bool] = True,
        length_penalty: Optional[float] = 1.0,
        num_beam_groups: Optional[int] = 1,
        diversity_penalty: Optional[float] = 0.0,
        do_sample: Optional[bool] = False,
        temperature: Optional[float] = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 1.0,
    ):
        outputs = super().generate(
            pixel_values=pixel_values,
            num_beams=num_beams,
            decoder_start_token_id=decoder_start_token_id,
            decoder_end_token_id=decoder_end_token_id,
            num_return_sequences=num_return_sequences,
            min_gen_seq_length=min_gen_seq_length,
            max_gen_seq_length=max_gen_seq_length,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            length_penalty=length_penalty,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        return GenerationOutputs(
            sequences=outputs.sequences,
            sequences_scores=outputs.sequences_scores,
        )
