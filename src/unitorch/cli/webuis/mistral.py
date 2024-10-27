# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import io
import torch
import gc
import gradio as gr
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.utils import nested_dict_value
from unitorch.cli import CoreConfigureParser, GenericWebUI
from unitorch.cli import register_webui
from unitorch.cli.pipelines.mistral import MistralForGenerationPipeline
from unitorch.cli.models.mistral import (
    pretrained_mistral_infos,
    pretrained_mistral_extensions_infos,
)
from unitorch.cli.webuis import (
    matched_pretrained_names,
    create_element,
    create_accordion,
    create_row,
    create_column,
    create_group,
    create_tab,
    create_tabs,
    create_blocks,
    create_pretrain_layout,
    create_lora_layout,
    create_freeu_layout,
)
from unitorch.cli.webuis import SimpleWebUI


@register_webui("core/webui/mistral")
class MistralWebUI(SimpleWebUI):
    pretrained_names = list(pretrained_mistral_infos.keys())
    supported_pretrained_names = matched_pretrained_names(
        pretrained_names,
        "^mistral-",
    )
    pretrained_extension_names = list(pretrained_mistral_extensions_infos.keys())
    supported_lora_names = matched_pretrained_names(
        pretrained_extension_names, "^mistral-lora-"
    )

    def __init__(self, config: CoreConfigureParser):
        self._config = config
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe
        self._status = "Stopped" if self._pipe is None else "Running"
        if len(self.supported_pretrained_names) == 0:
            raise ValueError("No supported pretrained models found.")
        self._name = self.supported_pretrained_names[0]

        # Create the elements
        pretrain_layout_group = create_pretrain_layout(
            self.supported_pretrained_names, self._name
        )
        name, status, start, stop, pretrain_layout = (
            pretrain_layout_group.name,
            pretrain_layout_group.status,
            pretrain_layout_group.start,
            pretrain_layout_group.stop,
            pretrain_layout_group.layout,
        )
        self.num_loras = 5
        lora_layout_group = create_lora_layout(
            self.supported_lora_names, num_loras=self.num_loras
        )
        loras = lora_layout_group.loras
        lora_layout = lora_layout_group.layout
        lora_params = []
        for lora in loras:
            lora_params += [
                lora.checkpoint,
                lora.weight,
                lora.alpha,
                lora.url,
                lora.file,
            ]

        prompt = create_element("text", "Input Prompt", lines=3)
        generate = create_element("button", "Generate")
        result = create_element("text", "Output Result", lines=3)

        # Create the blocks
        left = create_column(prompt, lora_layout, generate)
        right = create_column(result)
        iface = create_blocks(pretrain_layout, create_row(left, right))

        # Create the events
        iface.__enter__()

        start.click(self.start, inputs=[name], outputs=[status], trigger_mode="once")
        stop.click(self.stop, outputs=[status], trigger_mode="once")

        for lora in loras:
            lora.checkpoint.change(
                fn=lambda x: nested_dict_value(
                    pretrained_mistral_extensions_infos, x, "text"
                ),
                inputs=[lora.checkpoint],
                outputs=[lora.text],
            )

        generate.click(
            self.serve,
            inputs=[
                prompt,
                *lora_params,
            ],
            outputs=[result],
            trigger_mode="once",
        )

        iface.load(
            fn=lambda: [gr.update(value=self._name), gr.update(value=self._status)],
            outputs=[name, status],
        )

        iface.__exit__()

        super().__init__(config, iname="Mistral", iface=iface)

    def start(self, pretrained_name, **kwargs):
        if self._name == pretrained_name and self._status == "Running":
            return self._status
        if self._status == "Running":
            self.stop()
        self._name = pretrained_name
        self._pipe = MistralForGenerationPipeline.from_core_configure(
            self._config,
            pretrained_name=pretrained_name,
        )
        self._status = "Running"
        return self._status

    def stop(self, **kwargs):
        self._pipe.to("cpu")
        del self._pipe
        gc.collect()
        torch.cuda.empty_cache()
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe
        self._status = "Stopped" if self._pipe is None else "Running"
        return self._status

    def serve(
        self,
        text: str,
        *params,
    ):
        assert self._pipe is not None
        lora_params = params
        lora_checkpoints = lora_params[0::5]
        lora_weights = lora_params[1::5]
        lora_alphas = lora_params[2::5]
        lora_urls = lora_params[3::5]
        lora_files = lora_params[4::5]
        result = self._pipe(
            text,
            lora_checkpoints=lora_checkpoints,
            lora_weights=lora_weights,
            lora_alphas=lora_alphas,
            lora_urls=lora_urls,
            lora_files=lora_files,
        )
        return result
