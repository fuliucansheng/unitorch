# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import io
import torch
import gc
import gradio as gr
from PIL import Image
from unitorch.cli import CoreConfigureParser, GenericWebUI
from unitorch.cli import register_webui
from unitorch.cli.pipelines.bloom import BloomForGenerationPipeline
from unitorch.cli.pipelines.peft.bloom import BloomLoraForGenerationPipeline
from unitorch.cli.models.bloom import pretrained_bloom_infos
from unitorch.cli.webuis import matched_pretrained_names


class BloomGenerationWebUI(GenericWebUI):
    match_patterns = [
        "^bloom",
    ]
    pretrained_names = list(pretrained_bloom_infos.keys())
    supported_pretrained_names = matched_pretrained_names(
        pretrained_names,
        match_patterns,
    )

    def __init__(self, config: CoreConfigureParser):
        self.config = config
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe
        self._status = "stopped" if self._pipe is None else "running"
        if len(self.supported_pretrained_names) == 0:
            raise ValueError("No supported pretrained models found.")
        self._name = self.supported_pretrained_names[0]
        self._iface = gr.Blocks()
        with self._iface:
            with gr.Row(variant="panel"):
                pretrained_name = gr.Dropdown(
                    self.supported_pretrained_names,
                    value=self._name,
                    label="Pretrain Checkpoint Name",
                )
                status = gr.Textbox(label="Model Status", value=self._status)
                click_start = gr.Button(value="Start")
                click_stop = gr.Button(value="Stop")
                click_start.click(
                    self.start, inputs=[pretrained_name], outputs=[status]
                )
                click_stop.click(self.stop, outputs=[status])

            with gr.Row(variant="panel"):
                with gr.Column():
                    prompt = gr.Textbox(label="Input Prompt")
                    submit = gr.Button(value="Submit")

                result = gr.Textbox(label="Output Result")

                submit.click(self.serve, inputs=[prompt], outputs=[result])

            self._iface.load(
                fn=lambda: gr.update(value=self._name), outputs=[pretrained_name]
            )
            self._iface.load(fn=lambda: gr.update(value=self._status), outputs=[status])

    @property
    def name(self):
        return "Bloom"

    @property
    def iface(self):
        return self._iface

    @property
    def status(self):
        return self._status == "running"

    def start(self, pretrained_name, **kwargs):
        if self._status == "running":
            self.stop()
        self.config.set("core/pipeline/bloom", "pretrained_name", pretrained_name)
        self._name = pretrained_name
        self._pipe = BloomForGenerationPipeline.from_core_configure(self.config)
        self._status = "running"
        return self._status

    def stop(self, **kwargs):
        self._pipe.to("cpu")
        del self._pipe
        gc.collect()
        torch.cuda.empty_cache()
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe
        self._status = "stopped" if self._pipe is None else "running"
        return self._status

    def serve(
        self,
        text: str,
    ):
        assert self._pipe is not None
        result = self._pipe(text)
        return result


class PeftBloomLoraGenerationWebUI(GenericWebUI):
    match_patterns = [
        "^peft-lora-bloom",
    ]
    pretrained_names = list(pretrained_bloom_infos.keys())
    supported_pretrained_names = matched_pretrained_names(
        pretrained_names,
        match_patterns,
    )

    def __init__(self, config: CoreConfigureParser):
        self.config = config
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe
        self._status = "stopped" if self._pipe is None else "running"
        if len(self.supported_pretrained_names) == 0:
            raise ValueError("No supported pretrained models found.")
        self._name = self.supported_pretrained_names[0]
        self._iface = gr.Blocks()
        with self._iface:
            with gr.Row(variant="panel"):
                pretrained_name = gr.Dropdown(
                    self.supported_pretrained_names,
                    value=self._name,
                    label="Pretrain Checkpoint Name",
                )
                status = gr.Textbox(label="Model Status", value=self._status)
                click_start = gr.Button(value="Start")
                click_stop = gr.Button(value="Stop")
                click_start.click(
                    self.start, inputs=[pretrained_name], outputs=[status]
                )
                click_stop.click(self.stop, outputs=[status])

            with gr.Row(variant="panel"):
                with gr.Column():
                    prompt = gr.Textbox(label="Input Prompt")
                    submit = gr.Button(value="Submit")

                result = gr.Textbox(label="Output Result")

                submit.click(self.serve, inputs=[prompt], outputs=[result])

            self._iface.load(
                fn=lambda: gr.update(value=self._name), outputs=[pretrained_name]
            )
            self._iface.load(fn=lambda: gr.update(value=self._status), outputs=[status])

    @property
    def name(self):
        return "Peft-Bloom"

    @property
    def iface(self):
        return self._iface

    @property
    def status(self):
        return self._status == "running"

    def start(self, pretrained_name, **kwargs):
        if self._status == "running":
            self.stop()
        self.config.set(
            "core/pipeline/peft/bloom/lora", "pretrained_name", pretrained_name
        )
        self._name = pretrained_name
        self._pipe = BloomLoraForGenerationPipeline.from_core_configure(self.config)
        self._status = "running"
        return self._status

    def stop(self, **kwargs):
        self._pipe.to("cpu")
        del self._pipe
        gc.collect()
        torch.cuda.empty_cache()
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe
        self._status = "stopped" if self._pipe is None else "running"
        return self._status

    def serve(
        self,
        text: str,
    ):
        assert self._pipe is not None
        result = self._pipe(text)
        return result


@register_webui("core/webui/bloom")
class BloomWebUI(GenericWebUI):
    def __init__(self, config: CoreConfigureParser):
        webuis = [
            BloomGenerationWebUI(config),
            PeftBloomLoraGenerationWebUI(config),
        ]
        self._iface = gr.TabbedInterface(
            [webui.iface for webui in webuis],
            tab_names=[webui.name for webui in webuis],
        )

    def start(self):
        pass

    def stop(self):
        pass

    @property
    def name(self):
        return "Bloom"

    @property
    def iface(self):
        return self._iface
