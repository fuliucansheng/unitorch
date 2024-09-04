# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import io
import os
import torch
import gc
import random
import requests
import gradio as gr
import pandas as pd
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.cli import CoreConfigureParser, GenericWebUI
from unitorch.cli import register_webui
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
from unitorch_microsoft.china.alibaba.pipeline import (
    BletchleyAli1688ImageSelectionPipeline,
)
from unitorch_microsoft.webuis.labeling.classification import (
    GenericClassificationLabelingWebUI,
)

0, 0, 0, 64 = lambda url: Image.open(
    requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"
        },
        stream=True,
    ).raw
).convert("RGB")

@register_webui("microsoft/china/webui/ali1688")
class Ali1688ImageSelectionWebUI(SimpleWebUI):
    supported_pretrained_names = ["2.5B"]

    def __init__(self, config: CoreConfigureParser):
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe
        self._status = "Stopped" if self._pipe is None else "Running"
        if len(self.supported_pretrained_names) == 0:
            raise ValueError("No supported pretrained models found.")
        self._name = self.supported_pretrained_names[0]

        # create elements
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

        text = create_element("text", "Input Text", lines=3, scale=2)
        image = create_element("image", "Input Image", scale=2)
        topk = create_element(
            "slider", "Top K", default=100, min_value=1, max_value=250, step=1
        )
        search = create_element("button", "Search")
        result = create_element("dataframe", "Search Result")

        # create blocks
        top1 = create_row(text, image, create_column(topk, search))
        top2 = create_row(result)
        iface = create_blocks(pretrain_layout, top1, top2)

        # create events
        iface.__enter__()

        start.click(self.start, inputs=[name], outputs=[status])
        stop.click(self.stop, outputs=[status])
        search.click(self.serve, inputs=[text, image, topk], outputs=[result])

        iface.load(
            fn=lambda: [gr.update(value=self._name), gr.update(value=self._status)],
            outputs=[name, status],
        )

        iface.__exit__()

        super().__init__(config, iname="Ali1688", iface=iface)

    def start(self, config_type, **kwargs):
        if self._status == "Running":
            return self._status
        self._pipe = BletchleyAli1688ImageSelectionPipeline.from_core_configure(
            self._config,
            config_type="2.5B",
            pretrained_weight_path="https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/image/pytorch_model.bletchley.v1.retrieval.61.28.bin",
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
        text: Optional[str] = None,
        image: Optional[Image.Image] = None,
        topk: Optional[int] = 10,
    ):
        if text.strip() == "":
            text = None
        assert self._pipe is not None
        result = self._pipe(text, image, topk=topk)
        return result


@register_webui("microsoft/china/webui/ali1688/humanlabel")
class Ali1688ImageHumanlabelWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        names = [
            "offerid",
            "offerurl",
            "imageurl",
            "title",
            "1688_images",
            "image_name",
            "retrive_image_name",
            "image_path",
            "seg_path",
            "unique_id",
            "prompt",
        ]
        dataset = pd.read_csv(
            "/data/decu/1688/0805_controlnet_input_with_prompt.tsv",
            names=names,
            header=None,
            sep="\t",
            quoting=3,
        )
        dataset["offerid"] = dataset.offerid.astype(str)
        self.controlnet_foldeer = "/home/chunchen/1688/res_900_471_0805"
        dataset["controlnet_image"] = dataset.unique_id + ".png"
        self.dataset = dataset[
            ["offerid", "offerurl", "imageurl", "title", "controlnet_image"]
        ].drop_duplicates()
        self.dataset["label"] = -2

        if os.path.exists(
            "/data/decu/1688/0805_controlnet_input_with_prompt.results.tsv"
        ):
            self.dataset = pd.read_csv(
                "/data/decu/1688/0805_controlnet_input_with_prompt.results.tsv",
                sep="\t",
            )
            self.dataset["offerid"] = self.dataset.offerid.astype(str)

        # create elements
        offerid = create_element("text", "Offer ID", scale=1)
        offerurl = create_element("text", "Offer URL")
        offerimage = create_element("image", "Offer Image")
        title = create_element("text", "Title")
        images = create_element("gallery", "Images")
        images.allow_preview = False
        best_name = create_element("text", "Best Image Name")
        best_image = create_element("image", "Best Image Preview")
        sample = create_element("button", "Sample")
        submit = create_element("button", "Submit")

        # create blocks
        top = create_row(create_column(offerid, offerurl, title), offerimage)
        left = create_column(images)
        right = create_column(best_name, best_image, create_row(sample, submit))
        iface = create_blocks(
            top,
            create_row(left, right),
        )

        # create events
        iface.__enter__()

        def func(data: gr.SelectData):
            return data.value["image"]["path"], data.value["image"]["orig_name"]

        images.select(func, outputs=[best_image, best_name])

        sample.click(
            self.sample,
            inputs=[offerid],
            outputs=[offerid, offerurl, offerimage, title, images],
        )
        submit.click(
            self.serve,
            inputs=[offerid, images, best_name],
            outputs=[
                offerid,
                offerurl,
                offerimage,
                title,
                images,
                best_name,
                best_image,
            ],
        )

        iface.load(
            fn=self.sample, outputs=[offerid, offerurl, offerimage, title, images]
        )

        iface.__exit__()

        super().__init__(config, iname="Ali1688ImageHumanLabel", iface=iface)

    def sample(self, offerid=None, topk=40):
        non_labeled = self.dataset[self.dataset.label == -2]
        offerset = set(non_labeled.offerid)

        if offerid is None or offerid not in offerset:
            offerid = random.choice(list(offerset))
        items = non_labeled[non_labeled.offerid == offerid]
        if len(items) > topk:
            items = items.sample(topk)

        offerid, offerurl, imageurl, title = items.iloc[0][
            ["offerid", "offerurl", "imageurl", "title"]
        ]
        images = [
            self.controlnet_foldeer + f"/{p}" for p in items.controlnet_image.tolist()
        ]
        offerimage = 0, 0, 0, 64(imageurl)
        return offerid, offerurl, offerimage, title, images

    def serve(
        self,
        offerid,
        images,
        best,
        topk=40,
    ):
        if best is not None and best != "":
            self.dataset.loc[self.dataset.offerid == str(offerid), "label"] = 1
            self.dataset.loc[
                (self.dataset.offerid == str(offerid))
                & (self.dataset.controlnet_image == best),
                "label",
            ] = 2
        else:
            images = [os.path.basename(t[0]) for t in images]
            self.dataset.loc[
                (self.dataset.offerid == str(offerid))
                & (self.dataset.controlnet_image.isin(images)),
                "label",
            ] = 0
        print(self.dataset.label.value_counts())
        self.dataset.to_csv(
            "/data/decu/1688/0805_controlnet_input_with_prompt.results.tsv",
            sep="\t",
            index=False,
        )
        offerid, offerurl, offerimage, title, images = self.sample(offerid, topk)

        return offerid, offerurl, offerimage, title, images, None, None


@register_webui("microsoft/china/webui/ali1688/flight/image/labeling")
class FlightImageLabelingWebUI(GenericClassificationLabelingWebUI):
    def __init__(self, config: CoreConfigureParser):
        super().__init__(
            config,
            default_section="microsoft/china/webui/ali1688/flight/image/labeling",
        )

    def postprocess_htmls(self, *htmls, info=None):
        htmls = list(htmls)
        htmls[
            0
        ] = f"""<div style="height:157px;width:300px;position:relative;overflow:hidden;"> <div style="height: 100%;;background-repeat: no-repeat;;background-position: center;background-size: cover"></div> <img style="position: absolute;top: 0px;bottom:0px;left:50%;transform: translateX(-50%);object-fit: contain;max-width: 100%;max-height:100%" src="{htmls[0]}" alt="慵懒风宽松毛衣，金智秀同款黑白棋盘设计，质优价廉！" /> </div>"""
        if info["BackgroundType"] != "White":
            htmls[1] += "&pcl=f5f5f5"
        htmls[
            1
        ] = f"""<div style="height:157px;width:300px;position:relative;overflow:hidden;"> <div style="height: 100%;;background-repeat: no-repeat;;background-position: center;background-size: cover"></div> <img style="position: absolute;top: 0px;bottom:0px;left:50%;transform: translateX(-50%);object-fit: contain;max-width: 100%;max-height:100%" src="{htmls[1]}" alt="慵懒风宽松毛衣，金智秀同款黑白棋盘设计，质优价廉！" /> </div>"""
        htmls[
            2
        ] = f"""<div style="height:157px;width:300px;position:relative;overflow:hidden;"> <div style="height: 100%;;filter: blur(90px);background-repeat: no-repeat;;background-position: center;background-size: cover;background-image: url('{htmls[2]}');"></div> <img style="position: absolute;top: 0px;bottom:0px;left:50%;transform: translateX(-50%);object-fit: contain;max-width: 100%;max-height:100%" src="{htmls[2]}&amp;h=157" alt="慵懒风宽松毛衣，金智秀同款黑白棋盘设计，质优价廉！" /> </div>"""
        return tuple(htmls)

    def process_show_cols(self, results, show_cols=None):
        results["CenterCropURL"] = results["CenterCropURL"].map(
            lambda x: f"""<div style="height:157px;width:300px;position:relative;overflow:hidden;"> <div style="height: 100%;;background-repeat: no-repeat;;background-position: center;background-size: cover"></div> <img style="position: absolute;top: 0px;bottom:0px;left:50%;transform: translateX(-50%);object-fit: contain;max-width: 100%;max-height:100%" src="{x}" alt="慵懒风宽松毛衣，金智秀同款黑白棋盘设计，质优价廉！" /> </div>"""
        )
        results["FullROIImageURL"] = results["FullROIImageURL"].map(
            lambda x: f"""<div style="height:157px;width:300px;position:relative;overflow:hidden;"> <div style="height: 100%;;background-repeat: no-repeat;;background-position: center;background-size: cover"></div> <img style="position: absolute;top: 0px;bottom:0px;left:50%;transform: translateX(-50%);object-fit: contain;max-width: 100%;max-height:100%" src="{x}" alt="慵懒风宽松毛衣，金智秀同款黑白棋盘设计，质优价廉！" /> </div>"""
        )
        results["BlurURL"] = results["BlurURL"].map(
            lambda x: f"""<div style="height:157px;width:300px;position:relative;overflow:hidden;"> <div style="height: 100%;;filter: blur(90px);background-repeat: no-repeat;;background-position: center;background-size: cover;background-image: url('{x}');"></div> <img style="position: absolute;top: 0px;bottom:0px;left:50%;transform: translateX(-50%);object-fit: contain;max-width: 100%;max-height:100%" src="{x}&amp;h=157" alt="慵懒风宽松毛衣，金智秀同款黑白棋盘设计，质优价廉！" /> </div>"""
        )
        return results
