# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import io
import re
import time
import socket
import requests
import tempfile
import hashlib
import logging
import subprocess
import pandas as pd
import gradio as gr
from PIL import Image
from collections import Counter, defaultdict
from torch.hub import download_url_to_file
from unitorch import get_temp_dir
from unitorch.cli import CoreConfigureParser
from unitorch.cli import register_webui
from unitorch.cli.webuis import (
    create_element,
    create_accordion,
    create_row,
    create_column,
    create_flex_layout,
    create_group,
    create_tab,
    create_tabs,
    create_blocks,
)
from unitorch.cli.webuis import SimpleWebUI

_js = """
(() => {
    const shortcuts = (e) => {
        const event = document.all ? window.event : e;
        if(e.target.tagName.toLowerCase() == "body") {
            const code = e.key;
            if (code.toLowerCase() === "arrowright") {
                document.getElementById("ut-labeling-submit").click();
                document.activeElement.blur();
                window.focus();
            } else {
                const choices = document.getElementById("ut-labeling-choices").getElementsByTagName("label");
                if (/^[1-9]$/.test(code)) {
                    const index = parseInt(code, 10) - 1; // 数字键1对应choices[0]
                    if (index >= 0 && index < choices.length) {
                        choices[index].click();
                        document.activeElement.blur();
                        window.focus();
                    } else {
                        console.warn("Key index out of range:", index);
                    }
                }

            }
        }
    };
    document.addEventListener("keyup", shortcuts);
    console.log("Shortcut keys for labeling loaded.");
})();
"""

_css = """
img {
    transform-origin: bottom;
    transform: scale(0.85) !important;
}

video {
    transform-origin: bottom;
    transform: scale(0.85) !important;
}
"""


@register_webui("core/webui/labeling")
class LabelingWebUI(SimpleWebUI):
    def __init__(
        self,
        config: CoreConfigureParser,
    ):
        self._config = config
        config.set_default_section("core/webui/labeling")
        data_file = config.getoption("data_file", None)
        result_file = config.getoption("result_file", None)
        force_to_relabel = config.getoption("force_to_relabel", False)
        names = config.getoption("names", "*")
        quoting = config.getoption("quoting", 3)
        if isinstance(names, str) and names == "*":
            names = None
        if isinstance(names, str):
            names = re.split(r"[,;]", names)
            names = [n.strip() for n in names]

        sep = config.getoption("sep", "\t")
        temp_folder = config.getoption("temp_folder", get_temp_dir())
        os.makedirs(temp_folder, exist_ok=True)
        self.temp_folder = temp_folder
        self.tags = config.getoption("tags", "#Labeling")

        self.dataset = pd.read_csv(
            data_file,
            names=names,
            header="infer" if names is None else None,
            sep=sep,
            quoting=quoting,
        )
        self.dataset["Index"] = self.dataset.index.map(lambda x: f"No.{x}")
        self.result_file = result_file

        self.http_url = "/gradio_api/file={0}"
        # show columns
        self.group_text_cols = config.getoption("group_text_cols", None)
        self.text_cols = config.getoption("text_cols", None)
        self.image_cols = config.getoption("image_cols", None)
        self.video_cols = config.getoption("video_cols", None)
        self.url_cols = config.getoption("url_cols", None)
        self.html_cols = config.getoption("html_cols", None)
        self.show_cols = config.getoption("show_cols", None)
        self.zip_cols = config.getoption("zip_cols", None)
        self.zip_http_url = config.getoption("zip_http_url", None)
        self.group_col = config.getoption("group_col", None)
        self.pre_label_col = config.getoption("pre_label_col", None)
        self.num_group_texts_per_row = config.getoption("num_group_texts_per_row", 4)
        self.num_images_per_row = config.getoption("num_images_per_row", 4)
        self.num_videos_per_row = config.getoption("num_videos_per_row", 4)
        self.num_mix_images_videos_per_row = config.getoption(
            "num_mix_images_videos_per_row", 4
        )
        self.num_html_per_row = config.getoption("num_html_per_row", 4)

        # css settings
        self.min_image_width = config.getoption("min_image_width", "none")
        self.min_image_height = config.getoption("min_image_height", "none")
        self.max_image_width = config.getoption("max_image_width", "none")
        self.max_image_height = config.getoption("max_image_height", "100px")
        self.min_video_width = config.getoption("min_video_width", "none")
        self.min_video_height = config.getoption("min_video_height", "none")
        self.max_video_width = config.getoption("max_video_width", "none")
        self.max_video_height = config.getoption("max_video_height", "100px")

        if self.group_text_cols is not None:
            if isinstance(self.group_text_cols, str):
                self.group_text_cols = re.split(r"[,;]", self.group_text_cols)
                self.group_text_cols = [n.strip() for n in self.group_text_cols]
            assert all(
                [col in self.dataset.columns for col in self.group_text_cols]
            ), f"group_text_cols {self.group_text_cols} not found in dataset"
        else:
            self.group_text_cols = []

        if self.text_cols is not None:
            if isinstance(self.text_cols, str):
                self.text_cols = re.split(r"[,;]", self.text_cols)
                self.text_cols = [n.strip() for n in self.text_cols]

            self.text_cols = [
                col for col in self.text_cols if col not in self.group_text_cols
            ] + self.group_text_cols
            assert all(
                [col in self.dataset.columns for col in self.text_cols]
            ), f"text_cols {self.text_cols} not found in dataset"
        else:
            self.text_cols = self.group_text_cols

        if self.image_cols is not None:
            if isinstance(self.image_cols, str):
                self.image_cols = re.split(r"[,;]", self.image_cols)
                self.image_cols = [n.strip() for n in self.image_cols]
            assert all(
                [col in self.dataset.columns for col in self.image_cols]
            ), f"image_cols {self.image_cols} not found in dataset"
        else:
            self.image_cols = []

        if self.video_cols is not None:
            if isinstance(self.video_cols, str):
                self.video_cols = re.split(r"[,;]", self.video_cols)
                self.video_cols = [n.strip() for n in self.video_cols]
            assert all(
                [col in self.dataset.columns for col in self.video_cols]
            ), f"video_cols {self.video_cols} not found in dataset"
        else:
            self.video_cols = []

        if self.url_cols is not None:
            if isinstance(self.url_cols, str):
                self.url_cols = re.split(r"[,;]", self.url_cols)
                self.url_cols = [n.strip() for n in self.url_cols]
            assert all(
                [col in self.dataset.columns for col in self.url_cols]
            ), f"url_cols {self.url_cols} not found in dataset"
        else:
            self.url_cols = []

        if self.html_cols is not None:
            if isinstance(self.html_cols, str):
                self.html_cols = re.split(r"[,;]", self.html_cols)
                self.html_cols = [n.strip() for n in self.html_cols]
            assert all(
                [col in self.dataset.columns for col in self.html_cols]
            ), f"html_cols {self.html_cols} not found in dataset"
        else:
            self.html_cols = []

        if self.show_cols is not None:
            if isinstance(self.show_cols, str):
                self.show_cols = re.split(r"[,;]", self.show_cols)
                self.show_cols = [n.strip() for n in self.show_cols]
            assert all(
                [col in self.dataset.columns for col in self.show_cols]
            ), f"show_cols {self.show_cols} not found in dataset"
        else:
            self.show_cols = list(self.dataset.columns)

        if self.zip_cols is not None:
            if isinstance(self.zip_cols, str):
                self.zip_cols = re.split(r"[,;]", self.zip_cols)
                self.zip_cols = [n.strip() for n in self.zip_cols]
            assert all(
                [col in self.dataset.columns for col in self.zip_cols]
            ), f"zip_cols {self.zip_cols} not found in dataset"
        else:
            self.zip_cols = []

        if len(self.zip_cols) > 0:
            assert self.zip_http_url is not None, "zip_http_url is required."

        self.num_group_text_cols = len(self.group_text_cols)
        self.num_text_cols = 0 if self.text_cols is None else len(self.text_cols)
        self.num_image_cols = 0 if self.image_cols is None else len(self.image_cols)
        self.num_video_cols = 0 if self.video_cols is None else len(self.video_cols)
        self.num_html_cols = 0 if self.html_cols is None else len(self.html_cols)
        self.guideline = config.getoption("guideline", None)
        self.choices = config.getoption("choices", None)
        self.checkbox = config.getoption("checkbox", False)
        self.use_shortcuts = config.getoption("use_shortcuts", False)
        self.html_styles = config.getoption("html_styles", {})
        self.dataset["User"] = ""
        self.dataset["Comment"] = ""
        self.dataset["Label"] = ""

        if self.pre_label_col is not None:
            assert (
                self.pre_label_col in self.dataset.columns
            ), f"pre_label_col {self.pre_label_col} not found in dataset"
            self.dataset["Label"] = self.dataset[self.pre_label_col].map(
                lambda x: str(x).strip()
            )
            self.dataset["Label"].fillna("", inplace=True)

        if isinstance(self.choices, str):
            self.choices = re.split(r"[,;]", self.choices)
            self.choices = [c.strip() for c in self.choices]
        self.choices = [str(c).strip() for c in self.choices]

        if os.path.exists(result_file) and not force_to_relabel:
            self.dataset = pd.read_csv(result_file, sep="\t")
        else:
            self.dataset.to_csv(result_file, sep="\t", index=False)

        self.logs = f"* {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}: Start Labeling. \n"

        # format dataset columns
        str_columns = ["User", "Comment", "Label"]
        str_columns += [self.group_col] if self.group_col is not None else []
        str_columns += (
            self.image_cols
            + self.video_cols
            + self.html_cols
            + self.url_cols
            + self.zip_cols
        )
        for col in str_columns:
            self.dataset[col].fillna("", inplace=True)
            self.dataset[col] = self.dataset[col].map(str)

        # create elements
        index = create_element(
            "dropdown",
            label="Index",
            default=" - ",
            values=[" - "] + self.dataset["Index"].tolist(),
        )
        if self.group_col is not None:
            group_values = self.dataset[self.group_col].unique().tolist()
        else:
            group_values = []
        group = create_element(
            "dropdown",
            label="Group",
            default=" - ",
            values=[" - "] + group_values,
        )
        user = create_element(
            "text",
            label="User",
            interactive=False,
            scale=2,
        )
        guideline_header = create_element(
            "markdown",
            label="# <div style='margin-top:10px'>Guideline</div>",
            interactive=False,
        )
        guideline = create_element(
            "markdown",
            label=f"{self.guideline}",
            interactive=False,
        )
        texts = [
            create_element(
                "text",
                label=col,
                lines=2,
            )
            for col in self.text_cols
        ]
        images = [
            create_element(
                "image",
                label=col,
            )
            for col in self.image_cols
        ]
        videos = [
            create_element(
                "video",
                label=col,
            )
            for col in self.video_cols
        ]
        htmls = [
            create_element(
                "html",
                label=col,
            )
            for col in self.html_cols
        ]
        choices = create_element(
            "radio" if not self.checkbox else "checkboxgroup",
            label="Label",
            values=self.choices,
            scale=3,
            elem_id="ut-labeling-choices",
        )

        comment = create_element(
            "text",
            label="Comment",
            lines=4,
            scale=2,
        )
        random_type = create_element(
            "radio",
            values=["All", "Labeled", "Unlabeled"],
            default="Unlabeled",
            label="Data Type",
            scale=2,
        )
        random = create_element(
            "button",
            label="Sample",
            scale=2,
        )
        submit = create_element(
            "button",
            label="Submit",
            elem_id="ut-labeling-submit",
        )
        reset = create_element(
            "button",
            label="Reset",
            variant="secondary",
        )
        progress = create_element(
            "text",
            label="Progress",
            interactive=False,
            scale=2,
        )
        res_disp_cols = create_element(
            "dropdown",
            label="Display Columns",
            default=" - ",
            values=[" - "] + self.show_cols,
            interactive=True,
            multiselect=True,
            scale=2,
        )
        res_label = create_element(
            "dropdown",
            label="Label",
            default=" - ",
            values=[" - "]
            + self.choices
            + ([] if not self.checkbox else [f"NOT {ch}" for ch in self.choices]),
        )

        adv_stats_header = create_element(
            "markdown",
            label="## <div style='margin-top:10px'>Distribution</div>",
            interactive=False,
        )
        adv_stats = create_element(
            "markdown",
            label="",
            interactive=False,
        )
        adv_logs_header = create_element(
            "markdown",
            label="## <div style='margin-top:10px'>Logs</div>",
            interactive=False,
        )
        adv_logs = create_element(
            "markdown",
            label="",
            interactive=False,
        )
        # show results
        data_type = create_element(
            "radio",
            values=["All", "Labeled", "Unlabeled"],
            default="Labeled",
            label="Data Type",
            scale=2,
        )
        refresh = create_element(
            "button",
            label="Refresh",
        )
        download = create_element(
            "download_button",
            label="Download",
        )
        results = create_element(
            "dataframe",
            label="Results",
            interactive=False,
        )

        # create blocks
        text_layout = (
            create_column(
                *texts[: -self.num_group_text_cols],
                create_flex_layout(
                    *texts[-self.num_group_text_cols :],
                    num_per_row=self.num_group_texts_per_row,
                ),
            )
            if self.num_group_text_cols > 0
            else create_column(*texts)
        )
        if (
            self.num_image_cols + self.num_video_cols
            > self.num_mix_images_videos_per_row
        ):
            image_layout = create_flex_layout(
                *images, num_per_row=self.num_images_per_row
            )
            video_layout = create_flex_layout(
                *videos, num_per_row=self.num_videos_per_row
            )
            mix_image_video_layout = None
        else:
            mix_image_video_layout = create_flex_layout(
                *images, *videos, num_per_row=self.num_mix_images_videos_per_row
            )
            image_layout, video_layout = None, None
        html_layout = create_flex_layout(*htmls, num_per_row=self.num_html_per_row)
        label_layout = create_row(
            comment,
            create_column(
                choices,
                create_row(reset, submit),
                scale=3,
            ),
        )

        layouts = []
        if self.num_text_cols > 0:
            layouts.append(text_layout)

        if self.num_image_cols > 0 and image_layout is not None:
            layouts.append(image_layout)

        if self.num_video_cols > 0 and video_layout is not None:
            layouts.append(video_layout)

        if mix_image_video_layout is not None:
            layouts.append(mix_image_video_layout)

        if self.num_html_cols > 0:
            layouts.append(html_layout)

        tab1 = create_tab(
            create_row(index, progress, user),
            create_row(group, random_type, random),
            *layouts,
            label_layout,
            name="Labeling",
        )
        tab2 = create_tab(
            create_row(progress, group, res_label, download),
            create_row(data_type, res_disp_cols, refresh),
            results,
            name="Results",
        )
        tab3 = create_tab(
            adv_stats_header,
            adv_stats,
            adv_logs_header,
            adv_logs,
            name="Advanced",
        )
        tabs = create_tabs(tab1, tab2, tab3)
        if self.use_shortcuts:
            iface = create_blocks(guideline_header, guideline, tabs, js=_js, css=_css)
        else:
            iface = create_blocks(guideline_header, guideline, tabs, css=_css)

        # create events
        iface.__enter__()
        submit.click(
            self.label,
            inputs=[index, group, user, choices, comment, adv_logs],
            outputs=[download, adv_stats, adv_logs, index],
            trigger_mode="once",
        )
        random.click(
            self.sample,
            inputs=[group, random_type],
            outputs=[index],
            trigger_mode="once",
        )
        refresh.click(
            self.show,
            inputs=[group, data_type, res_label, res_disp_cols],
            outputs=[progress, results],
            trigger_mode="once",
        )
        reset.click(
            self.reset,
            inputs=[index],
            outputs=[choices, comment, progress],
            trigger_mode="once",
        )
        index.change(
            fn=self.load,
            inputs=[index],
            outputs=[
                progress,
                group,
                choices,
                comment,
                *texts,
                *images,
                *videos,
                *htmls,
            ],
            trigger_mode="once",
        )

        iface.load(
            fn=self.sample,
            inputs=[group, random_type],
            outputs=[index],
        )
        iface.load(
            fn=lambda: tuple(self.show())
            + (os.path.abspath(self.result_file), self.stats()),
            inputs=[],
            outputs=[progress, results, download, adv_stats],
        )

        def get_user(request: gr.Request):
            if request:
                return request.username
            return None

        iface.load(fn=get_user, inputs=None, outputs=user)
        iface.load(fn=lambda: self.logs, inputs=None, outputs=adv_logs)

        iface.__exit__()

        super().__init__(config, iname="Labeling Tools", iface=iface)

    def process_sample(self, sample):
        def save_url(url):
            if url.startswith("http"):
                name = os.path.join(
                    self.temp_folder, hashlib.md5(url.encode()).hexdigest()
                )
                try:
                    if not os.path.exists(name):
                        download_url_to_file(url, name, progress=False)
                    return name
                except Exception as e:
                    print(e)
            return url

        for col in set(self.zip_cols):
            sample[col] = self.zip_http_url.format(sample[col])

        for col in self.image_cols + self.video_cols:
            sample[col] = save_url(sample[col])

        url = lambda x: (
            x if x.startswith("http") else self.http_url.format(os.path.abspath(x))
        )
        for col in self.url_cols:
            sample[col] = url(sample[col])

        for col in self.html_cols:
            sample[col] = self.html_styles.get(sample[col], "{0}").format(sample[col])
        return sample

    def process_results(self, results):
        url = lambda x: (
            x if x.startswith("http") else self.http_url.format(os.path.abspath(x))
        )

        for col in set(self.zip_cols):
            results[col] = results[col].map(lambda x: self.zip_http_url.format(x))

        for col in set(self.image_cols):
            results[col] = results[col].map(url)
            results[col] = results[col].map(
                lambda x: f'<img src="{x}" style="min-width: {self.min_image_width}; max-width: {self.max_image_width}; min-height: {self.min_image_height}; max-height: {self.max_image_height}; overflow: hidden;">'
            )
        for col in set(self.video_cols):
            results[col] = results[col].map(
                lambda x: (
                    x
                    if x.startswith("http")
                    else self.http_url.format(os.path.abspath(x))
                )
            )
            results[col] = results[col].map(
                lambda x: f'<video src="{x}" style="min-width: {self.min_video_width}; max-width: {self.max_video_width}; min-height: {self.min_video_height}; max-height: {self.max_video_height}; overflow: hidden;" preload="none" controls>'
            )

        for col in self.url_cols:
            results[col] = results[col].map(url)

        for col in set(self.html_cols):
            results[col] = results[col].map(
                lambda x: self.html_styles.get(x, "{0}").format(x)
            )
        return results

    def sample(self, group=None, random_type="Unlabeled"):
        if random_type == "Labeled":
            sampled_data = self.dataset[self.dataset["Label"] != ""]
        elif random_type == "Unlabeled":
            sampled_data = self.dataset[self.dataset["Label"] == ""]
        else:
            sampled_data = self.dataset

        if group is not None and group not in ["", " - "]:
            sampled_data = sampled_data[sampled_data[self.group_col] == group]

        if len(sampled_data) == 0:
            return " - "

        return sampled_data.sample(1).iloc[0]["Index"]

    def show(self, group=None, data_type="Labeled", choice=None, disp_cols=None):
        total = self.dataset.shape[0]
        progress = f"{len(self.dataset[self.dataset['Label'] != ''])} / {total}"

        results = self.dataset.copy()
        if data_type == "Labeled":
            results = results[results["Label"] != ""]
        elif data_type == "Unlabeled":
            results = results[results["Label"] == ""]

        if choice is not None and choice not in ["", " - "] and len(results) > 0:
            if self.checkbox and str(choice).startswith("NOT "):
                results = results[
                    results["Label"].map(lambda x: str(choice)[4:] not in x)
                ]
            else:
                results = results[results["Label"].map(lambda x: str(choice) in x)]

        if group is not None and group not in ["", " - "] and len(results) > 0:
            results = results[results[self.group_col] == str(group)]

        if len(results) > 0:
            results = self.process_results(results)

        if disp_cols is None:
            disp_cols = self.show_cols
        elif " - " in disp_cols:
            disp_cols = self.show_cols

        results = results[
            ["Index"]
            + [
                col
                for col in disp_cols
                if col not in ["Comment", "User", "Label", "Index"]
            ]
            + ["Label", "Comment"]
        ]
        return (progress, results)

    def reset(self, index):
        self.dataset.loc[self.dataset.Index == index, "User"] = ""
        self.dataset.loc[self.dataset.Index == index, "Label"] = ""
        self.dataset.loc[self.dataset.Index == index, "Comment"] = ""
        self.dataset.to_csv(self.result_file, sep="\t", index=False)
        total = self.dataset.shape[0]
        labeled = self.dataset[self.dataset["Label"] != ""].shape[0]
        progress = f"{labeled} / {total}"
        gr.Info(f"Reset {index} Success.")
        return None, None, progress

    def load(self, index):
        total = self.dataset.shape[0]
        progress = f"{len(self.dataset[self.dataset['Label'] != ''])} / {total}"

        if len(self.dataset[self.dataset["Index"] == index]) == 0:
            return (progress, None, None, None) + tuple(
                [None]
                * (
                    self.num_text_cols
                    + self.num_image_cols
                    + self.num_video_cols
                    + self.num_html_cols
                )
            )

        new_one = self.dataset[self.dataset["Index"] == index].iloc[0]
        new_one = self.process_sample(new_one)
        new_texts = new_one[self.text_cols].tolist()
        new_images = new_one[self.image_cols].tolist()
        new_videos = new_one[self.video_cols].tolist()
        new_htmls = new_one[self.html_cols].tolist()
        new_group = new_one[self.group_col] if self.group_col is not None else None
        new_comment = new_one["Comment"]
        new_choices = (
            new_one["Label"].split("<label-gap>") if new_one["Label"] != "" else None
        )
        new_choices = (
            new_choices[0]
            if not self.checkbox and new_choices is not None
            else new_choices
        )
        new_texts = [text if text != "" else None for text in new_texts]
        new_images = [img if img != "" else None for img in new_images]
        new_videos = [vid if vid != "" else None for vid in new_videos]
        new_htmls = [html if html != "" else None for html in new_htmls]

        return (
            (progress, new_group, new_choices, new_comment)
            + tuple(new_texts)
            + tuple(new_images)
            + tuple(new_videos)
            + tuple(new_htmls)
        )

    def stats(self):
        dataset = self.dataset.copy()
        dataset["Num"] = 1
        labeled = dataset[dataset["Label"] != ""]
        choices = self.choices

        labeled["Label"] = labeled["Label"].map(
            lambda x: x.split("<label-gap>") if x != "" else []
        )
        labeled_exploded = labeled.explode("Label")

        percent = lambda x, y: f"{x / (y if y > 0 else 1):.2%}"

        if self.group_col is not None:
            group = labeled.groupby(self.group_col)["Num"].count().to_dict()

            stats = labeled_exploded.groupby([self.group_col, "Label"])["Num"].count()
            stats = [
                {
                    "Group": g,
                    "Label": c,
                    "Count": f"{stats.get((g, c), 0)} / {group[g]}",
                    "Percentage": percent(stats.get((g, c), 0), group[g]),
                }
                for g in group.keys()
                for c in choices
            ]
            stats = pd.DataFrame(stats)
            stats = stats.append(
                {
                    "Group": "-",
                    "Label": "Total",
                    "Count": f"{len(labeled)} / {len(dataset)}",
                    "Percentage": percent(len(labeled), len(dataset)),
                },
                ignore_index=True,
            )

        else:
            stats = labeled_exploded.groupby("Label")["Num"].count()
            stats = [
                {
                    "Label": c,
                    "Count": stats.get(c, 0),
                    "Percentage": percent(stats.get(c, 0), len(labeled)),
                }
                for c in choices
            ]
            stats = pd.DataFrame(stats)
            stats = stats.append(
                {
                    "Label": "Total",
                    "Count": len(labeled),
                    "Percentage": percent(len(labeled), len(dataset)),
                },
                ignore_index=True,
            )

        stats = stats.to_markdown(index=False)
        return stats

    def label(
        self,
        index,
        group,
        user,
        choice,
        comment,
        logs,
    ):
        if (
            choice is None
            or choice == ""
            or (isinstance(choice, (list, tuple)) and len(choice) == 0)
        ):
            gr.Warning("Please ensure the label field is not left empty.")
            return os.path.abspath(self.result_file), self.stats(), logs, index
        if isinstance(choice, list) or isinstance(choice, tuple):
            choice = "<label-gap>".join(choice)
        self.dataset.loc[self.dataset.Index == index, "User"] = user
        self.dataset.loc[self.dataset.Index == index, "Label"] = choice
        self.dataset.loc[self.dataset.Index == index, "Comment"] = comment
        self.dataset.to_csv(self.result_file, sep="\t", index=False)

        if user is not None and user != "":
            new_logs = (
                f"* {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}: User {user} Label {index} to {choice} Success. \n"
                + self.logs
            )
        else:
            new_logs = (
                f"* {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}: Label {index} to {choice} Success. \n"
                + self.logs
            )
        self.logs = new_logs
        return (
            os.path.abspath(self.result_file),
            self.stats(),
            new_logs,
            self.sample(group),
        )
