# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import re
import logging
import torch
import pandas as pd
from PIL import Image
from typing import Any, Dict, List, Optional, Union
from sklearn.metrics import roc_auc_score

from unitorch.models import GenericOutputs
from unitorch.models.clip import ClipForPretrain as _ClipForPretrain, ClipProcessor
from unitorch.utils import pop_value, nested_dict_value, read_file, read_json_file
from unitorch.cli import (
    hf_endpoint_url,
    cached_path,
    config_defaults_init,
    register_script,
)
from unitorch.cli import Config, GenericScript
from unitorch.cli.models.clip import pretrained_clip_infos

_INTERROGATOR_BASE = "/datasets/fuliucansheng/hubfiles/raw/main/clip-interrogator"


def _load_interrogator_labels(filename: str) -> List[str]:
    return read_file(
        cached_path(hf_endpoint_url(f"{_INTERROGATOR_BASE}/{filename}")),
        lines=True,
    )


class ClipInterrogatorPipeline(_ClipForPretrain):
    def __init__(
        self,
        config_path: str,
        vocab_path: str,
        merge_path: str,
        vision_config_path: str,
        max_seq_length: int = 77,
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        device: Union[str, int] = "cpu",
    ):
        projection_dim = nested_dict_value(read_json_file(config_path), "projection_dim")
        super().__init__(config_path=config_path, projection_dim=projection_dim)

        self.processor = ClipProcessor(
            vocab_path=vocab_path,
            merge_path=merge_path,
            vision_config_path=vision_config_path,
            max_seq_length=max_seq_length,
        )
        self._device = "cpu" if device == "cpu" else int(device)
        self.from_pretrained(weight_path, state_dict=state_dict)
        self.to(device=self._device)

        artists = _load_interrogator_labels("artists.txt")
        flavors = _load_interrogator_labels("flavors.txt")
        mediums = _load_interrogator_labels("mediums.txt")
        movements = _load_interrogator_labels("movements.txt")
        negative = _load_interrogator_labels("negative.txt")
        sites = _load_interrogator_labels("sites.txt")

        artist_phrases = [f"by {a}" for a in artists] + [f"inspired by {a}" for a in artists]
        trending_phrases = (
            sites
            + [f"trending on {s}" for s in sites]
            + [f"featured on {s}" for s in sites]
            + [f"{s} contest winner" for s in sites]
        )

        self.positive_labels = artist_phrases + flavors + mediums + movements + trending_phrases
        self.positive_artists_labels = artist_phrases
        self.positive_flavors_labels = flavors
        self.negative_labels = negative

        self.eval()
        self.positive_labels_embeds = self.get_text_embeds(self.positive_labels)

    @classmethod
    @config_defaults_init("core/interrogator/clip")
    def from_config(
        cls,
        config,
        pretrained_name: Optional[str] = None,
        config_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        merge_path: Optional[str] = None,
        vision_config_path: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        device: str = "cpu",
        **kwargs,
    ):
        config.set_default_section("core/interrogator/clip")
        pretrained_name = config.getoption("pretrained_name", pretrained_name)

        config_path = cached_path(pop_value(
            config.getoption("config_path", config_path),
            nested_dict_value(pretrained_clip_infos, pretrained_name, "config"),
        ))
        vocab_path = cached_path(pop_value(
            config.getoption("vocab_path", vocab_path),
            nested_dict_value(pretrained_clip_infos, pretrained_name, "vocab"),
        ))
        merge_path = cached_path(pop_value(
            config.getoption("merge_path", merge_path),
            nested_dict_value(pretrained_clip_infos, pretrained_name, "merge"),
        ))
        vision_config_path = cached_path(pop_value(
            config.getoption("vision_config_path", vision_config_path),
            nested_dict_value(pretrained_clip_infos, pretrained_name, "vision_config"),
        ))
        weight_path = pop_value(
            config.getoption("pretrained_weight_path", pretrained_weight_path),
            nested_dict_value(pretrained_clip_infos, pretrained_name, "weight"),
            check_none=False,
        )

        return cls(
            config_path=config_path,
            vocab_path=vocab_path,
            merge_path=merge_path,
            vision_config_path=vision_config_path,
            max_seq_length=config.getoption("max_seq_length", 77),
            weight_path=weight_path,
            device=config.getoption("device", device),
        )

    def get_image_embeds(self, images: List[Image.Image], batch_size: int = 128) -> torch.Tensor:
        inputs = [self.processor.image_classification(image) for image in images]
        keys = inputs[0].keys()
        batched = {k: torch.stack([i[k] for i in inputs]).to(device=self._device) for k in keys}

        results = []
        for i in range(0, len(batched["pixel_values"]), batch_size):
            vision_out = self.vision_model(pixel_values=batched["pixel_values"][i:i + batch_size])
            embeds = self.visual_projection(vision_out[1])
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            results.append(embeds.cpu())
        return torch.cat(results, dim=0)

    def get_text_embeds(self, texts: Union[str, List[str]], batch_size: int = 128) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        inputs = [self.processor.text_classification(text) for text in texts]
        keys = inputs[0].keys()
        batched = {k: torch.stack([i[k] for i in inputs]).to(device=self._device) for k in keys}

        results = []
        for i in range(0, len(batched["input_ids"]), batch_size):
            text_out = self.text_model(
                input_ids=batched["input_ids"][i:i + batch_size],
                attention_mask=batched["attention_mask"][i:i + batch_size],
                position_ids=batched["position_ids"][i:i + batch_size],
            )
            embeds = self.text_projection(text_out[1])
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            results.append(embeds.cpu())
        return torch.cat(results, dim=0)

    def _similarity_scores(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        labels: List[int],
    ) -> List[float]:
        scores = torch.einsum("te,be->tb", text_embeds, image_embeds)
        return [roc_auc_score(labels, s.tolist()) for s in scores]

    def get_score(self, image_embeds: torch.Tensor, text: str, labels: List[int]) -> float:
        text_embeds = self.get_text_embeds(text)
        return self._similarity_scores(image_embeds, text_embeds, labels)[0]

    def rank_top(
        self,
        image_embeds: torch.Tensor,
        texts: Optional[List[str]] = None,
        topk: int = 1,
        reverse: bool = True,
        text_embeds: Optional[torch.Tensor] = None,
        labels: Optional[List[int]] = None,
    ) -> List[tuple]:
        if text_embeds is None:
            text_embeds = self.get_text_embeds(texts or [])
        scores = self._similarity_scores(image_embeds, text_embeds, labels)
        ranked = sorted(zip(texts, scores), key=lambda x: x[1], reverse=reverse)
        return ranked[:topk]

    def chain(
        self,
        image_embeds: torch.Tensor,
        phrases: List[str],
        caption: Optional[str] = None,
        min_count: int = 8,
        max_count: int = 32,
        reverse: bool = True,
        labels: Optional[List[int]] = None,
    ) -> str:
        if caption is None:
            caption = self.rank_top(image_embeds, phrases, topk=1, reverse=reverse, labels=labels)[0][0]

        best_prompt = caption
        best_score = self.get_score(image_embeds, caption, labels=labels)
        if best_prompt in phrases:
            phrases.remove(best_prompt)

        prompt, score = best_prompt, best_score

        for i in range(max_count):
            candidates = [f"{prompt}, {p}" for p in phrases]
            new_prompt, new_score = self.rank_top(
                image_embeds, candidates, topk=1, reverse=reverse, labels=labels
            )[0]

            added_label = new_prompt[len(prompt) + 2:]
            phrases.remove(added_label)

            if (reverse and new_score > best_score) or (not reverse and new_score < best_score):
                best_prompt, best_score = new_prompt, new_score

            if i < min_count or (reverse and new_score >= score) or (not reverse and new_score <= score):
                prompt, score = new_prompt, new_score
            else:
                break

        return best_prompt

    def get_best_prompt(
        self,
        image_embeds: torch.Tensor,
        min_count: int = 8,
        max_count: int = 32,
        labels: Optional[List[int]] = None,
    ) -> str:
        top_labels = self.rank_top(
            image_embeds,
            texts=self.positive_labels,
            topk=1024,
            text_embeds=self.positive_labels_embeds,
            labels=labels,
        )
        return self.chain(
            image_embeds,
            [label for label, _ in top_labels],
            min_count=min_count,
            max_count=max_count,
            labels=labels,
        )

    def get_negative_prompt(
        self,
        image_embeds: torch.Tensor,
        max_count: int = 3,
        labels: Optional[List[int]] = None,
    ) -> str:
        flavor_start = len(self.positive_artists_labels)
        flavor_end = flavor_start + len(self.positive_flavors_labels)
        flavor_embeds = self.positive_labels_embeds[flavor_start:flavor_end]

        bottom_flavors = self.rank_top(
            image_embeds,
            texts=self.positive_flavors_labels,
            topk=max_count,
            reverse=False,
            text_embeds=flavor_embeds,
            labels=labels,
        )
        negative_phrases = [label for label, _ in bottom_flavors] + self.negative_labels
        return self.chain(
            image_embeds,
            negative_phrases,
            max_count=max_count,
            reverse=False,
            labels=labels,
        )

    @torch.no_grad()
    def __call__(
        self,
        images: List[Image.Image],
        labels: List[int],
    ) -> GenericOutputs:
        image_embeds = self.get_image_embeds(images)
        return GenericOutputs(
            best_prompt=self.get_best_prompt(image_embeds, labels=labels),
            negative_prompt=self.get_negative_prompt(image_embeds, labels=labels),
        )


@register_script("core/script/interrogator/clip")
class ClipInterrogatorScript(GenericScript):
    def __init__(self, config: Config):
        self.config = config

    def launch(self, **kwargs):
        config = self.config
        pipe = ClipInterrogatorPipeline.from_config(config)

        config.set_default_section("core/script/interrogator/clip")
        data_file = config.getoption("data_file", None)
        image_col = config.getoption("image_col", None)
        label_col = config.getoption("label_col", None)
        do_reverse = config.getoption("do_reverse", False)

        names = config.getoption("names", None)
        if isinstance(names, str):
            names = None if names.strip() == "*" else [n.strip() for n in re.split(r"[,;]", names)]

        data = pd.read_csv(data_file, names=names, sep="\t", quoting=3, header=None)
        assert image_col in data.columns, f"Column '{image_col}' not found in data."
        assert label_col in data.columns, f"Column '{label_col}' not found in data."

        images = [Image.open(path).convert("RGB") for path in data[image_col]]
        labels = [1 - int(v) if do_reverse else int(v) for v in data[label_col]]

        results = pipe(images=images, labels=labels)
        logging.info(f"Best Prompt: {results.best_prompt}")
        logging.info(f"Negative Prompt: {results.negative_prompt}")
