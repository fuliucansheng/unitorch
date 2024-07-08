# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import re
import torch
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.models import GenericOutputs
from unitorch.models.clip import (
    ClipForPretrain as _ClipForPretrain,
    ClipProcessor,
)
from unitorch.utils import pop_value, nested_dict_value, read_file, read_json_file

from unitorch.cli import (
    get_global_config,
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
)
from unitorch.cli.models.clip import pretrained_clip_infos
from unitorch.cli.pipelines.blip import BlipForImageCaptionPipeline


class ClipInterrogatorPipeline(_ClipForPretrain):
    def __init__(
        self,
        config_path: str,
        vocab_path: str,
        merge_path: str,
        vision_config_path: str,
        max_seq_length: Optional[int] = 77,
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, int]] = "cpu",
    ):
        projection_dim = nested_dict_value(
            read_json_file(config_path), "projection_dim"
        )
        super().__init__(
            config_path=config_path,
            projection_dim=projection_dim,
        )
        self.processor = ClipProcessor(
            vocab_path=vocab_path,
            merge_path=merge_path,
            vision_config_path=vision_config_path,
            max_seq_length=max_seq_length,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)
        self.to(device=self._device)

        config = get_global_config()
        self.blip_pipe = BlipForImageCaptionPipeline.from_core_configure(
            config, pretrained_name="blip-image-captioning-large"
        )
        self.blip_pipe._device = self._device
        self.blip_pipe.to(device=self._device)

        artists = read_file(
            cached_path(
                "https://huggingface.co/datasets/fuliucansheng/hubfiles/raw/main/clip-interrogator/artists.txt"
            ),
            lines=True,
        )
        flavors = read_file(
            cached_path(
                "https://huggingface.co/datasets/fuliucansheng/hubfiles/raw/main/clip-interrogator/flavors.txt"
            ),
            lines=True,
        )
        mediums = read_file(
            cached_path(
                "https://huggingface.co/datasets/fuliucansheng/hubfiles/raw/main/clip-interrogator/mediums.txt"
            ),
            lines=True,
        )
        movements = read_file(
            cached_path(
                "https://huggingface.co/datasets/fuliucansheng/hubfiles/raw/main/clip-interrogator/movements.txt"
            ),
            lines=True,
        )
        negative = read_file(
            cached_path(
                "https://huggingface.co/datasets/fuliucansheng/hubfiles/raw/main/clip-interrogator/negative.txt"
            ),
            lines=True,
        )
        sites = read_file(
            cached_path(
                "https://huggingface.co/datasets/fuliucansheng/hubfiles/raw/main/clip-interrogator/sites.txt"
            ),
            lines=True,
        )

        artists = [f"by {a}" for a in artists] + [f"inspired by {a}" for a in artists]
        trendings = (
            sites
            + [f"trending on {s}" for s in sites]
            + [f"featured on {s}" for s in sites]
            + [f"{s} contest winner" for s in sites]
        )

        self.positive_labels = artists + flavors + mediums + movements + trendings
        self.positive_artists_labels = artists
        self.positive_flavors_labels = flavors
        self.positive_mediums_labels = mediums
        self.positive_movements_labels = movements
        self.positive_trendings_labels = trendings
        self.negative_labels = negative

        self.eval()

        self.positive_labels_embeds = self.get_text_embeds(self.positive_labels)

    @classmethod
    @add_default_section_for_init("core/pipeline/interrogator/clip")
    def from_core_configure(
        cls,
        config,
        pretrained_name: Optional[str] = "default-clip",
        config_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        merge_path: Optional[str] = None,
        vision_config_path: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = "cpu",
        **kwargs,
    ):
        config.set_default_section("core/pipeline/interrogator/clip")
        pretrained_name = config.getoption("pretrained_name", pretrained_name)

        config_path = config.getoption("config_path", config_path)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)

        vocab_path = config.getoption("vocab_path", vocab_path)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "vocab"),
        )
        vocab_path = cached_path(vocab_path)

        merge_path = config.getoption("merge_path", merge_path)
        merge_path = pop_value(
            merge_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "merge"),
        )
        merge_path = cached_path(merge_path)

        vision_config_path = config.getoption("vision_config_path", vision_config_path)
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "vision_config"),
        )
        vision_config_path = cached_path(vision_config_path)

        max_seq_length = config.getoption("max_seq_length", 77)
        device = config.getoption("device", device)
        pretrained_weight_path = config.getoption(
            "pretrained_weight_path", pretrained_weight_path
        )
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "weight"),
            check_none=False,
        )

        inst = cls(
            config_path,
            vocab_path,
            merge_path,
            vision_config_path,
            max_seq_length=max_seq_length,
            weight_path=weight_path,
            device=device,
        )

        return inst

    def get_image_embeds(self, image: Image.Image):
        inputs = self.processor.image_classification(image)
        inputs = {k: v.unsqueeze(0) if v is not None else v for k, v in inputs.items()}
        inputs = {
            k: v.to(device=self._device) if v is not None else v
            for k, v in inputs.items()
        }
        vision_outputs = self.vision_model(pixel_values=inputs["pixel_values"])
        image_embeds = self.visual_projection(vision_outputs[1])
        return image_embeds / image_embeds.norm(dim=-1, keepdim=True)

    def get_text_embeds(self, texts: Union[str, List[str]], max_batch_size=1024):
        if isinstance(texts, str):
            texts = [texts]
        inputs = [self.processor.text_classification(text) for text in texts]
        keys = inputs[0].keys()
        inputs = {
            k: torch.stack([i[k] for i in inputs]).to(device=self._device) for k in keys
        }
        results = []
        for i in range(0, len(inputs["input_ids"]), max_batch_size):
            text_outputs = self.text_model(
                input_ids=inputs["input_ids"][i : i + max_batch_size],
                attention_mask=inputs["attention_mask"][i : i + max_batch_size],
                position_ids=inputs["position_ids"][i : i + max_batch_size],
            )
            text_embeds = self.text_projection(text_outputs[1])
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            results.append(text_embeds)
        return torch.cat(results, dim=0)

    def get_score(self, image_embeds, text):
        text_embeds = self.get_text_embeds(text)
        scores = torch.einsum("te,be->tb", text_embeds, image_embeds)
        return scores[0][0].item()

    def rank_top(self, image_embeds, texts=[], topk=1, reverse=True, text_embeds=None):
        if text_embeds is None:
            text_embeds = self.get_text_embeds(texts)
        scores = torch.einsum("te,be->tb", text_embeds, image_embeds)
        scores = scores[:, 0].tolist()
        return sorted(list(zip(texts, scores)), key=lambda x: x[1], reverse=reverse)[
            :topk
        ]

    def chain(
        self,
        image_embeds,
        phrases,
        caption=None,
        min_count=8,
        max_count=32,
        reverse=True,
    ):
        if caption is None:
            caption = self.rank_top(image_embeds, phrases, topk=1, reverse=reverse)[0][
                0
            ]
        best_prompt, best_score = caption, self.get_score(image_embeds, caption)
        if best_prompt in phrases:
            phrases.remove(best_prompt)

        prompt, score = best_prompt, best_score

        for i in range(max_count):
            new_prompt, new_score = self.rank_top(
                image_embeds,
                [f"{prompt}, {p}" for p in phrases],
                topk=1,
                reverse=reverse,
            )[0]

            label = new_prompt[len(prompt) + 2 :]
            phrases.remove(label)

            if (reverse and new_score > best_score) or (
                not reverse and new_score < best_score
            ):
                best_prompt, best_score = new_prompt, new_score
            if (
                i < min_count
                or (reverse and new_score < score)
                or (not reverse and new_score > score)
            ):
                prompt, score = new_prompt, new_score
            else:
                break

        return best_prompt

    def get_best_prompt(self, image_embeds, caption="", min_count=8, max_count=32):
        positive_labels = self.rank_top(
            image_embeds, topk=1024, text_embeds=self.positive_labels_embeds
        )
        positive_labels = list(map(lambda x: x[0], positive_labels))

        best_prompt = self.chain(
            image_embeds, positive_labels, caption, min_count, max_count
        )
        return best_prompt

    def get_fast_prompt(self, image_embeds, caption="", max_count=32):
        positive_labels = self.rank_top(
            image_embeds, topk=max_count, text_embeds=self.positive_labels_embeds
        )
        positive_labels = list(map(lambda x: x[0], positive_labels))

        fast_prompt = f"{caption}, {', '.join(positive_labels)}"
        return fast_prompt

    def get_classic_prompt(self, image_embeds, caption="", max_count=3):
        positive_embeds = self.positive_labels_embeds.clone()
        positive_artists_labels_embeds, positive_embeds = (
            positive_embeds[: len(self.positive_artists_labels)],
            positive_embeds[len(self.positive_artists_labels) :],
        )
        positive_flavors_labels_embeds, positive_embeds = (
            positive_embeds[: len(self.positive_flavors_labels)],
            positive_embeds[len(self.positive_flavors_labels) :],
        )
        positive_mediums_labels_embeds, positive_embeds = (
            positive_embeds[: len(self.positive_mediums_labels)],
            positive_embeds[len(self.positive_mediums_labels) :],
        )
        positive_movements_labels_embeds, positive_embeds = (
            positive_embeds[: len(self.positive_movements_labels)],
            positive_embeds[len(self.positive_movements_labels) :],
        )
        positive_trendings_labels_embeds, positive_embeds = (
            positive_embeds[: len(self.positive_trendings_labels)],
            positive_embeds[len(self.positive_trendings_labels) :],
        )

        artist = self.rank_top(
            image_embeds, topk=1, text_embeds=positive_artists_labels_embeds
        )[0][0]
        flavors = self.rank_top(
            image_embeds, topk=max_count, text_embeds=positive_flavors_labels_embeds
        )
        flavors = ", ".join(list(map(lambda x: x[0], flavors)))
        medium = self.rank_top(
            image_embeds, topk=1, text_embeds=positive_mediums_labels_embeds
        )[0][0]
        movement = self.rank_top(
            image_embeds, topk=1, text_embeds=positive_movements_labels_embeds
        )[0][0]
        trending = self.rank_top(
            image_embeds, topk=1, text_embeds=positive_trendings_labels_embeds
        )[0][0]

        if caption.startswith(medium):
            classic_prompt = f"{caption} {artist}, {trending}, {movement}, {flavors}"
        else:
            classic_prompt = (
                f"{caption}, {medium} {artist}, {trending}, {movement}, {flavors}"
            )

        return classic_prompt

    def get_negative_prompt(self, image_embeds, max_count=3):
        positive_flavors_labels_embeds = self.positive_labels_embeds[
            len(self.positive_artists_labels) : len(self.positive_artists_labels)
            + len(self.positive_flavors_labels)
        ]
        negative_labels = self.rank_top(
            image_embeds,
            topk=max_count,
            reverse=False,
            text_embeds=positive_flavors_labels_embeds,
        )
        negative_labels = list(map(lambda x: x[0], negative_labels))
        negative_labels += self.negative_labels
        negative_prompt = self.chain(
            image_embeds, negative_labels, max_count=max_count, reverse=False
        )
        return negative_prompt

    @torch.no_grad()
    @add_default_section_for_function("core/pipeline/interrogator/clip")
    def __call__(
        self,
        image: Image.Image,
    ):
        caption = self.blip_pipe(image, num_beams=5, max_gen_seq_length=32)
        image_embeds = self.get_image_embeds(image)
        fast_prompt = self.get_fast_prompt(image_embeds, caption)
        classic_prompt = self.get_classic_prompt(image_embeds, caption)
        best_prompt = self.get_best_prompt(image_embeds, caption)
        negative_prompt = self.get_negative_prompt(image_embeds)
        return GenericOutputs(
            fast_prompt=fast_prompt,
            classic_prompt=classic_prompt,
            best_prompt=best_prompt,
            negative_prompt=negative_prompt,
        )
