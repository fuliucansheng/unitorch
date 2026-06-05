# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Any, Dict, List, Optional

from transformers.models.gemma import GemmaTokenizer

from unitorch.models import HfLlmProcessor
from unitorch.utils import read_json_file


def _load_gemma_tokenizer(
    tokenizer_file: str,
    tokenizer_config: Optional[str] = None,
    chat_template: Optional[str] = None,
) -> GemmaTokenizer:
    tokenizer_kwargs = read_json_file(tokenizer_config) if tokenizer_config else {}
    tokenizer = GemmaTokenizer(
        tokenizer_file=tokenizer_file,
        **tokenizer_kwargs,
    )
    if chat_template:
        tokenizer.chat_template = read_json_file(chat_template)["chat_template"]

    tokenizer.cls_token = tokenizer.bos_token
    tokenizer.sep_token = tokenizer.eos_token
    return tokenizer


def _render_message_content(
    content: Any,
    image_token: Optional[str] = None,
) -> str:
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            item_type = item.get("type")
            if item_type == "text":
                parts.append(str(item.get("text", "")))
            elif item_type == "image":
                parts.append(image_token or "<|image|>")
            else:
                parts.append(str(item.get("text", item.get("content", ""))))
        return "\n".join(filter(None, map(str.strip, parts))).strip()

    if isinstance(content, dict):
        if content.get("type") == "image":
            return image_token or "<|image|>"
        return str(content.get("text", content.get("content", ""))).strip()

    return str(content).strip()


def _fallback_chat_template(
    messages: List[Dict[str, Any]],
    image_token: Optional[str] = None,
) -> str:
    rendered = []
    for message in messages:
        role = str(message.get("role", "user")).strip()
        content = _render_message_content(
            message.get("content", ""),
            image_token=image_token,
        )
        rendered.append(f"{role}: {content}" if content else f"{role}:")

    if not messages or messages[-1].get("role") != "assistant":
        rendered.append("assistant:")

    return "\n".join(rendered).strip()


class GemmaProcessor(HfLlmProcessor):
    """
    Gemma tokenizer-backed processor for decoder-only generation tasks.
    """

    def __init__(
        self,
        tokenizer_file: str,
        tokenizer_config: Optional[str] = None,
        chat_template: Optional[str] = None,
        max_seq_length: Optional[int] = 12800,
        max_gen_seq_length: Optional[int] = 512,
    ):
        tokenizer = _load_gemma_tokenizer(
            tokenizer_file=tokenizer_file,
            tokenizer_config=tokenizer_config,
            chat_template=chat_template,
        )
        super().__init__(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )

    def chat_template(
        self,
        messages: List[Dict[str, Any]],
    ) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return _fallback_chat_template(
            messages,
            image_token=getattr(self.tokenizer, "image_token", None),
        )
