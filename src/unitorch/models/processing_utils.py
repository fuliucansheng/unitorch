# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image
from transformers import PreTrainedTokenizer
from transformers.image_processing_utils import BaseImageProcessor

from unitorch.models import GenericOutputs
from unitorch.utils import pop_value, truncate_sequence_pair


class HfTextGenerationProcessor:
    """Processor for encoder-decoder text generation tasks."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 128,
        max_gen_seq_length: int = 48,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_gen_seq_length = max_gen_seq_length
        self.pad_token = tokenizer.pad_token
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.mask_token = tokenizer.mask_token
        self.pad_token_id = tokenizer.pad_token_id
        self.vocab_size = len(tokenizer.get_vocab())

    def generation_inputs(
        self,
        text: str,
        max_seq_length: Optional[int] = None,
    ) -> GenericOutputs:
        """Tokenise *text* into padded encoder input IDs and attention mask."""
        max_seq_length = pop_value(max_seq_length, self.max_seq_length)
        tokens = self.tokenizer.tokenize(str(text))
        tokens = tokens[:max_seq_length]
        if self.bos_token is not None:
            tokens = [self.bos_token] + tokens[: max_seq_length - 1]
        if self.eos_token is not None:
            tokens = tokens[: max_seq_length - 1] + [self.eos_token]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)[:max_seq_length]
        attention_mask = [1] * len(input_ids)
        pad_len = max_seq_length - len(input_ids)
        input_ids += [self.pad_token_id] * pad_len
        attention_mask += [0] * pad_len

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        return GenericOutputs(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
        )

    def generation_labels(
        self,
        text: str,
        max_gen_seq_length: Optional[int] = None,
    ) -> GenericOutputs:
        """Tokenise *text* into padded decoder label IDs and attention mask."""
        max_gen_seq_length = pop_value(max_gen_seq_length, self.max_gen_seq_length)
        tokens = self.tokenizer.tokenize(str(text))
        tokens = tokens[:max_gen_seq_length]
        if self.bos_token is not None:
            tokens = [self.bos_token] + tokens[: max_gen_seq_length - 1]
        if self.eos_token is not None:
            tokens = tokens[: max_gen_seq_length - 1] + [self.eos_token]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)[1:max_gen_seq_length]
        attention_mask = [1] * len(input_ids)
        pad_len = max_gen_seq_length - len(input_ids)
        input_ids += [self.pad_token_id] * pad_len
        attention_mask += [0] * pad_len

        assert len(input_ids) == max_gen_seq_length
        assert len(attention_mask) == max_gen_seq_length
        return GenericOutputs(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
        )

    def generation(
        self,
        text: str,
        text_pair: str,
        max_seq_length: Optional[int] = None,
        max_gen_seq_length: Optional[int] = None,
    ) -> GenericOutputs:
        """Return encoder inputs, decoder inputs, and decoder labels for a text pair."""
        max_seq_length = pop_value(max_seq_length, self.max_seq_length)
        max_gen_seq_length = pop_value(max_gen_seq_length, self.max_gen_seq_length)
        tokens = self.generation_inputs(text, max_seq_length)
        tokens_pair = self.generation_inputs(text_pair, max_gen_seq_length)
        labels = self.generation_labels(text_pair, max_gen_seq_length)
        return GenericOutputs(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            input_ids_pair=tokens_pair.input_ids,
            attention_mask_pair=tokens_pair.attention_mask,
            input_ids_label=labels.input_ids,
            attention_mask_label=labels.attention_mask,
        )

    def detokenize(
        self,
        sequences: torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> list:
        """Decode a 2-D or 3-D token-ID tensor back to strings."""
        if sequences.dim() == 3:
            _, num_return_sequences, seq_len = sequences.size()
            sequences = sequences.reshape(-1, seq_len).clamp(0, self.vocab_size)
            sequences[sequences == self.vocab_size] = self.pad_token_id
            decoded = self.tokenizer.batch_decode(sequences, skip_special_tokens=skip_special_tokens)
            return [decoded[i : i + num_return_sequences] for i in range(0, len(decoded), num_return_sequences)]
        elif sequences.dim() == 2:
            sequences = sequences.clamp(0, self.vocab_size)
            sequences[sequences == self.vocab_size] = self.pad_token_id
            return self.tokenizer.batch_decode(sequences, skip_special_tokens=skip_special_tokens)
        else:
            raise ValueError(f"Cannot decode tensor with shape {sequences.shape}")


class HfLlmProcessor:
    """Processor for causal / decoder-only language model tasks."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 2048,
        max_gen_seq_length: int = 512,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_gen_seq_length = max_gen_seq_length
        self.pad_token = tokenizer.pad_token
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.mask_token = tokenizer.mask_token
        self.pad_token_id = tokenizer.pad_token_id
        self.vocab_size = len(tokenizer.get_vocab())

    def chat_template(self, messages: List[Dict[str, Any]]) -> str:
        """Apply the tokenizer's chat template and return the rendered string."""
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def classification(
        self,
        text: str,
        text_pair: Optional[str] = None,
        max_seq_length: Optional[int] = None,
    ) -> GenericOutputs:
        """Tokenise text (and optional pair) for sequence classification (left-padded)."""
        max_seq_length = pop_value(max_seq_length, self.max_seq_length)
        tokens = self.tokenizer.tokenize(str(text))
        if text_pair is None:
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens[:max_seq_length])
        else:
            tokens_pair = self.tokenizer.tokenize(str(text_pair))
            truncate_sequence_pair(tokens, tokens_pair, max_seq_length)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens + tokens_pair)

        pad_len = max_seq_length - len(input_ids)
        attention_mask = [0] * pad_len + [1] * len(input_ids)
        input_ids = [self.pad_token_id] * pad_len + input_ids

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        return GenericOutputs(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
        )

    def generation_inputs(
        self,
        text: str,
        max_seq_length: Optional[int] = None,
    ) -> GenericOutputs:
        """Tokenise *text* as a left-padded generation prompt."""
        max_seq_length = pop_value(max_seq_length, self.max_seq_length)
        raw = self.tokenizer.tokenize(str(text))
        if self.bos_token is not None:
            tokens = [self.bos_token] + raw[-(max_seq_length - 1):]
        else:
            tokens = raw[-max_seq_length:]
        pad_len = max_seq_length - len(tokens)
        attention_mask = [0] * pad_len + [1] * len(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids([self.pad_token] * pad_len + tokens)
        assert len(input_ids) == max_seq_length
        return GenericOutputs(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
        )

    def generation_labels(
        self,
        text: str,
        max_gen_seq_length: Optional[int] = None,
    ) -> GenericOutputs:
        """Tokenise *text* as right-padded generation labels (with EOS)."""
        max_gen_seq_length = pop_value(max_gen_seq_length, self.max_gen_seq_length)
        tokens = self.tokenizer.tokenize(str(text))[: max_gen_seq_length - 1] + [self.eos_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        pad_len = max_gen_seq_length - len(input_ids)
        input_ids += [self.pad_token_id] * pad_len
        attention_mask += [0] * pad_len

        assert len(input_ids) == max_gen_seq_length
        assert len(attention_mask) == max_gen_seq_length
        return GenericOutputs(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
        )

    def generation(
        self,
        text: str,
        text_pair: str,
        max_seq_length: Optional[int] = None,
        max_gen_seq_length: Optional[int] = None,
    ) -> GenericOutputs:
        """Combine prompt and response into a single packed sequence for causal LM training."""
        max_seq_length = pop_value(max_seq_length, self.max_seq_length)
        max_gen_seq_length = pop_value(max_gen_seq_length, self.max_gen_seq_length)

        raw = self.tokenizer.tokenize(str(text))
        if self.bos_token is not None:
            tokens = [self.bos_token] + raw[-(max_seq_length - 1):]
        else:
            tokens = raw[-max_seq_length:]
        tokens_pair = self.tokenizer.tokenize(str(text_pair))[: max_gen_seq_length - 1] + [self.eos_token]

        pad_a = [self.pad_token] * (max_seq_length - len(tokens))
        pad_b = [self.pad_token] * (max_gen_seq_length - len(tokens_pair))
        attention_mask = [0] * len(pad_a) + [1] * (len(tokens) + len(tokens_pair)) + [0] * len(pad_b)
        input_ids = self.tokenizer.convert_tokens_to_ids(pad_a + tokens + tokens_pair + pad_b)

        label_tokens = tokens_pair + [self.pad_token] * (max_gen_seq_length - len(tokens_pair) + 1)
        input_ids_label = [0] * (max_seq_length - 1) + self.tokenizer.convert_tokens_to_ids(label_tokens)
        attention_mask_label = (
            [0] * (max_seq_length - 1)
            + [1] * len(tokens_pair)
            + [0] * (max_gen_seq_length - len(tokens_pair) + 1)
        )

        return GenericOutputs(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
            input_ids_label=torch.tensor(input_ids_label, dtype=torch.long),
            attention_mask_label=torch.tensor(attention_mask_label, dtype=torch.long),
        )

    def messages_generation(
        self,
        messages: List[Dict[str, Any]],
        max_seq_length: Optional[int] = None,
    ) -> GenericOutputs:
        """Format a chat message list into training inputs for causal LM generation."""
        while messages and messages[-1]["role"] != "assistant":
            messages.pop()
        return self.generation(
            text=self.chat_template(messages[:-1]),
            text_pair=self.chat_template(messages[-1:]),
            max_seq_length=max_seq_length,
        )

    def detokenize(
        self,
        sequences: torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> list:
        """Decode a 2-D or 3-D token-ID tensor back to strings."""
        if sequences.dim() == 3:
            _, num_return_sequences, seq_len = sequences.size()
            sequences = sequences.reshape(-1, seq_len).clamp(0, self.vocab_size)
            sequences[sequences == self.vocab_size] = self.pad_token_id
            decoded = self.tokenizer.batch_decode(sequences, skip_special_tokens=skip_special_tokens)
            return [decoded[i : i + num_return_sequences] for i in range(0, len(decoded), num_return_sequences)]
        elif sequences.dim() == 2:
            sequences = sequences.clamp(0, self.vocab_size)
            sequences[sequences == self.vocab_size] = self.pad_token_id
            return self.tokenizer.batch_decode(sequences, skip_special_tokens=skip_special_tokens)
        else:
            raise ValueError(f"Cannot decode tensor with shape {sequences.shape}")


class HfTextClassificationProcessor:
    """Processor for BERT-style text classification tasks."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 128,
        source_type_id: int = 0,
        target_type_id: int = 1,
        position_start_id: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token = tokenizer.pad_token
        self.sep_token = tokenizer.sep_token
        self.cls_token = tokenizer.cls_token
        self.mask_token = tokenizer.mask_token
        self.pad_token_id = tokenizer.pad_token_id
        self.source_type_id = source_type_id
        self.target_type_id = target_type_id
        self.position_start_id = position_start_id

    def classification(
        self,
        text: str,
        text_pair: Optional[str] = None,
        max_seq_length: Optional[int] = None,
    ) -> GenericOutputs:
        """Tokenise text (and optional pair) for sequence classification."""
        max_seq_length = pop_value(max_seq_length, self.max_seq_length)
        tokens = self.tokenizer.tokenize(str(text))

        if text_pair is None:
            if self.cls_token is not None:
                tokens = [self.cls_token] + tokens[: max_seq_length - 2] + [self.sep_token]
            else:
                tokens = tokens[: max_seq_length - 1] + [self.sep_token]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            token_type_ids = [self.source_type_id] * len(input_ids)
            attention_mask = [1] * len(input_ids)
        else:
            tokens_pair = self.tokenizer.tokenize(str(text_pair))
            if self.cls_token is not None:
                truncate_sequence_pair(tokens, tokens_pair, max_seq_length - 3)
                token_type_ids = (
                    [self.source_type_id] * (len(tokens) + 2)
                    + [self.target_type_id] * (len(tokens_pair) + 1)
                )
                tokens = [self.cls_token] + tokens + [self.sep_token] + tokens_pair + [self.sep_token]
            else:
                truncate_sequence_pair(tokens, tokens_pair, max_seq_length - 2)
                token_type_ids = (
                    [self.source_type_id] * len(tokens)
                    + [self.target_type_id] * len(tokens_pair)
                )
                tokens = tokens + [self.sep_token] + tokens_pair + [self.sep_token]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(input_ids)

        pad_len = max_seq_length - len(input_ids)
        input_ids += [self.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        token_type_ids += [self.target_type_id] * pad_len

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        return GenericOutputs(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            token_type_ids=torch.tensor(token_type_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
            position_ids=torch.arange(
                self.position_start_id,
                self.position_start_id + max_seq_length,
                dtype=torch.long,
            ),
        )


class HfImageClassificationProcessor:
    """Processor for image classification tasks using a HuggingFace vision processor."""

    def __init__(self, vision_processor: BaseImageProcessor) -> None:
        self.vision_processor = vision_processor

    def classification(self, image: Union[Image.Image, str]) -> GenericOutputs:
        """Preprocess *image* into pixel values ready for a vision model."""
        if isinstance(image, str):
            image = Image.open(image)
        pixel_values = self.vision_processor.preprocess(image, return_tensors="pt").pixel_values[0]
        return GenericOutputs(pixel_values=pixel_values)
