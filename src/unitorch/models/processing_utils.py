# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from transformers import PreTrainedTokenizer
from transformers.image_utils import to_numpy_array, ChannelDimension
from transformers.image_transforms import to_channel_dimension_format
from transformers.image_processing_utils import BaseImageProcessor
from unitorch.models import GenericOutputs
from unitorch.utils import pop_value, truncate_sequence_pair


class HfTextGenerationProcessor:
    """
    Processor for text generation tasks.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to use for text generation.
        max_seq_length (int, optional): Maximum sequence length. Defaults to 128.
        max_gen_seq_length (int, optional): Maximum generated sequence length. Defaults to 48.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: Optional[int] = 128,
        max_gen_seq_length: Optional[int] = 48,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_gen_seq_length = max_gen_seq_length
        self.pad_token = self.tokenizer.pad_token
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        self.mask_token = self.tokenizer.mask_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.vocab_size = len(self.tokenizer.get_vocab())

    def generation_inputs(
        self,
        text: str,
        max_seq_length: Optional[int] = None,
    ):
        """
        Generate inputs for text generation.

        Args:
            text (str): The input text.
            max_seq_length (int, optional): Maximum sequence length. Defaults to None.

        Returns:
            GenericOutputs: The generated input tokens and attention mask.
        """
        max_seq_length = pop_value(max_seq_length, self.max_seq_length)
        tokens = self.tokenizer.tokenize(str(text))
        tokens = tokens[: max_seq_length - 2]
        tokens = [self.bos_token] + tokens + [self.eos_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = input_ids[:max_seq_length]
        attention_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += [self.pad_token_id] * len(padding)
        attention_mask += padding

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
    ):
        """
        Generate labels for text generation.

        Args:
            text (str): The input text.
            max_gen_seq_length (int, optional): Maximum generated sequence length. Defaults to None.

        Returns:
            GenericOutputs: The generated label tokens and attention mask.
        """
        max_gen_seq_length = pop_value(max_gen_seq_length, self.max_gen_seq_length)
        tokens = self.tokenizer.tokenize(str(text))
        tokens = tokens[: max_gen_seq_length - 2]
        tokens = [self.bos_token] + tokens + [self.eos_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = input_ids[1:max_gen_seq_length]
        attention_mask = [1] * len(input_ids)

        padding = [0] * (max_gen_seq_length - len(input_ids))
        input_ids += [self.pad_token_id] * len(padding)
        attention_mask += padding

        assert len(input_ids) == max_gen_seq_length

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
    ):
        """
        Generate inputs, labels, and tokens for text generation.

        Args:
            text (str): The input text.
            text_pair (str): The paired text.
            max_seq_length (int, optional): Maximum sequence length. Defaults to None.
            max_gen_seq_length (int, optional): Maximum generated sequence length. Defaults to None.

        Returns:
            GenericOutputs: The generated input tokens, attention masks, label tokens, and attention masks.
        """
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
        skip_special_tokens: Optional[bool] = True,
    ):
        """
        Detokenize the sequences.

        Args:
            sequences (torch.Tensor): The sequences to detokenize.
            skip_special_tokens (bool, optional): Whether to skip special tokens. Defaults to True.

        Returns:
            list: The detokenized sequences.
        """
        if sequences.dim() == 3:
            _, num_return_sequences, sequences_length = sequences.size()
            sequences = sequences.reshape(-1, sequences_length).clamp_max(
                self.vocab_size
            )
            sequences = sequences.clamp_min(0)
            sequences[sequences == self.vocab_size] = self.pad_token_id
            decode_tokens = self.tokenizer.batch_decode(
                sequences,
                skip_special_tokens=skip_special_tokens,
            )
            decode_tokens = [
                decode_tokens[i : i + num_return_sequences]
                for i in range(0, len(decode_tokens), num_return_sequences)
            ]
        elif sequences.dim() == 2:
            sequences = sequences.clamp_min(0).clamp_max(self.vocab_size)
            sequences[sequences == self.vocab_size] = self.pad_token_id
            decode_tokens = self.tokenizer.batch_decode(
                sequences,
                skip_special_tokens=skip_special_tokens,
            )
        else:
            raise ValueError(f"Can't decode the tensor with shape {sequences.shape}")

        return decode_tokens


class HfTextClassificationProcessor:
    """
    Processor for text classification tasks.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to use for text classification.
        max_seq_length (int, optional): Maximum sequence length. Defaults to 128.
        source_type_id (int, optional): Source type ID. Defaults to 0.
        target_type_id (int, optional): Target type ID. Defaults to 1.
        position_start_id (int, optional): Start position ID. Defaults to 0.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: Optional[int] = 128,
        source_type_id: Optional[int] = 0,
        target_type_id: Optional[int] = 1,
        position_start_id: Optional[int] = 0,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token = self.tokenizer.pad_token
        self.sep_token = self.tokenizer.sep_token
        self.cls_token = self.tokenizer.cls_token
        self.mask_token = self.tokenizer.mask_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.source_type_id = source_type_id
        self.target_type_id = target_type_id
        self.position_start_id = position_start_id

    def classification(
        self,
        text: str,
        text_pair: Optional[str] = None,
        max_seq_length: Optional[int] = None,
    ):
        """
        Generate inputs for text classification.

        Args:
            text (str): The input text.
            text_pair (str, optional): The paired text. Defaults to None.
            max_seq_length (int, optional): Maximum sequence length. Defaults to None.

        Returns:
            GenericOutputs: The generated input tokens, token type IDs, attention mask, and position IDs.
        """
        max_seq_length = pop_value(
            max_seq_length,
            self.max_seq_length,
        )

        tokens = self.tokenizer.tokenize(str(text))
        if text_pair is None:
            tokens = tokens[: max_seq_length - 2]
            tokens = [self.cls_token] + tokens + [self.sep_token]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            token_type_ids = [self.source_type_id] * len(input_ids)
            attention_mask = [1] * len(input_ids)
        else:
            tokens_pair = self.tokenizer.tokenize(str(text_pair))
            truncate_sequence_pair(tokens, tokens_pair, max_seq_length - 3)
            token_type_ids = (
                [self.source_type_id]
                + [self.source_type_id] * len(tokens)
                + [self.source_type_id]
                + [self.target_type_id] * len(tokens_pair)
                + [self.target_type_id]
            )
            tokens = (
                [self.cls_token]
                + tokens
                + [self.sep_token]
                + tokens_pair
                + [self.sep_token]
            )
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += len(padding) * [self.pad_token_id]
        attention_mask += padding
        token_type_ids += len(padding) * [self.target_type_id]

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        return GenericOutputs(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            token_type_ids=torch.tensor(token_type_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
            position_ids=torch.tensor(
                list(
                    range(
                        self.position_start_id,
                        self.position_start_id + max_seq_length,
                    )
                ),
                dtype=torch.long,
            ),
        )


class HfImageClassificationProcessor:
    """
    Processor for image classification tasks.
    """

    def __init__(
        self,
        vision_processor: BaseImageProcessor,
    ):
        """
        Initialize the HfImageClassificationProcessor.

        Args:
            vision_processor (BaseImageProcessor): The vision processor object used for image transformations.
        """
        self.vision_processor = vision_processor

        self.size = getattr(self.vision_processor, "size", None)

        self.resample = getattr(self.vision_processor, "resample", None)

        self.crop_size = getattr(self.vision_processor, "crop_size", None)
        self.pad_size = getattr(self.vision_processor, "pad_size", None)

        self.rescale_factor = getattr(self.vision_processor, "rescale_factor", None)

        self.image_mean = getattr(self.vision_processor, "image_mean", None)
        self.image_std = getattr(self.vision_processor, "image_std", None)

    def classification(
        self,
        image: Union[Image.Image, str],
    ):
        """
        Perform image classification on the given image.

        Args:
            image (Image.Image): The input image.

        Returns:
            GenericOutputs: The output of the image classification, including pixel values.
        """
        if isinstance(image, str):
            image = Image.open(image)

        if self.size is not None:
            image = self.vision_processor.resize(
                image=to_numpy_array(image.convert("RGB")),
                size=self.size,
                resample=self.resample,
            )

        if self.crop_size is not None:
            image = self.vision_processor.center_crop(
                image,
                size=self.crop_size,
            )

        if self.rescale_factor is not None:
            image = self.vision_processor.rescale(
                image,
                self.rescale_factor,
            )

        if self.image_mean is not None and self.image_std is not None:
            image = self.vision_processor.normalize(
                image=image,
                mean=self.image_mean,
                std=self.image_std,
            )

        if self.pad_size is not None:
            image = self.vision_processor.pad_image(
                image,
                size=self.pad_size,
            )

        image = to_channel_dimension_format(image, ChannelDimension.FIRST)

        return GenericOutputs(
            pixel_values=torch.tensor(image),
        )
