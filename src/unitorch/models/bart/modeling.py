# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import List, Optional, Union

import torch
import torch.nn as nn
from transformers import BartConfig, BartForConditionalGeneration

from unitorch.models import GenericModel, GenericOutputs


class BartForGeneration(GenericModel):
    """BART model for sequence-to-sequence text generation."""

    prefix_keys_in_state_dict = {"^(?!model\.model\.|model\.).*": "model.model."}

    def __init__(
        self,
        config_path: str,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.config = BartConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.config.forced_bos_token_id = None
        self.model = BartForConditionalGeneration(self.config)
        self.init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=True,
        ).logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        num_beams: int = 5,
        decoder_start_token_id: int = 2,
        decoder_end_token_id: Union[int, List[int]] = 2,
        num_return_sequences: int = 1,
        min_gen_seq_length: int = 0,
        max_gen_seq_length: int = 48,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        early_stopping: bool = True,
        length_penalty: float = 1.0,
        num_beam_groups: int = 1,
        diversity_penalty: float = 0.0,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
    ) -> GenericOutputs:
        outputs = self.model.generate(
            input_ids,
            max_length=max_gen_seq_length,
            min_length=min_gen_seq_length,
            num_beams=num_beams,
            do_sample=do_sample,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            bos_token_id=decoder_start_token_id,
            eos_token_id=decoder_end_token_id,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            return_dict_in_generate=True,
            output_scores=True,
        )

        sequences = outputs.sequences.reshape(-1, num_return_sequences, outputs.sequences.size(-1))
        padded = torch.full(
            (sequences.size(0), num_return_sequences, max_gen_seq_length),
            fill_value=decoder_start_token_id,
            device=sequences.device,
            dtype=sequences.dtype,
        )
        padded[:, :, : sequences.size(-1)].copy_(sequences)

        if num_return_sequences == 1:
            padded = padded.reshape(-1, max_gen_seq_length)

        return GenericOutputs(sequences=padded, sequences_scores=outputs.sequences_scores)
