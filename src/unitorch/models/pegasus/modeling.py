# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Dict, List, Optional, Union
from transformers import PegasusConfig, PegasusForConditionalGeneration
from unitorch.models import GenericModel, GenericOutputs


class PegasusForGeneration(GenericModel):
    """
    Pegasus model for text generation tasks.
    """

    prefix_keys_in_state_dict = {
        "^(?!model\.model\.|model\.lm_head\.)model\.": "model.",
        "^lm_head.": "model.",
    }

    def __init__(
        self,
        config_path: str,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initializes the PegasusForGeneration model.

        Args:
            config_path (str): Path to the Pegasus configuration file.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()
        self.config = PegasusConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.model = PegasusForConditionalGeneration(self.config)
        self.init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
    ):
        """
        Forward pass of the PegasusForGeneration model.

        Args:
            input_ids (torch.Tensor): Encoder input token IDs.
            attention_mask (torch.Tensor): Encoder attention mask.
            decoder_input_ids (torch.Tensor): Decoder input token IDs.
            decoder_attention_mask (torch.Tensor): Decoder attention mask.

        Returns:
            torch.Tensor: Output logits.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=True,
        )
        return outputs.logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        num_beams: Optional[int] = 5,
        decoder_start_token_id: Optional[int] = 0,
        decoder_end_token_id: Optional[Union[int, List[int]]] = 1,
        num_return_sequences: Optional[int] = 1,
        min_gen_seq_length: Optional[int] = 0,
        max_gen_seq_length: Optional[int] = 48,
        repetition_penalty: Optional[float] = 1.0,
        no_repeat_ngram_size: Optional[int] = 0,
        early_stopping: Optional[bool] = True,
        length_penalty: Optional[float] = 1.0,
        num_beam_groups: Optional[int] = 1,
        diversity_penalty: Optional[float] = 0.0,
        do_sample: Optional[bool] = False,
        temperature: Optional[float] = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 1.0,
    ):
        """
        Generates sequences using the PegasusForGeneration model.

        Args:
            input_ids (torch.Tensor): Encoder input token IDs.
            num_beams (int, optional): Number of beams for beam search. Defaults to 5.
            decoder_start_token_id (int, optional): Decoder start token ID. Defaults to 0.
            decoder_end_token_id (int or List[int], optional): Decoder end token ID. Defaults to 1.
            num_return_sequences (int, optional): Number of sequences to return. Defaults to 1.
            min_gen_seq_length (int, optional): Minimum generated sequence length. Defaults to 0.
            max_gen_seq_length (int, optional): Maximum generated sequence length. Defaults to 48.
            repetition_penalty (float, optional): Repetition penalty. Defaults to 1.0.
            no_repeat_ngram_size (int, optional): N-gram size to avoid repeating. Defaults to 0.
            early_stopping (bool, optional): Whether to stop early. Defaults to True.
            length_penalty (float, optional): Length penalty. Defaults to 1.0.
            num_beam_groups (int, optional): Number of beam groups. Defaults to 1.
            diversity_penalty (float, optional): Diversity penalty. Defaults to 0.0.
            do_sample (bool, optional): Whether to use sampling. Defaults to False.
            temperature (float, optional): Sampling temperature. Defaults to 1.0.
            top_k (int, optional): Top-k sampling. Defaults to 50.
            top_p (float, optional): Top-p (nucleus) sampling. Defaults to 1.0.

        Returns:
            GenericOutputs: Generated sequences and their scores.
        """
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

        sequences = outputs.sequences.reshape(
            -1, num_return_sequences, outputs.sequences.size(-1)
        )
        padded = torch.full(
            (sequences.size(0), num_return_sequences, max_gen_seq_length),
            fill_value=decoder_start_token_id,
            device=sequences.device,
        )
        padded[:, :, : sequences.size(-1)].copy_(sequences)

        if num_return_sequences == 1:
            padded = padded.reshape(-1, max_gen_seq_length)

        return GenericOutputs(
            sequences=padded,
            sequences_scores=outputs.sequences_scores,
        )
