# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import numpy as np
import torch.nn.functional as F
import transformers
from itertools import accumulate
from collections import UserDict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from transformers.generation.logits_process import _calc_banned_ngram_tokens
from unitorch.utils.decorators import replace
from unitorch.ops.ngram_repeat_block import NGramRepeatBlock


@replace(transformers.generation.logits_process.NoRepeatNGramLogitsProcessor)
class NoRepeatNGramLogitsProcessorV2(
    transformers.generation.logits_process.LogitsProcessor
):
    r"""
    Args:
        ngram_size (:obj:`int`):
            All ngrams of size :obj:`ngram_size` can only occur once.
    """

    def __init__(self, ngram_size: int):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(
                f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}"
            )
        self.ngram_size = ngram_size
        if torch.cuda.is_available() and NGramRepeatBlock.is_available():
            self.no_repeat_ngram_op = NGramRepeatBlock()
        else:
            self.no_repeat_ngram_op = None

    def __call__(
        self, input_ids: torch.Tensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        num_batch_hypotheses = scores.shape[0]
        cur_len = input_ids.shape[-1]
        if self.no_repeat_ngram_op and input_ids.device.type == "cuda":
            scores = self.no_repeat_ngram_op(
                input_ids,
                scores.float(),
                num_batch_hypotheses,
                cur_len - 1,
                1,
                self.ngram_size,
            )
        else:
            banned_batch_tokens = _calc_banned_ngram_tokens(
                self.ngram_size, input_ids, num_batch_hypotheses, cur_len
            )

            for i, banned_tokens in enumerate(banned_batch_tokens):
                scores[i, banned_tokens] = -float("inf")

        return scores


@replace(transformers.generation.beam_search.BeamSearchScorer)
class BeamSearchScorerV2(transformers.generation.beam_search.BeamSearchScorer):
    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        device: torch.device,
        length_penalty: Optional[float] = 1.0,
        do_early_stopping: Optional[bool] = False,
        num_beam_hyps_to_keep: Optional[int] = 1,
        num_beam_groups: Optional[int] = 1,
        max_length: Optional[int] = None,
        use_reorder_cache_v2: Optional[bool] = False,
    ):
        super().__init__(
            batch_size,
            num_beams,
            device,
            length_penalty,
            do_early_stopping,
            num_beam_hyps_to_keep,
            num_beam_groups,
            max_length,
        )
        self.use_reorder_cache_v2 = use_reorder_cache_v2
        self.beams_offset = (
            (torch.arange(0, batch_size) * num_beams // self.num_beam_groups)
            .unsqueeze(1)
            .to(device)
        )
        self.cand_size = 2 * num_beams // self.num_beam_groups
        self.cand_offsets = torch.arange(0, self.cand_size).to(device)

    def process(
        self,
        input_ids: torch.Tensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.Tensor,
        next_indices: torch.Tensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        beam_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        cur_len = input_ids.shape[-1]
        batch_size = len(self._beam_hyps)
        if not (batch_size == (input_ids.shape[0] // self.group_size)):
            if self.num_beam_groups > 1:
                raise ValueError(
                    f"A group beam size of {input_ids.shape[0]} is used as the input, but a group beam "
                    f"size of {self.group_size} is expected by the beam scorer."
                )
            else:
                raise ValueError(
                    f"A beam size of {input_ids.shape[0]} is used as the input, but a beam size of "
                    f"{self.group_size} is expected by the beam scorer."
                )

        next_tokens_id = next_tokens
        next_beams_id = next_indices
        effective_beam_id = next_beams_id + self.beams_offset

        if eos_token_id is not None:
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_mask = torch.zeros_like(next_tokens)
            for eos_token in eos_token_id:
                eos_mask = eos_mask + next_tokens.eq(eos_token)
            eos_mask = eos_mask.bool()
        else:
            eos_mask = torch.zeros_like(next_tokens).bool()
        eos_effective_idx = torch.masked_select(
            effective_beam_id[:, : self.group_size], mask=eos_mask[:, : self.group_size]
        )

        finished_batch_idxs = []
        if self.use_reorder_cache_v2 and eos_effective_idx.numel() > 0:
            eos_effective_scores = torch.masked_select(
                next_scores[:, : self.group_size], mask=eos_mask[:, : self.group_size]
            )
            input_clone = input_ids.index_select(0, eos_effective_idx)
            unfin_offset = np.array(list(accumulate(map(int, self._done))))[
                np.array(list(map(int, self._done))) == 0
            ]
            for i in range(eos_effective_idx.size(0)):
                eos_idx = eos_effective_idx[i]
                eos_score = eos_effective_scores[i]
                unfin_batch_idx = eos_idx // self.group_size
                batch_idx = unfin_batch_idx + unfin_offset[unfin_batch_idx]
                if beam_indices is not None:
                    batch_beam_idx = batch_idx * self.group_size + (
                        eos_idx % self.group_size
                    )
                    beam_index = beam_indices[batch_beam_idx]
                    beam_index = beam_index + (batch_beam_idx,)
                else:
                    beam_index = None
                if not self._done[batch_idx]:
                    self._beam_hyps[batch_idx.item()].add(
                        input_clone[i],
                        eos_score.item(),
                        beam_indices=beam_index,
                    )
                is_done = bool(self._done[batch_idx])
                self._done[batch_idx] = self._done[batch_idx] or self._beam_hyps[
                    batch_idx
                ].is_done(next_scores[unfin_batch_idx].max().item(), cur_len)
                if is_done != bool(self._done[batch_idx]):
                    finished_batch_idxs.append(unfin_batch_idx)

        if not self.use_reorder_cache_v2:
            eos_effective_scores = torch.masked_select(
                next_scores[:, : self.group_size], mask=eos_mask[:, : self.group_size]
            )
            input_ids_cpu = input_ids.cpu()
            eos_effective_idx_cpu = eos_effective_idx.cpu()
            eos_effective_scores_cpu = eos_effective_scores.cpu()
            for i in range(0, eos_effective_idx_cpu.size()[-1]):
                eos_idx = eos_effective_idx_cpu[i]
                batch_idx = eos_idx // self.group_size
                if beam_indices is not None:
                    batch_beam_idx = batch_idx * self.group_size + (
                        eos_idx % self.group_size
                    )
                    beam_index = beam_indices[batch_beam_idx]
                    beam_index = beam_index + (batch_beam_idx,)
                else:
                    beam_index = None
                if not self._done[batch_idx]:
                    self._beam_hyps[batch_idx.item()].add(
                        input_ids_cpu[eos_idx].clone(),
                        eos_effective_scores_cpu[i],
                        beam_indices=beam_index,
                    )
                self._done[batch_idx] = self._done[batch_idx] or self._beam_hyps[
                    batch_idx
                ].is_done(next_scores[batch_idx].max().item(), cur_len)

        if self.use_reorder_cache_v2 and len(finished_batch_idxs) > 0:
            new_batch_size = batch_size - len(finished_batch_idxs)
            batch_mask = torch.ones(batch_size).to(next_tokens_id)
            batch_mask[torch.tensor(finished_batch_idxs)] = 0
            batch_idxs = batch_mask.nonzero(as_tuple=False).squeeze(-1)
            eos_mask = eos_mask[batch_idxs]
            next_beams_id = next_beams_id[batch_idxs]
            self.beams_offset.resize_(new_batch_size, 1)
            effective_beam_id = next_beams_id.add(self.beams_offset)
            next_scores = next_scores[batch_idxs]
            next_tokens = next_tokens[batch_idxs]
            next_tokens_id = next_tokens_id[batch_idxs]
            before_batch_size = batch_size
            batch_size = new_batch_size
        else:
            before_batch_size = batch_size
            batch_idxs = None

        active_mask = torch.add(
            eos_mask.type_as(self.cand_offsets) * self.cand_size,
            self.cand_offsets[: eos_mask.size(1)],
        )
        _, active_hypos = torch.topk(
            active_mask, k=self.group_size, dim=1, largest=False
        )
        active_effective_beam_id = torch.gather(
            effective_beam_id, dim=1, index=active_hypos
        )
        active_scores = torch.gather(next_scores, dim=1, index=active_hypos)
        active_tokens = torch.gather(next_tokens_id, dim=1, index=active_hypos)
        beam_idxs = active_effective_beam_id.view(-1)
        beam_scores = active_scores.view(-1)
        beam_tokens = active_tokens.view(-1)

        if batch_idxs is not None:
            new_beam_idxs = (
                torch.arange(before_batch_size * self.group_size)
                .reshape(before_batch_size, self.group_size)
                .to(input_ids)
            )
            beam_idxs = new_beam_idxs[batch_idxs].reshape(-1)[beam_idxs]

        userdict = UserDict(
            {
                "next_beam_scores": beam_scores.view(-1),
                "next_beam_tokens": beam_tokens.view(-1),
                "next_beam_indices": beam_idxs.view(-1),
            }
        )
        if self.use_reorder_cache_v2:
            userdict["next_batch_indices"] = batch_idxs

        return userdict

    def finalize(
        self,
        input_ids: torch.Tensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.Tensor,
        final_beam_indices: torch.Tensor,
        max_length: int,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        beam_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        batch_size = len(self._beam_hyps)

        unfin_offset = np.array(list(accumulate(map(int, self._done))))[
            np.array(list(map(int, self._done))) == 0
        ]
        if self.use_reorder_cache_v2:
            batch_size = len(unfin_offset)
        for batch_idx in range(batch_size):
            if not self.use_reorder_cache_v2 and self._done[batch_idx]:
                continue
            if self.use_reorder_cache_v2:
                final_batch_idx = batch_idx + unfin_offset[batch_idx]
            else:
                final_batch_idx = batch_idx
            # need to add best num_beams hypotheses to generated hyps
            for beam_id in range(self.num_beams):
                effective_beam_id = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                beam_index = (
                    beam_indices[effective_beam_id]
                    if beam_indices is not None
                    else None
                )
                self._beam_hyps[final_batch_idx].add(
                    final_tokens, final_score, beam_indices=beam_index
                )

        batch_size = len(self._beam_hyps)

        # select the best hypotheses
        sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
        best = []
        best_indices = []
        best_scores = torch.zeros(
            batch_size * self.num_beam_hyps_to_keep,
            device=self.device,
            dtype=torch.float32,
        )

        # retrieve best hypotheses
        for i, beam_hyp in enumerate(self._beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                best_hyp_tuple = sorted_hyps.pop()
                best_score = best_hyp_tuple[0]
                best_hyp = best_hyp_tuple[1]
                best_index = best_hyp_tuple[2]
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)

                # append to lists
                best.append(best_hyp)
                best_indices.append(best_index)
                best_scores[i * self.num_beam_hyps_to_keep + j] = best_score

        # prepare for adding eos
        sent_lengths_max = sent_lengths.max().item() + 1
        sent_max_len = (
            min(sent_lengths_max, max_length)
            if max_length is not None
            else sent_lengths_max
        )
        decoded: torch.Tensor = input_ids.new(
            batch_size * self.num_beam_hyps_to_keep, sent_max_len
        )
        if len(best_indices) > 0 and best_indices[0] is not None:
            indices: torch.Tensor = input_ids.new(
                batch_size * self.num_beam_hyps_to_keep, sent_max_len
            )
        else:
            indices = None
        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`pad_token_id` has to be defined"
            decoded.fill_(pad_token_id)

        if indices is not None:
            indices.fill_(-1)

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, (hypo, best_idx) in enumerate(zip(best, best_indices)):
            decoded[i, : sent_lengths[i]] = hypo
            if indices is not None:
                indices[i, : len(best_idx)] = torch.tensor(best_idx)
            if sent_lengths[i] < sent_max_len:
                decoded[i, sent_lengths[i]] = (
                    eos_token_id if isinstance(eos_token_id, int) else eos_token_id[0]
                )
        return UserDict(
            {
                "sequences": decoded,
                "sequence_scores": best_scores,
                "beam_indices": indices,
            }
        )
