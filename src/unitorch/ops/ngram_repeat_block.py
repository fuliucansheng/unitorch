# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

""" Wrapper for ngram_repeat_block cuda extension """
import torch.nn as nn
from torch.autograd import Function

try:
    import unitorch.clib.ngram_repeat_block_cuda as ngram_repeat_block_cuda

    is_kernel_available = True
except:
    is_kernel_available = False


class NGramRepeatBlockFunction(Function):
    def forward(
        self,
        tokens,
        lprobs,
        bsz,
        step,
        beam_size,
        no_repeat_ngram_size,
    ):
        outputs = ngram_repeat_block_cuda.forward(
            tokens, lprobs, bsz, step, beam_size, no_repeat_ngram_size
        )
        return outputs

    def backward(*args):
        raise NotImplementedError


class NGramRepeatBlock(nn.Module):
    def __init__(self):
        super(NGramRepeatBlock, self).__init__()

    def reset_parameters(self):
        pass

    @classmethod
    def is_available(cls):
        return is_kernel_available

    def forward(
        self,
        tokens,
        lprobs,
        bsz,
        step,
        beam_size,
        no_repeat_ngram_size,
    ):
        assert tokens.size(0) == bsz * beam_size
        assert lprobs.size(0) == bsz * beam_size

        return NGramRepeatBlockFunction.apply(
            tokens, lprobs, bsz, step, beam_size, no_repeat_ngram_size
        )
