# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel
from transformers import CLIPConfig
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.models import GenericModel, GenericOutputs


# residual
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


# rotary positional embedding
# https://arxiv.org/abs/2104.09864


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


# classic Noam Shazeer paper, except here they use SwiGLU instead of the more popular GEGLU for gating the feedforward
# https://arxiv.org/abs/2002.05202


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


# parallel attention and feedforward with residual
# discovered by Wang et al + EleutherAI from GPT-J fame


class ParallelTransformerBlock(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = (attn_inner_dim, dim_head, dim_head, (ff_inner_dim * 2))

        self.heads = heads
        self.scale = dim_head**-0.5
        self.rotary_emb = RotaryEmbedding(dim_head)

        self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims), bias=False)
        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)

        self.ff_out = nn.Sequential(SwiGLU(), nn.Linear(ff_inner_dim, dim, bias=False))

        self.register_buffer("pos_emb", None, persistent=False)

    def get_rotary_embedding(self, n, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n]

        pos_emb = self.rotary_emb(n, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def forward(self, x, attn_mask=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device, h = x.shape[1], x.device, self.heads

        # pre layernorm

        x = self.norm(x)

        # attention queries, keys, values, and feedforward inner

        q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)

        # split heads
        # they use multi-query single-key-value attention, yet another Noam Shazeer paper
        # they found no performance loss past a certain scale, and more efficient decoding obviously
        # https://arxiv.org/abs/1911.02150

        # q = rearrange(q, "b n (h d) -> b h n d", h=h)
        b, n, _ = q.shape
        q = q.view(b, n, h, -1).transpose(1, 2)

        # rotary embeddings

        positions = self.get_rotary_embedding(n, device)
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))

        # scale

        q = q * self.scale

        # similarity

        sim = torch.einsum("b h i d, b j d -> b h i j", q, k)

        # extra attention mask - for masking out attention from text CLS token to padding

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        # attention

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values

        out = torch.einsum("b h i j, b j d -> b h i d", attn, v)

        # merge heads

        # out = rearrange(out, "b h n d -> b n (h d)")
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.attn_out(out) + self.ff_out(ff)


# cross attention - using multi-query + one-headed key / values as in PaLM w/ optional parallel feedforward


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim=None,
        dim_head=64,
        heads=12,
        parallel_ff=False,
        ff_mult=4,
        norm_context=False,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = heads * dim_head
        context_dim = context_dim if context_dim is not None else dim

        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(context_dim) if norm_context else nn.Identity()

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether to have parallel feedforward

        ff_inner_dim = ff_mult * dim

        self.ff = (
            nn.Sequential(
                nn.Linear(dim, ff_inner_dim * 2, bias=False),
                SwiGLU(),
                nn.Linear(ff_inner_dim, dim, bias=False),
            )
            if parallel_ff
            else None
        )

    def forward(self, x, context, mask):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        # pre-layernorm, for queries and context

        x = self.norm(x)
        context = self.context_norm(context)

        # get queries

        q = self.to_q(x)
        # q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        b, n, _ = q.shape
        q = q.view(b, n, self.heads, -1).transpose(1, 2)
        # scale

        q = q * self.scale

        # get key / values

        k, v = self.to_kv(context).chunk(2, dim=-1)

        # query / key similarity

        sim = torch.einsum("b h i d, b j d -> b h i j", q, k)

        # attention
        mask = mask.unsqueeze(1).repeat(1, self.heads, 1, 1)
        sim = sim + mask  # context mask
        sim = sim - sim.amax(dim=-1, keepdim=True)
        attn = sim.softmax(dim=-1)

        # aggregate

        out = torch.einsum("b h i j, b j d -> b h i d", attn, v)

        # merge and combine heads

        # out = rearrange(out, "b h n d -> b n (h d)")
        out = out.transpose(1, 2).reshape(b, n, -1)
        out = self.to_out(out)

        # add parallel feedforward (for multimodal layers)

        if self.ff is not None:
            out = out + self.ff(x)

        return out


class CrossModel(nn.Module):
    def __init__(self, dim=512, layer_num=4, dim_head=64, heads=8, ff_mult=4):
        super().__init__()

        self.layers = nn.ModuleList([])

        for ind in range(layer_num):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(
                            CrossAttention(
                                dim=dim,
                                dim_head=dim_head,
                                heads=heads,
                                parallel_ff=True,
                                ff_mult=ff_mult,
                            )
                        ),
                        Residual(
                            ParallelTransformerBlock(
                                dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult
                            )
                        ),
                    ]
                )
            )

    def forward(self, query_tokens, context_tokens, mask):
        for cross_attn, self_attn_ff in self.layers:
            query_tokens = cross_attn(query_tokens, context_tokens, mask)
            query_tokens = self_attn_ff(query_tokens)

        return query_tokens


class KolorsMPSModel(GenericModel):
    def __init__(self, config_path):
        super().__init__()
        self.config = CLIPConfig.from_json_file(config_path)
        self.model = CLIPModel(self.config)
        self.cross_model = CrossModel(dim=1024, layer_num=4, heads=16)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        condition_input_ids: torch.Tensor,
        condition_attention_mask: torch.Tensor,
        condition_position_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        text_outputs = self.model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        text_features = self.model.text_projection(text_outputs[0])
        text_pooled_features = self.model.text_projection(text_outputs[1])

        image_outputs = self.model.vision_model(pixel_values=pixel_values)
        image_features = self.model.visual_projection(image_outputs[0])

        condition_outputs = self.model.text_model(
            input_ids=condition_input_ids,
            attention_mask=condition_attention_mask,
            position_ids=condition_position_ids,
        )
        condition_features = self.model.text_projection(condition_outputs[0])

        sim_text_condition = torch.einsum(
            "b i d, b j d -> b j i", text_features, condition_features
        )
        sim_text_condition = torch.max(sim_text_condition, dim=1, keepdim=True)[0]
        sim_text_condition = sim_text_condition / sim_text_condition.max()
        mask = torch.where(sim_text_condition > 0.01, 0, float("-inf"))  # B*1*77

        mask = mask.repeat(1, image_features.shape[1], 1)  # B*257*77
        cross_features = self.cross_model(image_features, text_features, mask)[:, 0]

        text_embeds = text_pooled_features / text_pooled_features.norm(
            dim=-1, keepdim=True
        )
        cross_embeds = cross_features / cross_features.norm(dim=-1, keepdim=True)
        scores = torch.sum(text_embeds * cross_embeds, dim=-1, keepdim=True)

        return self.model.logit_scale.exp() * scores
