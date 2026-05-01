# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPConfig
from typing import Optional
from unitorch.models import GenericModel, GenericOutputs


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoPE). https://arxiv.org/abs/2104.09864"""

    def __init__(self, dim: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len: int, *, device: torch.device) -> torch.Tensor:
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class ParallelTransformerBlock(nn.Module):
    """Parallel self-attention and feedforward block with rotary embeddings."""

    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8, ff_mult: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = (attn_inner_dim, dim_head, dim_head, ff_inner_dim * 2)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.rotary_emb = RotaryEmbedding(dim_head)
        self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims), bias=False)
        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)
        self.ff_out = nn.Sequential(SwiGLU(), nn.Linear(ff_inner_dim, dim, bias=False))
        self.register_buffer("pos_emb", None, persistent=False)

    def get_rotary_embedding(self, n: int, device: torch.device) -> torch.Tensor:
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n]
        pos_emb = self.rotary_emb(n, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        n, device, h = x.shape[1], x.device, self.heads
        x = self.norm(x)
        q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)

        b, n, _ = q.shape
        q = q.view(b, n, h, -1).transpose(1, 2)

        positions = self.get_rotary_embedding(n, device)
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))
        q = q * self.scale

        sim = torch.einsum("b h i d, b j d -> b h i j", q, k)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b j d -> b h i d", attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.attn_out(out) + self.ff_out(ff)


class CrossAttention(nn.Module):
    """Cross-attention with optional parallel feedforward."""

    def __init__(
        self,
        dim: int,
        *,
        context_dim: Optional[int] = None,
        dim_head: int = 64,
        heads: int = 12,
        parallel_ff: bool = False,
        ff_mult: int = 4,
        norm_context: bool = False,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        context_dim = context_dim if context_dim is not None else dim

        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(context_dim) if norm_context else nn.Identity()
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

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

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.norm(x)
        context = self.context_norm(context)

        q = self.to_q(x)
        b, n, _ = q.shape
        q = q.view(b, n, self.heads, -1).transpose(1, 2)
        q = q * self.scale

        k, v = self.to_kv(context).chunk(2, dim=-1)
        sim = torch.einsum("b h i d, b j d -> b h i j", q, k)

        mask = mask.unsqueeze(1).repeat(1, self.heads, 1, 1)
        sim = sim + mask
        sim = sim - sim.amax(dim=-1, keepdim=True)
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b j d -> b h i d", attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        out = self.to_out(out)

        if self.ff is not None:
            out = out + self.ff(x)
        return out


class CrossModel(nn.Module):
    def __init__(self, dim: int = 512, layer_num: int = 4, dim_head: int = 64, heads: int = 8, ff_mult: int = 4):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Residual(CrossAttention(dim=dim, dim_head=dim_head, heads=heads, parallel_ff=True, ff_mult=ff_mult)),
                        Residual(ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult)),
                    ]
                )
                for _ in range(layer_num)
            ]
        )

    def forward(
        self,
        query_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        for cross_attn, self_attn_ff in self.layers:
            query_tokens = cross_attn(query_tokens, context_tokens, mask)
            query_tokens = self_attn_ff(query_tokens)
        return query_tokens


class KolorsMPSModel(GenericModel):
    def __init__(self, config_path: str):
        """
        Initializes the KolorsMPSModel.

        Args:
            config_path (str): Path to the CLIP configuration file.
        """
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
        """
        Forward pass of the KolorsMPSModel.

        Args:
            input_ids (torch.Tensor): Text token IDs.
            attention_mask (torch.Tensor): Text attention mask.
            position_ids (torch.Tensor): Text position IDs.
            pixel_values (torch.Tensor): Image pixel values.
            condition_input_ids (torch.Tensor): Condition text token IDs.
            condition_attention_mask (torch.Tensor): Condition text attention mask.
            condition_position_ids (torch.Tensor): Condition text position IDs.
            labels (torch.Tensor, optional): Labels (unused). Defaults to None.

        Returns:
            torch.Tensor: Scaled similarity scores.
        """
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

        sim_text_condition = torch.einsum("b i d, b j d -> b j i", text_features, condition_features)
        sim_text_condition = torch.max(sim_text_condition, dim=1, keepdim=True)[0]
        sim_text_condition = sim_text_condition / sim_text_condition.max()
        mask = torch.where(sim_text_condition > 0.01, 0, float("-inf"))
        mask = mask.repeat(1, image_features.shape[1], 1)

        cross_features = self.cross_model(image_features, text_features, mask)[:, 0]
        text_embeds = text_pooled_features / text_pooled_features.norm(dim=-1, keepdim=True)
        cross_embeds = cross_features / cross_features.norm(dim=-1, keepdim=True)

        scores = torch.sum(text_embeds * cross_embeds, dim=-1, keepdim=True)
        return self.model.logit_scale.exp() * scores
