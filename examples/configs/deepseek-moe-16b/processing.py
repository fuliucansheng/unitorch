#!/usr/bin/env python3.10
# processing.py — Convert deepseek-ai/deepseek-moe-16b-base HuggingFace weights
# into per-(tp_rank, pp_rank) sharded files for MegatronGPTForGeneration.
#
# Usage:
#   python3.10 processing.py \
#       --hf_dir /path/to/hf_model \
#       --out_dir ./cache/deepseek-moe-16b \
#       --tp_size 8 \
#       --pp_size 2 \
#       --ep_size 1
#
# Outputs one file per rank pair:
#   <out_dir>/tp{i:02d}_pp{j:02d}.pt
#
# Set pretrained_weight_path in the .ini to the path matching
# each rank's file, or implement rank-aware loading in the trainer.

import fire
import os
import re
import glob
import math
import torch
from typing import Dict, List, Optional, Tuple


# ── Key remapping: HuggingFace → Megatron ────────────────────────────────────

def _remap_key(hf_key: str) -> Optional[str]:
    """Map a HF DeepSeek key to Megatron GPTModel namespace. Returns None to drop."""
    k = hf_key

    k = re.sub(r"^model\.embed_tokens\.weight$", "embedding.word_embeddings.weight", k)
    k = re.sub(r"^model\.norm\.weight$",         "decoder.final_layernorm.weight", k)
    k = re.sub(r"^lm_head\.weight$",             "output_layer.weight", k)
    k = re.sub(r"^model\.layers\.(\d+)\.",        lambda m: f"decoder.layers.{m.group(1)}.", k)

    k = re.sub(r"self_attn\.q_proj\.weight",      "self_attention.linear_q.weight", k)
    k = re.sub(r"self_attn\.k_proj\.weight",      "self_attention.linear_k.weight", k)
    k = re.sub(r"self_attn\.v_proj\.weight",      "self_attention.linear_v.weight", k)
    k = re.sub(r"self_attn\.o_proj\.weight",      "self_attention.linear_proj.weight", k)

    k = re.sub(r"mlp\.gate_proj\.weight",         "mlp.linear_fc1_gate.weight", k)
    k = re.sub(r"mlp\.up_proj\.weight",           "mlp.linear_fc1.weight", k)
    k = re.sub(r"mlp\.down_proj\.weight",         "mlp.linear_fc2.weight", k)
    k = re.sub(r"mlp\.gate\.weight",              "mlp.router.weight", k)

    k = re.sub(r"mlp\.shared_experts\.gate_proj\.weight", "mlp.shared_expert.linear_fc1_gate.weight", k)
    k = re.sub(r"mlp\.shared_experts\.up_proj\.weight",   "mlp.shared_expert.linear_fc1.weight", k)
    k = re.sub(r"mlp\.shared_experts\.down_proj\.weight", "mlp.shared_expert.linear_fc2.weight", k)

    k = re.sub(
        r"mlp\.experts\.(\d+)\.gate_proj\.weight",
        lambda m: f"mlp.experts.local_experts.{m.group(1)}.linear_fc1_gate.weight", k,
    )
    k = re.sub(
        r"mlp\.experts\.(\d+)\.up_proj\.weight",
        lambda m: f"mlp.experts.local_experts.{m.group(1)}.linear_fc1.weight", k,
    )
    k = re.sub(
        r"mlp\.experts\.(\d+)\.down_proj\.weight",
        lambda m: f"mlp.experts.local_experts.{m.group(1)}.linear_fc2.weight", k,
    )

    k = re.sub(r"input_layernorm\.weight",           "input_layernorm.weight", k)
    k = re.sub(r"post_attention_layernorm\.weight",  "pre_mlp_layernorm.weight", k)

    if "rotary_emb" in k:
        return None

    return k


# ── Fusion helpers ────────────────────────────────────────────────────────────

def _fuse_qkv(state: Dict, prefix: str) -> None:
    """In-place: fuse split Q/K/V → linear_qkv."""
    q = state.pop(f"{prefix}.self_attention.linear_q.weight")
    k = state.pop(f"{prefix}.self_attention.linear_k.weight")
    v = state.pop(f"{prefix}.self_attention.linear_v.weight")
    state[f"{prefix}.self_attention.linear_qkv.weight"] = torch.cat([q, k, v], dim=0)


def _fuse_fc1(state: Dict, key_prefix: str) -> None:
    """In-place: fuse gate + up → fc1 for SwiGLU (concat along output dim)."""
    gate_key = f"{key_prefix}linear_fc1_gate.weight"
    up_key   = f"{key_prefix}linear_fc1.weight"
    if gate_key in state and up_key in state:
        state[f"{key_prefix}linear_fc1.weight"] = torch.cat(
            [state.pop(gate_key), state.pop(up_key)], dim=0
        )


# ── TP sharding ───────────────────────────────────────────────────────────────
#
# Megatron column-parallel: shard output dim (dim=0)  → QKV, fc1, word_embed, output_layer
# Megatron row-parallel:    shard input  dim (dim=1)  → proj (attn out), fc2

def _shard_col(tensor: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    chunk_size = tensor.shape[0] // tp_size
    return tensor[tp_rank * chunk_size:(tp_rank + 1) * chunk_size].contiguous()


def _shard_row(tensor: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    chunk_size = tensor.shape[1] // tp_size
    return tensor[:, tp_rank * chunk_size:(tp_rank + 1) * chunk_size].contiguous()


def _apply_tp(state: Dict, tp_rank: int, tp_size: int, num_layers: int, first_k_dense: int, n_routed_experts: int) -> Dict:
    """Return a new state dict with tensors sharded for the given tp_rank."""
    if tp_size == 1:
        return state

    out = {}
    for key, tensor in state.items():
        # ── Embedding / output layer (vocab parallel, column) ──────────────
        if key in ("model.embedding.word_embeddings.weight", "model.output_layer.weight"):
            out[key] = _shard_col(tensor, tp_rank, tp_size)
            continue

        # ── Per-layer rules ────────────────────────────────────────────────
        m = re.match(r"model\.decoder\.layers\.(\d+)\.(.*)", key)
        if m:
            layer_idx = int(m.group(1))
            sub = m.group(2)

            # Attention QKV (column parallel)
            if sub == "self_attention.linear_qkv.weight":
                out[key] = _shard_col(tensor, tp_rank, tp_size)
                continue
            # Attention output projection (row parallel)
            if sub == "self_attention.linear_proj.weight":
                out[key] = _shard_row(tensor, tp_rank, tp_size)
                continue

            # Dense MLP
            if layer_idx < first_k_dense:
                if sub == "mlp.linear_fc1.weight":  # gate+up fused, column
                    out[key] = _shard_col(tensor, tp_rank, tp_size)
                    continue
                if sub == "mlp.linear_fc2.weight":  # down, row
                    out[key] = _shard_row(tensor, tp_rank, tp_size)
                    continue

            # MoE shared expert
            if sub == "mlp.shared_expert.linear_fc1.weight":
                out[key] = _shard_col(tensor, tp_rank, tp_size)
                continue
            if sub == "mlp.shared_expert.linear_fc2.weight":
                out[key] = _shard_row(tensor, tp_rank, tp_size)
                continue

            # MoE routed experts (TP shards each expert independently)
            em = re.match(r"mlp\.experts\.local_experts\.(\d+)\.(.*)", sub)
            if em:
                expert_sub = em.group(2)
                if expert_sub == "linear_fc1.weight":
                    out[key] = _shard_col(tensor, tp_rank, tp_size)
                    continue
                if expert_sub == "linear_fc2.weight":
                    out[key] = _shard_row(tensor, tp_rank, tp_size)
                    continue

            # Router, LayerNorm — replicated across TP
            out[key] = tensor
            continue

        # Replicated (final norm, etc.)
        out[key] = tensor

    return out


# ── PP sharding ───────────────────────────────────────────────────────────────

def _layers_for_pp_rank(num_layers: int, pp_size: int, pp_rank: int) -> Tuple[int, int]:
    """Return [start, end) layer indices owned by pp_rank."""
    layers_per_stage = math.ceil(num_layers / pp_size)
    start = pp_rank * layers_per_stage
    end   = min(start + layers_per_stage, num_layers)
    return start, end


def _apply_pp(state: Dict, pp_rank: int, pp_size: int, num_layers: int) -> Dict:
    """Return a new state dict containing only keys relevant to pp_rank."""
    if pp_size == 1:
        return state

    layer_start, layer_end = _layers_for_pp_rank(num_layers, pp_size, pp_rank)
    is_first = pp_rank == 0
    is_last  = pp_rank == pp_size - 1

    out = {}
    for key, tensor in state.items():
        # Embedding: first stage only
        if "embedding" in key:
            if is_first:
                out[key] = tensor
            continue
        # Final norm + output layer: last stage only
        if "final_layernorm" in key or "output_layer" in key:
            if is_last:
                out[key] = tensor
            continue
        # Decoder layers: filter by ownership
        m = re.match(r"model\.decoder\.layers\.(\d+)\.(.*)", key)
        if m:
            layer_idx = int(m.group(1))
            if layer_start <= layer_idx < layer_end:
                # Remap layer index to local index within this PP stage
                local_idx = layer_idx - layer_start
                new_key = re.sub(
                    r"(model\.decoder\.layers\.)\d+\.",
                    f"\\g<1>{local_idx}.",
                    key,
                )
                out[new_key] = tensor
            continue
        # Other global keys: keep on all ranks
        out[key] = tensor

    return out


# ── EP sharding ───────────────────────────────────────────────────────────────

def _apply_ep(state: Dict, ep_rank: int, ep_size: int, n_routed_experts: int) -> Dict:
    """Keep only the routed experts assigned to ep_rank."""
    if ep_size == 1:
        return state

    experts_per_rank = n_routed_experts // ep_size
    expert_start = ep_rank * experts_per_rank
    expert_end   = expert_start + experts_per_rank

    out = {}
    for key, tensor in state.items():
        m = re.match(r"(.*mlp\.experts\.local_experts\.)(\d+)\.(.*)", key)
        if m:
            expert_idx = int(m.group(2))
            if expert_start <= expert_idx < expert_end:
                local_expert_idx = expert_idx - expert_start
                new_key = f"{m.group(1)}{local_expert_idx}.{m.group(3)}"
                out[new_key] = tensor
            # drop experts not belonging to this ep_rank
        else:
            out[key] = tensor

    return out


# ── HF weight loading ─────────────────────────────────────────────────────────

def _load_hf_weights(hf_dir: str) -> Dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
        shards = sorted(glob.glob(os.path.join(hf_dir, "model-*.safetensors")))
        if not shards:
            shards = [os.path.join(hf_dir, "model.safetensors")]
        loader = load_file
    except ImportError:
        shards = sorted(glob.glob(os.path.join(hf_dir, "pytorch_model-*.bin")))
        if not shards:
            shards = [os.path.join(hf_dir, "pytorch_model.bin")]
        loader = lambda p, **_: torch.load(p, map_location="cpu", weights_only=False)

    state = {}
    for shard in shards:
        print(f"  loading {os.path.basename(shard)}")
        state.update(loader(shard, device="cpu"))
    return state


# ── Main conversion ───────────────────────────────────────────────────────────

def convert(
    hf_dir: str,
    out_dir: str,
    tp_size: int = 1,
    pp_size: int = 1,
    ep_size: int = 1,
):
    """Convert DeepSeek-MoE-16B HF weights to sharded Megatron format.

    Args:
        hf_dir:   Path to HuggingFace model directory.
        out_dir:  Output directory for sharded .pt files.
        tp_size:  Tensor model parallel degree.
        pp_size:  Pipeline model parallel degree.
        ep_size:  Expert model parallel degree (must divide n_routed_experts).
    """
    from transformers import AutoConfig
    hf_cfg = AutoConfig.from_pretrained(hf_dir, trust_remote_code=True)
    num_layers      = hf_cfg.num_hidden_layers
    first_k_dense   = getattr(hf_cfg, "first_k_dense_replace", 1)
    n_routed_experts = getattr(hf_cfg, "n_routed_experts", 64)

    assert n_routed_experts % ep_size == 0, \
        f"ep_size={ep_size} must divide n_routed_experts={n_routed_experts}"

    print(f"Config: layers={num_layers}, first_k_dense={first_k_dense}, "
          f"n_routed_experts={n_routed_experts}")
    print(f"Parallelism: TP={tp_size}, PP={pp_size}, EP={ep_size}")

    print(f"\nLoading HF weights from {hf_dir} ...")
    hf_state = _load_hf_weights(hf_dir)

    print("Remapping keys ...")
    mg_state: Dict[str, torch.Tensor] = {}
    for hf_key, tensor in hf_state.items():
        mg_key = _remap_key(hf_key)
        if mg_key is None:
            continue
        mg_state[f"model.{mg_key}"] = tensor.to(torch.bfloat16)
    del hf_state

    print("Fusing QKV and FC1 projections ...")
    for layer_idx in range(num_layers):
        prefix = f"model.decoder.layers.{layer_idx}"
        _fuse_qkv(mg_state, prefix)
        if layer_idx < first_k_dense:
            _fuse_fc1(mg_state, f"{prefix}.mlp.")
        else:
            for e in range(n_routed_experts):
                _fuse_fc1(mg_state, f"{prefix}.mlp.experts.local_experts.{e}.")
            _fuse_fc1(mg_state, f"{prefix}.mlp.shared_expert.")

    os.makedirs(out_dir, exist_ok=True)

    total = tp_size * pp_size
    print(f"\nSharding into {total} files (TP={tp_size} × PP={pp_size}) ...")

    for pp_rank in range(pp_size):
        # PP slicing is cheap; do it once per pp_rank then shard by TP
        pp_state = _apply_pp(mg_state, pp_rank, pp_size, num_layers)

        for tp_rank in range(tp_size):
            tp_state = _apply_tp(
                pp_state, tp_rank, tp_size,
                num_layers, first_k_dense, n_routed_experts,
            )
            # EP sharding is applied on top of TP state
            # (each TP rank within an EP rank sees the same expert subset)
            ep_rank = 0  # default: all ranks share ep_rank 0 unless ep_size > 1
            if ep_size > 1:
                # Assume EP ranks map to pp_rank * tp_size // (tp_size // ep_size)
                # Standard mapping: ep_rank = global_rank // (tp_size * pp_size // ep_size)
                # For simplicity we expose per-(tp,pp) files and let the trainer pick ep_rank
                ep_rank = (pp_rank * tp_size + tp_rank) % ep_size
            final_state = _apply_ep(tp_state, ep_rank, ep_size, n_routed_experts)

            out_path = os.path.join(out_dir, f"tp{tp_rank:02d}_pp{pp_rank:02d}.pt")
            torch.save(final_state, out_path)
            n_params = sum(v.numel() for v in final_state.values())
            print(f"  saved tp={tp_rank} pp={pp_rank} ep={ep_rank} "
                  f"→ {out_path}  ({n_params/1e9:.2f}B params)")

    print(f"\nDone. {total} shard files written to {out_dir}")
    print("Load each shard by setting pretrained_weight_path to the corresponding file.")


def main():
    fire.Fire(convert)


if __name__ == "__main__":
    main()
