#!/usr/bin/env python3.10
# convert_ckpt.py — Convert deepseek-ai/deepseek-moe-16b-base HuggingFace weights
# to the flat dict format expected by MegatronGPTForGeneration.from_pretrained().
#
# Usage:
#   python3.10 convert_ckpt.py --hf_dir /path/to/hf_model --out ./cache/deepseek-moe-16b/hf_converted.pt
#
# The output is a single .pt file (or sharded .pt files for large models).
# Set pretrained_weight_path in the .ini to the output path.

import fire
import os
import re
import torch
from typing import Dict


# ── HuggingFace → Megatron key mapping ───────────────────────────────────────
# DeepSeek-MoE uses the same backbone naming as LLaMA with MoE layers.

def _remap_key(hf_key: str) -> str | None:
    """Map a HuggingFace DeepSeek key to Megatron GPTModel namespace.

    Returns None to drop the parameter (e.g. unused router buffers).
    """
    k = hf_key

    # Embedding
    k = re.sub(r"^model\.embed_tokens\.weight$", "embedding.word_embeddings.weight", k)

    # Final norm + lm_head
    k = re.sub(r"^model\.norm\.weight$",  "decoder.final_layernorm.weight", k)
    k = re.sub(r"^lm_head\.weight$",      "output_layer.weight", k)

    # Layer prefix
    k = re.sub(r"^model\.layers\.(\d+)\.", lambda m: f"decoder.layers.{m.group(1)}.", k)

    # Self-attention
    k = re.sub(r"self_attn\.q_proj\.weight", "self_attention.linear_q.weight", k)
    k = re.sub(r"self_attn\.k_proj\.weight", "self_attention.linear_k.weight", k)
    k = re.sub(r"self_attn\.v_proj\.weight", "self_attention.linear_v.weight", k)
    k = re.sub(r"self_attn\.o_proj\.weight", "self_attention.linear_proj.weight", k)

    # Dense MLP (first_k_dense_replace layers use standard MLP)
    k = re.sub(r"mlp\.gate_proj\.weight",   "mlp.linear_fc1_gate.weight", k)
    k = re.sub(r"mlp\.up_proj\.weight",     "mlp.linear_fc1.weight", k)
    k = re.sub(r"mlp\.down_proj\.weight",   "mlp.linear_fc2.weight", k)

    # MoE gate router
    k = re.sub(r"mlp\.gate\.weight",        "mlp.router.weight", k)

    # MoE shared experts
    k = re.sub(r"mlp\.shared_experts\.gate_proj\.weight", "mlp.shared_expert.linear_fc1_gate.weight", k)
    k = re.sub(r"mlp\.shared_experts\.up_proj\.weight",   "mlp.shared_expert.linear_fc1.weight", k)
    k = re.sub(r"mlp\.shared_experts\.down_proj\.weight", "mlp.shared_expert.linear_fc2.weight", k)

    # MoE routed experts
    k = re.sub(
        r"mlp\.experts\.(\d+)\.gate_proj\.weight",
        lambda m: f"mlp.experts.local_experts.{m.group(1)}.linear_fc1_gate.weight", k
    )
    k = re.sub(
        r"mlp\.experts\.(\d+)\.up_proj\.weight",
        lambda m: f"mlp.experts.local_experts.{m.group(1)}.linear_fc1.weight", k
    )
    k = re.sub(
        r"mlp\.experts\.(\d+)\.down_proj\.weight",
        lambda m: f"mlp.experts.local_experts.{m.group(1)}.linear_fc2.weight", k
    )

    # LayerNorm / RMSNorm
    k = re.sub(r"input_layernorm\.weight",      "input_layernorm.weight", k)
    k = re.sub(r"post_attention_layernorm\.weight", "pre_mlp_layernorm.weight", k)

    # Drop rotary embedding buffers (recomputed)
    if "rotary_emb" in k:
        return None

    return k


def fuse_qkv(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """Megatron expects fused QKV weight; fuse Q/K/V from HF split weights."""
    q = state.pop(f"{prefix}.self_attention.linear_q.weight")
    k = state.pop(f"{prefix}.self_attention.linear_k.weight")
    v = state.pop(f"{prefix}.self_attention.linear_v.weight")
    state[f"{prefix}.self_attention.linear_qkv.weight"] = torch.cat([q, k, v], dim=0)
    return state


def fuse_fc1(state: Dict[str, torch.Tensor], prefix: str, expert_path: str = "") -> Dict[str, torch.Tensor]:
    """Fuse gate + up projections into a single linear_fc1 for SwiGLU."""
    gate_key = f"{prefix}.{expert_path}linear_fc1_gate.weight"
    up_key   = f"{prefix}.{expert_path}linear_fc1.weight"
    if gate_key in state and up_key in state:
        gate = state.pop(gate_key)
        up   = state.pop(up_key)
        state[f"{prefix}.{expert_path}linear_fc1.weight"] = torch.cat([gate, up], dim=0)
    return state


def convert(hf_dir: str, out_path: str):
    from transformers import AutoConfig
    hf_cfg = AutoConfig.from_pretrained(hf_dir, trust_remote_code=True)
    first_k_dense = getattr(hf_cfg, "first_k_dense_replace", 1)
    num_layers     = hf_cfg.num_hidden_layers

    print(f"Loading HF weights from {hf_dir} ...")
    # Support sharded safetensors / pytorch_model.bin
    try:
        from safetensors.torch import load_file
        import glob
        shards = sorted(glob.glob(os.path.join(hf_dir, "model-*.safetensors")))
        if not shards:
            shards = [os.path.join(hf_dir, "model.safetensors")]
        hf_state: Dict[str, torch.Tensor] = {}
        for shard in shards:
            print(f"  loading {os.path.basename(shard)}")
            hf_state.update(load_file(shard, device="cpu"))
    except ImportError:
        import glob
        shards = sorted(glob.glob(os.path.join(hf_dir, "pytorch_model-*.bin")))
        if not shards:
            shards = [os.path.join(hf_dir, "pytorch_model.bin")]
        hf_state: Dict[str, torch.Tensor] = {}
        for shard in shards:
            print(f"  loading {os.path.basename(shard)}")
            hf_state.update(torch.load(shard, map_location="cpu"))

    print("Remapping keys ...")
    megatron_state: Dict[str, torch.Tensor] = {}
    for hf_key, tensor in hf_state.items():
        mg_key = _remap_key(hf_key)
        if mg_key is None:
            continue
        megatron_state[mg_key] = tensor.to(torch.bfloat16)

    # Prefix with "model." to match MegatronGPTForGeneration state dict
    megatron_state = {f"model.{k}": v for k, v in megatron_state.items()}

    print("Fusing QKV and FC1 projections ...")
    for layer_idx in range(num_layers):
        prefix = f"model.decoder.layers.{layer_idx}"
        megatron_state = fuse_qkv(megatron_state, prefix)

        if layer_idx < first_k_dense:
            # Dense MLP layer
            megatron_state = fuse_fc1(megatron_state, prefix, "mlp.")
        else:
            # MoE layer — fuse each routed expert and the shared expert
            n_experts = hf_cfg.n_routed_experts
            for e in range(n_experts):
                megatron_state = fuse_fc1(
                    megatron_state, prefix,
                    f"mlp.experts.local_experts.{e}."
                )
            megatron_state = fuse_fc1(megatron_state, prefix, "mlp.shared_expert.")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    print(f"Saving converted weights to {out_path} ...")
    torch.save(megatron_state, out_path)
    print(f"Done. Total parameters: {len(megatron_state)}")


def main():
    fire.Fire(convert)


if __name__ == "__main__":
    main()
