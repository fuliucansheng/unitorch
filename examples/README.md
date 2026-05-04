# Examples

This directory contains ready-to-use configuration files and scripts for training, evaluation, inference, serving, and model conversion with unitorch.

## Directory Structure

```
examples/
├── configs/            # Task-specific training/inference configs
│   ├── caption/        # Image captioning
│   ├── classification/ # Text & image classification
│   ├── deepspeed/      # DeepSpeed optimizer configs
│   ├── detection/      # Object detection
│   ├── diffusion/      # Image & video generation (Diffusers)
│   ├── generation/     # Text generation (LLM / seq2seq)
│   ├── pretrain/       # Pretraining (CLIP, VAE)
│   ├── segmentation/   # Image segmentation
│   ├── services/       # Background file-serving services
│   └── clip-interrogator.ini
├── fastapis/           # FastAPI model server configs
└── llms/               # Large language model configs & tools
    └── deepseek-moe-16b/
```

---

## configs/

### caption/

| File | Model | Task |
|------|-------|------|
| `blip.ini` | BLIP | Image captioning |

```bash
unitorch-train configs/caption/blip.ini --train_file train.tsv --dev_file dev.tsv
```

TSV columns: `image`, `text`

---

### classification/

Text and image classification configs. All use `core/task/supervised`.

| File | Model | Input |
|------|-------|-------|
| `bert.ini` | BERT | text |
| `roberta.ini` | RoBERTa | text |
| `llama.ini` | LLaMA | text |
| `llama.lora.ini` | LLaMA + LoRA | text |
| `mistral.ini` | Mistral | text |
| `mistral.lora.ini` | Mistral + LoRA | text |
| `clip.ini` | CLIP | text + image |
| `text_clip.ini` | CLIP | text only |
| `image_clip.ini` | CLIP | image only |
| `swin.ini` | Swin Transformer | image |
| `dinov2.ini` | DINOv2 | image |
| `llava_mistral_clip.ini` | LLaVA (Mistral+CLIP) | text + image |
| `llava_mistral_clip.lora.ini` | LLaVA + LoRA | text + image |
| `kolors.mps.ini` | Kolors MPS | image |

```bash
unitorch-train configs/classification/roberta.ini \
    --train_file train.tsv --dev_file dev.tsv --test_file test.tsv
```

TSV columns: `text`, `label` (or `text`, `image`, `label` for multimodal)

---

### generation/

Sequence-to-sequence and autoregressive generation. Includes LoRA, DPO, and GRPO variants.

| File | Model | Notes |
|------|-------|-------|
| `bart.ini` | BART | seq2seq |
| `t5.ini` | T5 | seq2seq |
| `pegasus.ini` | Pegasus | summarization |
| `xpegasus.ini` | XPegasus | cross-lingual |
| `mbart.ini` | MBart | multilingual |
| `llama.ini` | LLaMA | causal LM |
| `llama.lora.ini` | LLaMA + LoRA | PEFT fine-tuning |
| `mistral.ini` | Mistral | causal LM |
| `mistral.lora.ini` | Mistral + LoRA | PEFT fine-tuning |
| `qwen.ini` | QWen3 | causal LM |
| `qwen.lora.ini` | QWen3 + LoRA | PEFT fine-tuning |
| `qwen.dpo.lora.ini` | QWen3 + DPO LoRA | preference learning |
| `qwen.grpo.lora.ini` | QWen3 + GRPO LoRA | RL fine-tuning |
| `qwen_vl.ini` | QWen3-VL | vision-language |
| `qwen_vl.dpo.lora.ini` | QWen3-VL + DPO LoRA | preference learning |
| `llava_mistral_clip.ini` | LLaVA (Mistral+CLIP) | vision-language |
| `llava_mistral_clip.lora.ini` | LLaVA + LoRA | PEFT fine-tuning |
| `llava_llama_siglip.ini` | LLaVA (LLaMA+SigLIP) | vision-language |
| `llava_llama_siglip.lora.ini` | LLaVA + LoRA | PEFT fine-tuning |

```bash
# Standard training
unitorch-train configs/generation/llama.lora.ini \
    --train_file train.tsv --dev_file dev.tsv

# Multi-GPU with torchrun
torchrun --nproc_per_node 4 \
    $(which unitorch-train) configs/generation/qwen.lora.ini \
    --train_file train.tsv --dev_file dev.tsv
```

TSV columns: `encode` (input), `decode` (target); DPO adds `win_decode`, `lose_decode`

---

### diffusion/

Image and video generation/editing with Stable Flux and Wan models.

| Subdirectory | Task | Files |
|---|---|---|
| `text2image/` | Text → image | `stable-flux.ini`, `stable-flux.lora.ini`, `stable-flux.dpo.lora.ini`, `qwen-image.ini`, `qwen-image.lora.ini` |
| `image2image/` | Image → image | `stable-flux-redux.ini` |
| `inpainting/` | Image inpainting | `stable-flux.ini`, `stable-flux.lora.ini`, `stable-flux.dpo.lora.ini` |
| `editing/` | Image editing | `stable-flux.ini`, `stable-flux.lora.ini`, `qwen-image.ini`, `qwen-image.lora.ini` |
| `text2video/` | Text → video | `wan.ini`, `wan.lora.ini` |
| `image2video/` | Image → video | `wan.ini`, `wan.lora.ini` |

```bash
unitorch-train configs/diffusion/text2image/stable-flux.lora.ini \
    --train_file train.tsv --dev_file dev.tsv --output_folder ./images
```

TSV columns: `text` (prompt), `image` (path)

---

### segmentation/

| File | Model | Task |
|------|-------|------|
| `sam.ini` | SAM | Segment Anything |
| `mask2former.ini` | Mask2Former | Panoptic segmentation |
| `bria.ini` | BRIA | Background removal |

---

### detection/

| File | Model | Task |
|------|-------|------|
| `detr.ini` | DETR | Object detection |
| `grounding_dino.ini` | Grounding DINO | Open-vocabulary detection |

---

### pretrain/

| File | Model | Task |
|------|-------|------|
| `clip.ini` | CLIP | Contrastive image-text pretraining |
| `vae.ini` | VAE | Variational autoencoder pretraining |

---

### services/

Background HTTP microservices launched with `unitorch-service`.

| File | Service | Description |
|------|---------|-------------|
| `zip_files/config.ini` | `core/service/zip_files` | Serve files from `.zip` archives over HTTP |
| `zip_files/config_v2.ini` | `core/service/zip_files` | Multi-folder variant |
| `zip_saver.ini` | `core/service/zip_saver` | Save incoming files into a `.zip` archive |
| `http_files.ini` | `core/service/http_files` | Proxy a local file directory over HTTP |
| `mirror_files.ini` | `core/service/mirror` | Mirror a remote HTTP file server |
| `mirror_folders.ini` | `core/service/mirror` | Mirror multiple remote folders |

```bash
# Start zip file server (serves files from ./ on port 11230)
unitorch-service start configs/services/zip_files/config.ini \
    --zip_folder /data/archives --daemon_mode True

# Stop
unitorch-service stop configs/services/zip_files/config.ini
```

---

### clip-interrogator.ini

Run the CLIP Interrogator script to generate positive and negative prompts from images.

```bash
unitorch-launch configs/clip-interrogator.ini \
    --data_file data.tsv --image_col image --label_col label
```

TSV columns: `image` (path), `label` (0/1 binary relevance)

---

## fastapis/

FastAPI model server configs launched with `unitorch-fastapi`.

```bash
unitorch-fastapi fastapis.ini --port 8000 --device cpu
```

---

## llms/

Large language model training with Megatron-Core parallelism.

### llms/deepseek-moe-16b/

| File | Description |
|------|-------------|
| `config.ini` | Megatron supervised training config for DeepSeek-MoE-16B (TP=8, PP=1) |
| `config.json` | Megatron `TransformerConfig` JSON (28 layers, 64 experts) |
| `processing.py` | HuggingFace → Megatron weight conversion with TP/PP/EP sharding |

```bash
# Step 1: Convert HF weights (TP=8, PP=2)
python3.10 llms/deepseek-moe-16b/processing.py \
    --hf_dir /path/to/hf_model \
    --out_dir ./cache/deepseek-moe-16b \
    --tp_size 8 --pp_size 2 --ep_size 1

# Step 2: Launch Megatron training (16 GPUs)
torchrun --nproc_per_node 16 \
    $(which unitorch-train) llms/deepseek-moe-16b/config.ini \
    --train_file train.tsv --dev_file dev.tsv
```

---

## Common CLI Patterns

```bash
# Train
unitorch-train  <config.ini> [--key value ...]

# Evaluate
unitorch-eval   <config.ini> [--key value ...]

# Inference
unitorch-infer  <config.ini> [--key value ...]

# Run a script (clip-interrogator, etc.)
unitorch-launch <config.ini> [--key value ...]

# FastAPI server
unitorch-fastapi <config.ini> --port 8000

# Background service
unitorch-service start <config.ini> [--daemon_mode True]
unitorch-service stop  <config.ini>

# Multi-GPU (torchrun)
torchrun --nproc_per_node <N> $(which unitorch-train) <config.ini>
```

Config parameters can be overridden on the command line:

```bash
unitorch-train configs/generation/llama.lora.ini \
    --train_file ./my_train.tsv \
    --core/optim/adamw@learning_rate 5e-5 \
    --train_batch_size 4
```
