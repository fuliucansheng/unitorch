<div align="center">

![unitorch](https://raw.githubusercontent.com/fuliucansheng/unitorch/master/unitorch.png)

[Documentation](https://fuliucansheng.github.io/unitorch) •
[Installation](https://fuliucansheng.github.io/unitorch/installation/) •
[Report Issues](https://github.com/fuliucansheng/unitorch/issues/new?assignees=&labels=&template=bug-report.yml)

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/unitorch)](https://pypi.org/project/unitorch/)
[![PyPI Version](https://badge.fury.io/py/unitorch.svg)](https://badge.fury.io/py/unitorch)
[![PyPI Downloads](https://pepy.tech/badge/unitorch)](https://pepy.tech/project/unitorch)
[![Github Downloads](https://img.shields.io/github/downloads/fuliucansheng/unitorch/total?color=blue&label=downloads&logo=github&logoColor=lightgrey)](https://img.shields.io/github/downloads/fuliucansheng/unitorch/total?color=blue&label=Downloads&logo=github&logoColor=lightgrey)
[![License](https://img.shields.io/github/license/fuliucansheng/unitorch?color=dfd)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-pink.svg)](https://github.com/fuliucansheng/unitorch/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22)

</div>

## Introduction

🔥 **unitorch** is a PyTorch-based library that unifies training, inference, and serving of state-of-the-art models across NLP, computer vision, multimodal learning, and more. It wraps 20+ model architectures with a configuration-driven CLI, integrating seamlessly with [transformers](https://github.com/huggingface/transformers), [peft](https://github.com/huggingface/peft), and [diffusers](https://github.com/huggingface/diffusers).

Get started with a single import or a one-line CLI command — no boilerplate required.

## Features

| | |
|---|---|
| **Unified Model Support** | 20+ architectures: LLMs, diffusion models, vision transformers, multimodal models |
| **Configuration-Driven CLI** | Train, evaluate, infer, and serve via `.ini` config files |
| **Multi-GPU & Distributed** | Native `torchrun` support + DeepSpeed integration for large-scale models |
| **CUDA Optimized** | Optional CUDA C++ extensions for accelerated kernels |
| **PEFT / LoRA** | Built-in parameter-efficient fine-tuning support |
| **Model Serving** | FastAPI-based serving with `unitorch-fastapi` |

## Installation

```bash
pip install unitorch
```

<details>
<summary>Optional extras</summary>

```bash
pip install "unitorch[all]"          # everything
pip install "unitorch[deepspeed]"    # DeepSpeed support
pip install "unitorch[diffusers]"    # image generation models
```

Requires **Python >= 3.10** and **PyTorch 2.5+**.
</details>

## Quick Start

**Python API**
```python
from unitorch.models.bart import BartForGeneration
model = BartForGeneration("path/to/bart/config.json")

# Configuration-driven setup
from unitorch.cli import Config
config = Config("path/to/config.ini")
```

**Multi-GPU Training**
```bash
torchrun --no_python --nproc_per_node 4 \
    unitorch-train examples/configs/generation/bart.ini \
    --train_file path/to/train.tsv --dev_file path/to/dev.tsv
```

**Inference**
```bash
unitorch-infer examples/configs/generation/bart.ini --test_file path/to/test.tsv
```

> See the [documentation](https://fuliucansheng.github.io/unitorch) for full tutorials and examples.

## Supported Models

<details>
<summary>View all supported models</summary>

| Domain | Models |
|--------|--------|
| **Language** | BERT, RoBERTa, XLM-RoBERTa, BART, MBart, LLaMA, Mistral, QWen3 |
| **Vision** | BEiT, Swin Transformer, DINOv2, CLIP, SigLIP |
| **Multimodal** | LLaVA, QWen3-VL, Chinese CLIP |
| **Image Generation** | FLUX (StableFlux), QWenImage |
| **Video Generation** | Wan |
| **Detection** | DETR, Grounding DINO |
| **Segmentation** | SAM, Mask2Former, SegFormer, BRIA |
| **Depth Estimation** | DPT |
| **PEFT** | LoRA, DPO, GRPO (via peft wrappers) |

</details>

## CLI Commands

| Command | Purpose |
|---------|---------|
| `unitorch-train` | Train models (supports `torchrun`) |
| `unitorch-eval` | Evaluate models |
| `unitorch-infer` | Run batch inference |
| `unitorch-fastapi` | Start a FastAPI model server |
| `unitorch-copilot` | unitorch-native agent (similar to Claude / OpenCode) |
| `unitorch-copilot-cli` | CLI tool for agent use — invokes registered copilot tools |

## License

Released under the [MIT License](LICENSE).
