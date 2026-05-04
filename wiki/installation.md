# Installation

## Requirements

- Python >= 3.10
- [torch](http://pytorch.org/) >= 2.5
- [torchvision](http://pytorch.org/) >= 0.20
- [torchaudio](http://pytorch.org/) >= 2.5
- [transformers](https://github.com/huggingface/transformers)
- [peft](https://github.com/huggingface/peft)
- [datasets](https://github.com/huggingface/datasets)

## Install from PyPI

```bash
pip install unitorch
```

## Install with Optional Extras

```bash
# All extras (recommended)
pip install "unitorch[all]"

# DeepSpeed support
pip install "unitorch[deepspeed]"

# Image generation (FLUX)
pip install "unitorch[diffusers]"

# Object detection (timm)
pip install "unitorch[detection]"

# Megatron-LM support
pip install "unitorch[megatron]"

# FastAPI / ONNX / W&B
pip install "unitorch[others]"
```

## Install from Source

```bash
pip install "git+https://github.com/fuliucansheng/unitorch"

# With extras
pip install "git+https://github.com/fuliucansheng/unitorch#egg=unitorch[all]"
```

## Install with CUDA Extensions

```bash
# Build the ngram CUDA kernel
UNITORCH_EXTENSIONS=NGRAM pip install .
```

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `UNITORCH_CACHE` | `~/.cache/unitorch` | Model/data cache (also sets `HF_HOME`, `TRANSFORMERS_CACHE`) |
| `UNITORCH_TEMP` | `/tmp/unitorch` | Temporary files |
| `UNITORCH_HOME` | `~/.unitorch` | Home directory |
| `UNITORCH_DEBUG` | `INFO` | Log level: `OFF`, `INFO`, `DETAIL`, `CPU`, `ALL` |
| `UNITORCH_EXTENSIONS` | _(unset)_ | Set to `NGRAM` to build CUDA extensions |