# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

unitorch is a unified modeling framework built on PyTorch that supports NLU, NLG, CV, multimodal learning, and reinforcement learning. It wraps 38+ model architectures (BERT, LLaMA, CLIP, Diffusers, etc.) with a configuration-driven CLI system for training, evaluation, inference, and serving.

## Build & Install

```bash
# Standard install
pip install .

# With optional extras
pip install ".[all]"           # everything
pip install ".[deepspeed]"     # DeepSpeed support
pip install ".[diffusers]"     # image generation models
pip install ".[detection]"     # object detection (timm)

# With CUDA C++ extensions (ngram kernel)
UNITORCH_EXTENSIONS=NGRAM pip install .
```

**Requires Python >= 3.10** and PyTorch 2.5+.

## Testing

```bash
# Run all tests
python3 -m pytest ./tests

# Run a single test file
python3 -m pytest ./tests/cli/test_decorators.py

# CI uses: PyTorch CPU, absl-py for test framework
```

Tests use `absl.testing` (not plain pytest fixtures). Test files are in `tests/cli/` and `tests/models/`.

## CLI Entry Points

Seven commands defined in `pyproject.toml` under `[project.scripts]`:

| Command | Console module | Purpose |
|---------|---------------|---------|
| `unitorch-train` | `cli.consoles.train` | Train models (supports torchrun) |
| `unitorch-eval` | `cli.consoles.eval` | Evaluate models |
| `unitorch-infer` | `cli.consoles.infer` | Run inference |
| `unitorch-launch` | `cli.consoles.launch` | Launch a quick script |
| `unitorch-fastapi` | `cli.consoles.fastapi` | FastAPI model server |
| `unitorch-service` | `cli.consoles.service` | Background non-model service |

All commands consume `.ini` config files. Examples in `examples/configs/`.

## Architecture

### Configuration System (`src/unitorch/cli/core.py`)

`CoreConfigureParser` extends Python's `configparser.ConfigParser` with:
- Extended interpolation between sections
- Safe AST-based value parsing (auto-converts strings to Python types)
- Section freezing for scoped parameter resolution
- CLI parameter override via `params=[[section, key, value], ...]`
- Remote file loading support

### Decorator Pattern (`src/unitorch/cli/decorators.py`)

Two key decorators wire config sections to class constructors and methods:
- `@add_default_section_for_init(section)` — on `from_core_configure` classmethods, auto-populates `__init__` params from the config section
- `@add_default_section_for_function(section)` — on instance methods, reads params from the config attached via `__unitorch_setting__`

Every model/pipeline class follows this pattern: `__init__` takes explicit args, `from_core_configure` classmethod reads them from config.

### Package Layout (`src/unitorch/`)

- `models/` — 38+ model wrappers (each with `modeling.py`, `processing.py`), thin layers over HuggingFace Transformers
- `cli/` — configuration-driven layer
  - `cli/models/` — model config adapters
  - `cli/pipelines/` — data processing pipelines (classification, generation, detection, segmentation, etc.)
  - `cli/tasks/` — task runners
  - `cli/consoles/` — entry point implementations
  - `cli/fastapis/` — FastAPI endpoint definitions
- `datasets/` — dataset implementations
- `losses/`, `scores/` — loss functions and evaluation metrics
- `optims/`, `schedulers/` — optimizer and LR scheduler wrappers
- `tasks/` — high-level task abstractions
- `clib/` — optional CUDA C++ extensions

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `UNITORCH_CACHE` | `~/.cache/unitorch` | Model/data cache (also sets HF_HOME, TRANSFORMERS_CACHE) |
| `UNITORCH_TEMP` | `/tmp/unitorch` | Temporary files |
| `UNITORCH_HOME` | `~/.unitorch` | Home directory |
| `UNITORCH_DEBUG` | `INFO` | Log level: OFF, INFO, DETAIL, CPU, ALL. CPU mode disables CUDA |
| `UNITORCH_EXTENSIONS` | unset | Set to "NGRAM" to build CUDA extensions |

## Key Patterns

- Models are thin wrappers around HuggingFace `transformers` classes — check upstream docs for model-specific behavior.
- PEFT (LoRA/QLoRA) integration lives in `models/peft/` and is applied as a wrapper around base models.
- Diffusion model support (Stable Diffusion, SDXL, ControlNet, Kolors) uses the `diffusers` library under `models/diffusers/`.
- Config files use INI format with section names like `core/model`, `core/pipeline`, `core/task` to wire components together.
