# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

unitorch is a unified modeling framework built on PyTorch that supports NLU, NLG, CV, multimodal learning, and reinforcement learning. It wraps 38+ model architectures (BERT, LLaMA, CLIP, Diffusers, etc.) with a configuration-driven CLI system for training, evaluation, inference, and serving.

## Architecture Boundary: Package vs CLI

Always preserve unitorch's package/CLI separation.

- The package layer (`src/unitorch/`) is the source of truth for reusable ML
  modules: model wrappers, processors, datasets, losses, scores,
  optimizers, schedulers, task abstractions, and utilities.
- The CLI layer (`src/unitorch/cli/`) is an adapter layer built on top of package
  modules. It adapts package modules to config-driven command flows such
  as train, eval, infer, serving, and agent-facing tools.
- Do not put core model, data, optimization, or metric behavior in the CLI layer
  when it belongs in the package layer. CLI modules should compose, register,
  configure, and orchestrate package modules.
- `src/unitorch/cli/models/`, `src/unitorch/cli/tasks/`,
  `src/unitorch/cli/fastapis/`, and `src/unitorch/cli/consoles/` should be
  treated as adapters for specific command flows, not as the canonical
  implementation of model behavior.
- Copilot tools and generated skills must follow the same boundary: reusable ML
  logic belongs in package modules; copilot/CLI code should expose, describe,
  plan, invoke, or orchestrate that logic for agents and command workflows.

## Project Skills

Project skills live in `skills/<skill-name>/SKILL.md`. This is the canonical
location for Claude, Codex, OpenCode, Studio, and other coding agents working in
this repository.

Before starting a task, inspect the frontmatter of each `skills/*/SKILL.md`.
If a skill's `description` matches the task, read that whole `SKILL.md` before
acting and follow its instructions. Do not use `.claude/skills` as the canonical
source; the project skills were migrated to `skills`.

Current skills:

- `config-ini`: use when writing, reviewing, or debugging unitorch `.ini`
  configs.
- `replace-decorator`: use when working with the process-global `@replace`
  decorator or replacement modules.

## Build & Install

```bash
# Standard install
pip install .

# With optional extras
pip install ".[all]"           # everything
pip install ".[deepspeed]"     # DeepSpeed support
pip install ".[diffusers]"     # image generation models

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

Six commands defined in `pyproject.toml` under `[project.scripts]`:

| Command | Console module | Purpose |
|---------|---------------|---------|
| `unitorch-train` | `cli.consoles.train` | Train models (supports torchrun) |
| `unitorch-eval` | `cli.consoles.eval` | Evaluate models |
| `unitorch-infer` | `cli.consoles.infer` | Run inference |
| `unitorch-fastapi` | `cli.consoles.fastapi` | FastAPI model server |
| `unitorch-copilot` | `cli.consoles.copilot:main` | Unitorch-native agent (similar to Claude / OpenCode) |
| `unitorch-copilot-cli` | `cli.consoles.copilot:cli_main` | CLI tool for agent use — invokes registered copilot tools |

All commands except `unitorch-copilot-cli` consume `.ini` config files. Examples in `examples/configs/`.

One-off scripts (previously `unitorch-launch`) are now run directly with:

```bash
python3 -m unitorch.cli.<module>
```

### `unitorch-copilot-cli`

Invokes a registered `copilot_tool` by name. Format:

```bash
unitorch-copilot-cli <name> [--key value ...]
# e.g.
unitorch-copilot-cli core/copilot/pkg_infos
```

Copilot tools live in `src/unitorch/cli/copilots/`. Each tool is a class that
extends `GenericCopilotTool` and is registered with `@register_copilot_tool("core/copilot/<name>")`.
It must implement `launch(**kwargs)`, `describe()`, and `usage()`.

To list all registered components (models, processes, fastapis, etc.):

```bash
unitorch-copilot-cli core/copilot/pkg_infos                  # list all types
unitorch-copilot-cli core/copilot/pkg_infos --name model      # list registered models only
unitorch-copilot-cli core/copilot/pkg_infos --name process    # list registered processes only
# available types: process, copilot_tool, model, fastapi,
#                  score, dataset, loss, optimizer, scheduler, task, writer
```

## Architecture

### Configuration System (`src/unitorch/cli/core.py`)

`Config` extends Python's `configparser.ConfigParser` with:
- Extended interpolation between sections
- Safe AST-based value parsing (auto-converts strings to Python types)
- Section freezing for scoped parameter resolution
- CLI parameter override via `params=[[section, key, value], ...]`
- Remote file loading support

### Decorator Pattern (`src/unitorch/cli/decorators.py`)

Two key decorators wire config sections to class constructors and methods:
- `@config_defaults_init(section)` — on `from_config` classmethods, auto-populates `__init__` params from the config section
- `@config_defaults_method(section)` — on instance methods, reads params from the config attached via `__unitorch_setting__`

Every model/pipeline class follows this pattern: `__init__` takes explicit args, `from_config` classmethod reads them from config.

### Package Layout (`src/unitorch/`)

- `models/` — 38+ model wrappers (each with `modeling.py`, `processing.py`), thin layers over HuggingFace Transformers
- `cli/` — configuration-driven layer
  - `cli/models/` — model config adapters
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

## `@replace` Decorator

Defined in `src/unitorch/utils/decorators.py`. Process-global monkey-patcher: replaces a target class across all loaded modules at import time, including rewriting subclass `__bases__`. Used to override upstream library behaviour (e.g. `diffusers`, `datasets`) without forking. Replacement classes inherit from the target and are named `<Original>V2` by convention. See `skills/replace-decorator/SKILL.md` for full details.

## CLI Argument Conventions

- All command-line scripts must use the `fire` package for argument parsing. Do **not** use `argparse`, `click`, or `sys.argv` directly.

## INI Config Generation Conventions

- When generating `.ini` config files for `unitorch-fastapi`, always set `device = cpu` in `[core/cli]`. Train / eval / infer configs do not need this key.

## Sync Requirements After Code Changes

After **any** code change (adding/removing models, CLI commands, processors, etc.), always keep the following in sync:

1. **`examples/configs/`** — add, update, or remove `.ini` config files to match the current set of supported models and pipelines.
2. **`mkdocs.yml`** & **`wiki/`** — update the docs navigation and any auto-generated API references to reflect the change.
3. **`README.md`** — update the Supported Models table and CLI Commands table.

## Code Constraints

> **No AutoClass — ever.**
> All model and processor classes in this repo must be instantiated and defined explicitly using their full concrete class names. Using any HuggingFace AutoClass (`AutoModel`, `AutoModelForCausalLM`, `AutoTokenizer`, `AutoProcessor`, `AutoConfig`, `AutoFeatureExtractor`, etc.) is **strictly prohibited**. Every class definition and its implementation must be fully transparent and visible in source — no dynamic dispatch, no opaque factory resolution.

## Key Patterns

- Models are thin wrappers around HuggingFace `transformers` classes — check upstream docs for model-specific behavior.
- PEFT (LoRA/DPO/GRPO) integration lives in `models/peft/` and is applied as a wrapper around base models.
- Diffusion model support (StableFlux, Wan video, QWenImage) uses the `diffusers` library under `models/diffusers/`.
- Config files use INI format with section names like `core/model`, `core/fastapi/pipeline`, `core/task` to wire components together.

## CLI Model Reference

> **Critical**: `cli/models/<name>/__init__.py` files only do side-effect imports (`import unitorch.cli.models.X.modeling`). They do **not** re-export classes. Wiki docstring references and any direct imports **must** use the full submodule path, e.g. `unitorch.cli.models.bart.processing.BartProcessor`.

### Required Data Flow Contract for `unitorch-train` / `unitorch-infer`

All CLI models that integrate with `unitorch-train` or `unitorch-infer` **must strictly follow** the data flow contracts below. Adherence is required for registered components (models, processors, writers) to be composable and reusable across the framework.

#### Training Flow

```
raw data (text, PIL image, etc.)
  → preprocess        → tensor(s)
  → collate_fn        → batched tensor(s)   # stack or concat, depending on preprocess output type
  → model forward     → tensor(s)
  → loss compute      → scalar loss
  → backward / optim update
```

- **preprocess**: takes raw data, returns tensor(s). Each output field's type determines how `collate_fn` handles it.
- **collate_fn**: stacks or concatenates tensors depending on whether the preprocess output is a fixed-shape tensor (stack) or variable-length sequence (concat). The collation strategy must be consistent with the preprocess output contract: **stack** produces a single batched tensor (shape `[B, ...]`); **concat** produces a `list` of tensors (one per sample, shapes may differ).
- **model forward**: receives batched tensors, returns tensor(s) fed into the loss.
- **loss compute**: operates on model outputs and labels, returns a scalar.

#### Inference Flow

```
raw data (text, PIL image, etc.)
  → preprocess        → tensor(s)
  → collate_fn        → batched tensor(s)   # stack or concat, depending on preprocess output type
  → model forward     → tensor(s)
  → postprocess       → raw data (pandas DataFrame format)
  → writer            → output file (jsonl, tsv, parquet, etc.)
```

- **preprocess** and **collate_fn**: same contract as training.
- **model forward**: receives batched tensors, returns tensor(s) passed to postprocess.
- **postprocess**: converts model output tensors back to human-readable form; **must return a `pandas.DataFrame`**.
- **writer**: consumes the DataFrame and writes to disk in the configured format (jsonl, tsv, parquet, etc.). Registered writers are reusable across models as long as the DataFrame schema is respected.

### Config Section Naming Convention

| Category | Section pattern | Example |
|----------|----------------|---------|
| Processor | `core/process/<name>` | `core/process/llama` |
| Classification model | `core/model/classification/<name>` | `core/model/classification/roberta` |
| Generation model | `core/model/generation/<name>` | `core/model/generation/llama` |
| Detection model | `core/model/detection/<name>` | `core/model/detection/detr` |
| Segmentation model | `core/model/segmentation/<name>` | `core/model/segmentation/sam` |
| Diffusion model | `core/model/diffusers/<task>/<name>` | `core/model/diffusers/text2image/stable_flux` |
| PEFT LoRA | `core/model/<task>/peft/lora/<name>` | `core/model/generation/peft/lora/llama` |
| PEFT DPO | `core/model/generation/peft/dpo/lora/<name>` | `core/model/generation/peft/dpo/lora/qwen3` |
| PEFT GRPO | `core/model/generation/peft/grpo/lora/<name>` | `core/model/generation/peft/grpo/lora/qwen3` |

## Current Model Inventory

| Module | Foundation classes | CLI config section(s) |
|--------|-------------------|----------------------|
| **bart** | BartProcessor, BartForGeneration | `core/process/bart`, `core/model/generation/bart` |
| **beit** | BeitProcessor, BeitForImageClassification | `core/process/beit`, `core/model/classification/beit` |
| **bert** | BertProcessor, BertForClassification | `core/process/bert`, `core/model/classification/bert` |
| **bria** | BRIAProcessor, BRIAForSegmentation | `core/process/bria`, `core/model/segmentation/bria` |
| **chinese_clip** | ChineseClipProcessor, ChineseClipForPretrain/Classification/TextClassification/ImageClassification | `core/process/chinese_clip`, `core/model/pretrain\|classification/chinese_clip` |
| **clip** | ClipProcessor, ClipForPretrain/Classification/TextClassification/ImageClassification | `core/process/clip`, `core/model/pretrain\|classification/clip` |
| **detr** | DetrProcessor, DetrForDetection | `core/process/detr`, `core/model/detection/detr` |
| **diffusers** | StableFluxProcessor, StableFluxFor{Text2Image,Image2Image,ImageRedux,ImageInpainting,Kontext2Image}, WanFor{Text2Video,Image2Video}, QWenImageProcessor, QWenImageText2ImageGeneration, QWenImageEditingGeneration | `core/process/diffusion/{stable_flux,wan,qwen_image}`, `core/model/diffusers/{text2image,image2image,image_redux,inpainting,kontext2image}/stable_flux`, `core/model/diffusers/{text2video,image2video}/wan`, `core/model/diffusers/{text2image,editing}/qwen_image` |
| **dinov2** | DinoV2Processor, DinoV2ForImageClassification | `core/process/dinov2`, `core/model/classification/dinov2` |
| **dpt** | DPTProcessor, DPTForDepthEstimation | `core/process/dpt`, `core/model/dpt` |
| **grounding_dino** | GroundingDinoProcessor, GroundingDinoForDetection | `core/process/grounding_dino`, `core/model/detection/grounding_dino` |
| **kolors** | KolorsMPSProcessor, KolorsMPSModel | `core/process/kolors/mps`, `core/model/classification/kolors/mps` |
| **llama** | LlamaProcessor, LlamaForClassification/Generation | `core/process/llama`, `core/model/classification\|generation/llama` |
| **llava** | LlavaMistralClipProcessor, LlavaLlamaSiglipProcessor, LlavaMistralClipFor{Classification,Generation}, LlavaLlamaSiglipForGeneration | `core/process/llava/{mistral_clip,llama_siglip}`, `core/model/{classification,generation}/llava/{mistral_clip,llama_siglip}` |
| **mask2former** | Mask2FormerProcessor, Mask2FormerForSegmentation | `core/process/mask2former`, `core/model/segmentation/mask2former` |
| **mbart** | MBartProcessor, MBartForGeneration | `core/process/mbart`, `core/model/generation/mbart` |
| **mistral** | MistralProcessor, MistralForClassification/Generation | `core/process/mistral`, `core/model/classification\|generation/mistral` |
| **peft** | ClipLoraForMatching/TextMatching, LlamaLoraFor{Classification,Generation}, LlavaMistralClipLoraFor{Classification,Generation}, LlavaLlamaSiglipLoraForGeneration, MistralLoraFor{Classification,Generation}, QWen3LoraForGeneration, QWen3DPOLoraForGeneration, QWen3GRPOLoraForGeneration, QWen3VLLoraForGeneration, QWen3VLDPOLoraForGeneration | `core/model/{matching,classification,generation}/peft/lora/...` |
| **qwen** | QWenProcessor, QWenVLProcessor, QWen3ForGeneration, QWen3VLForGeneration | `core/process/qwen`, `core/process/qwen_vl`, `core/model/generation/qwen3`, `core/model/generation/qwen3_vl` |
| **roberta** | RobertaProcessor, RobertaForClassification/MaskLM | `core/process/roberta`, `core/model/classification/roberta` |
| **sam** | SamProcessor, SamForSegmentation | `core/process/sam`, `core/model/segmentation/sam` |
| **segformer** | SegformerProcessor, SegformerForSegmentation | `core/process/segformer`, `core/model/segmentation/segformer` |
| **siglip** | SiglipProcessor, SiglipForPretrain/Classification/TextClassification/ImageClassification/Matching | `core/process/siglip`, `core/model/pretrain\|classification\|matching/siglip` |
| **swin** | SwinProcessor, SwinForImageClassification | `core/process/swin`, `core/model/classification/swin` |
| **xlm_roberta** | XLMRobertaProcessor, XLMRobertaForClassification/MaskLM, XLMRobertaXLForClassification | `core/process/xlm_roberta`, `core/model/classification/xlm_roberta` |
