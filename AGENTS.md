# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## Project Overview

`unitorch` is a unified modeling framework built on PyTorch that supports NLU, NLG, CV, multimodal learning, and reinforcement learning. It wraps 38+ model architectures, including BERT, LLaMA, CLIP, and Diffusers, with a configuration-driven CLI system for training, evaluation, inference, and serving.

Python `>=3.10` and PyTorch `>=2.5` are required.

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
this repository. Generated UniTorch copilot tool skills live under
`skills/unitorch-copilot-tools/`.

Before starting a task, inspect the frontmatter of each `skills/*/SKILL.md`.
If a skill's `description` matches the task, read that whole `SKILL.md` before
acting and follow its instructions. Do not use `.claude/skills` or `.skills` as
the canonical source.

Current skills:

- `config-ini`: use when writing, reviewing, or debugging unitorch `.ini`
  configs.
- `replace-decorator`: use when working with the process-global `@replace`
  decorator or replacement modules.

## Build And Install

```bash
# Standard install
pip install .

# Editable development install
python -m pip install -e .

# Optional extras
pip install ".[all]"
pip install ".[deepspeed]"
pip install ".[diffusers]"
python -m pip install -e ".[docs]"

# With CUDA C++ extensions (ngram kernel)
UNITORCH_EXTENSIONS=NGRAM pip install .
```

Use `".[all]"` only when every optional backend is needed.

For development commands that run `python3 -m unitorch...`, npm skill scripts,
and unit tests, install the package first, normally with
`python3 -m pip install -e .` for local development. Run those commands from
the installed package environment without `PYTHONPATH=src`; do not document
`PYTHONPATH=src` as the normal path.

## Testing

```bash
# Run all tests
python3 -m pytest ./tests

# Run a single test file
python3 -m pytest ./tests/cli/test_decorators.py
```

Tests use `absl.testing` and `absl.testing.parameterized` where helpful. Test files live in `tests/cli/` and `tests/models/`. Coverage is still fairly light, so new features and bug fixes should ship with at least one targeted test and, when relevant, an updated example config.

## Documentation

```bash
mkdocs serve
mkdocs build
```

Source documentation belongs in `wiki/`; `docs/` is generated output. Update `wiki/` first, then regenerate `docs/` as needed.

## CLI Entry Points

Six commands are defined in `pyproject.toml` under `[project.scripts]`:

| Command | Console module | Purpose |
|---------|---------------|---------|
| `unitorch-train` | `cli.consoles.train` | Train models (supports `torchrun`) |
| `unitorch-eval` | `cli.consoles.eval` | Evaluate models |
| `unitorch-infer` | `cli.consoles.infer` | Run inference |
| `unitorch-fastapi` | `cli.consoles.fastapi` | FastAPI model server |
| `unitorch-copilot` | `cli.consoles.copilot:main` | Unitorch-native agent |
| `unitorch-copilot-cli` | `cli.consoles.copilot:cli_main` | CLI tool for agent use |

All commands except `unitorch-copilot-cli` consume `.ini` config files. Examples live in `src/unitorch/cli/configs/`.

One-off scripts that previously used `unitorch-launch` should now run directly with:

```bash
python3 -m unitorch.cli.<module>
```

### `unitorch-copilot-cli`

`unitorch-copilot-cli` invokes a registered `copilot_tool` by name:

```bash
unitorch-copilot-cli <name> [--key value ...]
unitorch-copilot-cli core/copilot/pkg_infos
```

Copilot tools live in `src/unitorch/cli/copilots/`. Each tool is a class extending `GenericCopilotTool`, registered with `@register_copilot_tool("core/copilot/<name>")`, and must implement `launch(**kwargs)`, `describe()`, and `usage()`.

To list registered components:

```bash
unitorch-copilot-cli core/copilot/pkg_infos
unitorch-copilot-cli core/copilot/pkg_infos --name model
unitorch-copilot-cli core/copilot/pkg_infos --name process
```

Available listing types include `process`, `copilot_tool`, `model`, `fastapi`, `score`, `dataset`, `loss`, `optimizer`, `scheduler`, `task`, and `writer`.

## Project Structure

- `src/unitorch/models/`: model wrappers and processors
- `src/unitorch/cli/`: configuration-driven CLI layer
- `src/unitorch/cli/models/`: model config adapters
- `src/unitorch/cli/tasks/`: task runners
- `src/unitorch/cli/consoles/`: entry point implementations
- `src/unitorch/cli/fastapis/`: FastAPI endpoint definitions
- `src/unitorch/datasets/`: dataset implementations
- `src/unitorch/losses/`, `src/unitorch/scores/`: losses and metrics
- `src/unitorch/optims/`, `src/unitorch/schedulers/`: optimization wrappers
- `src/unitorch/tasks/`: high-level task abstractions
- `src/unitorch/clib/`: optional CUDA C++ extensions
- `src/unitorch/cli/configs/`: runnable `.ini` examples
- `tests/`: automated checks

Add new modules in the closest existing package and keep filenames in `snake_case`.

## Architecture

### Configuration System

`src/unitorch/cli/core.py` defines `Config`, which extends Python's `configparser.ConfigParser` with:

- Extended interpolation between sections
- Safe AST-based value parsing from strings into Python types
- Section freezing for scoped parameter resolution
- CLI parameter override via `params=[[section, key, value], ...]`
- Remote file loading support

### Decorator Pattern

`src/unitorch/cli/decorators.py` defines the two key decorators used across the codebase:

- `@config_defaults_init(section)`: used on `from_config` classmethods to auto-populate `__init__` parameters from a config section
- `@config_defaults_method(section)`: used on instance methods to read parameters from the config attached through `__unitorch_setting__`

Every model and pipeline class should follow the same pattern: explicit `__init__` arguments plus a `from_config` classmethod that reads from config.

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `UNITORCH_CACHE` | `~/.cache/unitorch` | Model and data cache; also sets `HF_HOME` and `TRANSFORMERS_CACHE` |
| `UNITORCH_TEMP` | `/tmp/unitorch` | Temporary files |
| `UNITORCH_HOME` | `~/.unitorch` | Home directory |
| `UNITORCH_DEBUG` | `INFO` | Log level: `OFF`, `INFO`, `DETAIL`, `CPU`, `ALL` |
| `UNITORCH_EXTENSIONS` | unset | Set to `NGRAM` to build CUDA extensions |

`UNITORCH_DEBUG=CPU` disables CUDA.

## `@replace` Decorator

`src/unitorch/utils/decorators.py` defines `@replace`, a process-global monkey patcher that replaces a target class across loaded modules at import time, including rewriting subclass `__bases__`. It is used to override upstream library behavior, such as in `diffusers` or `datasets`, without forking. Replacement classes inherit from the target and are named `<Original>V2` by convention.

## CLI Argument Conventions

- All command-line scripts must use the `fire` package for argument parsing.
- Do not use `argparse`, `click`, or direct `sys.argv` handling.

## INI Config Generation Conventions

- When generating `.ini` config files for `unitorch-fastapi`, always set `device = cpu` in `[core/cli]`.
- Train, eval, and infer configs do not require this key.

## Sync Requirements After Code Changes

After any code change that adds, removes, or modifies models, CLI commands, processors, or related components, keep the following in sync:

1. `src/unitorch/cli/configs/`: add, update, or remove `.ini` files
2. `mkdocs.yml` and `wiki/`: update navigation and API references
3. `README.md`: update the supported models table and CLI commands table

## Code Constraints

> No AutoClass, ever.
>
> All model and processor classes in this repository must be instantiated and defined explicitly using their full concrete class names. Using any HuggingFace AutoClass, including `AutoModel`, `AutoModelForCausalLM`, `AutoTokenizer`, `AutoProcessor`, `AutoConfig`, and `AutoFeatureExtractor`, is strictly prohibited. Every class definition and its implementation must remain fully transparent in source. Avoid dynamic dispatch and opaque factory resolution.

## Coding Style And Naming

Use 4-space indentation, `snake_case` for modules, functions, and files, and `CamelCase` for classes. Follow the surrounding import order and docstring style. New public APIs should include concise docstrings and type hints where practical. No repo-level formatter configuration is checked in, so match the local style instead of reformatting unrelated code.

## Key Patterns

- Models are thin wrappers around HuggingFace `transformers` classes; check upstream docs for model-specific behavior
- PEFT integration, including LoRA, DPO, and GRPO, lives under `models/peft/`
- Diffusion support, including StableFlux, Wan, and QWenImage, lives under `models/diffusers/`
- Config files use INI sections such as `core/model`, `core/fastapi/pipeline`, and `core/task` to wire components together

## CLI Model Reference

`cli/models/<name>/__init__.py` files only do side-effect imports such as `import unitorch.cli.models.X.modeling`. They do not re-export classes. Wiki docstring references and any direct imports must therefore use the full submodule path, for example `unitorch.cli.models.bart.processing.BartProcessor`.

### Required Data Flow Contract For `unitorch-train` And `unitorch-infer`

All CLI models that integrate with `unitorch-train` or `unitorch-infer` must follow the contracts below so processors, models, losses, post-processors, and writers remain composable.

#### Training Flow

```text
raw data (text, PIL image, etc.)
  -> preprocess        -> tensor(s)
  -> collate_fn        -> batched tensor(s)
  -> model forward     -> tensor(s)
  -> loss compute      -> scalar loss
  -> backward / optim update
```

- `preprocess` takes raw data and returns tensor(s)
- `collate_fn` stacks fixed-shape tensors or concatenates variable-length sequences consistently with the preprocess contract
- `model forward` receives batched tensors and returns outputs for the loss
- `loss compute` operates on model outputs and labels and returns a scalar

#### Inference Flow

```text
raw data (text, PIL image, etc.)
  -> preprocess        -> tensor(s)
  -> collate_fn        -> batched tensor(s)
  -> model forward     -> tensor(s)
  -> postprocess       -> raw data (pandas DataFrame format)
  -> writer            -> output file
```

- `preprocess` and `collate_fn` follow the same contract as training
- `model forward` returns tensors for post-processing
- `postprocess` must convert outputs into human-readable form and return a `pandas.DataFrame`
- `writer` consumes the DataFrame and writes configured formats such as `jsonl`, `tsv`, or `parquet`

### Config Section Naming Convention

| Category | Section pattern | Example |
|----------|-----------------|---------|
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
| `bart` | `BartProcessor`, `BartForGeneration` | `core/process/bart`, `core/model/generation/bart` |
| `beit` | `BeitProcessor`, `BeitForImageClassification` | `core/process/beit`, `core/model/classification/beit` |
| `bert` | `BertProcessor`, `BertForClassification` | `core/process/bert`, `core/model/classification/bert` |
| `bria` | `BRIAProcessor`, `BRIAForSegmentation` | `core/process/bria`, `core/model/segmentation/bria` |
| `chinese_clip` | `ChineseClipProcessor`, `ChineseClipForPretrain`, `ChineseClipForClassification`, `ChineseClipForTextClassification`, `ChineseClipForImageClassification` | `core/process/chinese_clip`, `core/model/pretrain\|classification/chinese_clip` |
| `clip` | `ClipProcessor`, `ClipForPretrain`, `ClipForClassification`, `ClipForTextClassification`, `ClipForImageClassification` | `core/process/clip`, `core/model/pretrain\|classification/clip` |
| `detr` | `DetrProcessor`, `DetrForDetection` | `core/process/detr`, `core/model/detection/detr` |
| `diffusers` | `StableFluxProcessor`, `StableFluxForText2Image`, `StableFluxForImage2Image`, `StableFluxForImageRedux`, `StableFluxForImageInpainting`, `StableFluxForKontext2Image`, `WanForText2Video`, `WanForImage2Video`, `QWenImageProcessor`, `QWenImageText2ImageGeneration`, `QWenImageEditingGeneration` | `core/process/diffusion/{stable_flux,wan,qwen_image}`, `core/model/diffusers/{text2image,image2image,image_redux,inpainting,kontext2image}/stable_flux`, `core/model/diffusers/{text2video,image2video}/wan`, `core/model/diffusers/{text2image,editing}/qwen_image` |
| `dinov2` | `DinoV2Processor`, `DinoV2ForImageClassification` | `core/process/dinov2`, `core/model/classification/dinov2` |
| `dpt` | `DPTProcessor`, `DPTForDepthEstimation` | `core/process/dpt`, `core/model/dpt` |
| `grounding_dino` | `GroundingDinoProcessor`, `GroundingDinoForDetection` | `core/process/grounding_dino`, `core/model/detection/grounding_dino` |
| `kolors` | `KolorsMPSProcessor`, `KolorsMPSModel` | `core/process/kolors/mps`, `core/model/classification/kolors/mps` |
| `llama` | `LlamaProcessor`, `LlamaForClassification`, `LlamaForGeneration` | `core/process/llama`, `core/model/classification\|generation/llama` |
| `llava` | `LlavaMistralClipProcessor`, `LlavaLlamaSiglipProcessor`, `LlavaMistralClipForClassification`, `LlavaMistralClipForGeneration`, `LlavaLlamaSiglipForGeneration` | `core/process/llava/{mistral_clip,llama_siglip}`, `core/model/{classification,generation}/llava/{mistral_clip,llama_siglip}` |
| `mask2former` | `Mask2FormerProcessor`, `Mask2FormerForSegmentation` | `core/process/mask2former`, `core/model/segmentation/mask2former` |
| `mbart` | `MBartProcessor`, `MBartForGeneration` | `core/process/mbart`, `core/model/generation/mbart` |
| `mistral` | `MistralProcessor`, `MistralForClassification`, `MistralForGeneration` | `core/process/mistral`, `core/model/classification\|generation/mistral` |
| `peft` | `ClipLoraForMatching`, `ClipLoraForTextMatching`, `LlamaLoraForClassification`, `LlamaLoraForGeneration`, `LlavaMistralClipLoraForClassification`, `LlavaMistralClipLoraForGeneration`, `LlavaLlamaSiglipLoraForGeneration`, `MistralLoraForClassification`, `MistralLoraForGeneration`, `QWen3LoraForGeneration`, `QWen3DPOLoraForGeneration`, `QWen3GRPOLoraForGeneration`, `QWen3VLLoraForGeneration`, `QWen3VLDPOLoraForGeneration` | `core/model/{matching,classification,generation}/peft/lora/...` |
| `qwen` | `QWenProcessor`, `QWenVLProcessor`, `QWen3ForGeneration`, `QWen3VLForGeneration` | `core/process/qwen`, `core/process/qwen_vl`, `core/model/generation/qwen3`, `core/model/generation/qwen3_vl` |
| `roberta` | `RobertaProcessor`, `RobertaForClassification`, `RobertaForMaskLM` | `core/process/roberta`, `core/model/classification/roberta` |
| `sam` | `SamProcessor`, `SamForSegmentation` | `core/process/sam`, `core/model/segmentation/sam` |
| `segformer` | `SegformerProcessor`, `SegformerForSegmentation` | `core/process/segformer`, `core/model/segmentation/segformer` |
| `siglip` | `SiglipProcessor`, `SiglipForPretrain`, `SiglipForClassification`, `SiglipForTextClassification`, `SiglipForImageClassification`, `SiglipForMatching` | `core/process/siglip`, `core/model/pretrain\|classification\|matching/siglip` |
| `swin` | `SwinProcessor`, `SwinForImageClassification` | `core/process/swin`, `core/model/classification/swin` |
| `xlm_roberta` | `XLMRobertaProcessor`, `XLMRobertaForClassification`, `XLMRobertaForMaskLM`, `XLMRobertaXLForClassification` | `core/process/xlm_roberta`, `core/model/classification/xlm_roberta` |

## Manual Integration Pattern

The standard multi-GPU CLI pattern for manual checks is:

```bash
torchrun --no_python --nproc_per_node 4 unitorch-train src/unitorch/cli/configs/generation/bart.ini ...
```

## Commit And Pull Request Guidelines

Recent commits use short, lower-case subjects such as `update qwen` and `clean up code`. Keep commit titles brief, imperative, and scoped to one change. Pull requests should describe the behavior change, list the commands you ran, and link any related issue plus the affected config or example paths. Include screenshots only when documentation or serving output changes.
