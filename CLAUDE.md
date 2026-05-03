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

Defined in `src/unitorch/utils/decorators.py`. Process-global monkey-patcher: replaces a target class across all loaded modules at import time, including rewriting subclass `__bases__`. Used to override upstream library behaviour (e.g. `diffusers`, `datasets`) without forking. Replacement classes inherit from the target and are named `<Original>V2` by convention. See `.claude/skills/replace-decorator.md` for full details.

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
| **blip** | BlipProcessor, BlipForPretrain/Classification/TextClassification/ImageClassification/ImageCaption | `core/process/blip`, `core/model/pretrain\|classification\|caption/blip` |
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
| **pegasus** | PegasusProcessor, PegasusForGeneration | `core/process/pegasus`, `core/model/generation/pegasus` |
| **qwen** | QWenProcessor, QWenVLProcessor, QWen3ForGeneration, QWen3VLForGeneration | `core/process/qwen`, `core/process/qwen_vl`, `core/model/generation/qwen3`, `core/model/generation/qwen3_vl` |
| **roberta** | RobertaProcessor, RobertaForClassification/MaskLM | `core/process/roberta`, `core/model/classification/roberta` |
| **sam** | SamProcessor, SamForSegmentation | `core/process/sam`, `core/model/segmentation/sam` |
| **segformer** | SegformerProcessor, SegformerForSegmentation | `core/process/segformer`, `core/model/segmentation/segformer` |
| **siglip** | SiglipProcessor, SiglipForPretrain/Classification/TextClassification/ImageClassification/Matching | `core/process/siglip`, `core/model/pretrain\|classification\|matching/siglip` |
| **swin** | SwinProcessor, SwinForImageClassification | `core/process/swin`, `core/model/classification/swin` |
| **t5** | T5Processor, T5ForGeneration | `core/process/t5`, `core/model/generation/t5` |
| **visualbert** | VisualBertProcessor, VisualBertForClassification/Pretrain | `core/process/visualbert`, `core/model/classification\|pretrain/visualbert` |
| **vit** | ViTProcessor, ViTForImageClassification | `core/process/vit`, `core/model/classification/vit` |
| **xlm_roberta** | XLMRobertaProcessor, XLMRobertaForClassification/MaskLM, XLMRobertaXLForClassification | `core/process/xlm_roberta`, `core/model/classification/xlm_roberta` |
| **xpegasus** | XPegasusProcessor, XPegasusForGeneration | `core/process/xpegasus`, `core/model/generation/xpegasus` |
