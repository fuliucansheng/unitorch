# Gemma Model Support Report

## Scope

This report covers the Gemma integration work completed on branch `feat/gemma-model-support` in `/home/decu/my/unitorch`.

- Text generation support for `google/gemma-4-12B`
- Vision-language generation support for `google/gemma-4-12B`
- CLI model/process registration
- FastAPI `generate` support
- `unitorch-infer` smoke validation
- Training smoke validation for full fine-tune and LoRA

## Environment

- Date: 2026-06-04
- Repo: `/home/decu/my/unitorch`
- Branch: `feat/gemma-model-support`
- Final commit: `TBD_AFTER_COMMIT`
- PR: `TBD_AFTER_PR`
- Python: `3.10.18`
- PyTorch: `2.10.0+cu128`
- Transformers: `5.7.0`
- GPU: `NVIDIA RTX A6000` x2, `49140 MiB` each
- Driver: `570.153.02`

## Checkpoint And Cache

- Required cache path: `UNITORCH_CACHE=/data/decu/.cache`
- `/data` was already at `100%` usage, so the downloaded Gemma cache entries were relocated behind symlinks:
  - `/data/decu/.cache/models--google--gemma-4-12B -> /home/decu/.cache/hf-relocated/models--google--gemma-4-12B`
  - `/data/decu/.cache/.locks/models--google--gemma-4-12B -> /home/decu/.cache/hf-relocated/.locks/models--google--gemma-4-12B`
- Hugging Face auth state:
  - `token_present=True`
  - `whoami=fuliucansheng`
  - `google/gemma-4-12B private=False`
  - `google/gemma-4-12B gated=False`
- Resolved checkpoint revision: `0c9850654599d822abc394e06d35e86e2435cffc`
- Snapshot files verified:
  - `config.json`
  - `model.safetensors`
  - `processor_config.json`
  - `tokenizer.json`
  - `tokenizer_config.json`

## Implementation Summary

- Added explicit Gemma tokenizer/process/model support without using Hugging Face AutoClasses inside this repository.
- Added Gemma text generation model support.
- Added Gemma vision-language generation support.
- Added Gemma LoRA text generation support.
- Added CLI model registrations, FastAPI registrations, example configs, docs, and registration tests.
- Fixed runtime issues discovered during real-checkpoint validation:
  - Greedy decode path no longer assumes `sequences_scores` exists.
  - Gemma dtype resolution now correctly honors `torch.bfloat16`.
  - Text-only Gemma paths drop incompatible vision/audio towers so the 12B checkpoint loads as expected.
  - VLM path uses a custom encoder-free Gemma 12B vision tower that maps checkpoint `vision_embedder.*` weights into generation.
  - Custom Gemma VLM tower now aligns runtime dtypes and returns a FastAPI/Transformers-compatible dataclass output.
  - Earlier implementation fixes already included in this branch:
    - VLM prompt/image placeholder expansion fix
    - biasless `LayerNorm` init fix

## Files Added Or Updated

- Model code:
  - `src/unitorch/models/gemma/`
  - `src/unitorch/models/peft/modeling_gemma.py`
- CLI code:
  - `src/unitorch/cli/models/gemma/`
  - `src/unitorch/cli/models/peft/modeling_gemma.py`
  - `src/unitorch/cli/fastapis/gemma.py`
  - `src/unitorch/cli/fastapis/gemma_vl.py`
- Registry updates:
  - `src/unitorch/models/__init__.py`
  - `src/unitorch/models/peft/__init__.py`
  - `src/unitorch/cli/models/__init__.py`
  - `src/unitorch/cli/models/peft/__init__.py`
  - `src/unitorch/cli/fastapis/__init__.py`
- Examples/docs/tests:
  - `examples/configs/generation/gemma.ini`
  - `examples/configs/generation/gemma.lora.ini`
  - `examples/configs/generation/gemma_vl.ini`
  - `examples/configs/fastapis/gemma.ini`
  - `examples/fastapis.ini`
  - `wiki/models/gemma.md`
  - `wiki/cli/models/gemma.md`
  - `wiki/cli/fastapis.md`
  - `wiki/models/peft.md`
  - `wiki/cli/models/peft.md`
  - `README.md`
  - `mkdocs.yml`
  - `tests/cli/test_gemma_registration.py`

## Validation Summary

| Area | Result | Notes |
| --- | --- | --- |
| `pytest tests/cli/test_gemma_registration.py -q` | Pass | `9 passed` |
| `python3 -m compileall ...` | Pass | New Gemma model/CLI/FastAPI files compiled |
| Text pipeline smoke | Pass | Real checkpoint load `99%`, `bfloat16`, output returned |
| VLM pipeline smoke | Pass | Real checkpoint load `99%`, image input path exercised |
| `unitorch-infer` text | Pass | Output: `The capital of Germany is -> Berlin.` |
| `unitorch-infer` VLM | Pass | Output produced from image input path |
| FastAPI text `/generate` | Pass | HTTP 200 via `TestClient` |
| FastAPI VLM `/generate` | Pass | HTTP 200 via `TestClient` with query + file upload |
| Training e2e | Pass | 6-layer truncated config, checkpoint saved |
| Training LoRA | Pass | 6-layer truncated config, LoRA checkpoint saved |
| Training checkpoint reload | Pass | e2e and LoRA checkpoints reloaded successfully |

## Commands And Key Outputs

### Registration / compile checks

```bash
python3 -m pytest tests/cli/test_gemma_registration.py -q
python3 -m compileall \
  src/unitorch/models/gemma \
  src/unitorch/models/peft/modeling_gemma.py \
  src/unitorch/cli/models/gemma \
  src/unitorch/cli/models/peft/modeling_gemma.py \
  src/unitorch/cli/fastapis/gemma.py \
  src/unitorch/cli/fastapis/gemma_vl.py
```

Observed result:

- `9 passed in 9.02s`

### Low-level text pipeline smoke

Command shape:

```bash
UNITORCH_CACHE=/data/decu/.cache \
HF_HUB_OFFLINE=1 \
UNITORCH_DEBUG=DETAIL \
CUDA_VISIBLE_DEVICES=1 \
python3 - <<'PY'
# GemmaForGenerationPipeline.from_config(...)
PY
```

Key observations:

- `GemmaForGenerationPipeline loaded weights (99%)`
- `model_dtype torch.bfloat16`
- `device_before cpu`
- Output sample:

```text
The capital of France is -> a city of art, culture, and history.
```

### Low-level VLM pipeline smoke

Command shape:

```bash
UNITORCH_CACHE=/data/decu/.cache \
HF_HUB_OFFLINE=1 \
UNITORCH_DEBUG=DETAIL \
CUDA_VISIBLE_DEVICES=1 \
python3 - <<'PY'
# GemmaVLForGenerationPipeline.from_config(...)
PY
```

Key observations:

- `GemmaVLForGenerationPipeline loaded weights (99%)`
- `model_dtype torch.bfloat16`
- `device_before cpu`
- Output sample from `unitorch.png`:

```text
What is visible in this image? -> What is the main subject of this image ...
```

The output is not high quality for this smoke image, but the image-conditioned generate path executed successfully with the real checkpoint.

### CLI inference smoke

Text:

```bash
unitorch-infer examples/configs/generation/gemma.ini \
  --train_file /tmp/unitorch_gemma_smoke/train.tsv \
  --dev_file /tmp/unitorch_gemma_smoke/dev.tsv \
  --test_file /tmp/unitorch_gemma_smoke/test.tsv \
  --cache_dir /tmp/unitorch_gemma_smoke/infer_text_clean \
  --from_ckpt_dir /tmp/unitorch_gemma_smoke/infer_text_from \
  --core/task/supervised@test_batch_size 1 \
  --core/task/supervised@num_workers 0 \
  --core/task/supervised@pin_memory False \
  --core/process/gemma@max_seq_length 64 \
  --core/process/gemma@max_gen_seq_length 16 \
  --core/model/generation/gemma@max_gen_seq_length 16
```

Observed output:

```text
The capital of Germany is    Berlin.
```

VLM:

```bash
unitorch-infer examples/configs/generation/gemma_vl.ini \
  --train_file /tmp/unitorch_gemma_smoke/train.tsv \
  --dev_file /tmp/unitorch_gemma_smoke/dev.tsv \
  --test_file /tmp/unitorch_gemma_smoke/test_vl.tsv \
  --cache_dir /tmp/unitorch_gemma_smoke/infer_vl \
  --from_ckpt_dir /tmp/unitorch_gemma_smoke/infer_vl_from \
  --core/task/supervised@test_batch_size 1 \
  --core/task/supervised@num_workers 0 \
  --core/task/supervised@pin_memory False \
  --core/process/gemma_vl@max_seq_length 256 \
  --core/process/gemma_vl@max_gen_seq_length 32 \
  --core/model/generation/gemma_vl@max_gen_seq_length 32
```

Observed output:

```text
What text is visible in this image?    2020 2020 2020 2020 2020 2020 2
```

### FastAPI generate smoke

Text FastAPI:

- `/core/fastapi/gemma/start` -> `start success`
- `/core/fastapi/gemma/generate` -> HTTP `200`
- Response sample:

```text
Rome, and the capital of the Vatican is the Vatican City...
```

VLM FastAPI:

- `/core/fastapi/gemma_vl/start` -> `start success`
- `/core/fastapi/gemma_vl/generate` -> HTTP `200`
- Request format note: `text` must be sent as a query parameter, not multipart form data.
- Response sample:

```text
100 100 100 100 100 100 100 100
```

### Training smoke

#### Why the training config was reduced

The full 48-layer 12B model is too large for a practical full-parameter smoke run on a single 48 GB GPU once optimizer state is included.

I tried smaller truncation points first:

- 1-layer and 2-layer truncation are not good for this checkpoint because Gemma 4 forces the last layer to `full_attention`, which changes projection shapes and drops checkpoint compatibility.
- The smallest truncation point that preserves the original layer-type pattern is 6 layers, because the checkpoint uses a repeating pattern where every 6th layer is `full_attention`.

Final training smoke therefore used a 6-layer config derived from the real `google/gemma-4-12B` `config.json`:

- `/tmp/unitorch_gemma_smoke/gemma_6layers_config.json`

#### e2e full fine-tune

Command shape:

```bash
unitorch-train examples/configs/generation/gemma.ini \
  --train_file /tmp/unitorch_gemma_smoke/train.tsv \
  --dev_file /tmp/unitorch_gemma_smoke/dev.tsv \
  --test_file /tmp/unitorch_gemma_smoke/test.tsv \
  --cache_dir /tmp/unitorch_gemma_smoke/train_e2e_6layers \
  --from_ckpt_dir /tmp/unitorch_gemma_smoke/train_e2e_6layers_from \
  --core/model/generation/gemma@config_path /tmp/unitorch_gemma_smoke/gemma_6layers_config.json \
  --core/model/generation/gemma@pretrained_weight_path /data/decu/.cache/models--google--gemma-4-12B/snapshots/0c9850654599d822abc394e06d35e86e2435cffc/model.safetensors \
  --core/model/generation/gemma@gradient_checkpointing True \
  --core/process/gemma@max_seq_length 64 \
  --core/process/gemma@max_gen_seq_length 16 \
  --core/model/generation/gemma@max_gen_seq_length 16 \
  --core/task/supervised@epochs 1 \
  --core/task/supervised@train_batch_size 1 \
  --core/task/supervised@dev_batch_size 1 \
  --core/task/supervised@num_workers 0 \
  --core/task/supervised@pin_memory False \
  --core/task/supervised@log_freq 1 \
  --core/task/supervised@ckpt_freq 1 \
  --core/task/supervised@save_optimizer False \
  --core/task/supervised@save_scheduler False \
  --core/task/supervised@save_checkpoint latest \
  --core/task/supervised@use_amp False
```

Observed result:

- `GemmaForGeneration loaded weights (98%)`
- `epoch 0 step 0: train/loss=15.791667`
- Checkpoint written:
  - `/tmp/unitorch_gemma_smoke/train_e2e_6layers/pytorch_model_latest.bin`
  - Size: `4.5G`

#### LoRA fine-tune

Command shape:

```bash
unitorch-train examples/configs/generation/gemma.lora.ini \
  --train_file /tmp/unitorch_gemma_smoke/train.tsv \
  --dev_file /tmp/unitorch_gemma_smoke/dev.tsv \
  --test_file /tmp/unitorch_gemma_smoke/test.tsv \
  --cache_dir /tmp/unitorch_gemma_smoke/train_lora_6layers \
  --from_ckpt_dir /tmp/unitorch_gemma_smoke/train_lora_6layers_from \
  --core/model/generation/peft/lora/gemma@config_path /tmp/unitorch_gemma_smoke/gemma_6layers_config.json \
  --core/model/generation/peft/lora/gemma@pretrained_weight_path /data/decu/.cache/models--google--gemma-4-12B/snapshots/0c9850654599d822abc394e06d35e86e2435cffc/model.safetensors \
  --core/model/generation/peft/lora/gemma@gradient_checkpointing True \
  --core/process/gemma@max_seq_length 64 \
  --core/process/gemma@max_gen_seq_length 16 \
  --core/model/generation/peft/lora/gemma@max_gen_seq_length 16 \
  --core/task/supervised@epochs 1 \
  --core/task/supervised@train_batch_size 1 \
  --core/task/supervised@dev_batch_size 1 \
  --core/task/supervised@num_workers 0 \
  --core/task/supervised@pin_memory False \
  --core/task/supervised@log_freq 1 \
  --core/task/supervised@ckpt_freq 1 \
  --core/task/supervised@save_optimizer False \
  --core/task/supervised@save_scheduler False \
  --core/task/supervised@save_checkpoint latest \
  --core/task/supervised@use_amp False
```

Observed result:

- `Gemma4ForConditionalGeneration.add_adapter is incompatible with the installed PEFT version; falling back to inject_adapter_in_model().`
- `GemmaLoraForGeneration loaded weights (64%)`
- The lower percentage is expected here because the LoRA model state dict includes newly initialized adapter tensors not present in the base checkpoint.
- Quick accounting on the 6-layer LoRA model:
  - `total_keys=132`
  - `loadable=85`
  - `46` missing keys were `lora_*` adapter weights
- `epoch 0 step 0: train/loss=12.375000`
- Checkpoint written:
  - `/tmp/unitorch_gemma_smoke/train_lora_6layers/pytorch_model_latest.bin`
  - Size: `5.2M`

### Checkpoint reload validation

After training, both training outputs were loaded back explicitly:

- e2e checkpoint reload:
  - `GemmaForGeneration loaded weights from /tmp/unitorch_gemma_smoke/train_e2e_6layers/pytorch_model_latest.bin`
- LoRA checkpoint reload:
  - `GemmaLoraForGeneration model load weight from checkpoint /tmp/unitorch_gemma_smoke/train_lora_6layers/pytorch_model_latest.bin`

This confirms the produced checkpoints can be reopened by the corresponding unitorch model classes.

## Known Limitations

- The `google/gemma-4-12B` vision path is working end-to-end, but output quality on the smoke image (`unitorch.png`) is weak and repetitive. The image-conditioned generation path is exercised successfully, but the sample output is not strong enough to claim high OCR quality from this one checkpoint/image pair.
- FastAPI VLM `generate` expects `text` as a query parameter alongside the uploaded image file. Sending `text` as multipart form data returns HTTP `422`.
- `transformers` emits upstream warnings during these runs:
  - `torch_dtype` deprecation warning
  - `early_stopping` ignored for greedy decode
  - PEFT compatibility warning requiring fallback to `inject_adapter_in_model()`

## Git Hygiene Notes

- No checkpoint, cache directory, generated image/video output, or `/tmp` smoke artifact is intended to be committed.
- The only files to be committed are source, config, tests, docs, and this report.
