# Lucy Real Checkpoint E2E Test Report

## Scope

- Repository: `/home/decu/my/unitorch`
- Branch: `test/lucy-real-ckpt-e2e`
- Date: `2026-06-05`
- Real checkpoint: `decart-ai/Lucy-Edit-1.1-Dev`
- Cache: `UNITORCH_CACHE=/data/decu/.cache`
- Large runtime artifacts/logs: `/data/decu/lucy_test`

This run covered:

- real checkpoint load verification with `UNITORCH_DEBUG=DETAIL`
- real GPU inference for Lucy video editing
- FastAPI startup and real `generate` API test with MP4 upload
- real training smoke for Lucy end-to-end training
- real training smoke for Lucy LoRA training
- repository fixes required to make the above pass

## Environment

- GPU: `2 x NVIDIA RTX A6000`
- Python: editable install from current repo via `python3 -m pip install -e .`
- PyTorch: `2.10.0`
- diffusers: `0.38.0`
- FastAPI: `0.124.4`
- DeepSpeed: `0.17.0`

## Runtime Data And Artifact Layout

- Runtime configs: `/data/decu/lucy_test/configs`
- Runtime dataset/videos: `/data/decu/lucy_test/datasets/lucy_smoke`
- Runtime logs: `/data/decu/lucy_test/logs`
- Runtime outputs: `/data/decu/lucy_test/outputs`

Key files:

- checkpoint load debug logs:
  - `/data/decu/lucy_test/logs/ckpt_load_detail.log`
  - `/data/decu/lucy_test/logs/ckpt_load_detail_after_fix2.log`
- inference log:
  - `/data/decu/lucy_test/logs/infer_cli_cfg.log`
- FastAPI request log:
  - `/data/decu/lucy_test/logs/fastapi_generate_test.log`
- FastAPI server log:
  - `/tmp/unitorch_fastapi_services1_bbb0eb1b@df28f0.stdout.log`
- training logs:
  - `/data/decu/lucy_test/logs/train_e2e.log`
  - `/data/decu/lucy_test/logs/train_e2e_offload.log`
  - `/data/decu/lucy_test/logs/train_lora.log`

## Issues Found And Fixes

### 1. Lucy text weights were not fully loaded from the real checkpoint

Symptom:

- Lucy real checkpoint load only reached `99%`.
- `text.encoder.embed_tokens.weight` was missed.

Root cause:

- Lucy loader used generic `load_weight(...)` for the text encoder shards.
- Lucy text encoder layout matches the Wan/T5-compatible loading path instead.

Fix:

- `src/unitorch/cli/models/diffusers/modeling_lucy.py`
  - switched Lucy text weight loading to `load_wan_text_weight(...)`
- `src/unitorch/cli/models/diffusers/__init__.py`
  - fixed `load_wan_text_weight(...)` to accept/pass `use_auth_token`

Verification:

- `/data/decu/lucy_test/logs/ckpt_load_detail_after_fix2.log`
- real checkpoint load reached `100%`

### 2. CLI exit left distributed process groups alive

Symptom:

- DeepSpeed-backed runs could leave NCCL process groups alive and hit shutdown instability.

Fix:

- added explicit distributed teardown in:
  - `src/unitorch/cli/consoles/infer.py`
  - `src/unitorch/cli/consoles/train.py`
  - `src/unitorch/cli/consoles/eval.py`

Verification:

- inference and both training runs completed without the earlier shutdown crash path

### 3. Lucy LoRA training path was missing

Symptom:

- repo had Lucy inference/training support but no Lucy PEFT LoRA model registration path

Fix:

- added Lucy LoRA model implementations:
  - `src/unitorch/models/peft/diffusers/modeling_lucy.py`
  - `src/unitorch/cli/models/peft/diffusers/modeling_lucy.py`
- wired imports/registration:
  - `src/unitorch/models/peft/__init__.py`
  - `src/unitorch/models/peft/diffusers/__init__.py`
  - `src/unitorch/cli/models/peft/diffusers/__init__.py`
- added example config:
  - `examples/configs/diffusion/editing/lucy.lora.ini`
- updated docs and test coverage

Verification:

- `python3 -m pytest tests/cli/test_lucy_registration.py -q`
- real LoRA training run succeeded with the real Lucy checkpoint

## Real Checkpoint Load Verification

Command pattern used:

```bash
UNITORCH_CACHE=/data/decu/.cache UNITORCH_DEBUG=DETAIL ...
```

Result:

- before fix: Lucy text load was incomplete
- after fix: real checkpoint load reached `100%`

Evidence:

- `/data/decu/lucy_test/logs/ckpt_load_detail.log`
- `/data/decu/lucy_test/logs/ckpt_load_detail_after_fix2.log`

## Inference

Command:

```bash
UNITORCH_CACHE=/data/decu/.cache CUDA_VISIBLE_DEVICES=0 \
unitorch-infer /data/decu/lucy_test/configs/lucy_infer_smoke.ini
```

Status: `PASS`

Notes:

- used installed console script `unitorch-infer`
- `python3 -m unitorch.cli.consoles.infer` is not a valid substitute here because the module does not call `cli_main()` under `__main__`

Artifacts:

- output CSV:
  - `/data/decu/lucy_test/outputs/infer_cli_cfg/result.csv`
- output video:
  - `/data/decu/lucy_test/outputs/infer_cli_cfg/33b0d37c611c679135ad78512414a6d5.mp4`
- log:
  - `/data/decu/lucy_test/logs/infer_cli_cfg.log`

Output video probe:

- resolution: `256x256`
- fps: `8`
- frames: `17`
- duration: `2.125s`

## FastAPI

Start command:

```bash
UNITORCH_CACHE=/data/decu/.cache \
unitorch-fastapi start /data/decu/lucy_test/configs/lucy_fastapi_smoke.ini --daemon_mode=False
```

Config notes:

- service: `core/fastapi/lucy/video_editing`
- host: `127.0.0.1`
- port: `8013`
- port `8001` was occupied, so smoke config was moved to `8013`

API flow executed:

1. `GET /health-check`
2. `GET /core/fastapi/lucy/video_editing/status`
3. `POST /core/fastapi/lucy/video_editing/start`
4. `POST /core/fastapi/lucy/video_editing/generate`
5. `GET /core/fastapi/lucy/video_editing/stop`

Generate request:

- text prompt: `Turn the scene into a bright cinematic daytime drive with warmer colors and soft sunlight.`
- negative prompt: `low quality, blurry, artifacts`
- uploaded real MP4: `/data/decu/lucy_test/datasets/lucy_smoke/videos/refer_test.mp4`
- generation params:
  - `height=256`
  - `width=256`
  - `num_frames=17`
  - `num_fps=8`
  - `guidance_scale=4.0`
  - `num_timesteps=4`
  - `seed=2026`

Status: `PASS`

Artifacts:

- generated video:
  - `/data/decu/lucy_test/outputs/fastapi_generate.mp4`
- request log:
  - `/data/decu/lucy_test/logs/fastapi_generate_test.log`
- server log:
  - `/tmp/unitorch_fastapi_services1_bbb0eb1b@df28f0.stdout.log`

Output video probe:

- resolution: `256x256`
- fps: `8`
- frames: `17`
- duration: `2.125s`

## Training: End-to-End

### Default smoke config

Command:

```bash
UNITORCH_CACHE=/data/decu/.cache CUDA_VISIBLE_DEVICES=0 \
unitorch-train /data/decu/lucy_test/configs/lucy_train_smoke.ini
```

Status: `FAIL`

Failure:

- DeepSpeed ZeRO stage 2 optimizer init OOM
- error from log:
  - `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 18.63 GiB.`

Failure log:

- `/data/decu/lucy_test/logs/train_e2e.log`

### CPU-offload smoke config used to complete real e2e training

Runtime config added outside git:

- `/data/decu/lucy_test/configs/deepspeed_adamw_bf16_stage2_cpu_offload.json`
- `/data/decu/lucy_test/configs/lucy_train_smoke_e2e_offload.ini`

Change from default smoke:

- kept real Lucy checkpoint and real dataset path
- kept `256x256`, `17` frames, `4` infer timesteps
- changed DeepSpeed config to ZeRO stage 2 with CPU optimizer offload

Command:

```bash
UNITORCH_CACHE=/data/decu/.cache CUDA_VISIBLE_DEVICES=0 \
unitorch-train /data/decu/lucy_test/configs/lucy_train_smoke_e2e_offload.ini
```

Status: `PASS`

Observed training milestones:

- optimizer states initialized successfully with CPU offload
- `epoch 0 step 0: train/loss=1.666654`
- `epoch 0 step 1: train/loss=2.220193`
- checkpoint saved repeatedly during the run

Artifacts:

- checkpoint:
  - `/data/decu/lucy_test/outputs/train_e2e_offload/ckpt/pytorch_model_latest.bin`
- checkpoint info:
  - `/data/decu/lucy_test/outputs/train_e2e_offload/ckpt/info.json`
- log:
  - `/data/decu/lucy_test/logs/train_e2e_offload.log`

## Training: LoRA

Command:

```bash
UNITORCH_CACHE=/data/decu/.cache CUDA_VISIBLE_DEVICES=0 \
unitorch-train /data/decu/lucy_test/configs/lucy_train_smoke_lora.ini
```

Status: `PASS`

Config notes:

- real Lucy checkpoint
- `lora_r = 4`
- `enable_text_adapter = False`
- same real dataset and same smoke generation shape as the e2e run

Observed training milestones:

- `epoch 0 step 0: train/loss=1.667019`
- `epoch 0 step 1: train/loss=2.279802`
- LoRA checkpoint save succeeded

Artifacts:

- checkpoint:
  - `/data/decu/lucy_test/outputs/train_lora/ckpt/pytorch_model_latest.bin`
- checkpoint info:
  - `/data/decu/lucy_test/outputs/train_lora/ckpt/info.json`
- log:
  - `/data/decu/lucy_test/logs/train_lora.log`

## Runtime Smoke Inputs

Dataset and videos used for inference/FastAPI/training:

- `/data/decu/lucy_test/datasets/lucy_smoke/train.tsv`
- `/data/decu/lucy_test/datasets/lucy_smoke/dev.tsv`
- `/data/decu/lucy_test/datasets/lucy_smoke/test.tsv`
- `/data/decu/lucy_test/datasets/lucy_smoke/videos/refer_a.mp4`
- `/data/decu/lucy_test/datasets/lucy_smoke/videos/refer_b.mp4`
- `/data/decu/lucy_test/datasets/lucy_smoke/videos/refer_dev.mp4`
- `/data/decu/lucy_test/datasets/lucy_smoke/videos/refer_test.mp4`
- `/data/decu/lucy_test/datasets/lucy_smoke/videos/target_a.mp4`
- `/data/decu/lucy_test/datasets/lucy_smoke/videos/target_b.mp4`
- `/data/decu/lucy_test/datasets/lucy_smoke/videos/target_dev.mp4`
- `/data/decu/lucy_test/datasets/lucy_smoke/videos/target_test.mp4`

## Repo-Level Validation

Commands:

```bash
git diff --check
python3 -m pytest tests/cli/test_lucy_registration.py -q
python3 -m compileall \
  src/unitorch/cli/consoles \
  src/unitorch/cli/models/diffusers/modeling_lucy.py \
  src/unitorch/cli/models/peft/diffusers/modeling_lucy.py \
  src/unitorch/models/diffusers/modeling_lucy.py \
  src/unitorch/models/peft/diffusers/modeling_lucy.py
```

Status:

- `git diff --check`: `PASS`
- `pytest tests/cli/test_lucy_registration.py -q`: `PASS` (`7 passed`)
- `compileall`: `PASS`

## Repo Files Changed

- `src/unitorch/cli/models/diffusers/modeling_lucy.py`
- `src/unitorch/cli/models/diffusers/__init__.py`
- `src/unitorch/cli/consoles/infer.py`
- `src/unitorch/cli/consoles/train.py`
- `src/unitorch/cli/consoles/eval.py`
- `src/unitorch/models/peft/diffusers/modeling_lucy.py`
- `src/unitorch/cli/models/peft/diffusers/modeling_lucy.py`
- `src/unitorch/models/peft/__init__.py`
- `src/unitorch/models/peft/diffusers/__init__.py`
- `src/unitorch/cli/models/peft/diffusers/__init__.py`
- `examples/configs/diffusion/editing/lucy.lora.ini`
- `tests/cli/test_lucy_registration.py`
- `README.md`
- `wiki/models/diffusers.md`
- `wiki/cli/models/diffusers.md`
- `wiki/models/peft.md`
- `wiki/cli/models/peft.md`

## Overall Result

- checkpoint load verification: `PASS`
- inference: `PASS`
- FastAPI generate: `PASS`
- training e2e with default stage-2 config: `FAIL (OOM)`
- training e2e with CPU-offload smoke config: `PASS`
- training LoRA: `PASS`
- targeted repo validation: `PASS`

The required Lucy real-checkpoint path is now exercised across inference, FastAPI generate, end-to-end training, and LoRA training. The remaining caveat is that full-model single-GPU DeepSpeed stage-2 training with the default smoke config OOMs on an RTX A6000, so the successful e2e validation used a CPU-offload DeepSpeed config that is recorded above.
