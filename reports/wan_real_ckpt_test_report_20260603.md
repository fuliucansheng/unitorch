# Wan Real Checkpoint Test Report (2026-06-03)

## Scope

This report covers real-checkpoint validation of Wan in `unitorch` on the GPU host for:

- text-to-video inference
- image-to-video inference
- FastAPI generate flow for text-to-video and image-to-video
- e2e/full training path
- LoRA training path

Required checkpoint:

- `Wan-AI/Wan2.2-TI2V-5B-Diffusers`

Persistent cache:

- `UNITORCH_CACHE=/data/decu/.cache`

Implementation branch:

- `test/wan-real-ckpt-0.0.2.1`

Implementation commit:

- `95e59a8` (`fix wan ti2v 5b`)

Base branch at start of work:

- `origin/master` = `f74391d7d3c7969c69276afafb4c49368ea98704`

## Environment

| Item | Value |
|---|---|
| Host | `br1t43-s3-17` |
| Repo | `/home/decu/my/unitorch` |
| Date | `2026-06-03 06:18:34 UTC` |
| Python | `3.10.18` |
| PyTorch | `2.10.0+cu128` |
| Diffusers | `0.38.0` |
| Transformers | `5.7.0` |
| PEFT | `0.18.1` |
| DeepSpeed | `0.17.0` |
| GPU | `NVIDIA RTX A6000` x2 |
| Driver | `570.153.02` |
| CUDA runtime | `12.8` |
| Test GPU | `CUDA_VISIBLE_DEVICES=1` |

Common test env:

```bash
export UNITORCH_CACHE=/data/decu/.cache
export UNITORCH_DEBUG=DETAIL
export PYTHONPATH=/home/decu/my/unitorch/src
export CUDA_VISIBLE_DEVICES=1
```

## Key Answers

- Verified `e2e/full training was T2V`, not I2V.
- Evidence: both e2e training commands used `examples/configs/diffusion/text2video/wan.ini` with `wan_t2v_train.noheader.tsv` and `wan_t2v_dev.noheader.tsv`.
- Evidence: `/data/decu/wan_real_ckpt/runs/20260603/logs/train_e2e_t2v.log` and `/data/decu/wan_real_ckpt/runs/20260603/logs/train_e2e_t2v_2l_z3.log` show `WanForText2VideoGeneration loaded weights (100%)`; the fallback log also reached `epoch 0 step 0`.

| Output | Path | Size |
|---|---|---:|
| T2V inference video path | `/data/decu/wan_real_ckpt/runs/20260603/infer_t2v/videos/0a85d18f8e9baa891f0be4ca350b8de6.mp4` | 44813 bytes |
| I2V inference video path | `/data/decu/wan_real_ckpt/runs/20260603/infer_i2v/videos/250c7591f36d2a309483be1015cb2af3.mp4` | 90018 bytes |
| FastAPI T2V output path | `/data/decu/wan_real_ckpt/runs/20260603/fastapi/t2v_generate.mp4` | 31973 bytes |
| FastAPI I2V output path | `/data/decu/wan_real_ckpt/runs/20260603/fastapi/i2v_generate.mp4` | 23489 bytes |

## Summary Matrix

| Area | Result | Notes |
|---|---|---|
| Wan text-to-video inference | PASS | Real 5B TI2V checkpoint loaded at 100% |
| Wan image-to-video inference | PASS | Fixed `expand_timesteps` and image path handling |
| FastAPI text-to-video generate | PASS | `/core/fastapi/wan/text2video/generate` returned MP4 |
| FastAPI image-to-video generate | PASS | `/core/fastapi/wan/image2video/generate` returned MP4 |
| e2e/full training default config (T2V) | FAIL | OOM during optimizer init with full 5B T2V train path |
| e2e/full training fallback (T2V) | PASS | 2-layer config + ZeRO-3 CPU offload + 1 train step + checkpoint |
| LoRA I2V initial attempt | FAIL | `transformers` PEFT version gate blocked `add_adapter()` |
| LoRA I2V second attempt | FAIL | UMT5 text target modules were wrong for fallback injection |
| LoRA I2V final attempt | PASS | 2-layer config + correct UMT5 targets + 1 train step + LoRA checkpoint |
| LoRA T2V follow-up smoke | PASS | 2-layer config + real TI2V 5B ckpt + `epoch 0 step 0` + LoRA checkpoint |

## Checkpoint Load Verification

Key log confirmations with `UNITORCH_DEBUG=DETAIL`:

- `/data/decu/wan_real_ckpt/runs/20260603/logs/infer_t2v.log`
  - `WanForText2VideoGeneration loaded weights (100%)`
- `/data/decu/wan_real_ckpt/runs/20260603/logs/infer_i2v.log`
  - `WanForImage2VideoGeneration loaded weights (100%)`
- `/data/decu/wan_real_ckpt/runs/20260603/logs/fastapi_wan_server.log`
  - `WanForText2VideoFastAPIPipeline loaded weights (100%)`
  - `WanForImage2VideoFastAPIPipeline loaded weights (100%)`
- `/data/decu/wan_real_ckpt/runs/20260603/logs/train_e2e_t2v_2l_z3.log`
  - `WanForText2VideoGeneration loaded weights (100%)`
- `/data/decu/wan_real_ckpt/runs/20260603/logs/train_lora_i2v_2l_after_fix2.log`
  - `WanLoraForImage2VideoGeneration loaded weights (56%)`
- `/data/decu/wan_real_ckpt/runs/20260603/logs/train_lora_t2v_2l_followup.log`
  - `WanLoraForText2VideoGeneration loaded weights (56%)`
  - `epoch 0 step 0: train/loss=5.662803`

Notes:

- `100%` load is expected for inference/FastAPI and the reduced-layer e2e fallback because all parameters present in the reduced 2-layer model were found in the real 5B checkpoint.
- `56%` load for LoRA training is expected for both I2V and T2V reduced-layer smoke runs because:
  - the base Wan model was reduced to 2 transformer layers for the training smoke
  - LoRA adapter parameters are newly initialized and therefore are not present in the base checkpoint

## Exact Commands

### Text-to-video inference

Successful smoke used the real 5B TI2V checkpoint and produced `/data/decu/wan_real_ckpt/runs/20260603/infer_t2v/output.tsv` and `0a85d18f8e9baa891f0be4ca350b8de6.mp4`.

```bash
env UNITORCH_CACHE=/data/decu/.cache UNITORCH_DEBUG=DETAIL \
  PYTHONPATH=/home/decu/my/unitorch/src CUDA_VISIBLE_DEVICES=1 \
  unitorch-infer examples/configs/diffusion/text2video/wan.ini \
  --core/cli@from_ckpt_dir /data/decu/wan_real_ckpt/runs/20260603/empty_ckpt \
  --core/cli@cache_dir /data/decu/wan_real_ckpt/runs/20260603/infer_t2v/cache \
  --core/cli@output_folder /data/decu/wan_real_ckpt/runs/20260603/infer_t2v/videos \
  --test_file /data/decu/wan_real_ckpt/data/wan_t2v_test.noheader.tsv \
  --core/task/deepspeed/supervised@output_path /data/decu/wan_real_ckpt/runs/20260603/infer_t2v/output.tsv \
  --core/task/deepspeed/supervised@test_batch_size 1 \
  --core/task/deepspeed/supervised@num_workers 0 \
  --core/task/deepspeed/supervised@postprocess_workers 1 \
  --core/model/diffusers/text2video/wan@num_infer_timesteps 4 \
  --core/model/diffusers/text2video/wan@height 192 \
  --core/model/diffusers/text2video/wan@width 320 \
  --core/model/diffusers/text2video/wan@num_frames 9 \
  --core/model/diffusers/text2video/wan@guidance_scale 4.5 \
  --core/process/diffusion/wan@max_seq_length 128
```

### Image-to-video inference

Successful smoke after the I2V fixes:

```bash
env UNITORCH_CACHE=/data/decu/.cache UNITORCH_DEBUG=DETAIL \
  PYTHONPATH=/home/decu/my/unitorch/src CUDA_VISIBLE_DEVICES=1 \
  unitorch-infer examples/configs/diffusion/image2video/wan.ini \
  --test_file /data/decu/wan_real_ckpt/data/wan_i2v_test.noheader.tsv \
  --core/cli@from_ckpt_dir /data/decu/wan_real_ckpt/runs/20260603/empty_ckpt \
  --core/cli@cache_dir /data/decu/wan_real_ckpt/runs/20260603/infer_i2v/cache \
  --core/cli@output_folder /data/decu/wan_real_ckpt/runs/20260603/infer_i2v/videos \
  --core/task/deepspeed/supervised@output_path /data/decu/wan_real_ckpt/runs/20260603/infer_i2v/output.tsv \
  --core/task/deepspeed/supervised@num_workers 0 \
  --core/task/deepspeed/supervised@test_batch_size 1 \
  --core/model/diffusers/image2video/wan@num_infer_timesteps 4 \
  --core/model/diffusers/image2video/wan@num_frames 9 \
  --core/model/diffusers/image2video/wan@guidance_scale 4.5
```

### FastAPI server

```bash
env UNITORCH_CACHE=/data/decu/.cache UNITORCH_DEBUG=DETAIL \
  PYTHONPATH=/home/decu/my/unitorch/src CUDA_VISIBLE_DEVICES=1 \
  unitorch-fastapi examples/configs/fastapis/wan.ini \
  --port=18080 --device=0 \
  > /data/decu/wan_real_ckpt/runs/20260603/logs/fastapi_wan_server.log 2>&1
```

Health check:

```bash
curl -sf http://127.0.0.1:18080/health-check
```

Text-to-video FastAPI generate:

```bash
curl -sS -X POST 'http://127.0.0.1:18080/core/fastapi/wan/text2video/start'
curl -sS -X POST -o /data/decu/wan_real_ckpt/runs/20260603/fastapi/t2v_generate.mp4 \
  'http://127.0.0.1:18080/core/fastapi/wan/text2video/generate?text=A%20small%20robot%20walking%20through%20snow&height=192&width=320&num_frames=9&num_fps=8&guidance_scale=4.5&num_timesteps=4&seed=1234'
curl -sS -X GET 'http://127.0.0.1:18080/core/fastapi/wan/text2video/stop'
```

Image-to-video FastAPI generate:

```bash
curl -sS -X POST 'http://127.0.0.1:18080/core/fastapi/wan/image2video/start'
curl -sS -X POST -F image=@/data/decu/wan_real_ckpt/assets/wan_input.png \
  -o /data/decu/wan_real_ckpt/runs/20260603/fastapi/i2v_generate.mp4 \
  'http://127.0.0.1:18080/core/fastapi/wan/image2video/generate?text=Animate%20this%20poster%20with%20gentle%20camera%20drift&num_frames=9&num_fps=8&guidance_scale=4.5&num_timesteps=4&seed=1234'
curl -sS -X GET 'http://127.0.0.1:18080/core/fastapi/wan/image2video/stop'
```

Post-stop check:

- `curl -sf http://127.0.0.1:18080/health-check` failed as expected after shutdown

### e2e/full training

Initial full-path attempt that failed with OOM:

```bash
env UNITORCH_CACHE=/data/decu/.cache UNITORCH_DEBUG=DETAIL \
  PYTHONPATH=/home/decu/my/unitorch/src CUDA_VISIBLE_DEVICES=1 \
  unitorch-train examples/configs/diffusion/text2video/wan.ini \
  --train_file /data/decu/wan_real_ckpt/data/wan_t2v_train.noheader.tsv \
  --dev_file /data/decu/wan_real_ckpt/data/wan_t2v_dev.noheader.tsv \
  --core/cli@from_ckpt_dir /data/decu/wan_real_ckpt/runs/20260603/empty_ckpt \
  --core/cli@cache_dir /data/decu/wan_real_ckpt/runs/20260603/train_e2e_t2v/ckpts \
  --core/cli@output_folder /data/decu/wan_real_ckpt/runs/20260603/train_e2e_t2v/videos \
  --core/process/diffusion/wan@video_size '(320, 192)' \
  --core/task/deepspeed/supervised@epochs 1 \
  --core/task/deepspeed/supervised@ckpt_freq 1 \
  --core/task/deepspeed/supervised@log_freq 1 \
  --core/task/deepspeed/supervised@num_workers 0 \
  --core/task/deepspeed/supervised@train_batch_size 1 \
  --core/task/deepspeed/supervised@dev_batch_size 1
```

Successful fallback:

```bash
env UNITORCH_CACHE=/data/decu/.cache UNITORCH_DEBUG=DETAIL \
  PYTHONPATH=/home/decu/my/unitorch/src CUDA_VISIBLE_DEVICES=1 \
  unitorch-train examples/configs/diffusion/text2video/wan.ini \
  --train_file /data/decu/wan_real_ckpt/data/wan_t2v_train.noheader.tsv \
  --dev_file /data/decu/wan_real_ckpt/data/wan_t2v_dev.noheader.tsv \
  --core/cli@from_ckpt_dir /data/decu/wan_real_ckpt/runs/20260603/empty_ckpt \
  --core/cli@cache_dir /data/decu/wan_real_ckpt/runs/20260603/train_e2e_t2v_2l_z3/ckpts \
  --core/cli@output_folder /data/decu/wan_real_ckpt/runs/20260603/train_e2e_t2v_2l_z3/videos \
  --core/process/diffusion/wan@video_size '(160, 96)' \
  --core/process/diffusion/wan@max_seq_length 64 \
  --core/model/diffusers/text2video/wan@config_path /data/decu/wan_real_ckpt/configs/wan_ti2v_5b_transformer_num_layers_2.json \
  --core/task/deepspeed/supervised@config_path examples/configs/deepspeed/adamw.bf16.z3.json \
  --core/task/deepspeed/supervised@cpu_offload True \
  --core/task/deepspeed/supervised@epochs 1 \
  --core/task/deepspeed/supervised@ckpt_freq 1 \
  --core/task/deepspeed/supervised@log_freq 1 \
  --core/task/deepspeed/supervised@num_workers 0 \
  --core/task/deepspeed/supervised@train_batch_size 1 \
  --core/task/deepspeed/supervised@dev_batch_size 1 \
  --core/task/deepspeed/supervised@merge_zero3_checkpoint False
```

### LoRA training

Initial Wan I2V LoRA attempt hit the PEFT version gate:

```bash
env UNITORCH_CACHE=/data/decu/.cache UNITORCH_DEBUG=DETAIL \
  PYTHONPATH=/home/decu/my/unitorch/src CUDA_VISIBLE_DEVICES=1 \
  unitorch-train examples/configs/diffusion/image2video/wan.lora.ini \
  --train_file /data/decu/wan_real_ckpt/data/wan_i2v_train.noheader.tsv \
  --dev_file /data/decu/wan_real_ckpt/data/wan_i2v_dev.noheader.tsv \
  --core/cli@from_ckpt_dir /data/decu/wan_real_ckpt/runs/20260603/empty_ckpt \
  --core/cli@cache_dir /data/decu/wan_real_ckpt/runs/20260603/train_lora_i2v_2l/ckpts \
  --core/cli@output_folder /data/decu/wan_real_ckpt/runs/20260603/train_lora_i2v_2l/videos \
  --core/process/diffusion/wan@video_size '(160, 96)' \
  --core/process/diffusion/wan@max_seq_length 64 \
  --core/model/diffusers/peft/lora/image2video/wan@config_path /data/decu/wan_real_ckpt/configs/wan_ti2v_5b_transformer_num_layers_2.json \
  --core/task/deepspeed/supervised@config_path examples/configs/deepspeed/adamw.bf16.json \
  --core/task/deepspeed/supervised@epochs 1 \
  --core/task/deepspeed/supervised@ckpt_freq 1 \
  --core/task/deepspeed/supervised@log_freq 1 \
  --core/task/deepspeed/supervised@num_workers 0 \
  --core/task/deepspeed/supervised@train_batch_size 1 \
  --core/task/deepspeed/supervised@dev_batch_size 1
```

Successful Wan I2V LoRA run after the compatibility and target-module fixes:

```bash
env UNITORCH_CACHE=/data/decu/.cache UNITORCH_DEBUG=DETAIL \
  PYTHONPATH=/home/decu/my/unitorch/src CUDA_VISIBLE_DEVICES=1 \
  unitorch-train examples/configs/diffusion/image2video/wan.lora.ini \
  --train_file /data/decu/wan_real_ckpt/data/wan_i2v_train.noheader.tsv \
  --dev_file /data/decu/wan_real_ckpt/data/wan_i2v_dev.noheader.tsv \
  --core/cli@from_ckpt_dir /data/decu/wan_real_ckpt/runs/20260603/empty_ckpt \
  --core/cli@cache_dir /data/decu/wan_real_ckpt/runs/20260603/train_lora_i2v_2l_after_fix2/ckpts \
  --core/cli@output_folder /data/decu/wan_real_ckpt/runs/20260603/train_lora_i2v_2l_after_fix2/videos \
  --core/process/diffusion/wan@video_size '(160, 96)' \
  --core/process/diffusion/wan@max_seq_length 64 \
  --core/model/diffusers/peft/lora/image2video/wan@config_path /data/decu/wan_real_ckpt/configs/wan_ti2v_5b_transformer_num_layers_2.json \
  --core/task/deepspeed/supervised@config_path examples/configs/deepspeed/adamw.bf16.json \
  --core/task/deepspeed/supervised@epochs 1 \
  --core/task/deepspeed/supervised@ckpt_freq 1 \
  --core/task/deepspeed/supervised@log_freq 1 \
  --core/task/deepspeed/supervised@num_workers 0 \
  --core/task/deepspeed/supervised@train_batch_size 1 \
  --core/task/deepspeed/supervised@dev_batch_size 1
```

Successful Wan T2V LoRA follow-up smoke using the repo's T2V LoRA config:

```bash
env UNITORCH_CACHE=/data/decu/.cache UNITORCH_DEBUG=DETAIL \
  PYTHONPATH=/home/decu/my/unitorch/src CUDA_VISIBLE_DEVICES=1 \
  unitorch-train examples/configs/diffusion/text2video/wan.lora.ini \
  --train_file /data/decu/wan_real_ckpt/data/wan_t2v_train.noheader.tsv \
  --dev_file /data/decu/wan_real_ckpt/data/wan_t2v_dev.noheader.tsv \
  --core/cli@from_ckpt_dir /data/decu/wan_real_ckpt/runs/20260603/empty_ckpt \
  --core/cli@cache_dir /data/decu/wan_real_ckpt/runs/20260603/train_lora_t2v_2l_followup/ckpts \
  --core/cli@output_folder /data/decu/wan_real_ckpt/runs/20260603/train_lora_t2v_2l_followup/videos \
  --core/process/diffusion/wan@video_size '(160, 96)' \
  --core/process/diffusion/wan@max_seq_length 64 \
  --core/model/diffusers/peft/lora/text2video/wan@config_path /data/decu/wan_real_ckpt/configs/wan_ti2v_5b_transformer_num_layers_2.json \
  --core/model/diffusers/peft/lora/text2video/wan@pretrained_name wan-v2.2-ti2v-5b \
  --core/task/deepspeed/supervised@config_path examples/configs/deepspeed/adamw.bf16.json \
  --core/task/deepspeed/supervised@epochs 1 \
  --core/task/deepspeed/supervised@ckpt_freq 1 \
  --core/task/deepspeed/supervised@log_freq 1 \
  --core/task/deepspeed/supervised@num_workers 0 \
  --core/task/deepspeed/supervised@train_batch_size 1 \
  --core/task/deepspeed/supervised@dev_batch_size 1
```

Result:

- log: `/data/decu/wan_real_ckpt/runs/20260603/logs/train_lora_t2v_2l_followup.log`
- checkpoint: `/data/decu/wan_real_ckpt/runs/20260603/train_lora_t2v_2l_followup/ckpts/pytorch_model.bin` (`6917011` bytes)
- load snippet: `WanLoraForText2VideoGeneration loaded weights (56%)`
- train-step snippet: `epoch 0 step 0: train/loss=5.662803`
- save snippet: `WanLoraForText2VideoGeneration model save checkpoint to /data/decu/wan_real_ckpt/runs/20260603/train_lora_t2v_2l_followup/ckpts/pytorch_model.bin`
- fixes required for this follow-up run: none beyond the already-landed PEFT compatibility fallback and UMT5 target-module fixes

## Issues Found and Fixed

### 1. Wan 5B TI2V I2V path was misconfigured

Observed failures:

- `/data/decu/wan_real_ckpt/runs/20260603/logs/infer_i2v_before_fix.log`
  - `expected input ... to have 48 channels, but got 100`
- `/data/decu/wan_real_ckpt/runs/20260603/logs/infer_i2v_typeerror.log`
  - `unexpected keyword argument 'expand_timesteps'`

Fixes:

- added and wired `expand_timesteps=True` for `wan-v2.2-ti2v-5b`
- threaded `expand_timesteps` through CLI, FastAPI, base model, and PEFT model constructors
- aligned I2V training forward for expand-timesteps mode
- corrected I2V generate/FastAPI path to pass raw `image_pixel_values` to the pipeline instead of VAE latents

### 2. Local dataset/media path handling blocked realistic asset-driven tests

Fixes:

- local media path precedence fixes in image/video utilities
- AST dataset variable substitution fixes in dataset loaders

These were required to drive the real local image/video assets under `/data/decu/wan_real_ckpt/assets`.

### 3. LoRA training was incompatible with installed `peft==0.18.1`

Observed failure:

- `UMT5EncoderModel.add_adapter is incompatible with the installed PEFT version`

Fixes:

- added `add_adapter_compat()` in `src/unitorch/models/peft/__init__.py`
- fallback now uses `peft.inject_adapter_in_model()` when `add_adapter()` is blocked by the Transformers PEFT version gate
- added focused regression coverage in `tests/cli/test_wan_registration.py`

### 4. Wan LoRA text adapter target modules were wrong

Observed failure after the PEFT compatibility fallback:

- `Target modules {'to_v', 'to_q', 'to_k'} not found in the base model`

Root cause:

- the Wan diffusion transformer module names (`to_q`, `to_k`, `to_v`) were being reused for the `UMT5EncoderModel` text encoder

Fix:

- split text and diffusion target modules
- UMT5 text adapter now uses `["q", "k", "v", "o"]`
- Wan diffusion transformer adapter still uses `["to_q", "to_k", "to_v"]`

## Artifacts

| Artifact | Path | Size |
|---|---|---:|
| T2V inference video | `/data/decu/wan_real_ckpt/runs/20260603/infer_t2v/videos/0a85d18f8e9baa891f0be4ca350b8de6.mp4` | 44813 bytes |
| I2V inference video | `/data/decu/wan_real_ckpt/runs/20260603/infer_i2v/videos/250c7591f36d2a309483be1015cb2af3.mp4` | 90018 bytes |
| FastAPI T2V output | `/data/decu/wan_real_ckpt/runs/20260603/fastapi/t2v_generate.mp4` | 31973 bytes |
| FastAPI I2V output | `/data/decu/wan_real_ckpt/runs/20260603/fastapi/i2v_generate.mp4` | 23489 bytes |
| e2e fallback ZeRO-3 model shard | `/data/decu/wan_real_ckpt/runs/20260603/train_e2e_t2v_2l_z3/ckpts/pytorch_model/global_step1/zero_pp_rank_0_mp_rank_00_model_states.pt` | 12771624490 bytes |
| LoRA I2V checkpoint | `/data/decu/wan_real_ckpt/runs/20260603/train_lora_i2v_2l_after_fix2/ckpts/pytorch_model.bin` | 6917011 bytes |
| LoRA T2V checkpoint | `/data/decu/wan_real_ckpt/runs/20260603/train_lora_t2v_2l_followup/ckpts/pytorch_model.bin` | 6917011 bytes |

Key logs:

- `/data/decu/wan_real_ckpt/runs/20260603/logs/infer_t2v.log`
- `/data/decu/wan_real_ckpt/runs/20260603/logs/infer_i2v.log`
- `/data/decu/wan_real_ckpt/runs/20260603/logs/fastapi_wan_server.log`
- `/data/decu/wan_real_ckpt/runs/20260603/logs/train_e2e_t2v.log`
- `/data/decu/wan_real_ckpt/runs/20260603/logs/train_e2e_t2v_2l_z3.log`
- `/data/decu/wan_real_ckpt/runs/20260603/logs/train_lora_i2v_2l.log`
- `/data/decu/wan_real_ckpt/runs/20260603/logs/train_lora_i2v_2l_after_fix.log`
- `/data/decu/wan_real_ckpt/runs/20260603/logs/train_lora_i2v_2l_after_fix2.log`
- `/data/decu/wan_real_ckpt/runs/20260603/logs/train_lora_t2v_2l_followup.log`
- `/data/decu/wan_real_ckpt/runs/20260603/logs/infer_i2v_before_fix.log`
- `/data/decu/wan_real_ckpt/runs/20260603/logs/infer_i2v_typeerror.log`

## Fallbacks and Limitations

- The original full 5B e2e train path OOMed before the first step:
  - log: `/data/decu/wan_real_ckpt/runs/20260603/logs/train_e2e_t2v.log`
  - error: `torch.OutOfMemoryError: Tried to allocate 18.63 GiB`
- To still validate real checkpoint loading and an actual train step, the e2e fallback used:
  - 2 transformer layers
  - `video_size=(160, 96)`
  - `max_seq_length=64`
  - batch size 1
  - single-sample train/dev sets
  - ZeRO-3 with CPU offload
- LoRA I2V and T2V training used the same reduced 2-layer TI2V config and minimal data settings for a real train-step smoke.
- No additional code changes were required for the follow-up T2V LoRA smoke; the previously fixed PEFT compatibility path and UMT5 target-module split already covered it.
- Large generated artifacts and checkpoints were kept under `/data/decu/wan_real_ckpt` and were not committed.
- Raw session logs in `reports/*.log` were intentionally excluded from commit via `.gitignore`.

## Git Verification

Pre-commit `git status --short`:

```text
M  README.md
M  examples/README.md
M  examples/configs/diffusion/image2video/wan.ini
M  examples/configs/diffusion/image2video/wan.lora.ini
M  examples/configs/diffusion/text2video/wan.ini
M  examples/configs/diffusion/text2video/wan.lora.ini
A  examples/configs/fastapis/wan.ini
M  examples/fastapis.ini
M  src/unitorch/cli/datasets/hf.py
M  src/unitorch/cli/datasets/megatron.py
M  src/unitorch/cli/fastapis/wan/image2video.py
M  src/unitorch/cli/fastapis/wan/text2video.py
M  src/unitorch/cli/models/diffusers/__init__.py
M  src/unitorch/cli/models/diffusers/modeling_wan.py
M  src/unitorch/cli/models/diffusers/processing_wan.py
M  src/unitorch/cli/models/image_utils.py
M  src/unitorch/cli/models/peft/diffusers/modeling_wan.py
M  src/unitorch/cli/models/video_utils.py
M  src/unitorch/models/diffusers/modeling_wan.py
M  src/unitorch/models/diffusers/processing_wan.py
M  src/unitorch/models/peft/__init__.py
M  src/unitorch/models/peft/diffusers/modeling_wan.py
A  tests/cli/test_ast_datasets.py
A  tests/cli/test_media_processors.py
M  tests/cli/test_wan_registration.py
M  wiki/cli/fastapis.md
M  wiki/cli/models/diffusers.md
?? reports/
```

`git diff --check` before commit:

```text
<no output>
```

Focused checks:

```bash
python3 -m pytest tests/cli/test_wan_registration.py tests/cli/test_media_processors.py tests/cli/test_ast_datasets.py -q
# 21 passed in 9.86s

python3 -m py_compile \
  src/unitorch/models/peft/__init__.py \
  src/unitorch/models/peft/diffusers/modeling_wan.py \
  tests/cli/test_wan_registration.py
# success
```

`git log --oneline -1` at implementation commit creation:

```text
95e59a8 fix wan ti2v 5b
```

Follow-up validation for the T2V LoRA smoke and report update:

```text
git status --short
M reports/wan_real_ckpt_test_report_20260603.md

git diff --check
<no output>
```

```bash
python3 -m pytest tests/cli/test_wan_registration.py -q
# 18 passed in 9.72s
```

## PR Status

Branch push result:

- pushed to `origin/test/wan-real-ckpt-0.0.2.1`

Automatic PR creation:

- not possible from this host
- `gh` is not installed
- no usable `GH_TOKEN` / `GITHUB_TOKEN` was available
- `git credential fill` failed because no credential store was configured and the configured VS Code askpass path was missing

Manual PR URL:

- `https://github.com/fuliucansheng/unitorch/pull/new/test/wan-real-ckpt-0.0.2.1`
