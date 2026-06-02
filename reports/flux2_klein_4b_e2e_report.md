# FLUX.2-klein-4B 端到端测试报告

## 1. 测试目标与范围

- 仓库：`/home/decu/my/unitorch-flux2-fulltest-v2`
- 分支：`feat/flux2-klein-4b-fulltest-0.0.2.1-v2`
- 基线提交：`08bda5c8034bc13781dc2023323df5557c38a6f7`
- 提交说明：最终提交请以包含本报告的 git commit 为准
- 基线来源：`origin/master`
- checkpoint：`black-forest-labs/FLUX.2-klein-4B`
- 缓存目录：`/data/decu/.cache`
- 运行产物根目录：`/data/decu/flux2_klein_4b_e2e_20260602`

本次实际覆盖：

| 项目 | 状态 | 说明 |
| --- | --- | --- |
| text-to-image inference | PASS | 真实 `FLUX.2-klein-4B`，输出 `jpg` |
| image editing inference | PASS | 真实 `FLUX.2-klein-4B`，输出 `jpg` |
| FastAPI `generate` | PASS | daemon 启动 + 真实 HTTP 请求 + 返回 `png` |
| full training e2e | PASS | 真实 checkpoint，全量训练单步 smoke，落 checkpoint |
| LoRA training | PASS | 真实 checkpoint，LoRA adapter 训练单步 smoke，落 adapter checkpoint |

## 2. 环境信息

| 项目 | 值 |
| --- | --- |
| OS | `Linux-5.15.0-139-generic-x86_64-with-glibc2.31` |
| Python | `3.10.18` |
| torch | `2.10.0+cu128` |
| torch CUDA | `12.8` |
| diffusers | `0.38.0` |
| transformers | `5.7.0` |
| fastapi | `0.124.4` |
| accelerate | `1.7.0` |
| peft | `0.18.1` |
| deepspeed | `0.17.0` |
| GPU0 | `NVIDIA RTX A6000 49140 MiB` |
| GPU1 | `NVIDIA RTX A6000 49140 MiB` |
| NVIDIA Driver | `570.153.02` |

统一环境变量：

```bash
export UNITORCH_CACHE=/data/decu/.cache
export HF_HOME=/data/decu/.cache
export TRANSFORMERS_CACHE=/data/decu/.cache
export HF_DATASETS_CACHE=/data/decu/.cache
export UNITORCH_DEBUG=DETAIL
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
```

## 3. checkpoint 来源与缓存验证

checkpoint 来源为 Hugging Face 上的 `black-forest-labs/FLUX.2-klein-4B`，所有下载都保留在 `/data/decu/.cache`。

缓存文件存在性验证：

```text
/data/decu/.cache/3da7dc60...e2fd  7.3G  transformer
/data/decu/.cache/447dfdbc...d56b  4.7G  text_encoder shard 1
/data/decu/.cache/30691079...b381  2.9G  text_encoder shard 2
/data/decu/.cache/3e664b60...ddc2  161M  vae
```

`UNITORCH_DEBUG=DETAIL` 下的关键 load 日志摘录（`text2image_infer_task.log` / FastAPI worker / training log 一致）：

```text
2026-06-02 17:45:31 PM : DEBUG : Resolving weight source https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/main/transformer/diffusion_pytorch_model.safetensors
2026-06-02 17:45:31 PM : DEBUG : Loading weight file /data/decu/.cache/3da7dc60...e2fd
2026-06-02 17:45:31 PM : DEBUG : Loaded 169 tensors from /data/decu/.cache/3da7dc60...e2fd
2026-06-02 17:45:31 PM : DEBUG : Loading weight file /data/decu/.cache/447dfdbc...d56b
2026-06-02 17:45:31 PM : DEBUG : Loaded 229 tensors from /data/decu/.cache/447dfdbc...d56b
2026-06-02 17:45:31 PM : DEBUG : Loading weight file /data/decu/.cache/30691079...b381
2026-06-02 17:45:31 PM : DEBUG : Loaded 169 tensors from /data/decu/.cache/30691079...b381
2026-06-02 17:45:32 PM : DEBUG : Loading weight file /data/decu/.cache/3e664b60...ddc2
2026-06-02 17:45:32 PM : DEBUG : Loaded 251 tensors from /data/decu/.cache/3e664b60...ddc2
```

## 4. 修复摘要

本次为了让 `flux2-klein-4b` 能完整跑通 inference / FastAPI / training，做了以下修复：

1. `process` 不再依赖 `AutoProcessor`，改为显式 tokenizer / processor 初始化参数。
   - 增加 `tokenizer_class`
   - 显式解析 `tokenizer.json`、`vocab.json`、`merges.txt`、`tokenizer_config.json`、`special_tokens_map.json`、`chat_template.jinja`、`added_tokens.json`

2. 更新 `pretrained_stable_infos` 中的 `flux2-klein-4b` 预训练信息。
   - 增加 `text.tokenizer_class = Qwen2Tokenizer`
   - 在 `text` 节点补齐 tokenizer 初始化所需文件路径

3. 修复 `Qwen2Tokenizer.apply_chat_template()` 与旧多模态 prompt 格式不兼容的问题。
   - 复现错误：
     ```text
     TypeError: can only concatenate str (not "list") to str
     ```
   - 修复方式：改为 string-based chat message，而不是 list-of-dicts multimodal content

4. 修复 FLUX2 文本编码器实例化逻辑。
   - `mistral3` -> `Mistral3ForConditionalGeneration`
   - `qwen3` -> `Qwen3ForCausalLM`

5. 修复 FastAPI FLUX2 路径的三个实际问题。
   - 缺少 `cached_path` 导入，worker 直接异常退出
   - `start(pretrained_name="flux2-dev")` 硬编码覆盖 config，导致错误拉起 `FLUX.2-dev`
   - daemon 启动的 health-check 固定 30 秒，4B 模型真实加载会被误判超时

6. FastAPI 生成逻辑改为走 unitorch 自己的 `_encode_prompt()` + `_generate_from_embeds()`。
   - 避开 diffusers `Flux2Pipeline(...)` 对 tokenizer/processor 路径的旧假设

7. 版本更新为 `0.0.2.1`。

## 5. 失败复现与证据

### 5.1 inference 初始失败

失败日志：`/data/decu/flux2_klein_4b_e2e_20260602/logs/text2image_infer.log`

```text
TypeError: can only concatenate str (not "list") to str
```

根因：`Qwen2Tokenizer.apply_chat_template()` 不接受旧的 multimodal list 内容格式。

### 5.2 FastAPI 初始失败

失败日志：`/tmp/unitorch_fastapi_services1_676d6661@09d3e6.stdout.log`

```text
NameError: name 'cached_path' is not defined
```

同一份失败日志还显示 FastAPI 在修复前错误访问了 `FLUX.2-dev`：

```text
HEAD /black-forest-labs/FLUX.2-dev/resolve/main/transformer/config.json
HEAD /black-forest-labs/FLUX.2-dev/resolve/main/text_encoder/config.json
HEAD /black-forest-labs/FLUX.2-dev/resolve/main/vae/config.json
HEAD /black-forest-labs/FLUX.2-dev/resolve/main/scheduler/scheduler_config.json
```

第二次失败为 daemon 健康检查超时，日志：`/data/decu/flux2_klein_4b_e2e_20260602/logs/fastapi_start_retry.log`

```text
RuntimeError: unitorch-fastapi health-check timeout after 30s
```

修复后增加 `core/cli@health_check_timeout`，最终使用 `300` 秒成功拉起服务。

## 6. 测试明细

### 6.1 文生图 inference

- 配置：`/data/decu/flux2_klein_4b_e2e_20260602/configs/text2image_runtime.ini`
- 日志：`/data/decu/flux2_klein_4b_e2e_20260602/logs/text2image_infer_task.log`
- 输出目录：`/data/decu/flux2_klein_4b_e2e_20260602/outputs/text2image/images`
- 输出文件：`eaa3ad78ddf935776b4b7894e7bc948f.jpg`
- 结果索引：`/data/decu/flux2_klein_4b_e2e_20260602/outputs/train_full/ckpt/output.txt`
- 关键参数：`256x256`、`num_timesteps=8`、`guidance_scale=3.5`

执行命令：

```bash
/usr/bin/time -f 'ELAPSED=%E\nMAX_RSS_KB=%M' python3 -u - <<'PY' \
  > /data/decu/flux2_klein_4b_e2e_20260602/logs/text2image_infer_task.log 2>&1
import time
from unitorch.cli import Config, init_registered_module, registered_task
cfg = Config('/data/decu/flux2_klein_4b_e2e_20260602/configs/text2image_runtime.ini')
task = init_registered_module(cfg.getdefault('core/cli', 'task_name', None), cfg, registered_task)
print('TASK_INIT_DONE', time.strftime('%Y-%m-%d %H:%M:%S'), flush=True)
task.infer()
print('TASK_INFER_DONE', time.strftime('%Y-%m-%d %H:%M:%S'), flush=True)
PY
```

结果：

```text
TASK_INIT_DONE 2026-06-02 17:45:36
TASK_INFER_DONE 2026-06-02 17:45:39
ELAPSED=1:11.25
MAX_RSS_KB=32427112
```

`output.txt`：

```text
an abstract icon with a gold circle, red square, and green triangle	eaa3ad78ddf935776b4b7894e7bc948f.jpg
```

状态：`PASS`

### 6.2 图片编辑 inference

- 配置：`/data/decu/flux2_klein_4b_e2e_20260602/configs/editing_runtime.ini`
- 日志：`/data/decu/flux2_klein_4b_e2e_20260602/logs/editing_infer_task.log`
- 输入图：`/data/decu/flux2_klein_4b_e2e_20260602/data/source.png`
- 输出目录：`/data/decu/flux2_klein_4b_e2e_20260602/outputs/editing/images`
- 输出文件：`378da782aa276f66b2d2e7766b492d2b.jpg`
- 关键参数：`256x256`、`num_timesteps=8`、`guidance_scale=3.5`

执行命令：

```bash
/usr/bin/time -f 'ELAPSED=%E\nMAX_RSS_KB=%M' python3 -u - <<'PY' \
  > /data/decu/flux2_klein_4b_e2e_20260602/logs/editing_infer_task.log 2>&1
import time
from unitorch.cli import Config, init_registered_module, registered_task
cfg = Config('/data/decu/flux2_klein_4b_e2e_20260602/configs/editing_runtime.ini')
task = init_registered_module(cfg.getdefault('core/cli', 'task_name', None), cfg, registered_task)
print('TASK_INIT_DONE', time.strftime('%Y-%m-%d %H:%M:%S'), flush=True)
task.infer()
print('TASK_INFER_DONE', time.strftime('%Y-%m-%d %H:%M:%S'), flush=True)
PY
```

结果：

```text
TASK_INIT_DONE 2026-06-02 17:47:17
TASK_INFER_DONE 2026-06-02 17:47:21
ELAPSED=1:11.68
MAX_RSS_KB=32425712
```

`output.txt`：

```text
change the shapes into blue, purple, and orange while keeping the composition	/data/decu/flux2_klein_4b_e2e_20260602/data/source.png	378da782aa276f66b2d2e7766b492d2b.jpg
```

状态：`PASS`

### 6.3 FastAPI `generate`

- 配置：`/data/decu/flux2_klein_4b_e2e_20260602/configs/fastapi_text2image.ini`
- 启动日志：`/tmp/unitorch_fastapi_services1_676d6661@06de25.stdout.log`
- 请求输出：`/data/decu/flux2_klein_4b_e2e_20260602/outputs/fastapi/fastapi_text2image.png`
- 关键参数：`height=256`、`width=256`、`num_timesteps=8`、`guidance_scale=3.5`、`seed=1234`

启动命令：

```bash
unitorch-fastapi start /data/decu/flux2_klein_4b_e2e_20260602/configs/fastapi_text2image.ini
```

HTTP 验证命令：

```bash
curl -sS -G 'http://127.0.0.1:18080/core/fastapi/flux2/text2image/generate' \
  --data-urlencode 'text=an abstract icon with a gold circle, red square, and green triangle' \
  --data 'height=256' \
  --data 'width=256' \
  --data 'num_timesteps=8' \
  --data 'guidance_scale=3.5' \
  --data 'seed=1234' \
  -o /data/decu/flux2_klein_4b_e2e_20260602/outputs/fastapi/fastapi_text2image.png
```

关键日志：

```text
2026-06-02 17:55:23 PM : INFO : autostarting fastapi service core/fastapi/flux2/text2image
2026-06-02 17:56:21,915 | INFO | Application startup complete.
2026-06-02 17:56:21,916 | INFO | Uvicorn running on http://0.0.0.0:18080
2026-06-02 17:56:22,047 | INFO | 127.0.0.1 - "GET /health-check HTTP/1.1" 200
2026-06-02 17:56:45,585 | INFO | 127.0.0.1 - "GET /core/fastapi/flux2/text2image/generate?... HTTP/1.1" 200
```

请求返回文件验证：

```text
PNG image data, 256 x 256, 8-bit/color RGB, non-interlaced
```

命令耗时（curl）：

```text
ELAPSED=0:01.16
MAX_RSS_KB=12020
```

状态：`PASS`

### 6.4 full training e2e

- 配置：`/data/decu/flux2_klein_4b_e2e_20260602/configs/text2image_runtime.ini`
- 日志：`/data/decu/flux2_klein_4b_e2e_20260602/logs/train_full_task.log`
- checkpoint 目录：`/data/decu/flux2_klein_4b_e2e_20260602/outputs/train_full/ckpt`
- 关键参数：`epochs=1`、`train_batch_size=1`、`dev_batch_size=1`、`256x256`、`num_timesteps=8`

执行命令：

```bash
/usr/bin/time -f 'ELAPSED=%E\nMAX_RSS_KB=%M' python3 -u - <<'PY' \
  > /data/decu/flux2_klein_4b_e2e_20260602/logs/train_full_task.log 2>&1
from unitorch.cli import Config, init_registered_module, registered_task
cfg = Config('/data/decu/flux2_klein_4b_e2e_20260602/configs/text2image_runtime.ini')
task = init_registered_module(cfg.getdefault('core/cli', 'task_name', None), cfg, registered_task)
print('TASK_INIT_DONE', flush=True)
task.train()
print('TASK_TRAIN_DONE', flush=True)
PY
```

结果摘录：

```text
TASK_INIT_DONE
epoch 0 step 0: train/loss=1.468836
val/score: -1.352700  best: -inf
Flux2ForText2ImageGeneration saved checkpoint to .../pytorch_model_latest.bin
AdamWOptimizer saved checkpoint to .../pytorch_optim_latest.bin
val/score: -0.903466  best: -inf
TASK_TRAIN_DONE
ELAPSED=3:09.01
MAX_RSS_KB=32429872
```

输出物：

```text
pytorch_model_latest.bin  15G
pytorch_optim_latest.bin  15G
info.json                 76B
```

状态：`PASS`

### 6.5 LoRA training

- 配置：`/data/decu/flux2_klein_4b_e2e_20260602/configs/editing_lora_runtime.ini`
- 日志：`/data/decu/flux2_klein_4b_e2e_20260602/logs/train_lora_task.log`
- checkpoint 目录：`/data/decu/flux2_klein_4b_e2e_20260602/outputs/train_lora/ckpt`
- 关键参数：`epochs=1`、`train_batch_size=1`、`lora_r=4`、`enable_text_adapter=False`、`enable_transformer_adapter=True`、`256x256`

执行命令：

```bash
/usr/bin/time -f 'ELAPSED=%E\nMAX_RSS_KB=%M' python3 -u - <<'PY' \
  > /data/decu/flux2_klein_4b_e2e_20260602/logs/train_lora_task.log 2>&1
from unitorch.cli import Config, init_registered_module, registered_task
cfg = Config('/data/decu/flux2_klein_4b_e2e_20260602/configs/editing_lora_runtime.ini')
task = init_registered_module(cfg.getdefault('core/cli', 'task_name', None), cfg, registered_task)
print('TASK_INIT_DONE', flush=True)
task.train()
print('TASK_TRAIN_DONE', flush=True)
PY
```

结果摘录：

```text
TASK_INIT_DONE
epoch 0 step 0: train/loss=1.474273
val/score: -1.382570  best: -inf
Flux2LoraForImageEditingGeneration model save checkpoint to .../pytorch_model_latest.bin
AdamWOptimizer saved checkpoint to .../pytorch_optim_latest.bin
val/score: -0.917655  best: -inf
TASK_TRAIN_DONE
ELAPSED=1:11.46
MAX_RSS_KB=32440528
```

输出物：

```text
pytorch_model_latest.bin  1.5M
pytorch_optim_latest.bin  2.9M
info.json                 76B
```

状态：`PASS`

## 7. OOM / 降配说明

- 本次实际跑测未触发 OOM。
- 因此未做 layer 裁剪、未降低到 `192` 或 `128`、未切换到 mock、未回退到假 checkpoint。
- 最终保留的实测配置就是：
  - 单卡 `CUDA_VISIBLE_DEVICES=0`
  - `256x256`
  - `batch_size=1`
  - `epochs=1`
  - `num_timesteps=8`
  - LoRA 使用 `lora_r=4` 且 `enable_text_adapter=False`

## 8. 最终校验

已执行：

```text
pytest tests/cli/test_flux2_registration.py                -> 11 passed
python3 -m compileall <touched files>                      -> passed
git diff --check                                           -> passed
git status --short                                         -> 仅包含本次源码修改、`examples/configs/fastapis/flux2.ini`、本报告，以及未提交的本地日志 `reports/flux2_klein_codex_20260602_172010.log`
```

说明：

- 未运行 `pytest tests` 全量套件；本仓库测试覆盖本身较轻，且本次重点是 FLUX2 真实 checkpoint 的端到端验证。
- 已补充并执行与本次修复最相关的回归测试：`tests/cli/test_flux2_registration.py`
- 大文件、checkpoint、训练输出、生成图片均保留在 `/data/decu/...`，不会提交到仓库
- `reports/flux2_klein_codex_20260602_172010.log` 为现有日志文件，不应提交
