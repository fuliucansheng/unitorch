# FLUX.2 [dev] 32B 与 [klein] 4B 微调

支持以下两个**确切的官方检查点**，每个提供文生图、图像编辑，以及 LoRA、全 diffusion transformer 参数微调，共 8 个组合。

| UniTorch 名称 | 官方仓库 | 文本编码器 / hidden-state 层 | Guidance | 默认推理步数 |
|---|---|---|---|---|
| `flux2-dev` | `black-forest-labs/FLUX.2-dev` | `Mistral3ForConditionalGeneration`，`(10, 20, 30)` | transformer 有 guidance embedding，默认 `4.0` | 50 |
| `flux2-klein-4b` | `black-forest-labs/FLUX.2-klein-4B` | `Qwen3ForCausalLM`，`(9, 18, 27)` | 无 guidance embedding，默认 `1.0` | 4 |

这里的 klein 是 **distilled 版本**。上游推荐 `FLUX.2-klein-base-4B` 用于微调；蒸馏版继续训练可能损伤原有 4 步生成能力，必须重新评估质量。本实现保留用户指定的 distilled 检查点，未替换为 base，也没有为其他尺寸新增训练配置。

## 环境、权重与缓存

在仓库根目录激活已安装 UniTorch 的虚拟环境，再选择本机的数据与产物目录：

```bash
FLUX2_BIN="${VIRTUAL_ENV:?Activate your installed virtual environment}/bin"
FLUX2_RUN_ROOT="$PWD/flux2-runs"
FLUX2_DATA_ROOT="$PWD/data"
export TORCH_EXTENSIONS_DIR="$FLUX2_RUN_ROOT/torch_extensions"
"$FLUX2_BIN/python" -m pip show unitorch torch diffusers transformers peft deepspeed
```

使用本仓库的 editable 安装，不设置 `PYTHONPATH`。system-site-packages venv 不一定有 `torchrun` console script，所以下文通过该 venv 的 `python -m torch.distributed.run` 启动同一入口。验证环境为 Torch 2.10.0、Diffusers 0.38.0、Transformers 5.7.0、PEFT 0.19.1、DeepSpeed 0.17.0。PEFT 0.18.1 与此 Transformers 版本的文本 adapter API 不兼容，请使用兼容的 PEFT 版本。

先检查已有 Hub/UniTorch 缓存，优先复用官方 Diffusers snapshot。`UNITORCH_CACHE` 会影响 `HF_HOME`，改变它之前先定位缓存。配置的 `pretrained_model_path` 可指向已有 snapshot，包含 `transformer/`、`text_encoder/`、`vae/`、`scheduler/`、`tokenizer/`。所有组件都通过明确的 concrete 类加载。普通加载使用各组件的分片读取；ZeRO 初始化时逐 shard、逐 tensor 复制到分区，不同时保留完整模型的多份权重。

未设置 `pretrained_model_path` 时使用注册表中的官方资源。dev 的受限文件需要**已有、有效**的 Hub 授权和用户自行接受的许可；此代码不会代替用户接受许可。也可以分别覆盖 `config_path`、`text_config_path`、`vae_config_path`、`scheduler_config_path` 和 `pretrained_weight_path`。缓存、训练产物均应位于数据盘。

本次核查的上游 revisions：

- dev：`26afe3a78bb242c0a8bb181dcc8937bb16e5c66c`
- distilled klein-4B：`e7b7dc27f91deacad38e78976d1f2b499d76a294`

## 数据格式

默认使用无表头 TSV，列顺序如下。`image` 是目标图像，`reference` 是编辑条件图像；路径可为本地文件。`train` 与 `dev` 使用相同 schema，`test` 不含目标图像。

| 任务 | train / dev | test |
|---|---|---|
| 文生图 | `prompt<TAB>image` | `prompt` |
| 编辑 | `prompt<TAB>reference<TAB>image` | `prompt<TAB>reference` |

将 TSV 放入 `$FLUX2_DATA_ROOT/{text2image,editing}.{train,dev,test}.tsv`。下列命令显式覆盖配置中的任务环境路径；也可通过 `--train_file=... --dev_file=... --test_file=...` 指向其它位置。默认将图像处理为 256×256，prompt 长度为 512。官方 VAE 下尺寸必须为 16 的倍数；配置中的 `image_size` 顺序是 `(width, height)`。训练和推理应使用一致的处理尺寸。

执行 smoke 时可使用明确标记的合成数据验证数值和 IO；正式训练需要真实数据及独立验证集。官方 pipeline 的参考图像输入要求至少 64×64。

多个参考图受当前官方 API 支持。要使用固定的两个参考图，可显式改为：

```ini
[core/dataset/ast]
names = ['prompt', 'ref1', 'ref2', 'image']

[core/dataset/ast/train]
preprocess_functions = ['core/process/diffusion/flux2/editing(prompt, [ref1, ref2], image)']

[core/dataset/ast/dev]
preprocess_functions = ['core/process/diffusion/flux2/editing(prompt, [ref1, ref2], image)']

[core/dataset/ast/test]
names = ['prompt', 'ref1', 'ref2']
preprocess_functions = ['core/process/diffusion/flux2/editing/inputs(prompt, [ref1, ref2])']
```

单参考图的 batch 为 `[B,C,H,W]`，多参考图为 `[B,R,C,H,W]`；`R` 不会当作 batch。一个 batch 内要求相同的 `R` 与处理后尺寸；位置 IDs 的参考图时间坐标是 `10,20,...`，每个样本保留自己的图像内容。推理输出为 JSONL，保留配置的输入列并增加 `diffusion` 图片文件名。

## 八个训练命令

下列命令在仓库根目录执行，使用上述 `FLUX2_BIN`。权重已在本地时，给每条命令追加 `--pretrained_model_path=/实际的官方snapshot路径`。各配置的输出目录互不重叠。**按顺序运行，不同时启动多个大模型任务**。dev e2e 命令需要下文所述的资源，不能在内存预检失败时直接运行。

```bash
# distilled klein 4B：单卡 LoRA
"$FLUX2_BIN/unitorch-train" examples/configs/diffusion/text2image/flux2.klein-4b.lora.ini \
  --cache_dir="$FLUX2_RUN_ROOT/flux2.klein-4b.text2image.lora" --from_ckpt_dir="$FLUX2_RUN_ROOT/no-initial-checkpoint" \
  --train_file="$FLUX2_DATA_ROOT/text2image.train.tsv" --dev_file="$FLUX2_DATA_ROOT/text2image.dev.tsv" \
  --test_file="$FLUX2_DATA_ROOT/text2image.test.tsv"
"$FLUX2_BIN/unitorch-train" examples/configs/diffusion/editing/flux2.klein-4b.lora.ini \
  --cache_dir="$FLUX2_RUN_ROOT/flux2.klein-4b.editing.lora" --from_ckpt_dir="$FLUX2_RUN_ROOT/no-initial-checkpoint" \
  --train_file="$FLUX2_DATA_ROOT/editing.train.tsv" --dev_file="$FLUX2_DATA_ROOT/editing.dev.tsv" \
  --test_file="$FLUX2_DATA_ROOT/editing.test.tsv"

# distilled klein 4B：双卡、全 transformer、ZeRO-3 + CPU offload
"$FLUX2_BIN/python" -m torch.distributed.run --standalone --no_python --nproc_per_node=2 "$FLUX2_BIN/unitorch-train" examples/configs/diffusion/text2image/flux2.klein-4b.e2e.ini \
  --cache_dir="$FLUX2_RUN_ROOT/flux2.klein-4b.text2image.e2e" --from_ckpt_dir="$FLUX2_RUN_ROOT/no-initial-checkpoint" \
  --train_file="$FLUX2_DATA_ROOT/text2image.train.tsv" --dev_file="$FLUX2_DATA_ROOT/text2image.dev.tsv" \
  --test_file="$FLUX2_DATA_ROOT/text2image.test.tsv"
"$FLUX2_BIN/python" -m torch.distributed.run --standalone --no_python --nproc_per_node=2 "$FLUX2_BIN/unitorch-train" examples/configs/diffusion/editing/flux2.klein-4b.e2e.ini \
  --cache_dir="$FLUX2_RUN_ROOT/flux2.klein-4b.editing.e2e" --from_ckpt_dir="$FLUX2_RUN_ROOT/no-initial-checkpoint" \
  --train_file="$FLUX2_DATA_ROOT/editing.train.tsv" --dev_file="$FLUX2_DATA_ROOT/editing.dev.tsv" \
  --test_file="$FLUX2_DATA_ROOT/editing.test.tsv"

# dev 32B：双卡 LoRA、ZeRO-3 + CPU offload
"$FLUX2_BIN/python" -m torch.distributed.run --standalone --no_python --nproc_per_node=2 "$FLUX2_BIN/unitorch-train" examples/configs/diffusion/text2image/flux2.dev.lora.ini \
  --cache_dir="$FLUX2_RUN_ROOT/flux2.dev.text2image.lora" --from_ckpt_dir="$FLUX2_RUN_ROOT/no-initial-checkpoint" \
  --train_file="$FLUX2_DATA_ROOT/text2image.train.tsv" --dev_file="$FLUX2_DATA_ROOT/text2image.dev.tsv" \
  --test_file="$FLUX2_DATA_ROOT/text2image.test.tsv"
"$FLUX2_BIN/python" -m torch.distributed.run --standalone --no_python --nproc_per_node=2 "$FLUX2_BIN/unitorch-train" examples/configs/diffusion/editing/flux2.dev.lora.ini \
  --cache_dir="$FLUX2_RUN_ROOT/flux2.dev.editing.lora" --from_ckpt_dir="$FLUX2_RUN_ROOT/no-initial-checkpoint" \
  --train_file="$FLUX2_DATA_ROOT/editing.train.tsv" --dev_file="$FLUX2_DATA_ROOT/editing.dev.tsv" \
  --test_file="$FLUX2_DATA_ROOT/editing.test.tsv"

# dev 32B：全 transformer；约 160 GiB 可用 RAM 不满足此 AdamW CPU offload 配置
"$FLUX2_BIN/python" -m torch.distributed.run --standalone --no_python --nproc_per_node=2 "$FLUX2_BIN/unitorch-train" examples/configs/diffusion/text2image/flux2.dev.e2e.ini \
  --cache_dir="$FLUX2_RUN_ROOT/flux2.dev.text2image.e2e" --from_ckpt_dir="$FLUX2_RUN_ROOT/no-initial-checkpoint" \
  --train_file="$FLUX2_DATA_ROOT/text2image.train.tsv" --dev_file="$FLUX2_DATA_ROOT/text2image.dev.tsv" \
  --test_file="$FLUX2_DATA_ROOT/text2image.test.tsv"
"$FLUX2_BIN/python" -m torch.distributed.run --standalone --no_python --nproc_per_node=2 "$FLUX2_BIN/unitorch-train" examples/configs/diffusion/editing/flux2.dev.e2e.ini \
  --cache_dir="$FLUX2_RUN_ROOT/flux2.dev.editing.e2e" --from_ckpt_dir="$FLUX2_RUN_ROOT/no-initial-checkpoint" \
  --train_file="$FLUX2_DATA_ROOT/editing.train.tsv" --dev_file="$FLUX2_DATA_ROOT/editing.dev.tsv" \
  --test_file="$FLUX2_DATA_ROOT/editing.test.tsv"
```

这些是全部 8 个组合的训练入口；官方权重实测已通过其中 6 个组合，dev 全 transformer 的两个任务仍未验证，详见下文。可先用小数据和较低分辨率验证，再扩大训练规模。

## 模型与训练语义

E2E 表示 diffusion transformer 的**全部参数**可训练。默认 `freeze_text_encoder=True`、`freeze_vae_encoder=True`，不会把 24B/4B 文本编码器或 VAE 加入优化器。冻结组件保持 `eval()`，验证后会恢复 transformer 的训练状态。

- E2E 中 `freeze_text_encoder=False` 允许 prompt hidden states 向文本编码器反传。损失只依赖所选层：未使用的 LM head、Mistral 视觉分支和所选层之后的层没有梯度。它不是语言建模或视觉编码器训练目标；不建议在内存受限的 smoke 中启用。
- `freeze_vae_encoder=False` 训练 VAE encoder 与存在时的 `quant_conv`。decoder、`post_quant_conv` 和 BN 校准统计保持冻结；latent flow loss 不训练解码器。保存检查点时包含显式解冻的编码部分。
- LoRA 默认仅训练 transformer adapters；`enable_text_adapter=False`，`enable_transformer_adapter=True`。如显式打开文本 adapters，则增加文本 Q/K/V adapters；不会解冻文本底座。LoRA 类不把 E2E 的 freeze flags 当作训练底座的开关。
- 默认 LoRA targets 为双流注意力的 `to_q`、`to_k`、`to_v`、`add_q_proj`、`add_k_proj`、`add_v_proj`、`to_out.0`、`to_add_out`，以及单流块的 **`to_qkv_mlp_proj`** 和 `single_transformer_blocks.<i>.attn.to_out`。`to_qkv_mlp` 不是当前 Diffusers 的有效名称。
- 普通 LoRA 使用 BF16 底座、FP32 adapter 参数；ZeRO 管理分片参数与 FP32 optimizer master weights。启用 gradient checkpointing；LoRA + ZeRO 对冻结底座采用可重入的 checkpoint 路径。

dev prompt 使用官方 system message，去掉 `[IMG]` 标记且不添加 generation prompt。klein 使用 user-only chat、`add_generation_prompt=True`、`enable_thinking=False`。所选 hidden states 按官方顺序拼接；直接调用具体文本 backbone，避免计算未使用的巨大词表 logits。

目标图与参考图均使用 VAE posterior **mode**，先 2×2 patchify，再按 BN 的 running mean/variance 和 epsilon 归一化。训练使用 FlowMatchEuler 的 1000 个 timestep、随机噪声插值和 `noise - latent` 的标量 MSE。训练 scheduler 与推理 scheduler 分离；4 步推理不会把训练 schedule 改成 4 步。保留旧 `snr_gamma` 参数以兼容配置，它不改变此 flow-matching 损失。distilled klein 不使用 CFG 或 guidance embedding；其 guidance 默认值为 1。

## 检查点、恢复与推理

普通 LoRA 的 `pytorch_model*.bin` 仅包含 adapters；E2E 的文件包含完整 `transformer.*` 和显式解冻的编码组件。冻结文本/VAE 从同一官方底座复用。旧的完整 UniTorch 模型检查点仍可读取。

原生 adapter keys 形如 `transformer.transformer_blocks.0.attn.to_q.lora_A.default.weight` / `lora_B.default.weight`。它们是 UniTorch native checkpoint，不是自动带 `adapter_config.json` 的 Diffusers LoRA 发布目录。恢复前先校验所有必需的训练组件/adapters keys 和完整 tensor shape（ZeRO 参数使用 `ds_shape`）；校验失败不会先恢复部分权重。普通基础权重的逐 shard 加载仍保持宽松。加载时必须保持 `lora_r`、`lora_alpha`、targets 和底座一致；底座 linear keys 会根据实际 adapter 的 `base_layer` 自动匹配，也支持自定义 targets。Mistral3 的旧/新 checkpoint key 布局都支持。

普通 LoRA 恢复：使用同一个 `cache_dir`，增加 `epochs`，latest 模型、optimizer 和 `info.json` 会自动恢复。例如：

```bash
"$FLUX2_BIN/unitorch-train" examples/configs/diffusion/text2image/flux2.klein-4b.lora.ini \
  --core/task/supervised@epochs=2 \
  --cache_dir="$FLUX2_RUN_ROOT/flux2.klein-4b.text2image.lora" \
  --train_file="$FLUX2_DATA_ROOT/text2image.train.tsv" --dev_file="$FLUX2_DATA_ROOT/text2image.dev.tsv"

"$FLUX2_BIN/unitorch-infer" examples/configs/diffusion/text2image/flux2.klein-4b.lora.ini \
  --from_ckpt_dir="$FLUX2_RUN_ROOT/flux2.klein-4b.text2image.lora" \
  --test_file="$FLUX2_DATA_ROOT/text2image.test.tsv" --cache_dir="$FLUX2_RUN_ROOT/infer-klein-lora"
```

`unitorch-infer` 默认读取 `pytorch_model.bin`（best）；训练使用 `save_checkpoint=latest` 时先将 latest 导出/复制到单独目录的 `pytorch_model.bin`，再用该目录作为 `from_ckpt_dir`。`from_ckpt_dir` 只初始化权重，已有 `cache_dir` 的 latest 状态优先；开始新实验要使用新目录。

六个 DeepSpeed 配置使用 `examples/configs/deepspeed/flux2.zero3.cpu.json`、`zero_init=True`：模型在构造时分区；参数和 AdamW 状态 offload 到 CPU。`merge_zero3_checkpoint=False` 避免训练保存时构造完整 FP32 模型。原生模型与 optimizer shards 保存在 `pytorch_model_latest/checkpoint/`，同一 latest tag 被覆盖，冻结底座不重复写入。保留整个目录及根目录 `info.json`。同一个训练命令追加 `--core/task/deepspeed/supervised@epochs=2` 即可恢复。

GPU 配置使用 DeepSpeed CPUAdam；首次运行可能需要编译其扩展，构建缓存放在上述数据盘目录。CPU/Gloo 自动化测试使用真实 PyTorch AdamW，已覆盖单进程恢复与双进程分片；这不代替 CUDA/NCCL 和 CPUAdam 扩展的官方权重实测。

DeepSpeed task 的输入、标签和验证聚合跟随 accelerator/engine 的执行设备。选择 CPU/Gloo 时，即使宿主机 CUDA 可见也使用 CPU；CUDA/NCCL 的 CPU 参数 offload 仍以 GPU 为执行设备。

将原生 ZeRO-3 latest 导出为推理文件，使用仓库自带的 CLI（只读取自己信任的原生检查点）：

```bash
"$FLUX2_BIN/python" -m unitorch.cli.scripts.flux2 \
  --checkpoint="$FLUX2_RUN_ROOT/flux2.dev.text2image.lora/pytorch_model_latest" \
  --output="$FLUX2_RUN_ROOT/exports/dev-text2image-lora" \
  --mode=lora --dtype=float32

"$FLUX2_BIN/unitorch-infer" examples/configs/diffusion/text2image/flux2.dev.lora.ini \
  --task_name=core/task/supervised \
  --from_ckpt_dir="$FLUX2_RUN_ROOT/exports/dev-text2image-lora" \
  --test_file="$FLUX2_DATA_ROOT/text2image.test.tsv" --cache_dir="$FLUX2_RUN_ROOT/infer-dev-lora" \
  --core/task/supervised@cpu_offload=True \
  --core/model/diffusers/peft/lora/text2image/flux2@enable_cpu_offload=True
```

E2E 导出改为 `--mode=e2e --dtype=bfloat16`，推理使用对应 E2E 配置及 `core/model/diffusers/<task>/flux2@enable_cpu_offload=True`。导出器采用 DeepSpeed lazy/mmap 读取，逐 tensor 转换，仅保存训练组件或 adapters；E2E 包括显式解冻的 text、VAE encoder/quant_conv。内存预检包含最终 tensors、最大 FP32 gather 临时空间、模型元数据和默认 16 GiB 余量；`--max_memory_bytes` 可进一步限制预算。磁盘需额外保留 1 GiB。输出目录必须没有 `pytorch_model.bin`，写完临时文件后原子发布，不覆盖已有文件。它不支持 ZeRO-2 的 eager 合并。原生 optimizer shards 用于恢复，导出 `.bin` 用于推理，不能互相代替。

顺序 CPU offload 是**推理功能**；先加载底座和训练检查点，再加 hooks。需要同时设置 task 的 `cpu_offload=True`（避免初始化时整体上 GPU）和 model 的 `enable_cpu_offload=True`。训练后直接启用这些 hooks 的模型不应再用于优化；重新从 checkpoint 构造训练模型。

## 内存与验证边界

以下是官方配置推导的容量估计，不是 GPU 实测峰值。除 transformer 外必须计算文本编码器的内存。

| 模型 | transformer 参数量 / BF16 | 文本编码器 BF16 | 含 VAE 的总 BF16 权重 |
|---|---|---|---|
| dev | 32,223,281,152 / 60.02 GiB | 43.47 GiB | 103.65 GiB |
| klein | 3,875,544,576 / 7.22 GiB | 7.49 GiB | 14.87 GiB |

双卡 RTX A6000 48 GiB 的 96 GiB 总显存小于 dev 的全部 BF16 权重，因此需要分片和 offload。AdamW 的 FP32 master weights 与两个动量共约 12 bytes/可训练参数：dev E2E 约 **360 GiB**，再加底座、梯度和临时缓冲。可用主存约 160 GiB 的机器不满足本 CPU offload 配置。NVMe offload、不同优化器或更大机器需要另行验证，不能据此宣称 32B E2E 已成功。

最小 smoke 的保守预检阈值：klein LoRA 24 GiB 可用 RAM；klein E2E 96 GiB；dev LoRA 140 GiB。dev E2E 的实际 CPUAdam / ZeRO 验证配置要求至少 **540 GiB 可用 RAM 和 560 GiB 空闲磁盘**；验证主机当时约有 160 GiB 可用 RAM、385 GiB 空闲磁盘，资源预检未通过，未启动训练。e2e 的 native optimizer checkpoint 也可能达数百 GiB，必须检查数据盘空间。示例只保留 latest，避免同时写 best/latest 两份大状态；不要依赖已接近满的 swap。

以下官方权重验证使用上文列出的确切 repo/revision，均为 **64×64 合成数据、共 2 个 optimizer steps 的执行 smoke**，不代表收敛或图像质量评估。实现支持全部 8 个组合；官方权重验证状态如下，不能宣称全部 8 个组合通过。

| 官方检查点 | 微调方式 | 文生图 | 图像编辑 | 验证范围 |
|---|---|---|---|---|
| `black-forest-labs/FLUX.2-klein-4B` | LoRA | PASSED | PASSED | 实际官方权重 GPU train/save/resume（optimizer step 1 → 2）/export/infer |
| `black-forest-labs/FLUX.2-klein-4B` | 全 transformer | PASSED | PASSED | 实际官方权重 GPU train/save/resume（optimizer step 1 → 2）/export/infer |
| `black-forest-labs/FLUX.2-dev` | LoRA | PASSED | PASSED | 实际官方权重 GPU train/save/resume（optimizer step 1 → 2）/export/infer |
| `black-forest-labs/FLUX.2-dev` | 全 transformer | NOT VERIFIED | NOT VERIFIED | CPUAdam / ZeRO 资源预检未通过，未启动训练 |

dev 全 transformer 的未验证原因是可用 RAM 与磁盘不足，不是 GPU、驱动或 gated 权重访问失败。parent 完整测试套件已完成：249 passed、0 failed、0 skipped；parent 增量测试已完成：58 passed、0 failed。两个套件有重叠，计数不相加。

验证证据说明：`/data/decu/flux2-finetune/scripts/official_smoke.py` 是本次任务专用的验证 harness，不是公开训练或导出的必需依赖。用户训练、恢复、导出和推理使用上文的仓库入口与可移植命令；原生 ZeRO 导出由 `python -m unitorch.cli.scripts.flux2` 提供。

CPU 上真实 tiny Transformer/Mistral3 或 Qwen3/VAE 的优化、CLI train/resume/infer 和 ZeRO/Gloo 测试属于独立验证层级，不能证明官方大权重训练成功。CUDA 不可见的沙箱探测也不能证明宿主机驱动故障。实机验证应逐组合记录官方 repo/revision、资源预检、训练、optimizer step 恢复以及导出检查点推理结果；资源不足的组合标记为未验证。

```bash
# 真实 tiny 回归；只运行 CPU checkpoint 修正测试时可使用第二条命令。
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$FLUX2_BIN/python" -m pytest tests/cli/test_flux2_finetune.py -q
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$FLUX2_BIN/python" -m pytest tests/cli/test_flux2_checkpoint_review.py -q
```

CPU checkpoint 回归用真实双进程 DeepSpeed/Gloo 训练、原生恢复并导出，逐 tensor 比较 FP32/BF16 导出值与 gather 参数。合成图像只用于数值与 IO smoke，不能用作生成质量或收敛评估结果。
