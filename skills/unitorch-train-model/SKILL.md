---
name: unitorch-train-model
description: Guidance for creating, reviewing, running, and debugging unitorch model training workflows with unitorch-train. Use when preparing training INI configs, choosing registered models/processors/losses/scores, launching single-GPU or distributed training, checking checkpoint loading, monitoring loss and validation metrics, or diagnosing unstable training.
---

# Train unitorch Models

## Overview

Use this skill to plan and verify a `unitorch-train` workflow end to end:
choose registered components, write or review the INI config, launch training,
and inspect whether checkpoints, loss, and validation metrics behave correctly.
When editing detailed INI syntax, also use the `config-ini` skill.

## Training Workflow

1. Start from the closest example in `cli/configs/`.
2. Query the current registries before inventing section names.
3. Wire raw data through `preprocess_functions`, `collate_fn`, model forward,
   loss, and metrics.
4. Run a small single-process smoke test before distributed training.
5. Verify checkpoint loading and resume behavior.
6. Monitor `train/loss`, `score_fn`, and `monitor_fns`.
7. Scale to `torchrun` only after the small run is correct.

## Discover Registered Components

Use `unitorch-copilot-cli core/copilot/pkg_infos` to inspect the live registry.
This is more reliable than guessing from filenames because extension packages
can add registrations.

```bash
# List every registry type.
unitorch-copilot-cli core/copilot/pkg_infos

# Check registered training model sections.
unitorch-copilot-cli core/copilot/pkg_infos --name model

# Check registered preprocess and postprocess functions.
unitorch-copilot-cli core/copilot/pkg_infos --name process

# Check metrics, losses, tasks, optimizers, and schedulers.
unitorch-copilot-cli core/copilot/pkg_infos --name score
unitorch-copilot-cli core/copilot/pkg_infos --name loss
unitorch-copilot-cli core/copilot/pkg_infos --name task
unitorch-copilot-cli core/copilot/pkg_infos --name optimizer
unitorch-copilot-cli core/copilot/pkg_infos --name scheduler
```

If a model, processor, or score comes from an extension package, make sure its
Python module is imported before config initialization. Use
`depends_libraries` in `[core/cli]` when the extension needs an explicit import.

## Build The Training Config

Use these sections for a standard supervised task:

```ini
[core/cli]
task_name = core/task/supervised
from_ckpt_dir = ./cache/from
cache_dir = ./cache/run
train_file = ./train.tsv
dev_file = ./dev.tsv

[core/model/<task>/<name>]
pretrained_name = <model-id-or-local-path>

[core/process/<name>]
pretrained_name = <model-id-or-local-path>

[core/dataset/ast]
names = ['encode', 'decode']

[core/dataset/ast/train]
data_files = ${core/cli:train_file}
preprocess_functions = ['core/process/<name>/generation(encode, decode)']

[core/dataset/ast/dev]
data_files = ${core/cli:dev_file}
preprocess_functions = [
    'core/process/<name>/generation/inputs(encode)',
    'core/process/<name>/generation/labels(decode)',
  ]

[core/optim/adamw]
learning_rate = 0.0001

[core/scheduler/linear_warmup]
num_warmup_rate = 0.001

[core/task/supervised]
model = core/model/<task>/<name>
dataset = core/dataset/ast
optim = core/optim/adamw
scheduler = core/scheduler/linear_warmup
loss_fn = core/loss/lm
score_fn = core/score/bleu
monitor_fns = ['core/score/bleu', 'core/score/rouge1', 'core/score/rouge2', 'core/score/rougel']
from_ckpt_dir = ${core/cli:from_ckpt_dir}
to_ckpt_dir = ${core/cli:cache_dir}
train_batch_size = 4
dev_batch_size = 8
num_workers = 4
epochs = 5
log_freq = 100
ckpt_freq = 10000
grad_acc_step = 1
max_grad_norm = 1.0
save_checkpoint = default
use_amp = True
```

In `preprocess_functions`, the full text before the outer call parentheses is
the registered process function name. For example,
`core/process/<name>/generation`, `core/process/<name>/generation/inputs`, and
`core/process/<name>/generation/labels` are three separate registered process
functions, not a namespace plus a sub-mode. Query `--name process` when unsure.

`preprocess_functions` can also compose registered process functions with nested
calls. The inner return value is passed to the outer registered function, like
`func_a(func_b(col))`:

```ini
preprocess_functions = [
    'core/process/<name>/generation/inputs(encode, core/process/image/read(image))',
    'core/process/<name>/generation/labels(decode)',
  ]
```

Do not add `device = cpu` for train configs unless there is a very specific
reason. That key is required by generated FastAPI configs, not normal training.

## Data Flow Checks

Training must follow this contract:

```text
raw data -> preprocess -> tensor(s) -> collate_fn -> batched tensor(s)
         -> model forward -> tensor(s) -> loss compute -> scalar loss
```

Check these points before launching a long run:

- `names` must match the columns in the input file.
- Train preprocessing must produce both model inputs and labels/targets.
- Dev preprocessing must produce outputs that `score_fn` and `monitor_fns` can
  compare against targets.
- Treat each `preprocess_functions` item as a call to one registered process
  function. For names such as `core/process/foo/generation/inputs`, the whole
  slash-separated string is the registered name.
- Compose preprocessors by nesting registered function calls when a raw column
  needs another transform first, such as reading an image before tokenization.
- Image/video configs should use registered read/process functions rather than
  passing raw file paths to a model that expects tensors.
- For generation tasks, use separate `inputs` and `labels` preprocessors on dev
  when metrics need decoded targets.
- If a new model wrapper is involved, keep concrete class names explicit. Do not
  introduce HuggingFace AutoClass usage.

## Launch Commands

Run a single-process smoke test first:

```bash
unitorch-train cli/configs/generation/bart.ini \
  --cache_dir=/tmp/unitorch-train-smoke \
  --"core/task/supervised@epochs"=1 \
  --"core/task/supervised@log_freq"=1 \
  --"core/task/supervised@ckpt_freq"=10 \
  --"core/task/supervised@num_workers"=0
```

Use `@` to override keys outside `[core/cli]`:

```bash
unitorch-train config.ini --from_ckpt_dir=/path/to/init
unitorch-train config.ini --"core/model/generation/qwen3@pretrained_name"=/path/to/model
unitorch-train config.ini --"core/task/supervised@train_batch_size"=1
```

For distributed training with the standard CLI:

```bash
torchrun --no_python --nproc_per_node 4 unitorch-train config.ini
```

For deepspeed or megatron configs, set `task_name` to the registered variant
and keep the section name aligned:

```ini
[core/cli]
task_name = core/task/deepspeed/supervised

[core/task/deepspeed/supervised]
...
```

## Distributed Training Notes

- `torchrun` sets `LOCAL_RANK`; do not hardcode `local_rank` in the config.
- `train_batch_size` is per process/GPU. Effective batch size is roughly
  `train_batch_size * nproc_per_node * grad_acc_step`.
- `dev_batch_size` is also per process/GPU.
- `num_workers` is per process. Four GPUs with `num_workers = 4` create sixteen
  DataLoader workers.
- Rank 0 writes checkpoints; other ranks should not be expected to write files.
- Use a shared filesystem path for `to_ckpt_dir` when running across processes.
- If a single-GPU smoke test fails, distributed training will usually fail with
  noisier stack traces. Fix the single-process run first.

## Checkpoint Loading And Resume

Understand the load order in `core/task/supervised`:

1. If `from_ckpt_dir` exists, model weights and `pytorch_optim.bin` are loaded
   from that directory.
2. If `to_ckpt_dir` exists, `pytorch_model_latest.bin` and
   `pytorch_optim_latest.bin` are loaded from that directory afterward.
3. If `info.json` exists in `to_ckpt_dir`, `best_score`, `global_epoch`, and
   `global_step` are restored.

This means a stale `to_ckpt_dir` can override the initialization from
`from_ckpt_dir`. For a fresh run, use a new `cache_dir`/`to_ckpt_dir` or remove
old latest checkpoints first.

After startup, check logs for the restored best score:

```text
best score so far: ...
```

After the first checkpoint, inspect the output directory:

```text
pytorch_model.bin              # best checkpoint when score improves
pytorch_model_latest.bin       # latest checkpoint
pytorch_optim.bin              # optimizer for best checkpoint when enabled
pytorch_optim_latest.bin       # latest optimizer when enabled
pytorch_scheduler_latest.bin   # latest scheduler when enabled
info.json                      # best_score/global_epoch/global_step
```

If you expect pretrained weights to load but the model behaves randomly:

- Confirm `from_ckpt_dir` exists from the training process working directory.
- Confirm the expected weight filename is present.
- Confirm `to_ckpt_dir` is not silently resuming another run.
- Run with a tiny dataset and compare loss before and after one optimizer step.
- Use `UNITORCH_DEBUG=DETAIL` when parameter/device logs are needed.

## Loss Monitoring

`train/loss` is logged every `log_freq` steps on rank 0:

```text
epoch 0 step 100: train/loss=...
```

Expect noisy but generally improving loss over enough updates. Do not require
strict monotonic decrease at every log line.

When loss is not normal:

- If loss is `nan` or `inf`, reduce learning rate, disable AMP with
  `use_amp = False`, reduce batch size, or inspect labels for invalid values.
- If loss is flat, verify labels are produced by preprocessing and consumed by
  the configured `loss_fn`.
- If loss is exactly zero from the start, check whether targets are empty,
  masked out, or already identical to model outputs.
- If loss decreases on train but metrics do not improve, inspect dev
  preprocessing and metric target format.
- If loss improves only with `grad_acc_step = 1`, verify effective batch size
  and learning rate scaling.

For debugging, set:

```ini
[core/task/supervised]
log_freq = 1
ckpt_freq = 10
num_workers = 0
epochs = 1
train_batch_size = 1
dev_batch_size = 1
```

## Validation Metrics And Checkpoint Selection

`score_fn` is the primary metric used to decide whether the best checkpoint
improves. `monitor_fns` are extra metrics logged during snapshots. Snapshots run
every `ckpt_freq` steps and at the end of each epoch.

Use classification metrics for classification:

```ini
score_fn = core/score/acc
monitor_fns = ['core/score/acc']
```

Use generation metrics for text generation:

```ini
score_fn = core/score/bleu
monitor_fns = ['core/score/bleu', 'core/score/rouge1', 'core/score/rouge2', 'core/score/rougel']
```

Before setting metrics, query the score registry:

```bash
unitorch-copilot-cli core/copilot/pkg_infos --name score
```

Important details:

- `score_fn` must be a registered score and should match the task objective.
- `monitor_fns` entries that are not registered are filtered out, so misspelled
  monitor names may simply not log.
- Dev data must include targets compatible with every metric.
- Best checkpoint comparison assumes higher score is better.
- If metric computation is slow, increase `ckpt_freq` or reduce dev size during
  smoke tests.

## Common Failure Patterns

- Wrong registry name: query `model`, `process`, `loss`, and `score` before
  writing section names.
- Section mismatch: `task_name = core/task/supervised` requires a matching
  `[core/task/supervised]` section.
- Accidental resume: old `to_ckpt_dir` latest files override `from_ckpt_dir`.
- No labels on dev: metrics cannot validate generation/classification quality.
- Per-GPU batch too high: distributed launch multiplies memory usage by rank.
- Too many workers: `num_workers` multiplies by process count.
- Silent monitor omission: invalid `monitor_fns` names are skipped.
- Incompatible preprocessing: train/dev preprocessors must return data shapes
  expected by the model, loss, and metrics.

## Completion Checklist

Before considering a training setup ready:

- Registry names were checked with `unitorch-copilot-cli`.
- The config was based on a nearby `cli/configs/` file.
- A single-process smoke run completed.
- `train/loss` logged and was finite.
- The first snapshot wrote expected checkpoint files.
- `info.json` contains expected progress and best score.
- `score_fn` and `monitor_fns` logged on dev data.
- `from_ckpt_dir` and `to_ckpt_dir` behavior is intentional.
- Distributed effective batch size and worker count are understood.
