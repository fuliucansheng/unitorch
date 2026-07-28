---
name: unitorch-config-ini
description: Guidance for writing, reviewing, and updating unitorch INI configuration files for train, eval, infer, and FastAPI workflows. Use when creating or modifying cli/configs/*.ini, reasoning about Config interpolation and CLI overrides, composing preprocess_functions, choosing registered component names, or debugging unitorch CLI config behavior.
---

# unitorch Config INI Writing Guide

## Overview

All unitorch CLI commands consume a single `.ini` config file. The config uses
Python's `configparser` extended-interpolation syntax. Values are auto-parsed
from strings into Python types such as `int`, `float`, `bool`, `list`, and
`dict` through a safe AST evaluator.

## CLI Parameter Override

Any `[section]` key can be overridden from the command line without editing the
file:

```bash
# Override a key in [core/cli]
unitorch-train config.ini --from_ckpt_dir=/new/path

# Override a key in another section. Use @ as the section/key separator.
unitorch-train config.ini --"core/model/generation/qwen3@pretrained_name"=qwen3-8b
```

## Cross-section Interpolation

```ini
[core/cli]
cache_dir = ./cache

[core/task/supervised]
output_path = ${core/cli:cache_dir}/output.txt
```

## Common `[core/cli]` Keys

| Key | Used by | Description |
|-----|---------|-------------|
| `task_name` | train / eval / infer | Registered task name. Required. |
| `enabled_services` | fastapi | List of registered FastAPI service names. Required. |
| `device` | fastapi | Use `cpu` for generated FastAPI configs. Train, eval, and infer configs do not require this key. |
| `from_ckpt_dir` | train / eval / infer | Checkpoint input directory. |
| `cache_dir` | train / eval / infer | Output / cache directory. |
| `train_file` / `dev_file` / `test_file` | train / eval / infer | Data file paths. |
| `local_rank` | train (DDP) | Set automatically by torchrun; rarely hardcoded. |
| `depends_libraries` | all | Extra Python libraries to import before task init. |
| `host` / `port` | fastapi | Server bind address. Defaults are `0.0.0.0` and `5000`. |
| `wandb/team` / `wandb/project` / `wandb/token` | train | Optional W&B integration. |

## `preprocess_functions` Syntax

`preprocess_functions` is a Python list of call-expression strings. In each
string, the full text before the outer call parentheses is the registered
process function name, and the parentheses contain column names, literals, or
nested registered process calls.

### Rules

- Each string calls one registered process function:
  `'core/process/registered/name(arg1, arg2)'`.
- The whole slash-separated prefix is the registered name. For example,
  `core/process/foo/generation`, `core/process/foo/generation/inputs`, and
  `core/process/foo/generation/labels` are three separate registered process
  function names, not a namespace plus a mode selector.
- Arguments are column names from `names`, nested calls to other registered
  process functions, or inline Python literals such as lists, dicts, and
  `.format()` expressions.
- Column names are treated as local variables inside the call expression. They
  can be used directly or embedded inside string literals with
  `"{0}".format(col)`.
- Nesting is arbitrary. The inner call's return value is passed to the outer
  call.
- Multiple functions in the list are applied independently and their outputs
  are merged into the batch dict.
- Multi-line list syntax with a trailing comma is valid INI.
- For complex structured arguments such as message arrays, use a
  triple-quoted string (`'''...'''`) as the list element so the call expression
  can span multiple lines without escaping.

### Examples

```ini
# Single function, two columns.
preprocess_functions = ['core/process/qwen_vl/generation/inputs(encode, image)']

# Nested call: read image from path first, then pass the PIL image.
preprocess_functions = ['core/process/clip/classification(text, core/process/image/read(image))']

# Two independent functions merged into one batch dict.
preprocess_functions = [
    'core/process/foo/generation/inputs(encode)',
    'core/process/foo/generation/labels(decode)',
  ]

# Nested registered process calls compose like func_a(func_b(col)).
preprocess_functions = [
    'core/process/foo/generation/inputs(text, core/process/image/read(image))',
  ]

# Vision encoder plus label encoder.
preprocess_functions = [
    'core/process/clip/image_classification(core/process/image/read(image))',
    'core/process/label(label)',
  ]

# Structured message array, with column refs via .format().
preprocess_functions = [
    '''
    core/process/qwen/messages/grpo/generation(
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "Question: {0}".format(encode)}]
            }
        ],
        [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "{0}".format(decode)}]
            }
        ]
    )
    '''
  ]
```

### Typical split patterns

| Split | Common pattern |
|-------|----------------|
| `train` | Single call covering all columns, or nested image read. |
| `dev` | Separate `inputs` and `labels` calls for metric computation. |
| `test` | `inputs` only. `names` should drop label columns when labels are absent. |

## Commands to `task_name` / Entry Key Mapping

| CLI command | Entry key in `[core/cli]` | Valid values |
|-------------|---------------------------|--------------|
| `unitorch-train` | `task_name` | `core/task/supervised`, `core/task/deepspeed/supervised`, `core/task/megatron/supervised` |
| `unitorch-eval` | `task_name` | Same as train. |
| `unitorch-infer` | `task_name` | Same as train. Calls `.infer()`. |
| `unitorch-fastapi` | `enabled_services` | List of registered `core/fastapi/*` names. |

## `unitorch-train` / `unitorch-eval` / `unitorch-infer`

These three commands all use `task_name = core/task/supervised` or a deepspeed
/ megatron variant, then call `.train()`, `.eval()`, or `.infer()` on the task
object.

### Skeleton

```ini
[core/cli]
task_name = core/task/supervised
from_ckpt_dir = ./cache
cache_dir     = ./cache
train_file    = ./train.tsv
dev_file      = ./dev.tsv
test_file     = ./test.tsv

# Model.
[core/model/<task>/<name>]
pretrained_name = <model-id>
# model-specific params...

# Dataset.
[core/dataset/ast]
names = ['col1', 'col2']

[core/dataset/ast/train]
data_files = ${core/cli:train_file}
preprocess_functions = ['core/process/foo/generation(col1, col2)']

[core/dataset/ast/dev]
data_files = ${core/cli:dev_file}
preprocess_functions = [
    'core/process/foo/generation/inputs(col1)',
    'core/process/foo/generation/labels(col2)',
  ]

[core/dataset/ast/test]
names = ['col1']
data_files = ${core/cli:test_file}
preprocess_functions = ['core/process/foo/generation/inputs(col1)']

# Processor.
[core/process/<name>]
pretrained_name = <model-id>
max_seq_length     = 512
max_gen_seq_length = 512

# Output writer.
[core/writer/csv]
escapechar = \

# Optimizer / scheduler. Used by train.
[core/optim/adamw]
learning_rate = 0.0001

[core/scheduler/linear_warmup]
num_warmup_rate = 0.001

# Task.
[core/task/supervised]
model      = core/model/<task>/<name>
dataset    = core/dataset/ast

# Train params.
optim      = core/optim/adamw
scheduler  = core/scheduler/linear_warmup
loss_fn    = core/loss/lm
score_fn   = core/score/bleu
monitor_fns = ['core/score/bleu', 'core/score/rouge1', 'core/score/rouge2', 'core/score/rougel']
from_ckpt_dir   = ${core/cli:from_ckpt_dir}
to_ckpt_dir     = ${core/cli:cache_dir}
train_batch_size = 4
dev_batch_size   = 8
epochs           = 5

# Infer params.
output_header  = ['col1']
postprocess_fn = core/postprocess/qwen/detokenize
writer         = core/writer/csv
output_path    = ${core/cli:cache_dir}/output.txt
test_batch_size = 8
```

### Notes

- `[core/task/supervised]` section name must match `task_name`.
- For `unitorch-infer`, `optim`, `loss_fn`, and `score_fn` are ignored but do
  no harm if present.
- `from_ckpt_dir` under `[core/task/supervised]` controls which checkpoint is
  loaded at infer time. If the directory does not exist, it is silently skipped.
- `postprocess_fn`, `preprocess_functions`, `model`, and `enabled_services`
  must all be registered names. Query the current registry when uncertain:

```bash
unitorch-copilot-cli core/copilot/pkg_infos
unitorch-copilot-cli core/copilot/pkg_infos --name model
unitorch-copilot-cli core/copilot/pkg_infos --name process
unitorch-copilot-cli core/copilot/pkg_infos --name fastapi
```

Valid `--name` values include `process`, `copilot_tool`, `model`, `fastapi`,
`score`, `dataset`, `loss`, `optimizer`, `scheduler`, `task`, and `writer`.

## `unitorch-fastapi`

Always set `device = cpu` in `[core/cli]` when generating FastAPI configs.

```ini
[core/cli]
enabled_services = ['core/fastapi/<name1>', 'core/fastapi/<name2>']
device = cpu
host   = 0.0.0.0
port   = 5000

[core/fastapi/<name1>]
pretrained_name = <model-id>
router = /core/fastapi/<name1>

[core/fastapi/<name2>]
pretrained_name = <model-id>
```

- `enabled_services` is a Python list of registered `core/fastapi/*` names.
- Each service gets its own `[core/fastapi/<name>]` section.
- `host` and `port` live directly in `[core/cli]`.
- `router` is the URL prefix. The default is the section name.

## Key Rules and Gotchas

1. Generate `device = cpu` for `unitorch-fastapi` configs. Train, eval, and
   infer configs do not require this key.
2. `task_name` section name must match. `task_name = core/task/supervised`
   requires a `[core/task/supervised]` section, not `[core/task/infer]`.
3. Do not use HuggingFace AutoClass APIs. Model and processor classes must be
   explicit concrete classes.
4. vLLM inference uses `core/task/supervised` plus `unitorch-infer`; vLLM model
   sections live under `core/model/vllm/generation/<name>`.
5. `[core/dataset/ast/test]` must override `names` if the test file has fewer
   columns than train, such as no `decode` column for inference-only input.
6. Interpolation syntax is `${section:key}` with a colon.
7. List values use Python list syntax such as `['a', 'b']`. Dict values use
   `{}`.
8. Comments use `;` or `#`.
