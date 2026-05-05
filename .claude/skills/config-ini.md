# unitorch Config INI Writing Guide

## Overview

All unitorch CLI commands consume a single `.ini` config file. The config uses
Python's `configparser` extended-interpolation syntax. Values are auto-parsed
from strings into Python types (int, float, bool, list, dict) via a safe AST
evaluator.

---

## CLI Parameter Override

Any `[section]` key can be overridden from the command line without editing the
file:

```bash
# Override a key in [core/cli]
unitorch-train config.ini --from_ckpt_dir=/new/path

# Override a key in another section  (use @ as section/key separator)
unitorch-train config.ini --"core/model/generation/qwen3@pretrained_name"=qwen3-8b
```

---

## Cross-section Interpolation

```ini
[core/cli]
cache_dir = ./cache

[core/task/supervised]
output_path = ${core/cli:cache_dir}/output.txt   # resolves to ./cache/output.txt
```

---

## Common `[core/cli]` Keys

| Key | Used by | Description |
|-----|---------|-------------|
| `task_name` | train / eval / infer | Registered task name (required) |
| `enabled_services` | fastapi | List of registered FastAPI service names (required) |
| `service_name` | service | Registered service name (required) |
| `script_name` | launch | Registered script name (required) |
| `device` | all | `cpu` — **always default to `cpu`** |
| `from_ckpt_dir` | train / eval / infer | Checkpoint input directory |
| `cache_dir` | train / eval / infer | Output / cache directory |
| `train_file` / `dev_file` / `test_file` | train / eval / infer | Data file paths |
| `local_rank` | train (DDP) | Set automatically by torchrun; rarely hardcoded |
| `depends_libraries` | all | Extra Python libraries to import before task init |
| `host` / `port` | fastapi | Server bind address (default `0.0.0.0` / `5000`) |
| `daemon_mode` | service | Run service as background daemon (default `True`) |
| `wandb/team` `wandb/project` `wandb/token` | train | W&B integration (optional) |

---

## `preprocess_functions` Syntax

`preprocess_functions` is a Python list of **call-expression strings**. Each
string is a registered process function called with column names or nested
calls as arguments — exactly like Python function calls.

### Rules

- Each string calls one registered process function:
  `'core/process/registered/name(arg1, arg2)'`
- Arguments are **column names** from `names`, or **nested calls** to other
  registered process functions.
- Nesting is arbitrary: the inner call's return value is passed as an argument
  to the outer call.
- Multiple functions in the list are applied independently and their outputs
  are merged into the batch dict.
- Multi-line list syntax (with a trailing comma) is valid INI.

### Examples

```ini
# Single function, two columns
preprocess_functions = ['core/process/qwen_vl/generation/inputs(encode, image)']

# Nested call: read image from path first, then pass PIL image to processor
preprocess_functions = ['core/process/clip/classification(text, core/process/image/read(image))']

# Two independent functions merged (inputs + labels)
preprocess_functions = [
    'core/process/foo/generation/inputs(encode)',
    'core/process/foo/generation/labels(decode)',
  ]

# Two independent functions: vision encoder + label encoder
preprocess_functions = [
    'core/process/clip/image_classification(core/process/image/read(image))',
    'core/process/label(label)',
  ]
```

### Typical split patterns

| Split | Common pattern |
|-------|---------------|
| `train` | single call covering all columns, or nested image read |
| `dev` | separate `inputs` + `labels` calls (for metric computation) |
| `test` | `inputs`-only call; `names` should also drop label columns |

---

## Commands → `task_name` / entry key mapping

| CLI command | Entry key in `[core/cli]` | Valid values |
|-------------|--------------------------|--------------|
| `unitorch-train` | `task_name` | `core/task/supervised` · `core/task/deepspeed/supervised` · `core/task/megatron/supervised` |
| `unitorch-eval` | `task_name` | same as train |
| `unitorch-infer` | `task_name` | same as train (calls `.infer()`) |
| `unitorch-fastapi` | `enabled_services` | list of registered `core/fastapi/*` names |
| `unitorch-service` | `service_name` | registered `core/service/*` name |
| `unitorch-launch` | `script_name` | registered `core/script/*` name |

---

## `unitorch-train` / `unitorch-eval` / `unitorch-infer`

These three commands all use `task_name = core/task/supervised` (or deepspeed /
megatron variant) and call `.train()`, `.eval()`, or `.infer()` on the task
object respectively.

### Skeleton

```ini
[core/cli]
task_name = core/task/supervised
device    = cpu
from_ckpt_dir = ./cache
cache_dir     = ./cache
train_file    = ./train.tsv
dev_file      = ./dev.tsv
test_file     = ./test.tsv

# ── model ──────────────────────────────────────────────────────────────────
[core/model/<task>/<name>]
pretrained_name = <model-id>
# model-specific params …

# ── dataset ────────────────────────────────────────────────────────────────
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

# ── processor ──────────────────────────────────────────────────────────────
[core/process/<name>]
pretrained_name = <model-id>
max_seq_length     = 512
max_gen_seq_length = 512

# ── output writer ──────────────────────────────────────────────────────────
[core/writer/csv]
escapechar = \

# ── optimizer / scheduler (train only) ────────────────────────────────────
[core/optim/adamw]
learning_rate = 0.0001

[core/scheduler/linear_warmup]
num_warmup_rate = 0.001

# ── task ───────────────────────────────────────────────────────────────────
[core/task/supervised]
model      = core/model/<task>/<name>
dataset    = core/dataset/ast

# train params
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

# infer params
output_header  = ['col1']
postprocess_fn = core/postprocess/qwen/detokenize   # must be a registered postprocess name
writer         = core/writer/csv
output_path    = ${core/cli:cache_dir}/output.txt
test_batch_size = 8
```

### Notes
- `[core/task/supervised]` section name **must** match `task_name`.
- For `unitorch-infer`, the `optim` / `loss_fn` / `score_fn` keys are ignored
  (only `.infer()` is called), but they do no harm if present.
- For vLLM models (`core/model/vllm/generation/*`), `device = cpu` in
  `[core/cli]` is correct — vLLM manages its own GPU allocation.
- `from_ckpt_dir` under `[core/task/supervised]` controls which checkpoint is
  loaded at infer time; if the directory doesn't exist it is silently skipped.
- `postprocess_fn`, `preprocess_functions`, `model`, `enabled_services`,
  `service_name`, `script_name` must all be **registered** names. The registry
  is dynamic (grows as new models are added). Query the current list:

```bash
unitorch-copilot-cli core/copilot/pkg_infos                  # list all types
unitorch-copilot-cli core/copilot/pkg_infos --name model      # models only
unitorch-copilot-cli core/copilot/pkg_infos --name process    # preprocess/postprocess only
unitorch-copilot-cli core/copilot/pkg_infos --name fastapi    # fastapi services only
# available --name values: process, copilot_tool, model, fastapi, service,
#                          script, score, dataset, loss, optimizer, scheduler, task, writer
```

---

## `unitorch-fastapi`

```ini
[core/cli]
enabled_services = ['core/fastapi/<name1>', 'core/fastapi/<name2>']
device = cpu
host   = 0.0.0.0
port   = 5000

[core/fastapi/<name1>]
# service-specific params, e.g.:
pretrained_name = <model-id>
router = /core/fastapi/<name1>   # URL prefix (default = section name)

[core/fastapi/<name2>]
pretrained_name = <model-id>
```

- `enabled_services` is a Python list of registered `core/fastapi/*` names.
- Each service gets its own `[core/fastapi/<name>]` section.
- `host` and `port` live directly in `[core/cli]`.

---

## `unitorch-service`

```bash
unitorch-service start  config.ini
unitorch-service stop   config.ini
unitorch-service restart config.ini
```

```ini
[core/cli]
service_name = core/service/http_files   # use a registered core/service/* name
daemon_mode  = True                      # False = foreground

[core/service/http_files]
port     = 11220
html_dir = /path/to/static
```

---

## `unitorch-launch`

```ini
[core/cli]
script_name = core/script/interrogator/clip
device      = cpu
data_file   = ./data.tsv

[core/script/interrogator/clip]
data_file  = ${core/cli:data_file}
names      = image,label
image_col  = image
label_col  = label
do_reverse = False
```

---

## Key Rules & Gotchas

1. **`device = cpu` is the default** in `[core/cli]` for all configs.
2. **`task_name` section name must match** — `task_name = core/task/supervised`
   requires a `[core/task/supervised]` section, not `[core/task/infer]` (which
   is not a registered task).
3. **No AutoClass** — model/processor sections must use explicit class names
   (enforced by `pretrained_name` lookup, never `AutoModel`).
4. **vLLM inference** — use `core/task/supervised` + `unitorch-infer`; vLLM
   model sections live under `core/model/vllm/generation/<name>`.
5. **`[core/dataset/ast/test]`** must override `names` if the test file has
   fewer columns than train (e.g. no `decode` column for inference-only).
6. **Interpolation syntax** is `${section:key}` (colon, not slash).
7. **List values** use Python list syntax: `['a', 'b']`. Dict values use `{}`.
8. **Comments** use `;` or `#`.
