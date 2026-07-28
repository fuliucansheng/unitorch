---
name: unitorch-infer-model
description: Guidance for creating, reviewing, running, and debugging unitorch batch model inference workflows with unitorch-infer. Use when preparing inference INI configs, choosing registered models/processors/postprocess functions/writers, running inference from checkpoints, configuring vLLM inference, preserving raw input columns in outputs, or diagnosing empty, duplicated, or incorrect inference results.
---

# Infer unitorch Models

## Overview

Use this skill to prepare and verify a `unitorch-infer` workflow end to end:
choose registered components, write or review the INI config, load the intended
checkpoint, run test data through preprocessing, postprocess model outputs, and
write results. When editing detailed INI syntax, also use the `config-ini`
skill.

## Inference Workflow

1. Start from the closest training or inference example in `cli/configs/`.
2. Query the live registries before inventing section names.
3. Keep the dataset `test` split inputs-only unless labels are needed in output.
4. Wire `preprocess_functions`, model, `postprocess_fn`, writer, and output path.
5. Verify `from_ckpt_dir` exists and points to the intended checkpoint.
6. Run a tiny output-file smoke test before running a large batch.
7. Inspect output rows, copied raw columns, generated columns, and throughput.

## Discover Registered Components

Use `unitorch-copilot-cli core/copilot/pkg_infos` to inspect registrations:

```bash
unitorch-copilot-cli core/copilot/pkg_infos --name model
unitorch-copilot-cli core/copilot/pkg_infos --name process
unitorch-copilot-cli core/copilot/pkg_infos --name writer
unitorch-copilot-cli core/copilot/pkg_infos --name task
```

Check `process` for both preprocess and postprocess names. Extension packages
can add registrations, so add `depends_libraries` in `[core/cli]` when an
extension module must be imported before config initialization.

## Build The Inference Config

Use these sections for standard batch inference:

```ini
[core/cli]
task_name = core/task/supervised
from_ckpt_dir = ./cache
cache_dir = ./cache
test_file = ./test.tsv

[core/model/generation/<name>]
pretrained_name = <model-id-or-local-path>
max_gen_seq_length = 128

[core/process/<name>]
pretrained_name = <model-id-or-local-path>
max_seq_length = 512
max_gen_seq_length = 128

[core/dataset/ast]
names = ['encode']

[core/dataset/ast/test]
data_files = ${core/cli:test_file}
preprocess_functions = ['core/process/<name>/generation/inputs(encode)']

[core/writer/csv]
escapechar = \

[core/task/supervised]
model = core/model/generation/<name>
dataset = core/dataset/ast
from_ckpt_dir = ${core/cli:from_ckpt_dir}
output_header = ['encode']
postprocess_fn = core/postprocess/<name>/detokenize
writer = core/writer/csv
output_path = ${core/cli:cache_dir}/output.txt
test_batch_size = 8
num_workers = 4
postprocess_workers = 2
```

Do not add optimizer, scheduler, loss, or score sections only for inference.
They are ignored by `unitorch-infer` when present in a reused training config,
but a new inference-only config should stay smaller.

## Preprocessing Rules

`preprocess_functions` entries are call-expression strings. The full text before
the outer parentheses is the registered process function name. For example,
`core/process/<name>/generation/inputs` is one registered function name.

Compose registered process functions with nested calls when a raw column needs
another transform first:

```ini
preprocess_functions = [
    'core/process/<name>/generation/inputs(encode, core/process/image/read(image))',
  ]
```

For generation inference, use `inputs` preprocessors on `test`. Do not use a
train preprocessor such as `core/process/<name>/generation(encode, decode)`
unless the test file actually has target columns and the model/loss path expects
them.

If the test file has fewer columns than train/dev, override `names` in
`[core/dataset/ast/test]` or set `[core/dataset/ast] names` for inference only.

## Postprocess And Writers

`postprocess_fn` must be a registered process function that converts model
outputs into a pandas DataFrame-compatible result. Common generation configs use
`core/postprocess/<name>/detokenize`.

`writer` must be a registered writer:

- `core/writer/csv` writes TSV/CSV-style files; set `sep`, `header`, `columns`,
  `quoting`, and `escapechar` in `[core/writer/csv]`.
- `core/writer/jsonl` writes JSON lines; set `columns` when output should be
  restricted to a stable schema.
- `core/writer/parquet` writes Parquet and is better for large structured
  outputs.

`output_header` copies raw test columns into model outputs before postprocessing.
Use it for traceability, such as preserving prompt, id, image path, or query
columns. The listed names must exist in the raw dataset.

Existing output files can cause writer-level skipping when the writer can
compute `skip_n_samples`. For deterministic resume, configure
`nrows_per_sample` when every input produces a fixed number of output rows.
Use a fresh `output_path` for a clean run.

## Launch Commands

Run a small smoke test first:

```bash
unitorch-infer config.ini \
  --test_file=./tiny-test.tsv \
  --cache_dir=/tmp/unitorch-infer-smoke \
  --"core/task/supervised@test_batch_size"=1 \
  --"core/task/supervised@num_workers"=0
```

Use `@` for non-`[core/cli]` overrides:

```bash
unitorch-infer config.ini --from_ckpt_dir=/path/to/checkpoint
unitorch-infer config.ini --"core/task/supervised@output_path"=/tmp/output.jsonl
unitorch-infer config.ini --"core/writer/jsonl@columns"="['prompt', 'text']"
```

`core/task/supervised.infer()` asserts single-GPU mode. Do not launch normal
batch inference with `torchrun`; use backend-specific parallelism such as vLLM
`tensor_parallel_size` only when the model section supports it.

## vLLM Inference Pattern

vLLM inference still uses `unitorch-infer` and `task_name =
core/task/supervised`, but the model section is under
`core/model/vllm/generation/<name>`:

```ini
[core/model/vllm/generation/qwen3]
pretrained_name = qwen3-4b-thinking
tensor_parallel_size = 1
pipeline_parallel_size = 1
gpu_memory_utilization = 0.90
max_num_seqs = 256
dtype = auto

[core/dataset/ast]
names = ['prompt']

[core/dataset/ast/test]
data_files = ${core/cli:test_file}
preprocess_functions = [
    '''
    core/process/object/inputs(
      prompt=core/process/qwen/chat_template(
        [
          {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
          }
        ]
      )
    )
    '''
  ]

[core/task/supervised]
model = core/model/vllm/generation/qwen3
dataset = core/dataset/ast
postprocess_fn = core/postprocess/qwen/detokenize
writer = core/writer/csv
output_header = ['prompt']
```

Match `max_seq_length`, `max_gen_seq_length`, and backend memory settings to the
actual model and hardware. Start with small `test_batch_size` and increase only
after output correctness is verified.

## Checkpoint Loading

`from_ckpt_dir` under the task section controls inference weight loading. If the
directory does not exist, it is skipped and the model remains at whatever state
its model config created, such as pretrained weights or random initialization.

Before trusting outputs:

- Confirm `from_ckpt_dir` is correct from the process working directory.
- Confirm expected checkpoint files exist, usually `pytorch_model.bin` or
  `pytorch_model_latest.bin` depending on how the training run was saved.
- Confirm model architecture and processor config match the checkpoint.
- Avoid accidentally pointing inference at a stale training `cache_dir`.

## Data Flow Checks

Inference must follow this contract:

```text
raw data -> preprocess -> tensor(s) -> collate_fn -> batched tensor(s)
         -> model forward/generate -> tensor(s) -> postprocess -> DataFrame
         -> writer -> output file
```

Check these points before a large run:

- Test `names` match the file columns exactly.
- Test preprocessing returns only inputs unless targets are deliberately needed.
- Nested preprocessors return the type expected by the outer registered
  function.
- `postprocess_fn` matches the model output type.
- `output_header` names exist in raw input rows.
- `writer` schema and file extension match the intended output format.
- `test_batch_size`, `num_workers`, and `postprocess_workers` fit memory and CPU
  limits.

## Common Failure Patterns

- Empty output: test file path is wrong, dataset split is empty, or writer is
  appending/skipping because an old output file exists.
- Missing copied columns: `output_header` contains names not present in raw
  rows.
- Random-looking output: `from_ckpt_dir` does not exist or does not match the
  model section.
- Postprocess error: `postprocess_fn` is missing, misspelled, or incompatible
  with model outputs.
- Shape/device error: `preprocess_functions` returns labels or a wrong input
  class for inference.
- Slow inference: `num_workers` or `postprocess_workers` is too low, batch size
  is too small, or backend generation settings are too expensive.
- CUDA out of memory: reduce `test_batch_size`, generation length, or vLLM
  concurrency settings.

## Completion Checklist

Before considering an inference setup ready:

- Registry names were checked for model, process, writer, and task.
- The config was based on a nearby `cli/configs/` file.
- `from_ckpt_dir` existence and intended checkpoint files were verified.
- A tiny smoke run completed and wrote output rows.
- Raw trace columns from `output_header` appear in the output.
- Postprocessed generated fields are non-empty and human-inspectable.
- Re-running behavior with existing `output_path` is intentional.
- Batch size, worker count, and backend parallelism are documented.
