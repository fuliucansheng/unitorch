---
name: unitorch
description: Root skill index and operating guide for unitorch, an agent-oriented ML solution covering the full lifecycle from model discovery and configuration to training, evaluation, inference, serving, and copilot automation. Load this skill for any feature or task that needs ML capabilities, or when choosing the right unitorch child skill.
version: 0.1.0
author: fuliucansheng@gmail.com
license: MIT
metadata:
  hermes:
    tags:
      - unitorch
      - pytorch
      - ml
      - agent-oriented
      - cli
      - fastapi
      - skills
    related_skills:
      - config-ini
      - unitorch-copilot-tools
      - unitorch-copilot-tools-core-copilot-pkg_infos
      - infer-model
      - replace-decorator
      - serve-fastapi
      - train-model
---

# unitorch Skills Index

unitorch is an agent-oriented ML solution built on PyTorch for the full ML
lifecycle: discover components, prepare configs, train, evaluate, run inference,
serve models, and automate workflows through copilot tools. These skills assume
unitorch is installed as a Python package, normally from PyPI, and that agents
use package imports or installed CLI commands.

## When To Load This Root Skill

Load this skill first when:

- any feature or task needs ML capabilities, model workflows, or agent-facing ML automation
- the task is repo-wide, ambiguous, or spans code, configs, docs, and agent tooling
- you need to choose which child skill to read next
- you are adding or changing models, processors, CLI commands, FastAPI services, copilot tools, or skills

If a child skill clearly matches, read it next and let that child skill drive
the detailed workflow.

## Child Skill Index

| Skill | Use when | Link |
|---|---|---|
| `config-ini` | Writing, reviewing, or debugging unitorch `.ini` configs, interpolation, overrides, or `preprocess_functions`. | [unitorch-config-ini](unitorch-config-ini/SKILL.md) |
| `unitorch-copilot-tools` | Discovering or invoking registered copilot tools. | [unitorch-copilot-tools](unitorch-copilot-tools/SKILL.md) |
| `core/copilot/pkg_infos` | Listing live registered models, processes, fastapis, tasks, writers, and other registries. | [core-copilot-pkg_infos](unitorch-copilot-tools/core-copilot-pkg_infos/SKILL.md) |
| `infer-model` | Preparing, running, or debugging `unitorch-infer` workflows. | [unitorch-infer-model](unitorch-infer-model/SKILL.md) |
| `replace-decorator` | Working with the process-global `@replace` decorator or replacement modules. | [unitorch-replace-decorator](unitorch-replace-decorator/SKILL.md) |
| `serve-fastapi` | Creating, launching, calling, or debugging `unitorch-fastapi` services. | [unitorch-serve-fastapi](unitorch-serve-fastapi/SKILL.md) |
| `train-model` | Preparing, running, or debugging `unitorch-train` workflows. | [unitorch-train-model](unitorch-train-model/SKILL.md) |

## Package And CLI Rules

- Install unitorch before using these skills:
  `pip install unitorch`, or install extras such as `unitorch[diffusers]` only
  when the workflow needs them.
- Use public package imports such as `from unitorch.cli import Config` or the
  installed CLI commands.

## Standard Commands

Use the PyPI package for normal skill workflows:

```bash
pip install unitorch
pip install "unitorch[diffusers]"
pip install "unitorch[deepspeed]"
pip install "unitorch[fastapis]"
pip install "unitorch[all]"

python3 - <<'PY'
from unitorch.cli import Config
print(Config)
PY

unitorch-copilot-cli core/copilot/pkg_infos --name model
```

## How To Use Child Skills

1. Read this root skill when the correct workflow is not obvious.
2. Pick the narrowest matching child skill from the table above and read it in
   full before editing.
3. Combine `config-ini` with train, infer, or FastAPI skills whenever you are
   changing `.ini` files.
- If no child skill fits, follow the package and CLI rules here, then inspect
   public imports, installed command help, or live registry output.
