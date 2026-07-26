---
name: "unitorch-copilot-tools"
description: "Use when an agent needs to discover or invoke UniTorch copilot tools, inspect registered models, processors, tasks, FastAPI services, and writers, or automate ML workflows through unitorch-copilot-cli."
version: "0.0.2.1"
author: "FULIUCANSHENG"
license: "MIT"
metadata:
  hermes:
    tags: ["unitorch", "copilot", "cli", "skills", "ml", "clawhub", "hermeshub"]
    related_skills: ["unitorch-config-ini", "unitorch-train-model", "unitorch-infer-model", "unitorch-serve-fastapi"]
  clawhub:
    tags: ["unitorch", "copilot", "cli", "skills", "ml", "clawhub", "hermeshub"]
related_skills: ["unitorch-config-ini", "unitorch-train-model", "unitorch-infer-model", "unitorch-serve-fastapi"]
---

# unitorch-copilot-tools

## Overview

`unitorch-copilot-tools` is the generated skill index for UniTorch copilot tools. Use it to discover registered components, invoke tools through `unitorch-copilot-cli`, call the same tools from Python, and bridge to remote services exposed by `unitorch-fastapi` when a tool declares a FastAPI adapter.

## Install

Generate the skills into the project-standard `.skills` directory:

```bash
npm run generate-skills
```

Install or export into another folder with the npx wrapper:

```bash
npx unitorch install all --folder .skills --force true
npx unitorch export all --folder ./agent-skills
```

The Python entrypoint remains available for environments without Node:

```bash
python3 -m unitorch.cli.copilots.skills install all --folder .skills --force true
python3 -m unitorch.cli.copilots.skills validate --folder .skills
```

## When To Use

- Discover registered UniTorch models, processors, tasks, writers, and services.
- Invoke small agent-facing utilities through `unitorch-copilot-cli`.
- Publish generated skill markdown to ClawHub, HermesHub, or compatible skill registries.

## CLI

```bash
unitorch-copilot-cli
unitorch-copilot-cli <tool-name> --arg=value
```

## Python

```python
from unitorch.cli.copilots import get_copilot_tool

tool = get_copilot_tool("core/copilot/pkg_infos")
result = tool.invoke(name="model")
```

## Registered Tools

| Tool | Skill | Description |
|------|-------|-------------|
| `core/copilot/pkg_infos` | [core-copilot-pkg_infos](core-copilot-pkg_infos/SKILL.md) | List registered unitorch packages and copilot tools. |

## Verification Checklist

- Run `python3 -m unitorch.cli.copilots.skills validate --folder .skills` after generation.
- Confirm the parent index lists every generated child skill.
- For publishing, confirm the CI artifact contains `.skills/unitorch-copilot-tools/SKILL.md` and child `SKILL.md` files.

## Common Pitfalls

- Generate into `.skills` for project skills; `skills/` is only the legacy folder used by older installs.
- Run generation from an environment where UniTorch and any extension packages are importable.
- Do not publish on normal pushes unless ClawHub/HermesHub credentials are intentionally configured.
