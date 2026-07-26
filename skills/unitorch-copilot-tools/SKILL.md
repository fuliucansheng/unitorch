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

`unitorch-copilot-tools` is the generated skill index for UniTorch copilot tools. Use it to discover registered components, invoke tools through `unitorch-copilot-cli`, call the same tools from Python, and bridge to remote services exposed by `unitorch-fastapi` when a tool declares a FastAPI adapter. It also preserves access to model and algorithm related workflows, including package info discovery for registered UniTorch components.

## Install

Generate the canonical project skills under the root `skills/` directory:

```bash
npm run generate-skills
```

Install the published root skills into an agent skill folder with the external `skills` npm package:

```bash
npx skills add fuliucansheng/unitorch
npx skills add fuliucansheng/unitorch --folder ./agent-skills
```

That external installer copies the root `skills/` package and can report:

```json
{
  "repo": "fuliucansheng/unitorch",
  "folder": "/home/decu/.hermes/skills",
  "copied": [
    "unitorch-config-ini",
    "unitorch-copilot-tools",
    "unitorch-infer-model",
    "unitorch-replace-decorator",
    "unitorch-serve-fastapi",
    "unitorch-train-model"
  ]
}
```

Generate, export, and validate from Python when updating this repository:

```bash
python3 -m unitorch.cli.copilots.skills install all --folder ./skills --force true
python3 -m unitorch.cli.copilots.skills validate --folder ./skills
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

- Run `python3 -m unitorch.cli.copilots.skills validate --folder ./skills` after generation.
- Confirm the parent index lists every generated child skill.
- For publishing, confirm the CI artifact contains `skills/unitorch-copilot-tools/SKILL.md` and child `SKILL.md` files.

## Common Pitfalls

- Generate into `skills/` so the generated copilot skill sits beside the hand-written root skills.
- Use `npx skills add fuliucansheng/unitorch` only through the external open-agent skills ecosystem; this repository does not publish or alias that installer.
- The external installer copies published skills into an agent-local folder; it does not regenerate this repository's skill markdown.
- Run generation from an environment where UniTorch and any extension packages are importable.
- Do not publish on normal pushes unless ClawHub/HermesHub credentials are intentionally configured.
