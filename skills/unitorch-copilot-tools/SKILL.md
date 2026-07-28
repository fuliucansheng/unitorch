---
name: "unitorch-copilot-tools"
description: "Use when an agent needs to discover or invoke unitorch copilot tools, inspect registered models, processors, tasks, FastAPI services, and writers, or automate ML workflows through unitorch-copilot-cli after installing the unitorch package."
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

`unitorch-copilot-tools` is the generated skill index for unitorch copilot tools. Use it to discover registered components, invoke tools through `unitorch-copilot-cli`, call the same tools from Python, and bridge to remote services exposed by `unitorch-fastapi` when a tool declares a FastAPI adapter. It assumes the `unitorch` package is installed and available through normal Python imports and installed CLI commands.

## Install

Install unitorch from PyPI before using these tools:

```bash
pip install unitorch
# Optional extras only when needed:
pip install "unitorch[fastapis]"
pip install "unitorch[diffusers]"
```

After installation, use Python imports or installed CLI commands directly:

```bash
unitorch-copilot-cli core/copilot/pkg_infos
unitorch-copilot-cli core/copilot/pkg_infos --name model
```

```python
from unitorch.cli.copilots import get_copilot_tool

tool = get_copilot_tool("core/copilot/pkg_infos")
result = tool.invoke(name="model")
```

## When To Use

- Discover registered unitorch models, processors, tasks, writers, and services.
- Invoke small agent-facing utilities through `unitorch-copilot-cli`.

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

- Run `unitorch-copilot-cli core/copilot/pkg_infos` after installing unitorch.
- Confirm the parent index lists every generated child skill.

## Common Pitfalls

- Use the installed package and CLI for normal workflows.
- Install unitorch and any extension packages before discovery.
