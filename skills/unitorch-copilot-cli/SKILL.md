---
name: "unitorch-copilot-cli"
description: "Use when an agent needs to discover and invoke registered UniTorch copilot tools through `unitorch-copilot-cli`."
---

# unitorch-copilot-cli

`unitorch-copilot-cli` exposes registered UniTorch copilot tools as a single skill package. Each child directory documents one registered tool, including CLI usage, Python invocation, parameters, and any remote FastAPI adapter.

## Usage

```bash
unitorch-copilot-cli
unitorch-copilot-cli <tool-name> --arg=value
```

## Registered Tools

| Tool | Skill | Description |
|------|-------|-------------|
| `core/copilot/pkg_infos` | [core-copilot-pkg_infos](core-copilot-pkg_infos/SKILL.md) | List registered unitorch packages and copilot tools. |
