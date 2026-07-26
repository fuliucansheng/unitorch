---
name: "unitorch-copilot-tools-core-copilot-pkg_infos"
description: "Use when an agent needs to invoke the core/copilot/pkg_infos UniTorch copilot tool: List registered unitorch packages and copilot tools."
version: "0.0.2.1"
author: "FULIUCANSHENG"
license: "MIT"
metadata:
  hermes:
    tags: ["unitorch", "copilot", "cli", "tool", "core", "pkg", "infos", "metadata"]
    related_skills: ["unitorch-copilot-tools"]
  clawhub:
    tags: ["unitorch", "copilot", "cli", "tool", "core", "pkg", "infos", "metadata"]
related_skills: ["unitorch-copilot-tools"]
---

# core/copilot/pkg_infos

## Overview

List registered unitorch packages and copilot tools.

This is a subskill of `unitorch-copilot-tools`. Use the parent skill index to discover other UniTorch model, algorithm, and package info tools.

## When To Use

- Invoke `core/copilot/pkg_infos` from an agent workflow.
- Inspect the tool's parameter contract before calling it.
- Use local CLI or Python invocation because this tool has no remote adapter.

## CLI

Run the tool through `unitorch-copilot-cli`:

```bash
unitorch-copilot-cli core/copilot/pkg_infos --name "model"
```

## Python

```python
from unitorch.cli.copilots import get_copilot_tool

tool = get_copilot_tool("core/copilot/pkg_infos")
result = tool.invoke()
```

## Parameters

| Name | Type | Required | Default |
|------|------|----------|---------|
| `name` | `Optional[str]` | no |  |

## Remote FastAPI

This copilot tool does not declare a remote FastAPI adapter.

## Verification Checklist

- Confirm `unitorch-copilot-cli` is available in the active Python environment.
- Run the CLI example with the smallest useful inputs before wiring it into a larger agent workflow.

## Common Pitfalls

- Do not invent parameter names; use the table above or `tool.signature`.
- Import extension libraries before discovery when they register additional copilot tools.
