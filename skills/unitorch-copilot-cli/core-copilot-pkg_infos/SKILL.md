---
name: "unitorch-copilot-cli-core-copilot-pkg_infos"
description: "Use when an agent needs to invoke the `core/copilot/pkg_infos` unitorch copilot tool."
---

# core/copilot/pkg_infos

List registered unitorch packages and copilot tools.

This is a subskill of `unitorch-copilot-cli`. Use the parent skill index to discover other registered copilot tools.

## When To Use

Use this skill when you need to call `core/copilot/pkg_infos` through unitorch copilot.

## CLI

```bash
unitorch-copilot-cli core/copilot/pkg_infos
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
