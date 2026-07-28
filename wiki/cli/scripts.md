# Scripts (`python3 -m`)

One-off utility scripts are run directly as Python modules:

```bash
python3 -m unitorch.cli.<module> [args...]
```

Install UniTorch before running module commands. For local development, use an
editable install such as `python3 -m pip install -e .`, then run commands
without `PYTHONPATH=src`.

---

## Copilot Skill Document Exporter

Generates the `unitorch-copilot-tools` skill package from registered copilot
tools into a target folder. These skill documents make unitorch's ML lifecycle
capabilities easier for agents to discover, plan, and reuse.

```bash
python3 -m unitorch.cli.copilots.skills install all --folder ./skills --force true
python3 -m unitorch.cli.copilots.skills validate --folder ./skills
python3 -m unitorch.cli.copilots.skills uninstall all --folder ./skills
```

Each tool's output is written to
`<folder>/unitorch-copilot-tools/<skill-safe-tool-name>/SKILL.md`,
containing skill frontmatter, CLI/Python usage, parameter metadata, and any
remote `unitorch-fastapi` route metadata declared by the copilot tool.
`install` defaults to `./skills`, and `uninstall` removes the generated
copilot skill package from that folder. Use `--name all` or omit `--name` to
include every registered copilot tool.

The root `skills/` layout can then be installed into a local agent skill folder
through the external open-agent `skills` npm package:

```bash
npx skills add fuliucansheng/unitorch
npx skills add fuliucansheng/unitorch --folder ./agent-skills
```

A default external install can report:

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
