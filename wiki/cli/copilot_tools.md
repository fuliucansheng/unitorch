# Copilot Tools (`unitorch-copilot-cli`)

Copilot tools are designed for agent use. `unitorch-copilot-cli` invokes a
registered tool by name and exits — making it composable in agent pipelines.

```bash
unitorch-copilot-cli <name> [--key value ...]
```

---

## core/copilot/pkg_infos

Lists all registered components in the current unitorch installation.

```bash
# List every registered type
unitorch-copilot-cli core/copilot/pkg_infos

# Filter by type
unitorch-copilot-cli core/copilot/pkg_infos --name model
unitorch-copilot-cli core/copilot/pkg_infos --name process
unitorch-copilot-cli core/copilot/pkg_infos --name fastapi
```

Available `--name` values: `process`, `copilot_tool`, `model`, `fastapi`,
`score`, `dataset`, `loss`, `optimizer`, `scheduler`, `task`, `writer`.
