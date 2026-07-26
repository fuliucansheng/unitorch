# Copilot Tools (`unitorch-copilot-cli`)

Copilot is the agent-facing integration layer for unitorch. It adapts
registered copilot tools into Python functions, CLI commands, remote clients,
and skill metadata so agents can discover and invoke ML capabilities with
stable parameter contracts.

Copilot tools work alongside the lifecycle CLIs. Use the workflow commands
directly for training, inference, evaluation, and FastAPI serving, and use
`unitorch-copilot-cli` for component discovery, tool invocation, remote client
access, and skill-oriented automation.

```bash
unitorch-copilot-cli <tool-name> [--key value ...]
```

Use workflow commands directly:

```bash
unitorch-train
unitorch-infer
unitorch-eval
unitorch-fastapi
```

---

## Discovery

```bash
# List copilot tool names
unitorch-copilot-cli

# Install, export, validate, or uninstall generated skill markdown files
python3 -m unitorch.cli.copilots.skills install all --folder .skills --force true
python3 -m unitorch.cli.copilots.skills export all --folder ./agent-skills
python3 -m unitorch.cli.copilots.skills validate --folder .skills
python3 -m unitorch.cli.copilots.skills uninstall all --folder .skills
```

---

## Generated Skills

UniTorch copilot skill files are generated into the project-standard
`.skills/<skill-name>/SKILL.md` layout. Each generated `SKILL.md` includes
Hermes/OpenClaw-friendly frontmatter, CLI and Python invocation examples,
parameter tables, FastAPI adapter details when available, and verification
notes.

Use npm scripts from this repository:

```bash
npm run generate-skills
npm run validate-skills
```

Use the npx wrapper when installing or exporting to another folder:

```bash
npx unitorch install all --folder .skills --force true
npx unitorch export all --folder ./agent-skills
```

The wrapper invokes `python3 -m unitorch.cli.copilots.skills`, so the active
Python environment must be able to import UniTorch and any extension packages
whose copilot tools should be registered.

The `Publish UniTorch Skills to ClawHub/HermesHub` GitHub Actions workflow
generates `.skills`, validates all generated `SKILL.md` frontmatter, packages
the result as an artifact, and publishes only on tags, published releases, or
manual dispatch with publishing enabled. Hub publishing is optional and uses
these repository secrets when present: `CLAWHUB_TOKEN`,
`CLAWHUB_PUBLISH_URL`, `HERMESHUB_TOKEN`, and `HERMESHUB_PUBLISH_URL`.

---

## Python Adapter

```python
from unitorch.cli.copilots import classify_image

result = classify_image(
    image_path="a.jpg",
    config="config.ini",
    top_k=3,
)
```

For local execution, pass either an existing pipeline instance or a config path
that can build the underlying unitorch pipeline.

Copilot adapters register with the same CLI registry style used by other
unitorch components:

```python
from unitorch.cli import register_copilot_tool

@register_copilot_tool(name="core/copilot/classify_image")
def classify_image(...):
    ...
```

---

## Remote Client Adapter

```python
from unitorch.cli.copilots import CopilotClient

client = CopilotClient(endpoint="http://127.0.0.1:5000")
result = client.classify_image(image_path="a.jpg")
```

`CopilotClient` can also start a local `unitorch-fastapi` subprocess from a
FastAPI config file:

```python
from unitorch.cli.copilots import CopilotClient

with CopilotClient(config="fastapi.ini", host="127.0.0.1", port=5000) as client:
    result = client.classify_image(image_path="a.jpg")
```

If `port` is omitted, the client chooses an available local port and passes it
to the subprocess.

The remote client talks to services started with `unitorch-fastapi`.
Remote calls use the same media contract as `unitorch-fastapi`: parameters are
sent as query params, images are uploaded as JPEG multipart files, videos are
uploaded as video multipart files, and responses can be JSON, image, or video
bytes.

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
