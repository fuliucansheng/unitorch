# Copilot Tools (`unitorch-copilot-cli`)

Copilot is the agent-facing integration layer for unitorch. It adapts
registered copilot tools into Python functions, CLI commands, remote clients,
and skill metadata so agents can discover and invoke ML capabilities with
stable parameter contracts.

Copilot tools work alongside the lifecycle CLIs. Use the workflow commands
directly for training, inference, evaluation, and FastAPI serving, and use
`unitorch-copilot-cli` for component discovery, tool invocation, remote client
access, and skill-oriented automation.

Install UniTorch before running `python3 -m` commands or npm skill scripts. For
local development, use `python3 -m pip install -e .`, then run commands without
`PYTHONPATH=src`.

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
python3 -m unitorch.cli.copilots.skills install all --folder ./skills --force true
python3 -m unitorch.cli.copilots.skills export all --folder ./agent-skills
python3 -m unitorch.cli.copilots.skills validate --folder ./skills
python3 -m unitorch.cli.copilots.skills uninstall all --folder ./skills
```

---

## Generated Skills

UniTorch copilot skill files are generated under the root
`skills/unitorch-copilot-tools/` package. That generated package lives beside
the hand-written root skills such as `skills/unitorch-config-ini` and
`skills/unitorch-train-model`. Each generated `SKILL.md` includes
Hermes/OpenClaw-friendly frontmatter, CLI and Python invocation examples,
parameter tables, FastAPI adapter details when available, and verification
notes.

Use npm scripts from this repository:

```bash
npm run generate-skills
npm run validate-skills
```

Use the npm-distributed `skills` installer when installing UniTorch repository
skills into a local agent skill folder:

```bash
npx skills add fuliucansheng/unitorch
npx skills add fuliucansheng/unitorch --folder ./agent-skills
```

The command above is provided by the external open-agent `skills` npm package.
UniTorch only provides the root `skills/` layout for that ecosystem to copy;
this repository does not publish or alias a local installer bin. A default
install can report:

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

The external installer copies the published root `skills/` package; it does not
generate, export, or validate skill markdown.

Use the repository npm or `python3 -m` commands above when updating the
checked-in `skills/` tree. Use `npx skills add fuliucansheng/unitorch` when you
only need to install the packaged repository skills into an agent environment.

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
