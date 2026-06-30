# Copilot Tools (`unitorch-copilot-cli`)

Copilot is an agent compatibility layer. It adapts existing unitorch
copilot tools into Python functions, CLI commands, remote clients, and skill
metadata. It does not wrap workflow runtimes such as training, inference, or
FastAPI serving.

```bash
unitorch-copilot-cli <tool> [--key value ...]
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

# Show tool metadata
unitorch-copilot describe classify_image

# Install or uninstall skill markdown files
python3 -m unitorch.cli.copilots.skills install --folder ./skills
python3 -m unitorch.cli.copilots.skills uninstall --folder ./skills
```

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

@register_copilot_tool(name="classify_image")
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

## pkg_infos

Lists all registered components in the current unitorch installation.

```bash
# List every registered type
unitorch-copilot-cli pkg_infos

# Filter by type
unitorch-copilot-cli pkg_infos --name model
unitorch-copilot-cli pkg_infos --name process
unitorch-copilot-cli pkg_infos --name fastapi
```

Available `--name` values: `process`, `copilot_tool`, `model`, `fastapi`,
`score`, `dataset`, `loss`, `optimizer`, `scheduler`, `task`, `writer`.
