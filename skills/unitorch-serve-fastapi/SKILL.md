---
name: serve-fastapi
description: Guidance for creating, reviewing, launching, calling, and debugging unitorch FastAPI services with unitorch-fastapi and CopilotClient. Use when preparing FastAPI INI configs, choosing registered core/fastapi services, configuring routers and pipeline sections, starting or stopping daemon/foreground servers, calling /start /status /generate endpoints, using autostart_services, or diagnosing health-check, port, route, service loading, and request-format issues.
---

# Serve unitorch FastAPI

## Overview

Use this skill to expose unitorch models or utility services through
`unitorch-fastapi`, verify service lifecycle, and call routes from Python or
HTTP clients. When editing detailed INI syntax, also use the `config-ini` skill.

## FastAPI Workflow

1. Query registered FastAPI services before choosing names.
2. Start from the closest config in `examples/configs/fastapis/` or
   `examples/configs/servers/`.
3. Configure `[core/cli] enabled_services`, `device`, `host`, `port`, and
   optional `autostart_services`.
4. Configure the service router section and, for model services, the pipeline
   section.
5. Start the server, wait for `/health-check`, and check service `/status`.
6. Load the model with `/start` or `autostart_services`.
7. Call `/generate`, then stop the service or daemon cleanly.

## Discover Services

Use the live registry instead of guessing service names:

```bash
unitorch-copilot-cli core/copilot/pkg_infos --name fastapi
```

If a service comes from an extension package, add its import path to
`depends_libraries` in `[core/cli]` so it registers before the server starts.

## Build The FastAPI Config

Use this skeleton for model services:

```ini
[core/cli]
enabled_services = [
    "core/fastapi/<service>",
  ]
device = cpu
host = 0.0.0.0
port = 5000
health_check_timeout = 300
# autostart_services = ${core/cli:enabled_services}

[core/fastapi/<service>]
router = /core/fastapi/<service>

[core/fastapi/pipeline/<service>]
pretrained_name = <model-id-or-local-path>
device = ${core/cli:device}
enable_cpu_offload = True
```

Always set `device = cpu` in `[core/cli]` when generating FastAPI configs.
Pipeline sections can inherit it with `${core/cli:device}`. Only set a GPU
device intentionally after checking the service supports it and memory fits.

Most model services use a registered service section for router/lifecycle config
and a separate `core/fastapi/pipeline/...` section for model-loading parameters.
Server utility services, such as `core/fastapi/servers/zip_files`, usually read
their parameters directly from the service section instead.

When unsure, inspect the service class:

- `@register_fastapi("...")` gives the `enabled_services` name.
- `config.set_default_section("...")` in `__init__` usually gives the router
  section.
- `@config_defaults_init("...")` on the pipeline `from_config` gives the model
  pipeline section.

## Common Config Patterns

Generation service:

```ini
[core/cli]
enabled_services = ["core/fastapi/gemma"]
device = cpu
health_check_timeout = 1800

[core/fastapi/pipeline/gemma]
pretrained_name = gemma-4-12b
device = ${core/cli:device}
enable_cpu_offload = True
max_seq_length = 2048
max_gen_seq_length = 256
```

Image classification service:

```ini
[core/cli]
enabled_services = ["core/fastapi/clip/image/v2"]
device = cpu
health_check_timeout = 300

[core/fastapi/pipeline/clip/image/v2]
pretrained_name = clip-vit-base-patch16
device = ${core/cli:device}
enable_cpu_offload = True
label_dict = {"cat": "a photo of a cat", "dog": "a photo of a dog"}
act_fn = softmax
```

Utility server with autostart:

```ini
[core/cli]
enabled_services = ["core/fastapi/servers/zip_files"]
autostart_services = ${core/cli:enabled_services}

[core/fastapi/servers/zip_files]
router = /core/fastapi/servers/zip_files
zip_folder = ./
zip_extension = .zip
num_thread = 20
```

## Start, Stop, And Restart

Run foreground mode while developing:

```bash
unitorch-fastapi examples/configs/fastapis/gemma.ini
```

Run daemon mode when using explicit lifecycle actions:

```bash
unitorch-fastapi start examples/configs/fastapis/gemma.ini --port=5001
unitorch-fastapi stop examples/configs/fastapis/gemma.ini --port=5001
unitorch-fastapi restart examples/configs/fastapis/gemma.ini --port=5001
```

Use the same config path and CLI overrides for `stop` as for `start`, because
the daemon name is derived from enabled services and config hash. Daemon logs
are written under `/tmp` and the exact log path is printed after successful
startup.

`host = 0.0.0.0` binds publicly, but local health checks use `127.0.0.1`.
If port `5000` is occupied, pass a different `--port` or set `port` in
`[core/cli]`.

## Service Lifecycle

`/health-check` only confirms the FastAPI process is alive. It does not mean a
model pipeline is loaded.

Most services expose these routes under their router prefix:

```text
GET  /<router>/status
GET  /<router>/start
GET  /<router>/stop
POST /<router>/generate
```

If `autostart_services` is absent, call `/start` before `/generate`.
Use `autostart_services = ${core/cli:enabled_services}` or
`autostart_services = all` when the service should load during server startup.
For large models, increase `health_check_timeout` because daemon mode waits for
health and autostart to finish.

## Calling Services From Python

Use `CopilotClient` when Python code should manage a local server. Passing
`config` starts a `unitorch-fastapi` subprocess. If no endpoint or port is
provided, the client chooses an available local port and cleans it up on exit.

```python
from unitorch.cli.copilots import CopilotClient

with CopilotClient(
    config="examples/configs/fastapis/gemma.ini",
    startup_timeout=1800,
) as client:
    client.request("/core/fastapi/gemma/start", method="GET")
    result = client.request(
        "/core/fastapi/gemma/generate",
        params={"text": "Write one sentence about unitorch."},
    )
```

For an already-running server, pass `endpoint`:

```python
from unitorch.cli.copilots import CopilotClient

client = CopilotClient(endpoint="http://127.0.0.1:5000")
result = client.request(
    "/core/fastapi/gemma/generate",
    params={"text": "hello"},
)
```

For image or video uploads, use `images` or `videos`; the client sends multipart
requests with the field names expected by FastAPI:

```python
result = client.request(
    "/core/fastapi/clip/image/v2/generate",
    images={"image": "./cat.jpg"},
)
```

Use `call_fastapi` for a single direct request when lifecycle management is not
needed:

```python
from unitorch.cli.copilots import call_fastapi

result = call_fastapi(
    "http://127.0.0.1:5000/core/fastapi/gemma/generate",
    params={"text": "hello"},
)
```

## Request Format Checks

Before debugging model quality, verify request shape:

- Text and numeric arguments are query parameters in `params`.
- Image upload fields must match the service method parameter, usually
  `image`.
- Video upload fields must match the service method parameter, usually `video`.
- Binary image responses should use `resp_type="image"` with `CopilotClient` or
  `call_fastapi`.
- Binary video responses should use `resp_type="video"`.
- Generation services often accept runtime decoding parameters such as
  `max_gen_seq_length`, `num_beams`, `do_sample`, `temperature`, and `top_p`.

## Adding Or Modifying A FastAPI Service

When implementing a new service:

- Extend `GenericFastAPI`.
- Register it with `@register_fastapi("core/fastapi/<name>")`.
- Build an `APIRouter(prefix=router)`.
- Expose at least `start`, `stop`, `status`, and the task route such as
  `generate`.
- Keep model-loading parameters in a pipeline class with
  `@config_defaults_init("core/fastapi/pipeline/<name>")` when the service wraps
  a model.
- Ensure the service module is imported before startup so the registration happens.
  For installed workflows, use package imports or `depends_libraries` to load
  extension services.
- Add or update an example config under `examples/configs/fastapis/`.
- Do not introduce HuggingFace AutoClass usage.

## Common Failure Patterns

- `enabled_services` not found: the module was not imported or an extension
  package was not listed in `depends_libraries`.
- `/health-check` succeeds but `/generate` fails: the model service was not
  started; call `/start` or set `autostart_services`.
- Daemon startup times out: model autostart is slow, model download is blocked,
  or `health_check_timeout` is too low.
- Port occupied: choose another `--port`; `CopilotClient(config=...)` can pick
  a free port automatically.
- Route 404: the router prefix differs from the URL; check the service `router`
  config and class default.
- Upload validation error: multipart field name does not match the FastAPI
  method parameter.
- CUDA memory error: use `device = cpu`, enable CPU offload, reduce generation
  settings, or start fewer services in one process.
- Stop does not find the daemon: call `stop` with the same config path and
  overrides used for `start`.

## Completion Checklist

Before considering a FastAPI setup ready:

- `enabled_services` was checked against the live `fastapi` registry.
- The config was based on a nearby example.
- `[core/cli] device = cpu` is present in generated configs.
- Router and pipeline sections match the service implementation.
- Foreground startup or daemon startup reached `/health-check`.
- Each model service reports `running` after `/start` or autostart.
- A representative `/generate` request works with the expected request fields.
- Port, host, timeout, and service shutdown behavior are intentional.
