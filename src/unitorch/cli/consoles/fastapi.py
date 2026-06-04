# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import fire
import atexit
import asyncio
import inspect
import signal
import shutil
import tempfile
import subprocess
import sys
import logging
import time
import urllib.error
import urllib.request
import hashlib
import unitorch.cli
from unitorch.cli import Config
from unitorch.cli import (
    import_library,
    cached_path,
    registered_fastapi,
)
import unitorch.cli.fastapis

_DAEMON_WORKER_ENV = "_UNITORCH_FASTAPI_DAEMON_WORKER"


def _tmp_path(filename):
    return os.path.join(tempfile.gettempdir(), filename)


def _pid_file(name):
    safe_name = name.replace("/", "_")
    return _tmp_path(f"unitorch_fastapi_{safe_name}.pid")


def _log_file(name):
    safe_name = name.replace("/", "_")
    return _tmp_path(f"unitorch_fastapi_{safe_name}.stdout.log")


def _qualified_name(config):
    enabled_services = _get_enabled_services(config)
    hexsha = config.hexsha(6)
    digest = hashlib.sha1("_".join(enabled_services).encode()).hexdigest()[:8]
    service_name = f"services{len(enabled_services)}_{digest}"
    return f"{service_name}@{hexsha}"


def _get_enabled_services(config):
    enabled_services = config.getdefault("core/cli", "enabled_services", None)
    assert enabled_services is not None
    if isinstance(enabled_services, str):
        enabled_services = [enabled_services]
    return list(enabled_services)


def _get_autostart_services(config, enabled_services):
    autostart_services = config.getdefault("core/cli", "autostart_services", [])
    if autostart_services in (None, False):
        return []
    if autostart_services is True:
        return list(enabled_services)
    if isinstance(autostart_services, str):
        if autostart_services.lower() in ("all", "*"):
            return list(enabled_services)
        autostart_services = [autostart_services]
    else:
        autostart_services = list(autostart_services)

    for service_name in autostart_services:
        if service_name == "*" or service_name == "all":
            return list(enabled_services)
        if service_name not in enabled_services:
            raise ValueError(
                f"autostart service {service_name!r} is not in enabled_services"
            )
    return autostart_services


async def _call_fastapi_method(method, *args, **kwargs):
    result = method(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _run_fastapi_method(method, *args, **kwargs):
    return asyncio.run(_call_fastapi_method(method, *args, **kwargs))


def _is_running_status(status):
    if isinstance(status, str):
        return status.lower() == "running"
    if isinstance(status, dict) and "status" in status:
        value = status["status"]
        if isinstance(value, str):
            return value.lower() == "running"
    return bool(status)


def _get_service_status(fastapi_instance):
    try:
        status = _run_fastapi_method(fastapi_instance.status)
        return {
            "ok": True,
            "status": status,
        }
    except Exception as e:
        return {
            "ok": False,
            "status": "error",
            "detail": f"{type(e).__name__}: {e}",
        }


def _autostart_fastapi_services(
    autostart_services,
    fastapi_instances,
):
    for service_name in autostart_services:
        fastapi_instance = fastapi_instances[service_name]
        status_info = _get_service_status(fastapi_instance)
        if status_info["ok"] and _is_running_status(status_info["status"]):
            logging.info("fastapi service %s already running", service_name)
            continue

        logging.info("autostarting fastapi service %s", service_name)
        _run_fastapi_method(fastapi_instance.start)


def _stop_fastapi_services(
    enabled_services,
    fastapi_instances,
):
    for service_name in reversed(enabled_services):
        fastapi_instance = fastapi_instances[service_name]
        status_info = _get_service_status(fastapi_instance)
        if status_info["ok"] and not _is_running_status(status_info["status"]):
            continue

        try:
            _run_fastapi_method(fastapi_instance.stop)
        except Exception:
            logging.exception("failed to stop fastapi service %s", service_name)


def _health_check_host(config):
    host = config.getdefault("core/cli", "host", "0.0.0.0")
    if host in ("", "0.0.0.0", "::", "[::]"):
        host = "127.0.0.1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return host


def _health_check_url(config):
    host = _health_check_host(config)
    port = config.getdefault("core/cli", "port", 5000)
    return f"http://{host}:{port}/health-check"


def _health_check_timeout(config):
    return float(config.getdefault("core/cli", "health_check_timeout", 30))


def _terminate_process(process):
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


def _wait_for_health_check(config, process, log_file, timeout=None):
    timeout = _health_check_timeout(config) if timeout is None else timeout
    url = _health_check_url(config)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"unitorch-fastapi worker exited before health-check passed "
                f"(log: {log_file})"
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            pass
        time.sleep(0.5)

    _terminate_process(process)
    raise RuntimeError(
        f"unitorch-fastapi health-check timeout after {timeout}s "
        f"(url: {url}, log: {log_file})"
    )


def _run_foreground(config, pid_file):
    import uvicorn
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware

    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))
    atexit.register(lambda: os.path.exists(pid_file) and os.remove(pid_file))

    def _handler(signo, frame):
        raise SystemExit(1)

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)

    depends_libraries = config.getdefault("core/cli", "depends_libraries", None)

    if depends_libraries:
        for library in depends_libraries:
            import_library(library)

    enabled_services = _get_enabled_services(config)
    autostart_services = _get_autostart_services(config, enabled_services)

    for enabled_service in enabled_services:
        assert enabled_service in registered_fastapi, f"{enabled_service} not found"

    fastapi_instances = {
        fastapi_service: registered_fastapi.get(fastapi_service)["obj"](config)
        for fastapi_service in enabled_services
    }

    _autostart_fastapi_services(
        autostart_services=autostart_services,
        fastapi_instances=fastapi_instances,
    )

    app = FastAPI()

    @app.get("/health-check")
    async def health_check():
        return {"status": "ok"}

    for fastapi_instance in fastapi_instances.values():
        app.include_router(fastapi_instance.router)

    statics = config.getdefault("core/cli", "static", {})
    for name, path in statics.items():
        app.mount(f"/{name}", StaticFiles(directory=path), name=name)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    host = config.getdefault("core/cli", "host", "0.0.0.0")
    port = config.getdefault("core/cli", "port", 5000)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    uvicorn.config.LOGGING_CONFIG["formatters"]["access"][
        "fmt"
    ] = "%(asctime)s | %(levelname)s | %(message)s"
    uvicorn.config.LOGGING_CONFIG["formatters"]["default"][
        "fmt"
    ] = "%(asctime)s | %(levelname)s | %(message)s"

    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        _stop_fastapi_services(
            enabled_services=enabled_services,
            fastapi_instances=fastapi_instances,
        )


def start(name, config, daemon_mode):
    pid_file = _pid_file(name)
    if daemon_mode:
        if os.path.exists(pid_file):
            raise RuntimeError(f"unitorch-fastapi {name} already running")

        log_file = _log_file(name)
        log_fd = open(log_file, "a")

        child_args = ["start", config._source_path, "--daemon_mode=False"]
        for section, key, value in config._source_params:
            if section == "core/cli" and key == "daemon_mode":
                continue
            if section == "core/cli":
                child_args.append(f"--{key}={value}")
            else:
                child_args.append(f"--{section}@{key}={value}")

        env = os.environ.copy()
        env[_DAEMON_WORKER_ENV] = pid_file

        kwargs = dict(
            stdout=log_fd,
            stderr=log_fd,
            stdin=subprocess.DEVNULL,
            env=env,
            cwd=os.getcwd(),
        )
        if sys.platform == "win32":
            kwargs["creationflags"] = (
                subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            kwargs["start_new_session"] = True

        unitorch_fastapi_cmd = shutil.which("unitorch-fastapi") or os.path.join(
            os.path.dirname(sys.executable), "unitorch-fastapi"
        )
        process = subprocess.Popen(
            [unitorch_fastapi_cmd] + child_args,
            **kwargs,
        )

        _wait_for_health_check(config, process, log_file)
        print(f"unitorch-fastapi {name} started (log: {log_file})")


def stop(name):
    pid_file = _pid_file(name)
    if os.path.exists(pid_file):
        with open(pid_file) as f:
            pid = int(f.read().strip())
        os.kill(pid, signal.SIGTERM)


def restart(name, config, daemon_mode):
    stop(name)
    start(name, config, daemon_mode)


@fire.decorators.SetParseFn(str)
def fastapi(fastapi_action_or_config: str, config_path: str = None, **kwargs):

    daemon_mode = True
    if config_path is None:
        fastapi_action = "start"
        daemon_mode = False
        config_path = fastapi_action_or_config
    else:
        fastapi_action = fastapi_action_or_config

    config_path = cached_path(config_path)

    params = []
    for k, v in kwargs.items():
        if k.count("@") > 0:
            k0 = k.split("@")[0]
            k1 = "@".join(k.split("@")[1:])
        else:
            k0 = "core/cli"
            k1 = k
        params.append((k0, k1, v))

    if os.environ.get(_DAEMON_WORKER_ENV):
        config = Config(config_path, params=params)
        _run_foreground(config, os.environ[_DAEMON_WORKER_ENV])
        return

    config = Config(config_path, params=params)
    config._source_path = config_path
    config._source_params = params

    qualified_name = _qualified_name(config)

    if fastapi_action == "start":
        if daemon_mode:
            start(qualified_name, config, daemon_mode)
        else:
            _run_foreground(config, _pid_file(qualified_name))
    elif fastapi_action == "stop":
        stop(qualified_name)
    elif fastapi_action == "restart":
        restart(qualified_name, config, daemon_mode)
    else:
        raise ValueError(f"unknown fastapi action: {fastapi_action!r}")


def cli_main():
    import traceback

    try:
        fire.Fire(fastapi)
    except Exception:
        traceback.print_exc()
        raise
