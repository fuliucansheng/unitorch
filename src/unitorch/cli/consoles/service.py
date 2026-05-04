# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import fire
import atexit
import signal
import shutil
import tempfile
import subprocess
import sys
import unitorch.cli
import unitorch.cli.services
from unitorch.cli import Config
from unitorch.cli import (
    import_library,
    cached_path,
    registered_service,
)

# Sentinel env var: when set, the process runs in foreground (worker) mode.
_DAEMON_WORKER_ENV = "_UNITORCH_DAEMON_WORKER"


def _tmp_path(filename):
    """Return a platform-appropriate path under the system temp directory."""
    return os.path.join(tempfile.gettempdir(), filename)


def _pid_file(name):
    safe_name = name.replace("/", "_")
    return _tmp_path(f"unitorch_service_{safe_name}.pid")


def _log_file(name):
    safe_name = name.replace("/", "_")
    return _tmp_path(f"unitorch_service_{safe_name}.stdout.log")


def _run_foreground(config, pid_file):
    """Run the service in the current process (foreground / worker mode)."""
    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))
    atexit.register(lambda: os.path.exists(pid_file) and os.remove(pid_file))

    service_inst_ref = [None]

    def _handler(signo, frame):
        if service_inst_ref[0] is not None:
            service_inst_ref[0].stop()
        raise SystemExit(1)

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)

    depends_libraries = config.getdefault("core/cli", "depends_libraries", None)
    if depends_libraries:
        for library in depends_libraries:
            import_library(library)

    service_name = config.getdefault("core/cli", "service_name", None)
    assert service_name is not None

    main_service_cls = registered_service.get(service_name)
    if main_service_cls is None:
        raise ValueError(f"service {service_name!r} not found")

    inst = main_service_cls["obj"](config)
    service_inst_ref[0] = inst
    inst.start()


def start(name, config, daemon_mode):
    """Launch the service, daemonising it when daemon_mode is True."""
    pid_file = _pid_file(name)
    if daemon_mode:
        if os.path.exists(pid_file):
            raise RuntimeError(f"unitorch-service {name} already running")

        log_file = _log_file(name)
        log_fd = open(log_file, "a")

        # config._source_path and config._source_params are set in service()
        child_args = ["start", config._source_path, "--daemon_mode=False"]
        for section, key, value in config._source_params:
            if section == "core/cli" and key == "daemon_mode":
                continue  # already forced to False above
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

        unitorch_service_cmd = shutil.which("unitorch-service") or os.path.join(
            os.path.dirname(sys.executable), "unitorch-service"
        )
        subprocess.Popen(
            [unitorch_service_cmd] + child_args,
            **kwargs,
        )

        print(f"unitorch-service {name} started (log: {log_file})")


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
def service(service_action: str, config_path: str, **kwargs):
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

    # Worker mode: this process was launched by start() as a background child.
    if os.environ.get(_DAEMON_WORKER_ENV):
        config = Config(config_path, params=params)
        _run_foreground(config, os.environ[_DAEMON_WORKER_ENV])
        return

    config = Config(config_path, params=params)
    config._source_path = config_path
    config._source_params = params

    daemon_mode = config.getdefault("core/cli", "daemon_mode", True)
    service_name = config.getdefault("core/cli", "service_name", None)
    assert service_name is not None

    if registered_service.get(service_name) is None:
        raise ValueError(f"service {service_name!r} not found")

    hexsha = config.hexsha(6)
    qualified_name = f"{service_name}@{hexsha}"

    if service_action == "start":
        if daemon_mode:
            start(qualified_name, config, daemon_mode)
        else:
            _run_foreground(config, _pid_file(qualified_name))
    elif service_action == "stop":
        stop(qualified_name)
    elif service_action == "restart":
        restart(qualified_name, config, daemon_mode)
    else:
        raise ValueError(f"unknown service action: {service_action!r}")


def cli_main():
    import traceback
    try:
        fire.Fire(service)
    except Exception:
        traceback.print_exc()
        raise
