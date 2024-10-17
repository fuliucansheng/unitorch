# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import sys
import fire
import atexit
import signal
import time
import logging
import torch
import importlib
import unitorch.cli
import unitorch.cli.services
from pathlib import Path
from unitorch.cli import CoreConfigureParser
from unitorch.cli import (
    import_library,
    cached_path,
    registered_service,
    init_registered_module,
)

__service_inst__ = None


def _sigterm_handler(signo, frame):
    if __service_inst__ is not None:
        __service_inst__.stop()
    raise SystemExit(1)


def daemonize(pid_file, name):
    if os.path.exists(pid_file):
        raise RuntimeError(f"unitorch-service {name} already Running")
    pid = os.fork()
    if pid > 0:
        # sys.exit(0)
        os._exit(0)

    os.chdir("/")
    os.setsid()
    os.umask(0)

    _pid = os.fork()
    if _pid:
        # sys.exit(0)
        os._exit(0)

    sys.stdout.flush()
    sys.stderr.flush()

    stdin_file = f"/tmp/unitorch_service_{name}.stdin.log"
    stdout_file = f"/tmp/unitorch_service_{name}.stdout.log"

    Path(stdin_file).touch()
    Path(stdout_file).touch()

    with open(stdin_file, "rb") as read_file, open(stdout_file, "ab") as write_file:
        os.dup2(read_file.fileno(), sys.stdin.fileno())
        os.dup2(write_file.fileno(), sys.stdout.fileno())
        os.dup2(write_file.fileno(), sys.stderr.fileno())

    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))

    atexit.register(lambda: os.remove(pid_file))


signal.signal(signal.SIGTERM, _sigterm_handler)
signal.signal(signal.SIGINT, _sigterm_handler)


def start(name, inst, daemon_mode):
    name = name.replace("/", "_")
    pid_file = f"/tmp/unitorch_service_{name}.pid"
    if daemon_mode:
        daemonize(pid_file, name)
    inst.start()


def stop(name, inst):
    name = name.replace("/", "_")
    pid_file = f"/tmp/unitorch_service_{name}.pid"
    if os.path.exists(pid_file):
        with open(pid_file) as f:
            os.kill(int(f.read()), signal.SIGTERM)


def restart(name, inst):
    stop(name, inst)
    start(name, inst)


@fire.decorators.SetParseFn(str)
def service(service_action: str, config_path: str, **kwargs):
    global __service_inst__
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

    config = CoreConfigureParser(config_path, params=params)

    depends_libraries = config.getdefault("core/cli", "depends_libraries", None)

    if depends_libraries:
        for library in depends_libraries:
            import_library(library)

    daemon_mode = config.getdefault("core/cli", "daemon_mode", True)
    service_name = config.getdefault("core/cli", "service_name", None)
    assert service_name is not None
    main_service_cls = registered_service.get(service_name)
    if main_service_cls is None:
        raise ValueError(f"service {service_name} not found")

    if service_action in ["start", "restart"]:
        service_inst = main_service_cls["obj"](config)
    else:
        service_inst = None

    __service_inst__ = service_inst

    hexsha = config.hexsha(6)
    service_name = service_name + f"@{hexsha}"
    if service_action == "start":
        start(service_name, service_inst, daemon_mode)
    elif service_action == "stop":
        stop(service_name, service_inst)
    elif service_action == "restart":
        restart(service_name, service_inst)
    else:
        raise ValueError(f"service action {service_action} not found")


def cli_main():
    fire.Fire(service)
