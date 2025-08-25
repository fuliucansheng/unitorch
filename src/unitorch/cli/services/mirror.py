# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import time
import logging
import shutil
from unitorch.cli import CoreConfigureParser
from unitorch.cli import register_service, GenericService


@register_service("core/service/mirror_files")
class MirrorFilesService(GenericService):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        config.set_default_section("core/service/mirror_files")
        self.ip = config.getoption("ip", "0.0.0.0")
        self.port = config.getoption("port", 11221)
        self.name = config.getoption("processname", "core_mirror_files_service")
        self.mirror_files = config.getoption("mirror_files", {})
        self.mirror_files = {k: v for k, v in self.mirror_files.items() if k != v}
        self.mirror_interval = config.getoption("mirror_interval", 120)
        self.mirror_checks = config.getoption("mirror_checks", 2)
        self.last_mtime = {src: None for src in self.mirror_files}

    def is_stable(self, path):
        """Check whether the file write has completed"""
        try:
            prev_size = os.path.getsize(path)
        except FileNotFoundError:
            return False

        for _ in range(self.mirror_checks):
            time.sleep(30)
            try:
                size = os.path.getsize(path)
            except FileNotFoundError:
                return False
            if size != prev_size:
                return False
            prev_size = size
        return True

    def mirror(self):
        for src, dst in self.mirror_files.items():
            if not os.path.exists(src):
                logging.warning(f"Source file does not exist: {src}")
                continue

            try:
                mtime = os.path.getmtime(src)
            except OSError as e:
                logging.error(f"Failed to get file modification time: {src} - {e}")
                continue

            if self.last_mtime[src] is None or mtime > self.last_mtime[src]:
                logging.info(f"File update detected: {src}")
                if self.is_stable(src):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    try:
                        shutil.copy2(src, dst)
                        self.last_mtime[src] = mtime
                        logging.info(f"Backup completed: {src} -> {dst}")
                    except Exception as e:
                        logging.error(f"Backup failed: {src} -> {dst} - {e}")
                else:
                    logging.info(f"File is still being written, skipping: {src}")

    def start(self, **kwargs):
        logging.info("Mirror Files Service started")
        while True:
            self.mirror()
            time.sleep(self.mirror_interval)

    def stop(self, **kwargs):
        pass

    def restart(self, **kwargs):
        pass


@register_service("core/service/mirror_folders")
class MirrorFoldersService(GenericService):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        config.set_default_section("core/service/mirror_folders")
        self.ip = config.getoption("ip", "0.0.0.0")
        self.port = config.getoption("port", 11222)
        self.name = config.getoption("processname", "core_mirror_folders_service")
        self.mirror_folders = config.getoption(
            "mirror_folders", {}
        )  # {src_dir: dst_dir}
        self.mirror_folders = {k: v for k, v in self.mirror_folders.items() if k != v}
        self.mirror_interval = config.getoption("mirror_interval", 120)
        self.mirror_checks = config.getoption("mirror_checks", 2)

        # Store the last modification time for each file
        self.last_mtime = {}

    def is_stable(self, path):
        """Check whether the file write has completed"""
        try:
            prev_size = os.path.getsize(path)
        except FileNotFoundError:
            return False

        for _ in range(self.mirror_checks):
            time.sleep(30)
            try:
                size = os.path.getsize(path)
            except FileNotFoundError:
                return False
            if size != prev_size:
                return False
            prev_size = size
        return True

    def mirror(self):
        """Traverse all folders and synchronize"""
        for src_dir, dst_dir in self.mirror_folders.items():
            if not os.path.exists(src_dir):
                logging.warning(f"Source folder does not exist: {src_dir}")
                continue

            for root, _, files in os.walk(src_dir):
                rel_path = os.path.relpath(root, src_dir)
                target_root = os.path.join(dst_dir, rel_path)

                for file_name in files:
                    src_path = os.path.join(root, file_name)
                    dst_path = os.path.join(target_root, file_name)

                    try:
                        mtime = os.path.getmtime(src_path)
                    except OSError as e:
                        logging.error(
                            f"Failed to get file modification time: {src_path} - {e}"
                        )
                        continue

                    if (
                        self.last_mtime.get(src_path) is None
                        or mtime > self.last_mtime[src_path]
                    ):
                        logging.info(f"File update detected: {src_path}")
                        if self.is_stable(src_path):
                            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                            try:
                                shutil.copy2(src_path, dst_path)
                                self.last_mtime[src_path] = mtime
                                logging.info(
                                    f"Backup completed: {src_path} -> {dst_path}"
                                )
                            except Exception as e:
                                logging.error(
                                    f"Backup failed: {src_path} -> {dst_path} - {e}"
                                )
                        else:
                            logging.info(
                                f"File is still being written, skipping: {src_path}"
                            )

    def start(self, **kwargs):
        logging.info("Mirror Folders Service started")
        while True:
            self.mirror()
            time.sleep(self.mirror_interval)

    def stop(self, **kwargs):
        pass

    def restart(self, **kwargs):
        pass
