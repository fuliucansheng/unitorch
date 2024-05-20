# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import logging
import hashlib
import ast
import abc
import torch
import torch.nn as nn
import configparser
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from copy import copy, deepcopy
from transformers.utils import is_remote_url


# core config class
class CoreConfigureParser(configparser.ConfigParser):
    def __init__(
        self,
        fpath: Optional[str] = "./config.ini",
        params: Optional[Tuple[str]] = [],
    ):
        super().__init__(interpolation=configparser.ExtendedInterpolation())
        self.fpath = fpath
        self.read(fpath)
        for param in params:
            assert len(param) == 3
            k0, k1, v = param
            if not self.has_section(k0):
                self.add_section(k0)
            self.set(k0, k1, str(v))

        self._freeze_section = None

        self._default_section = self.getdefault("core/config", "default_section", None)

    def _getdefault(self, section, option, value=None):
        if not self.has_section(section):
            return value
        if self.has_option(section, option):
            return self.get(section, option)
        return value

    def _ast_replacement(self, node):
        value = node.id
        if value in ("True", "False", "None"):
            return node
        return ast.Str(value)

    def _ast_literal_eval(self, value):
        root = ast.parse(value, mode="eval")
        if isinstance(root.body, ast.BinOp):
            raise ValueError(value)

        for node in ast.walk(root):
            for field, child in ast.iter_fields(node):
                if isinstance(child, list):
                    for index, subchild in enumerate(child):
                        if isinstance(subchild, ast.Name):
                            child[index] = self._ast_replacement(subchild)
                elif isinstance(child, ast.Name):
                    replacement = self._ast_replacement(child)
                    node.__setattr__(field, replacement)
        return ast.literal_eval(root)

    def get(
        self,
        section,
        option,
        raw=False,
        vars=None,
        fallback=configparser._UNSET,
    ):
        value = super().get(
            section,
            option,
            raw=raw,
            vars=vars,
            fallback=fallback,
        )
        if raw:
            return value
        try:
            return self._ast_literal_eval(value)
        except (SyntaxError, ValueError):
            return value

    def set_default_section(self, section):
        self._default_section = section

    def getdefault(self, section, option, value=None):
        return self._getdefault(section, option, value)

    def getoption(self, option, value=None):
        return self.getdefault(self._default_section, option, value)

    def logging(self):
        logging.info("#" * 30, "Config Info".center(20, " "), "#" * 30)
        for sec, item in self.items():
            for k, v in item.items():
                logging.info(
                    sec.rjust(10, " "),
                    ":".center(5, " "),
                    k.ljust(30, " "),
                    ":".center(5, " "),
                    str(v).ljust(30, " "),
                )
        logging.info("#" * 30, "Config Info End".center(20, " "), "#" * 30)

    def save(self, save_path="./config.ini"):
        self.write(open(save_path, "w"))
        return save_path

    def hexsha(self, length=None):
        string = sorted(
            [f"{k}_{kk}_{vv}" for k, v in self.items() for kk, vv in v.items()]
        )
        string = "|".join(string)
        hexsha = hashlib.sha1(string.encode()).hexdigest()
        if length is not None:
            hexsha = hexsha[:length]
        return hexsha

    def __copy__(self):
        setting = [(sec, k, v) for sec in self.sections() for k, v in self[sec].items()]
        return type(self)(self.fpath, setting)

    def __deepcopy__(self, memo):
        setting = [(sec, k, v) for sec in self.sections() for k, v in self[sec].items()]
        return type(self)(self.fpath, setting)
