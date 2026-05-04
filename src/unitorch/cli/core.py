# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import ast
import hashlib
import logging
import configparser
from typing import Any, List, Optional, Tuple, Union


class Config(configparser.ConfigParser):
    """Config parser with extended interpolation and safe AST-based value parsing."""

    def __init__(
        self,
        fpath: Optional[str] = "./config.ini",
        params: Optional[List[Tuple[str, str, str]]] = None,
    ):
        super().__init__(interpolation=configparser.ExtendedInterpolation())
        self.fpath = fpath
        self.read(fpath)
        for param in params or []:
            assert len(param) == 3
            section, key, value = param
            if not self.has_section(section):
                self.add_section(section)
            self.set(section, key, str(value))

        self._freeze_section = None
        self._default_section = self._getdefault("core/config", "default_section", None)

    def _getdefault(self, section: str, option: str, value=None):
        if not self.has_section(section):
            return value
        if self.has_option(section, option):
            return self.get(section, option)
        return value

    def _ast_replacement(self, node: ast.Name) -> ast.AST:
        if node.id in ("True", "False", "None"):
            return node
        return ast.Constant(value=node.id)

    def _ast_literal_eval(self, value: str):
        root = ast.parse(value, mode="eval")
        if isinstance(root.body, ast.BinOp):
            raise ValueError(value)

        for node in ast.walk(root):
            for field, child in ast.iter_fields(node):
                if isinstance(child, list):
                    for i, subchild in enumerate(child):
                        if isinstance(subchild, ast.Name):
                            child[i] = self._ast_replacement(subchild)
                elif isinstance(child, ast.Name):
                    node.__setattr__(field, self._ast_replacement(child))

        return ast.literal_eval(root)

    def get(
        self,
        section,
        option,
        raw=False,
        vars=None,
        fallback=configparser._UNSET,
    ):
        value = super().get(section, option, raw=raw, vars=vars, fallback=fallback)
        if raw:
            return value
        try:
            return self._ast_literal_eval(value)
        except (SyntaxError, ValueError):
            return value

    def set_default_section(self, section: str) -> None:
        self._default_section = section

    def getdefault(self, section: str, option: str, value=None):
        return self._getdefault(section, option, value)

    def getoption(self, option: str, value=None):
        return self._getdefault(self._default_section, option, value)

    def logging(self) -> None:
        sep = "#" * 30
        logging.info("%s %s %s", sep, "Config Info".center(20), sep)
        for sec, items in self.items():
            for k, v in items.items():
                logging.info("%-10s : %-30s : %s", sec, k, v)
        logging.info("%s %s %s", sep, "Config Info End".center(20), sep)

    def save(self, save_path: str = "./config.ini") -> str:
        with open(save_path, "w") as f:
            self.write(f)
        return save_path

    def hexsha(self, length: Optional[int] = None) -> str:
        parts = sorted(f"{k}_{kk}_{vv}" for k, v in self.items() for kk, vv in v.items())
        digest = hashlib.sha1("|".join(parts).encode()).hexdigest()
        return digest[:length] if length is not None else digest

    def __copy__(self):
        params = [(sec, k, v) for sec in self.sections() for k, v in self[sec].items()]
        return type(self)(self.fpath, params)

    def __deepcopy__(self, memo):
        params = [(sec, k, v) for sec in self.sections() for k, v in self[sec].items()]
        return type(self)(self.fpath, params)
