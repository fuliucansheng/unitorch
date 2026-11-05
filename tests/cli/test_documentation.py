# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

"""Portable public examples, independent of optional model backends."""

import ast
import configparser
import html
import inspect
import json
from pathlib import Path
import re
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


def _public_text_paths(root):
    """Yield public text files in a stable order, without consulting Git."""
    public_entries = (
        "README.md", "AGENTS.md", "CLAUDE.md", "LICENSE", "MANIFEST.in",
        "mkdocs.yml", "package.json", "pyproject.toml", "requirements.txt",
        "setup.cfg", "setup.py", ".gitignore", ".github", "dockers",
        "src", "tests", "wiki", "docs", "examples", "skills",
    )
    text_suffixes = {
        ".py", ".pyi", ".md", ".rst", ".txt", ".ini", ".cfg", ".toml",
        ".yaml", ".yml", ".json", ".jsonl", ".html", ".htm", ".css",
        ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx", ".map",
        ".sh", ".bash", ".zsh", ".c", ".cc", ".cpp", ".h", ".hpp",
        ".cu", ".cuh", ".svg", ".xml", ".csv", ".tsv", ".ipynb",
    }
    text_names = {"LICENSE", "Dockerfile", "Makefile", "MANIFEST.in", ".gitignore"}
    excluded_dirs = {"__pycache__", "venv", "node_modules"}

    def visit(path):
        # Check before is_dir/is_file: both follow symlinks, including links
        # at an allowlisted root and links whose names look like text files.
        if path.is_symlink():
            return
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.is_symlink():
                    continue
                if child.is_dir() and (
                    child.name.startswith(".")
                    or child.name in excluded_dirs
                    or child.name.endswith(".egg-info")
                ):
                    continue
                yield from visit(child)
        elif path.is_file() and (
            path.suffix.lower() in text_suffixes or path.name in text_names
        ):
            yield path

    for name in sorted(public_entries):
        yield from visit(root / name)


def _forbidden_public_content(root):
    forbidden = re.compile(
        rb"unitorch\.cli\.scripts\.flux[23]|"
        rb"/(?:home|data|Users)/decu\b|[.]hermes[/]skills"
    )
    matches = []
    for path in _public_text_paths(root):
        with path.open("rb") as source:
            if any(forbidden.search(line) for line in source):
                matches.append(path.relative_to(root).as_posix())
    return matches


def _source_signature(module, name):
    """Read a function or constructor signature without executing its module."""
    path = ROOT / "src" / (module.replace(".", "/") + ".py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definition = next(node for node in tree.body if getattr(node, "name", None) == name)
    constructor = isinstance(definition, ast.ClassDef)
    if constructor:
        definition = next(
            node for node in definition.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
    args = definition.args
    positional = args.posonlyargs + args.args
    required = len(positional) - len(args.defaults)
    parameters = []
    for index, arg in enumerate(positional):
        if constructor and index == 0:
            continue
        kind = (inspect.Parameter.POSITIONAL_ONLY if index < len(args.posonlyargs)
                else inspect.Parameter.POSITIONAL_OR_KEYWORD)
        default = inspect.Parameter.empty if index < required else None
        parameters.append(inspect.Parameter(arg.arg, kind, default=default))
    if args.vararg:
        parameters.append(inspect.Parameter(args.vararg.arg, inspect.Parameter.VAR_POSITIONAL))
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        parameters.append(inspect.Parameter(
            arg.arg, inspect.Parameter.KEYWORD_ONLY,
            default=inspect.Parameter.empty if default is None else None,
        ))
    if args.kwarg:
        parameters.append(inspect.Parameter(args.kwarg.arg, inspect.Parameter.VAR_KEYWORD))
    return inspect.Signature(parameters)


class DocumentationTest(unittest.TestCase):
    def test_tracked_content_has_no_private_paths_or_obsolete_commands(self):
        self.assertEqual(_forbidden_public_content(ROOT), [])

    def test_public_content_scan_is_scoped_and_skips_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # No .git directory: source archives must get the same coverage.
            public = ["README.md", "src/a.py", "tests/b.py", "wiki/c.md",
                      "docs/d.html", "examples/e.ini", "skills/f/SKILL.md",
                      "pyproject.toml"]
            forbidden = [
                "unitorch.cli.scripts." + f"flux{version}" for version in (2, 3)
            ] + [f"/{parent}/decu/private" for parent in ("home", "data", "Users")]
            forbidden.append(".hermes" + "/skills")
            for index, name in enumerate(public):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("safe first line\n" + forbidden[index % len(forbidden)])
            for name in ("local-notes.md", "runs/log.txt", ".cache/log.txt",
                         "src/weights.bin", "src/.cache/log.txt",
                         "src/__pycache__/log.txt", "src/node_modules/log.js"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(forbidden[0])
            (root / "src/link.py").symlink_to(root / "local-notes.md")
            (root / "src/link-dir").symlink_to(root / "runs", target_is_directory=True)
            (root / "src/loop").symlink_to(root, target_is_directory=True)
            (root / "src/broken.py").symlink_to(root / "missing")
            (root / "setup.py").symlink_to(root / "local-notes.md")
            (root / "dockers").symlink_to(root / "runs", target_is_directory=True)
            expected = sorted(public)
            self.assertEqual(
                [path.relative_to(root).as_posix() for path in _public_text_paths(root)],
                expected,
            )
            self.assertEqual(_forbidden_public_content(root), expected)

    def test_flux_guide_python_examples_match_source_signatures(self):
        guide_apis = {
            "flux2_finetuning.md": {
                ("unitorch.models.diffusers.checkpoint_flux2", "export_zero_checkpoint"),
            },
            "flux3_finetuning.md": {
                ("unitorch.models.diffusers.caption_flux3", "Flux3SourceCaptioner"),
                ("unitorch.datasets.flux3_assets", "prepare_captions"),
                ("unitorch.datasets.flux3_assets", "validate_assets"),
                ("unitorch.tasks.flux3", "train_assets"),
            },
        }
        for guide, expected in guide_apis.items():
            path = ROOT / "wiki/guides" / guide
            source = path.read_text(encoding="utf-8")
            signatures = {api: _source_signature(*api) for api in sorted(expected)}
            seen = set()
            blocks = re.finditer(r"^```python\s*\n(.*?)^```\s*$", source, re.M | re.S)
            for block in blocks:
                line = source.count("\n", 0, block.start(1)) + 1
                tree = ast.parse(block[1], filename=f"{guide}:{line}")
                imports = {
                    alias.asname or alias.name: (node.module, alias.name)
                    for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                    for alias in node.names
                }
                for call in ast.walk(tree):
                    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
                        continue
                    api = imports.get(call.func.id)
                    if api not in signatures:
                        continue
                    seen.add(api)
                    with self.subTest(guide=guide, line=line + call.lineno - 1, api=api[1]):
                        self.assertFalse(any(isinstance(arg, ast.Starred) for arg in call.args))
                        self.assertTrue(all(keyword.arg is not None for keyword in call.keywords))
                        # Bind placeholders, never evaluate example expressions,
                        # defaults, annotations, or heavyweight optional imports.
                        try:
                            signatures[api].bind(
                                *[None for arg in call.args],
                                **{keyword.arg: None for keyword in call.keywords},
                            )
                        except TypeError as error:
                            self.fail(str(error))
            self.assertEqual(seen, expected, f"Missing documented API calls in {guide}")

    def test_flux_scripts_removed_and_package_apis_preserved(self):
        for version in (2, 3):
            self.assertFalse((ROOT / f"src/unitorch/cli/scripts/flux{version}.py").exists())
        for module in (
            "models/diffusers/checkpoint_flux2.py",
            "models/diffusers/caption_flux3.py",
            "datasets/flux3_assets.py",
            "tasks/flux3.py",
        ):
            self.assertTrue((ROOT / "src/unitorch" / module).is_file())

    def test_flux2_example_paths_are_relative(self):
        examples = sorted((ROOT / "examples/configs/diffusion").glob("*/flux2.*.*.ini"))
        self.assertEqual(len(examples), 8)
        for path in examples:
            with self.subTest(config=path.name, task=path.parent.name):
                config = configparser.ConfigParser(
                    interpolation=configparser.ExtendedInterpolation(),
                )
                config.read(path)
                for key in ("cache_dir", "from_ckpt_dir", "train_file", "dev_file", "test_file"):
                    self.assertTrue(config["core/cli"][key].startswith("./"), key)
                self.assertEqual(
                    config["core/dataset/ast/train"]["data_files"],
                    config["core/cli"]["train_file"],
                )

    def test_generated_skill_install_payload_matches_wiki(self):
        search = json.loads((ROOT / "docs/search/search_index.json").read_text())
        for name in ("copilot_tools", "scripts"):
            with self.subTest(page=name):
                source = (ROOT / f"wiki/cli/{name}.md").read_text()
                payload = json.loads(re.search(r"```json\n(.*?)\n```", source, re.S)[1])
                self.assertEqual(payload["folder"], "./agent-skills")
                rendered = (ROOT / f"docs/cli/{name}/index.html").read_text()
                rendered = html.unescape(re.sub(r"<[^>]+>", "", rendered))
                indexed = "\n".join(
                    doc["text"] for doc in search["docs"]
                    if doc["location"].startswith(f"cli/{name}/")
                )
                for text in (rendered, indexed):
                    # Compare parsed payloads so syntax highlighting and search
                    # whitespace do not obscure a stale installation example.
                    example = re.search(r'\{\s*"repo":.*?\}', text, re.S)
                    self.assertIsNotNone(example)
                    self.assertEqual(json.loads(example[0]), payload)
