# Repository Guidelines

## Project Structure & Module Organization
`src/unitorch/` contains the library code. Core model implementations live under `src/unitorch/models/`, while `src/unitorch/cli/` and `src/unitorch/cli/models/` provide the configuration-driven CLI layer. Shared runtime pieces are split across `datasets/`, `tasks/`, `losses/`, `ops/`, `scores/`, `schedulers/`, and `utils/`. Add new modules in the closest existing package and keep filenames in `snake_case`.

`tests/` holds automated checks; mirror the package area you touch when adding coverage, for example `tests/cli/test_decorators.py`. Runnable `.ini` examples live in `examples/configs/`. Documentation source belongs in `wiki/`; `docs/` is the generated MkDocs site output, so update `wiki/` first.

## Build, Test, and Development Commands
`python -m pip install -e .` installs an editable development copy.

`python -m pip install -e ".[docs]"` adds the MkDocs toolchain. Use `".[all]"` only when you need every optional backend.

`pytest tests` runs the test suite. During iteration, target a file directly, such as `pytest tests/cli/test_decorators.py`.

`mkdocs serve` previews the documentation locally from `wiki/`, and `mkdocs build` regenerates the static site in `docs/`.

`torchrun --no_python --nproc_per_node 4 unitorch-train examples/configs/generation/bart.ini ...` is the standard multi-GPU CLI pattern for manual integration checks.

## Coding Style & Naming Conventions
Use 4-space indentation, `snake_case` for modules/functions/files, and `CamelCase` for classes. Follow the surrounding import order and docstring style; new public APIs should include concise docstrings and type hints where practical. No repo-level Ruff or Black configuration is checked in, so match the local style instead of reformatting unrelated code.

## Testing Guidelines
Write tests with `pytest`; existing tests also use `absl.testing.parameterized` when parameterization helps. Name files `test_<feature>.py` and keep each test focused on one behavior or regression. Coverage is currently light, so new features and bug fixes should ship with at least one targeted test and, when relevant, an updated example config.

## Commit & Pull Request Guidelines
Recent commits use short, lower-case subjects such as `update qwen` and `clean up code`. Keep commit titles brief, imperative, and scoped to one change. Pull requests should describe the behavior change, list the commands you ran (for example `pytest tests` or `mkdocs build`), and link any related issue plus the affected config or example paths. Include screenshots only when documentation or serving output changes.
