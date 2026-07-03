# Scripts (`python3 -m`)

One-off utility scripts are run directly as Python modules:

```bash
python3 -m unitorch.cli.<module> [args...]
```

---

## Copilot Skill Document Exporter

Installs generated copilot-tool skills and hand-written skills from
`src/unitorch/cli/skills` into a target folder. These skill documents make
unitorch's ML lifecycle capabilities easier for agents to discover, plan, and
reuse.

```bash
python3 -m unitorch.cli.copilots.skills install --folder ./skills
python3 -m unitorch.cli.copilots.skills uninstall --folder ./skills
```

Each tool's output is written to
`<folder>/unitorch-<skill-safe-tool-name>/SKILL.md`,
containing skill frontmatter, CLI/Python usage, parameter metadata, and any
remote `unitorch-fastapi` route metadata declared by the copilot tool.
`install` defaults to `./skills`, and `uninstall` removes the matching
`SKILL.md` files from that folder. Use `--name all` or omit `--name` to include
all generated and hand-written skills.

Extension packages can expose additional hand-written skills by shipping a
`unitorch.cli.skills` namespace package that contains skill subfolders in
`<skill-name>/SKILL.md` format. Installed extension package skill directories
are discovered automatically.