# Scripts (`python3 -m`)

One-off utility scripts are run directly as Python modules:

```bash
python3 -m unitorch.cli.<module> [args...]
```

---

## generate_skills

Generates per-tool `SKILL.md` files for all registered copilot tools into a
target folder. Useful for bootstrapping Claude skill documentation.

```bash
python3 -m unitorch.cli.copilots.tools.generate_skills --folder ./skills
```

Each tool's output is written to
`<folder>/<tool_name_with_slashes_replaced_by_underscores>/SKILL.md`,
containing the tool's `describe()` text and `usage()` snippet.
