# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import json
from pathlib import Path

from unitorch.cli import (
    GenericCopilotRemoteSpec,
    register_copilot_tool,
    registered_copilot_tool,
)
from unitorch.cli.copilots.skills import (
    _extract_frontmatter,
    _load_frontmatter,
    export_copilot_skill_documents,
    render_copilot_skill_markdown,
    validate_copilot_skill_documents,
)


def test_render_copilot_skill_frontmatter_metadata_and_sections():
    tool_name = "tests/copilot/render_skill"

    @register_copilot_tool(
        name=tool_name,
        description="Echo a prompt for generated skill tests.",
        tags=("testing", "metadata"),
        remote=GenericCopilotRemoteSpec(
            route="/core/fastapi/tests/render",
            param_fields={"prompt": "text"},
            resp_type="json",
        ),
    )
    def render_skill(prompt: str, count: int = 1):
        return {"prompt": prompt, "count": count}

    try:
        markdown = render_copilot_skill_markdown(tool_name)
        frontmatter = _load_frontmatter(_extract_frontmatter(markdown))
    finally:
        registered_copilot_tool.pop(tool_name, None)

    assert frontmatter["name"] == "unitorch-copilot-tools-tests-copilot-render_skill"
    assert frontmatter["version"]
    assert frontmatter["author"] == "FULIUCANSHENG"
    assert frontmatter["license"] == "MIT"
    assert len(frontmatter["description"]) <= 1024
    assert "Use when an agent needs to invoke" in frontmatter["description"]
    hermes = frontmatter["metadata"]["hermes"]
    assert "testing" in hermes["tags"]
    assert "fastapi" in hermes["tags"]
    assert "unitorch-copilot-tools" in hermes["related_skills"]
    assert "unitorch-serve-fastapi" in hermes["related_skills"]
    assert "serve-fastapi" not in hermes["related_skills"]
    assert frontmatter["related_skills"] == hermes["related_skills"]

    assert "## Overview" in markdown
    assert "## When To Use" in markdown
    assert "## CLI" in markdown
    assert "## Python" in markdown
    assert "## Parameters" in markdown
    assert "## Remote FastAPI" in markdown
    assert "## Verification Checklist" in markdown
    assert "## Common Pitfalls" in markdown
    assert "unitorch-copilot-cli tests/copilot/render_skill --prompt" in markdown
    assert "prompt=\"value\"" in markdown


def test_export_and_validate_copilot_skill_documents(tmp_path):
    folder = tmp_path / ".skills"

    generated = export_copilot_skill_documents(
        name="core/copilot/pkg_infos",
        folder=str(folder),
    )
    result = validate_copilot_skill_documents(folder=str(folder))

    assert result["valid"] is True
    assert result["count"] == 2
    assert "unitorch-copilot-tools" in generated
    assert "core/copilot/pkg_infos" in generated

    parent = folder / "unitorch-copilot-tools" / "SKILL.md"
    child = folder / "unitorch-copilot-tools" / "core-copilot-pkg_infos" / "SKILL.md"
    assert parent.is_file()
    assert child.is_file()

    parent_markdown = parent.read_text(encoding="utf-8")
    parent_frontmatter = _load_frontmatter(_extract_frontmatter(parent_markdown))
    parent_related_skills = parent_frontmatter["metadata"]["hermes"]["related_skills"]

    assert "unitorch-config-ini" in parent_related_skills
    assert "unitorch-train-model" in parent_related_skills
    assert "unitorch-infer-model" in parent_related_skills
    assert "unitorch-serve-fastapi" in parent_related_skills
    assert "config-ini" not in parent_related_skills
    assert "npm run generate-skills" in parent_markdown
    assert "npx unitorch install all --folder .skills --force true" in parent_markdown
    assert "## Registered Tools" in parent_markdown


def test_package_json_exposes_npx_skill_wrapper():
    package = json.loads(Path("package.json").read_text(encoding="utf-8"))
    wrapper = Path("bin/unitorch-skills.js").read_text(encoding="utf-8")

    assert package["bin"]["unitorch"] == "bin/unitorch-skills.js"
    assert package["bin"]["unitorch-skills"] == "bin/unitorch-skills.js"
    assert "--folder .skills" in package["scripts"]["generate-skills"]
    assert package["scripts"]["validate-skills"] == (
        "node ./bin/unitorch-skills.js validate --folder .skills"
    )
    assert "unitorch.cli.copilots.skills" in wrapper
    assert "PYTHONPATH" in wrapper


def test_clawhub_hermeshub_workflow_sanity():
    workflow_path = Path(".github/workflows/publish-skills.yml")
    workflow = workflow_path.read_text(encoding="utf-8")

    assert "ClawHub/HermesHub" in workflow
    assert "workflow_dispatch" in workflow
    assert "npm run generate-skills" in workflow
    assert "npm run validate-skills" in workflow
    assert "CLAWHUB_TOKEN" in workflow
    assert "CLAWHUB_PUBLISH_URL" in workflow
    assert "HERMESHUB_TOKEN" in workflow
    assert "HERMESHUB_PUBLISH_URL" in workflow
    assert "Authorization: Bearer ${CLAWHUB_TOKEN}" in workflow
    assert "Authorization: Bearer ${HERMESHUB_TOKEN}" in workflow
    assert "Authorization: Bearer CLAWHUB_TOKEN" not in workflow
    assert "Authorization: Bearer HERMESHUB_TOKEN" not in workflow
    assert "actions/upload-artifact" in workflow
    assert "SHOULD_PUBLISH" in workflow
    assert "github.event.inputs.publish == 'true'" in workflow

    try:
        import yaml
    except Exception:
        return

    assert yaml.safe_load(workflow)
