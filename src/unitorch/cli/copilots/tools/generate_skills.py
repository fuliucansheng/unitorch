# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import fire
import unitorch.cli
import unitorch.cli.copilots
from unitorch.cli import registered_copilot_tool


def generate(folder):
    os.makedirs(folder, exist_ok=True)
    for name, entry in registered_copilot_tool.items():
        inst = entry["obj"]()
        description = inst.describe()
        usage = inst.usage()
        safe_name = name.replace("/", "_")
        skill_dir = os.path.join(folder, safe_name)
        os.makedirs(skill_dir, exist_ok=True)
        skill_path = os.path.join(skill_dir, "SKILL.md")
        with open(skill_path, "w") as f:
            f.write(f"# {name}\n\n")
            f.write(f"{description}\n\n")
            f.write("## Usage\n\n")
            if isinstance(usage, str):
                f.write(f"```bash\n{usage}\n```\n")
            elif isinstance(usage, list):
                f.write("```bash\n")
                for usage_item in usage:
                    f.write(f"{usage_item}\n")
                f.write("```\n")


if __name__ == "__main__":
    fire.Fire(generate)