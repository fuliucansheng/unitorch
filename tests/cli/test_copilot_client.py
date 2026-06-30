# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import io

from PIL import Image

from unitorch.cli import (
    GenericCopilotRemoteSpec,
    register_copilot_tool,
    registered_copilot_tool,
)
from unitorch.cli.copilots import CopilotClient, call_fastapi, get_copilot_tool
from unitorch.cli.copilots import client as client_module
from unitorch.cli.copilots.skills import (
    export_copilot_skill_documents,
    install_copilot_skill_documents,
    render_copilot_skill_markdown,
    uninstall_copilot_skill_documents,
)


class DummyResponse:
    def __init__(self, content=b"", json_data=None):
        self.content = content
        self.text = (
            content.decode("utf-8", errors="ignore")
            if isinstance(content, bytes)
            else str(content)
        )
        self._json_data = json_data

    def raise_for_status(self):
        return None

    def json(self):
        return self._json_data


def test_call_fastapi_posts_image_as_jpeg(monkeypatch):
    seen = {}

    def fake_post(url, params=None, files=None, timeout=None):
        seen["url"] = url
        seen["params"] = params
        seen["timeout"] = timeout
        filename, file_obj, content_type = files["image"]
        seen["filename"] = filename
        seen["content_type"] = content_type
        seen["payload"] = file_obj.read()
        return DummyResponse(json_data={"ok": True})

    monkeypatch.setattr(client_module.requests, "post", fake_post)

    image = Image.new("RGB", (2, 2), (255, 0, 0))
    result = call_fastapi(
        "http://127.0.0.1:8000/generate",
        params={"text": "hello", "skip": None},
        images={"image": image},
        timeout=3,
    )

    assert result == {"ok": True}
    assert seen["url"] == "http://127.0.0.1:8000/generate"
    assert seen["params"] == {"text": "hello"}
    assert seen["timeout"] == 3
    assert seen["filename"] == "image.jpg"
    assert seen["content_type"] == "image/jpeg"
    assert seen["payload"].startswith(b"\xff\xd8")


def test_call_fastapi_posts_video_and_returns_bytes(monkeypatch, tmp_path):
    seen = {}
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"fake-video")

    def fake_post(url, params=None, files=None, timeout=None):
        filename, file_obj, content_type = files["video"]
        seen["filename"] = filename
        seen["content_type"] = content_type
        seen["payload"] = file_obj.read()
        return DummyResponse(content=b"generated-video")

    monkeypatch.setattr(client_module.requests, "post", fake_post)

    result = call_fastapi(
        "http://127.0.0.1:8000/generate",
        videos={"video": str(video_path)},
        resp_type="video",
    )

    assert result == b"generated-video"
    assert seen == {
        "filename": "input.mp4",
        "content_type": "video/mp4",
        "payload": b"fake-video",
    }


def test_call_fastapi_gets_image_response(monkeypatch):
    buffer = io.BytesIO()
    Image.new("RGB", (3, 4), (0, 255, 0)).save(buffer, format="PNG")

    def fake_get(url, params=None, timeout=None):
        return DummyResponse(content=buffer.getvalue())

    monkeypatch.setattr(client_module.requests, "get", fake_get)

    image = call_fastapi(
        "http://127.0.0.1:8000/generate",
        req_type="GET",
        resp_type="image",
    )

    assert image.mode == "RGB"
    assert image.size == (3, 4)


def test_client_invoke_maps_remote_media_fields(monkeypatch):
    tool_name = "tests/copilot/remote_media"
    calls = {}

    @register_copilot_tool(
        name=tool_name,
        remote=GenericCopilotRemoteSpec(
            route="/core/fastapi/test/generate",
            param_fields={"prompt": "text"},
            image_fields={"image_path": "image"},
            video_fields={"video_path": "video"},
            resp_type="video",
        ),
    )
    def remote_media_tool():
        return None

    def fake_call_fastapi(**kwargs):
        calls.update(kwargs)
        return b"ok"

    monkeypatch.setattr(client_module, "call_fastapi", fake_call_fastapi)

    try:
        result = CopilotClient("http://127.0.0.1:8000/").invoke(
            tool_name,
            prompt="hello",
            image_path="image.jpg",
            video_path="input.mp4",
            seed=1123,
        )
    finally:
        registered_copilot_tool.pop(tool_name, None)

    assert result == b"ok"
    assert calls == {
        "url": "http://127.0.0.1:8000/core/fastapi/test/generate",
        "params": {"text": "hello", "seed": 1123},
        "images": {"image": "image.jpg"},
        "videos": {"video": "input.mp4"},
        "req_type": "POST",
        "resp_type": "video",
        "timeout": 60,
    }


def test_client_starts_fastapi_from_config(monkeypatch, tmp_path):
    calls = {}

    class DummyProcess:
        def __init__(self, cmd, stdin=None):
            calls["cmd"] = cmd
            calls["stdin"] = stdin
            self.terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            calls["wait_timeout"] = timeout
            return 0

        def kill(self):
            calls["killed"] = True

    class HealthResponse:
        status_code = 200

    config_path = tmp_path / "fastapi.ini"
    config_path.write_text("[core/cli]\n", encoding="utf-8")

    monkeypatch.setattr(client_module.shutil, "which", lambda name: "/bin/unitorch-fastapi")
    monkeypatch.setattr(client_module.subprocess, "Popen", DummyProcess)
    monkeypatch.setattr(
        client_module.requests,
        "get",
        lambda url, timeout=None: calls.setdefault("health", (url, timeout))
        and HealthResponse(),
    )

    client = CopilotClient(
        config=str(config_path),
        host="0.0.0.0",
        port=8765,
        startup_timeout=1,
    )
    try:
        assert client.endpoint == "http://127.0.0.1:8765"
        assert calls["cmd"] == [
            "/bin/unitorch-fastapi",
            str(config_path),
            "--host=0.0.0.0",
            "--port=8765",
        ]
        assert calls["health"] == ("http://127.0.0.1:8765/health-check", 2)
    finally:
        client.stop()

    assert calls["wait_timeout"] == 10


def test_client_auto_selects_port_for_fastapi_config(monkeypatch, tmp_path):
    calls = {}

    class DummyProcess:
        def __init__(self, cmd, stdin=None):
            calls["cmd"] = cmd

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

        def kill(self):
            return None

    class HealthResponse:
        status_code = 200

    config_path = tmp_path / "fastapi.ini"
    config_path.write_text("[core/cli]\n", encoding="utf-8")

    monkeypatch.setattr(client_module, "_find_available_port", lambda host: 9876)
    monkeypatch.setattr(client_module.shutil, "which", lambda name: "/bin/unitorch-fastapi")
    monkeypatch.setattr(client_module.subprocess, "Popen", DummyProcess)
    monkeypatch.setattr(
        client_module.requests,
        "get",
        lambda url, timeout=None: HealthResponse(),
    )

    client = CopilotClient(config=str(config_path))
    try:
        assert client.port == 9876
        assert client.endpoint == "http://127.0.0.1:9876"
        assert calls["cmd"][-1] == "--port=9876"
    finally:
        client.stop()


def test_copilot_tool_invoke_is_canonical_call_path():
    tool_name = "tests/copilot/local_invoke"

    @register_copilot_tool(name=tool_name)
    def local_invoke(prompt: str, top_k: int = 3):
        return {"prompt": prompt, "top_k": top_k}

    try:
        copilot_tool = get_copilot_tool(tool_name)
        result = copilot_tool.invoke(prompt="hello", top_k=2)
    finally:
        registered_copilot_tool.pop(tool_name, None)

    assert result == {"prompt": "hello", "top_k": 2}
    assert "prompt" in copilot_tool.signature.parameters
    assert copilot_tool.type_hints["prompt"] is str


def test_export_copilot_skill_documents_writes_skill_markdown(tmp_path):
    tool_name = "tests/copilot/skill_writer"

    @register_copilot_tool(
        name=tool_name,
        description="Write a test copilot skill.",
        remote=GenericCopilotRemoteSpec(
            route="/core/fastapi/test/generate",
            param_fields={"prompt": "text"},
        ),
    )
    def skill_writer(prompt: str, top_k: int = 3):
        return {"prompt": prompt, "top_k": top_k}

    try:
        markdown = render_copilot_skill_markdown(tool_name)
        outputs = export_copilot_skill_documents(tool_name, folder=str(tmp_path))
    finally:
        registered_copilot_tool.pop(tool_name, None)

    skill_path = tmp_path / "unitorch-tests-copilot-skill_writer" / "SKILL.md"
    assert outputs == {tool_name: str(skill_path)}
    assert skill_path.exists()
    content = skill_path.read_text(encoding="utf-8")
    assert content == markdown
    assert content.startswith("---\n")
    assert 'name: "unitorch-tests-copilot-skill_writer"' in content
    assert "# tests/copilot/skill_writer" in content
    assert "unitorch-copilot-cli tests/copilot/skill_writer" in content
    assert "| `prompt` | `str` | yes |  |" in content
    assert "| `top_k` | `int` | no | `3` |" in content
    assert "from unitorch.cli.copilots import get_copilot_tool" in content
    assert "result = tool.invoke()" in content
    assert "CopilotClient" in content


def test_install_and_uninstall_copilot_skill_documents(tmp_path):
    tool_name = "tests/copilot/skill_installer"

    @register_copilot_tool(name=tool_name)
    def skill_installer(prompt: str):
        return {"prompt": prompt}

    try:
        installed = install_copilot_skill_documents(tool_name, folder=str(tmp_path))
        skill_path = tmp_path / "unitorch-tests-copilot-skill_installer" / "SKILL.md"
        assert installed == {tool_name: str(skill_path)}
        assert skill_path.exists()

        removed = uninstall_copilot_skill_documents(tool_name, folder=str(tmp_path))
        assert removed == {tool_name: str(skill_path)}
        assert not skill_path.exists()
        assert not skill_path.parent.exists()
    finally:
        registered_copilot_tool.pop(tool_name, None)


def test_install_and_uninstall_all_includes_manual_skills(tmp_path):
    installed = install_copilot_skill_documents("all", folder=str(tmp_path))

    skill_path = tmp_path / "unitorch-config-ini" / "SKILL.md"
    assert installed["config-ini"] == str(skill_path)
    assert skill_path.exists()
    assert 'name: "unitorch-config-ini"' in skill_path.read_text(encoding="utf-8")

    removed = uninstall_copilot_skill_documents("all", folder=str(tmp_path))
    assert removed["config-ini"] == str(skill_path)
    assert not skill_path.exists()
