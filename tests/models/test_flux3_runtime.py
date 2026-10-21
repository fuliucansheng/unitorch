"""Formal training cannot start from incomplete or unreviewed artifacts."""
import json
from pathlib import Path

import pytest

from unitorch.tasks.flux3 import check_admission, check_resources


def test_formal_requires_concrete_review(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        check_admission(str(tmp_path / "data.json"), str(tmp_path / "train.jsonl"),
                        str(tmp_path / "val.jsonl"), str(tmp_path / "proof.json"), None)


def test_resource_gate_refuses_impossible_reserve(tmp_path):
    with pytest.raises(RuntimeError, match="disk|memory"):
        check_resources(str(tmp_path), min_disk_bytes=10**30)


def test_completed_pilot_refuses_another_update(tmp_path, monkeypatch):
    pytest.importorskip("diffusers", minversion="0.41.0.dev0")
    pytest.importorskip("peft")
    from unitorch.tasks import flux3

    latest = tmp_path / "latest"
    latest.mkdir()
    (latest / "trainer.json").write_text(json.dumps({"step": 20}))
    monkeypatch.setattr(flux3, "check_admission", lambda *args: {})
    monkeypatch.setattr(flux3.torch.cuda, "is_available", lambda: False)
    # Reaching the saved limit must reject even before CUDA/model allocation.
    with pytest.raises(ValueError, match="pilot limit already reached"):
        flux3.train_assets("unused", "unused", "unused", "unused", str(tmp_path),
                           pilot_steps=20, resume=True)
