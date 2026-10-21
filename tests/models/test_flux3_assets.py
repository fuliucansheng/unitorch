"""Provenance checks do not require model weights or diffusion extras."""
import hashlib
import json

import pytest
from PIL import Image

from unitorch.datasets.flux3_assets import validate_assets, prepare_captions


def make_dataset(root):
    dataset = root / "dataset"
    dataset.mkdir(parents=True)
    assets = root / "assets" / "1024x1536"
    assets.mkdir(parents=True)
    records = []
    for i, split in enumerate(("train", "val")):
        row = dict(pair_id=str(i), group_id=str(i), split=split, prompt="extract the visible clothing")
        for side, color in (("source", (i * 80, 0, 0)), ("result", (0, i * 80 + 50, 0))):
            path = assets / f"{i}-{side}.png"
            Image.new("RGB", (16, 16), color).save(path)
            row[side + "_local"] = str(path)
            row[side + "_asset_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(row)
        (dataset / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
    (dataset / "test.jsonl").write_text("")
    (dataset / "full.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return records


def test_manifest_images_and_leakage(tmp_path):
    rows = make_dataset(tmp_path)
    result = validate_assets(str(tmp_path), str(tmp_path / "report"), expected_counts={"train": 1, "val": 1, "test": 0},
                             expected_image_count=4, geometry=(16, 16), require_source_manifest=False)
    assert result["valid"] and result["images_verified"] == 4
    (tmp_path / "assets/1024x1536/0-source.png").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA|image"):
        validate_assets(str(tmp_path), str(tmp_path / "report2"), expected_counts={"train": 1, "val": 1, "test": 0},
                        expected_image_count=4, geometry=(16, 16), require_source_manifest=False)


def test_captions_restart_and_source_only_boundary(tmp_path):
    make_dataset(tmp_path)
    calls = []
    def captioner(source, instruction):
        calls.append((source, instruction))
        assert source.endswith("source.png")
        return "A full-color overhead product photograph on a plain white background with one red cotton garment and a “玫瑰” motif."
    options = dict(dataset_path=str(tmp_path / "dataset/train.jsonl"), output_root=str(tmp_path / "captions"),
                   captioner=captioner, captioner_id="unit-source-only", max_records=None)
    first = prepare_captions(**options)
    second = prepare_captions(**options)
    assert first["complete"] and second["complete"] and len(calls) == 1
    row = json.loads((tmp_path / "captions/train.jsonl").read_text())
    assert row["caption_provenance"]["conditioning"] == "source_image_and_instruction_only"
    assert "result" not in row["caption_provenance"]["inputs"]
