# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.
"""Immutable WearIt validation and resumable source-only scene captions."""

from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile

from PIL import Image


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _rows(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_assets(dataset_root: str, output_root: str, expected_counts=None,
                    expected_image_count: int = 58026, geometry=(1024, 1536),
                    require_source_manifest: bool = True, workers: int = 8):
    """Check SHA256, full PNG decoding, exact inherited split and no shared assets.

    Source files are only read. Writes a new checksummed image manifest and
    DATA_VALIDATED.json under output_root only after every check succeeds.
    """
    root, output = Path(dataset_root), Path(output_root)
    dataset = root / "dataset"
    output.mkdir(parents=True, exist_ok=True)
    counts = expected_counts or {"train": 31640, "val": 200, "test": 0}
    manifest = json.loads((root / "DATASET_READY.json").read_text()) if require_source_manifest else {}
    if require_source_manifest:
        expected_input = "20402a0003242cb52c02c097128c3e8735d14500b9aa84f2081b41816e645356"
        if manifest.get("input_sha256") != expected_input or file_sha256(root / "input.tsv") != expected_input:
            raise ValueError("Original input snapshot SHA256 mismatch")
        for name, digest in manifest["file_sha256"].items():
            if file_sha256(dataset / name) != digest:
                raise ValueError(f"Dataset manifest SHA256 mismatch: {name}")
    owners, images, pairs = {}, {}, {}
    split_rows = {}
    for split, count in counts.items():
        rows = _rows(dataset / f"{split}.jsonl")
        split_rows[split] = rows
        if len(rows) != count:
            raise ValueError(f"Expected {count} {split} rows, found {len(rows)}")
        for row in rows:
            if row["split"] != split or row["pair_id"] in pairs:
                raise ValueError("Duplicate pair or split mismatch")
            pairs[row["pair_id"]] = row
            identifiers = ["group:" + row["group_id"]]
            for side in ("source", "result"):
                path = str(Path(row[side + "_local"]).resolve())
                if not Path(path).is_relative_to(root.resolve() / "assets"):
                    raise ValueError("Image path escapes the read-only dataset assets")
                digest = row[side + "_asset_sha256"]
                if path in images and images[path] != digest:
                    raise ValueError("Conflicting image SHA256 records")
                images[path] = digest
                identifiers.append("asset:" + digest)
                for field in ("content_hash", "normalized_hash"):
                    if side + "_" + field in row:
                        identifiers.append(field + ":" + row[side + "_" + field])
            for identity in identifiers:
                if identity in owners and owners[identity] != split:
                    raise ValueError("Train/validation/test leakage detected")
                owners[identity] = split
    full = _rows(dataset / "full.jsonl")
    if len(full) != len(pairs) or any(pairs.get(row["pair_id"]) != row for row in full):
        raise ValueError("Inherited split differs from the full manifest")
    files = sorted((root / "assets" / "1024x1536").rglob("*.png"))
    if len(files) != expected_image_count:
        raise ValueError(f"Expected {expected_image_count} PNGs, found {len(files)}")
    if not set(images).issubset({str(p.resolve()) for p in files}):
        raise ValueError("A paired image is missing")
    def check(path):
        digest = file_sha256(path)
        if str(path.resolve()) in images and digest != images[str(path.resolve())]:
            raise ValueError(f"Image SHA256 mismatch: {path}")
        try:
            with Image.open(path) as image:
                image.load()
                if image.format != "PNG" or image.size != tuple(geometry) or image.mode != "RGB":
                    raise ValueError(f"Unexpected image geometry/mode: {path}")
        except OSError as exc:
            raise ValueError(f"Invalid image: {path}") from exc
        return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}
    fd, temporary = tempfile.mkstemp(dir=output, suffix=".jsonl")
    verified = total_bytes = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=workers) as pool:
            for item in pool.map(check, files):
                handle.write(json.dumps(item) + "\n")
                total_bytes += item["bytes"]
                verified += 1
                if verified % 1000 == 0:
                    _publish(output / "validation_progress.json", {"verified": verified, "total": len(files)})
        destination = output / "images.sha256.jsonl"
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    result = dict(valid=True, dataset_root=str(root), geometry=list(geometry), split_counts=counts,
                  images_verified=verified, paired_images_verified=len(images), total_bytes=total_bytes,
                  split_group_counts={s: len({r["group_id"] for r in rows}) for s, rows in split_rows.items()},
                  source_manifest_sha256=file_sha256(dataset / "full.jsonl"),
                  image_manifest_sha256=file_sha256(destination), leakage_free=True,
                  original_detail_note=manifest.get("original_detail_note", "Not evaluated in synthetic test"))
    _publish(output / "DATA_VALIDATED.json", result)
    return result


def prepare_captions(dataset_path: str, output_root: str, captioner, captioner_id: str,
                     max_records=None, stop_file=None):
    """Expand captions from source image + instruction; commit each result.

    captioner is called with ONLY (source_path, instruction), never with a row
    or target. Its immutable identity must include model/prompt configuration.
    A SQLite cache survives interruption; JSONL/TSV publication is atomic.
    The split, pair/group IDs, paths and algorithm metadata are inherited.
    """
    rows = _rows(dataset_path)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    split = Path(dataset_path).stem
    db = sqlite3.connect(output / "captions.sqlite3", timeout=120)
    db.execute("CREATE TABLE IF NOT EXISTS captions (key TEXT PRIMARY KEY, caption TEXT NOT NULL, provenance TEXT NOT NULL)")
    generated = []
    try:
        for row in rows:
            if (max_records is not None and len(generated) >= max_records) or (stop_file and Path(stop_file).exists()):
                break
            source = row["source_local"]
            digest = file_sha256(source)
            if digest != row["source_asset_sha256"]:
                raise ValueError("Source SHA256 changed before caption generation")
            identity = dict(source_sha256=digest, instruction=row["prompt"], captioner_id=captioner_id)
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            cached = db.execute("SELECT caption, provenance FROM captions WHERE key=?", (key,)).fetchone()
            if cached is None:
                caption = captioner(source, row["prompt"]).strip()
                if len(caption.split()) < 12:
                    raise ValueError("Caption expander did not produce a full scene description")
                provenance = dict(conditioning="source_image_and_instruction_only", inputs=["source_local", "prompt"],
                                  **identity, caption_sha256=hashlib.sha256(caption.encode()).hexdigest())
                db.execute("INSERT INTO captions VALUES (?, ?, ?)", (key, caption, json.dumps(provenance)))
                db.commit()
            else:
                caption, provenance = cached[0], json.loads(cached[1])
            generated.append({**row, "instruction": row["prompt"], "prompt": caption, "caption_provenance": provenance})
            _publish(output / f"{split}.progress.json", {"completed": len(generated), "total": len(rows),
                                                        "captioner_id": captioner_id, "complete": False})
        for suffix in ("jsonl", "tsv"):
            fd, temporary = tempfile.mkstemp(dir=output, suffix="." + suffix)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                if suffix == "jsonl":
                    for row in generated:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                else:
                    writer = csv.writer(handle, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE, escapechar="\\")
                    for row in generated:
                        writer.writerow([row["prompt"].replace("\n", " ").replace("\t", " "), row["source_local"], row["result_local"]])
            os.replace(temporary, output / f"{split}.{suffix}")
        result = dict(completed=len(generated), total=len(rows), complete=len(generated) == len(rows),
                      captioner_id=captioner_id, input_sha256=file_sha256(dataset_path),
                      output_sha256=file_sha256(output / f"{split}.jsonl"),
                      conditioning="source_image_and_instruction_only")
        _publish(output / f"{split}.progress.json", result)
        return result
    finally:
        db.close()
