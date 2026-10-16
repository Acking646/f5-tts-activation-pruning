"""Validate real speech data and create leakage-resistant splits/F5 metadata."""

import csv
import hashlib
from pathlib import Path

from .audio import wav_info
from .io import read_jsonl, write_jsonl


def _bucket(group, seed):
    digest = hashlib.sha256((str(seed) + ":" + group).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def prepare_dataset(manifest, output_dir, seed=42, min_duration=1.0, max_duration=30.0):
    output_dir = Path(output_dir)
    accepted, rejected = [], []
    ids = set()
    texts = set()
    for row in read_jsonl(manifest):
        reason = None
        for field in ("id", "audio", "text", "speaker", "source_recording"):
            if not str(row.get(field, "")).strip():
                reason = "missing_%s" % field
                break
        audio = Path(row.get("audio", "")).expanduser()
        if reason is None and not audio.is_absolute():
            audio = (Path(manifest).parent / audio).resolve()
        if reason is None and not audio.is_file():
            reason = "audio_not_found"
        if reason is None and audio.suffix.lower() != ".wav":
            reason = "not_pcm_wav"
        if reason is None:
            try:
                info = wav_info(audio)
            except Exception:
                reason = "invalid_wav"
        if reason is None and not min_duration <= info["duration_s"] <= max_duration:
            reason = "duration_out_of_range"
        if reason is None and row["id"] in ids:
            reason = "duplicate_id"
        normalized_text = " ".join(str(row.get("text", "")).split())
        if reason is None and normalized_text in texts:
            reason = "duplicate_text"
        if reason:
            rejected.append({"id": row.get("id"), "reason": reason, "line": row["_line"]})
            continue
        ids.add(row["id"])
        texts.add(normalized_text)
        group = "%s/%s" % (row["speaker"], row["source_recording"])
        value = _bucket(group, seed)
        split = "train" if value < 0.8 else ("valid" if value < 0.9 else "test")
        accepted.append({
            "id": row["id"], "audio": str(audio), "text": normalized_text,
            "speaker": row["speaker"], "source_recording": row["source_recording"],
            "duration_s": info["duration_s"], "sample_rate": info["sample_rate"], "split": split,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        write_jsonl(output_dir / (split + ".jsonl"), (row for row in accepted if row["split"] == split))
    write_jsonl(output_dir / "rejected.jsonl", rejected)
    with (output_dir / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|", lineterminator="\n")
        writer.writerow(["audio_file", "text"])
        for row in accepted:
            if row["split"] == "train":
                writer.writerow([row["audio"], row["text"]])
    return {"accepted": len(accepted), "rejected": len(rejected), "splits": {
        split: sum(row["split"] == split for row in accepted) for split in ("train", "valid", "test")
    }}

