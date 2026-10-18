import csv
import json
import math
from pathlib import Path
import struct
import sys
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car_tts.audio import wav_duration
from car_tts.backend import MockBackend
from car_tts.benchmark import run_benchmark
from car_tts.dataset import prepare_dataset
from car_tts.experts import make_masks
from car_tts.intelligibility import evaluate_intelligibility, transcript_errors
from car_tts.mos import aggregate_ratings, create_trial_manifest


def make_wav(path, seconds=0.5, sample_rate=8000):
    samples = [0] * int(seconds * sample_rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(struct.pack("<%dh" % len(samples), *samples))


def test_benchmark_aggregate_rtf(tmp_path):
    reference = tmp_path / "ref.wav"
    make_wav(reference)
    manifest = tmp_path / "eval.jsonl"
    manifest.write_text(json.dumps({"id": "a", "text": "hello", "ref_audio": str(reference),
                                    "ref_text": "reference"}) + "\n", encoding="utf-8")
    result = run_benchmark(MockBackend(), manifest, tmp_path / "out", warmup=0, repeats=2)
    summary = result["summary"]
    assert summary["samples"] == 2
    assert math.isclose(summary["aggregate_rtf"], summary["total_latency_s"] /
                        summary["total_audio_duration_s"])
    assert len(result["repeats"]) == 2
    assert result["per_case"][0]["id"] == "a"
    assert "repeat_mean_rtf" in summary and "repeat_stdev_rtf" in summary
    assert (tmp_path / "out" / "mos_samples.jsonl").is_file()
    assert wav_duration(result["items"][0]["output"]) > 0


def test_prepare_rejects_duplicate_text_and_groups_split(tmp_path):
    audio_a, audio_b = tmp_path / "a.wav", tmp_path / "b.wav"
    make_wav(audio_a, 1.2)
    make_wav(audio_b, 1.2)
    source = tmp_path / "source.jsonl"
    rows = [
        {"id": "a", "audio": str(audio_a), "text": "same text", "speaker": "s", "source_recording": "r"},
        {"id": "b", "audio": str(audio_b), "text": "same  text", "speaker": "s", "source_recording": "r"},
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    result = prepare_dataset(source, tmp_path / "prepared")
    assert result["accepted"] == 1
    assert result["rejected"] == 1


def test_masks_support_global_and_equal():
    profile = {"n_experts": 4, "layers": {
        "a": {"activation_counts": [9, 8, 1, 0]},
        "b": {"activation_counts": [7, 6, 5, 4]},
    }}
    equal = make_masks(profile, 0.5, "equal")
    assert sum(equal["a"]) == sum(equal["b"]) == 2
    global_masks = make_masks(profile, 0.5, "global")
    assert any(global_masks["a"]) and any(global_masks["b"])
    random_masks = make_masks(profile, 0.5, "random", seed=42)
    assert random_masks == make_masks(profile, 0.5, "random", seed=42)
    assert {name: sum(mask) for name, mask in random_masks.items()} == {
        name: sum(mask) for name, mask in global_masks.items()
    }


def test_mos_blinding_and_aggregation(tmp_path):
    manifest = tmp_path / "system.jsonl"
    manifest.write_text(json.dumps({"id": "x", "audio": "x.wav"}) + "\n", encoding="utf-8")
    trials = tmp_path / "trials.jsonl"
    assert create_trial_manifest({"baseline": str(manifest)}, trials) == 1
    trial = json.loads(trials.read_text(encoding="utf-8"))
    ratings = tmp_path / "ratings.csv"
    with ratings.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["listener_id", "trial_id", "naturalness"])
        writer.writeheader()
        writer.writerow({"listener_id": "l1", "trial_id": trial["trial_id"], "naturalness": 4})
    result = aggregate_ratings(ratings, trials)
    assert result["baseline"]["mean"] == 4
    assert result["baseline"]["ci95_low"] == result["baseline"]["ci95_high"] == 4


def test_transcript_errors_and_intelligibility_report(tmp_path):
    errors = transcript_errors("Hello, world!", "hello word")
    assert errors["word_errors"] == 1
    assert errors["reference_words"] == 2
    chinese = transcript_errors("你好，世界！", "你好世")
    assert chinese["character_errors"] == 1
    assert chinese["reference_characters"] == 4
    normalized = transcript_errors(
        "eight minutes",
        "8 minutes",
        lambda text: text.replace("8", "eight"),
    )
    assert normalized["word_errors"] == 0

    manifest = tmp_path / "eval.jsonl"
    manifest.write_text(
        json.dumps({"id": "a", "text": "hello world"}) + "\n",
        encoding="utf-8",
    )
    audio_dir = tmp_path / "system"
    audio_dir.mkdir()
    (audio_dir / "a-r0.wav").write_bytes(b"test")
    output = tmp_path / "intelligibility.json"
    report = evaluate_intelligibility(
        manifest,
        {"baseline": audio_dir},
        lambda _: "hello word",
        output,
    )
    assert report["systems"]["baseline"]["wer"] == 0.5
    assert output.is_file()
