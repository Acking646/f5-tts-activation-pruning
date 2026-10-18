"""Reproducible end-to-end real-time-factor benchmark."""

import json
import platform
from pathlib import Path
import statistics
import time

from .audio import wav_duration
from .io import atomic_json, read_jsonl, write_jsonl


def _sync_device():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def validate_cases(path):
    cases = list(read_jsonl(path))
    required = {"id", "text", "ref_audio", "ref_text"}
    seen = set()
    for case in cases:
        missing = required.difference(case)
        if missing:
            raise ValueError("line %d is missing %s" % (case["_line"], sorted(missing)))
        if case["id"] in seen:
            raise ValueError("duplicate case id: %s" % case["id"])
        seen.add(case["id"])
        if not Path(case["ref_audio"]).is_file():
            raise FileNotFoundError("reference audio does not exist: %s" % case["ref_audio"])
    if not cases:
        raise ValueError("evaluation manifest is empty")
    return cases


def run_benchmark(backend, manifest, output_dir, warmup=1, repeats=3, seed=42, nfe_step=32, speed=1.0):
    cases = validate_cases(manifest)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for index in range(warmup):
        case = cases[index % len(cases)]
        backend.synthesize(
            case["ref_audio"], case["ref_text"], case["text"], output_dir / "_warmup.wav",
            seed=seed, nfe_step=nfe_step, speed=speed,
        )

    rows = []
    for repeat in range(repeats):
        for case in cases:
            destination = output_dir / ("%s-r%d.wav" % (case["id"], repeat))
            _sync_device()
            started = time.perf_counter()
            backend.synthesize(
                case["ref_audio"], case["ref_text"], case["text"], destination,
                seed=seed, nfe_step=nfe_step, speed=speed,
            )
            _sync_device()
            latency = time.perf_counter() - started
            duration = wav_duration(destination)
            rows.append({
                "id": case["id"], "repeat": repeat, "text": case["text"],
                "output": str(destination), "latency_s": latency,
                "audio_duration_s": duration, "rtf": latency / duration,
            })

    total_latency = sum(row["latency_s"] for row in rows)
    total_audio = sum(row["audio_duration_s"] for row in rows)
    repeat_summaries = []
    for repeat in range(repeats):
        repeat_rows = [row for row in rows if row["repeat"] == repeat]
        repeat_latency = sum(row["latency_s"] for row in repeat_rows)
        repeat_audio = sum(row["audio_duration_s"] for row in repeat_rows)
        repeat_summaries.append({
            "repeat": repeat,
            "latency_s": repeat_latency,
            "audio_duration_s": repeat_audio,
            "aggregate_rtf": repeat_latency / repeat_audio,
        })
    repeat_rtfs = [row["aggregate_rtf"] for row in repeat_summaries]
    per_case = []
    for case in cases:
        case_rows = [row for row in rows if row["id"] == case["id"]]
        case_rtfs = [row["rtf"] for row in case_rows]
        per_case.append({
            "id": case["id"],
            "category": case.get("category", "unspecified"),
            "text": case["text"],
            "mean_latency_s": statistics.mean(row["latency_s"] for row in case_rows),
            "mean_audio_duration_s": statistics.mean(row["audio_duration_s"] for row in case_rows),
            "mean_rtf": statistics.mean(case_rtfs),
            "stdev_rtf": statistics.stdev(case_rtfs) if len(case_rtfs) > 1 else 0.0,
        })
    result = {
        "protocol": {
            "scope": "end_to_end_excluding_model_load_including_wav_write",
            "warmup": warmup, "repeats": repeats, "seed": seed,
            "nfe_step": nfe_step, "speed": speed,
        },
        "environment": environment_info(),
        "summary": {
            "samples": len(rows),
            "total_latency_s": total_latency,
            "total_audio_duration_s": total_audio,
            "aggregate_rtf": total_latency / total_audio,
            "median_item_rtf": statistics.median(row["rtf"] for row in rows),
            "repeat_mean_rtf": statistics.mean(repeat_rtfs),
            "repeat_stdev_rtf": statistics.stdev(repeat_rtfs) if len(repeat_rtfs) > 1 else 0.0,
        },
        "repeats": repeat_summaries,
        "per_case": per_case,
        "items": rows,
    }
    atomic_json(output_dir / "benchmark.json", result)
    write_jsonl(output_dir / "mos_samples.jsonl", [
        {"id": row["id"], "text": row["text"], "audio": row["output"]}
        for row in rows if row["repeat"] == 0
    ])
    return result


def environment_info():
    info = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        import torch

        info.update({"torch": torch.__version__, "cuda_available": torch.cuda.is_available()})
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except ImportError:
        info["torch"] = None
    return info


def format_summary(result):
    summary = result["summary"]
    return json.dumps(summary, ensure_ascii=False, indent=2)
