"""Create and aggregate a small blinded naturalness listening test."""

import csv
import hashlib
import math
import random
from pathlib import Path
import statistics

from .io import read_jsonl, write_jsonl


def create_trial_manifest(systems, output, seed=42):
    """Systems maps an anonymized internal name to an evaluation JSONL file."""
    trials = []
    for system, manifest in systems.items():
        for row in read_jsonl(manifest):
            audio = row.get("audio") or row.get("output")
            if not audio:
                raise ValueError("trial %s has no audio/output field" % row.get("id"))
            opaque = hashlib.sha256((system + ":" + str(row["id"]) + ":" + str(seed)).encode()).hexdigest()[:16]
            trials.append({"trial_id": opaque, "item_id": row["id"], "system": system, "audio": audio})
    random.Random(seed).shuffle(trials)
    write_jsonl(output, trials)
    return len(trials)


def aggregate_ratings(ratings_csv, trials_jsonl):
    trials = {row["trial_id"]: row for row in read_jsonl(trials_jsonl)}
    values = {}
    listener_trials = set()
    with Path(ratings_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            trial = row["trial_id"]
            listener = row["listener_id"]
            if trial not in trials:
                raise ValueError("unknown trial_id: %s" % trial)
            key = (listener, trial)
            if key in listener_trials:
                raise ValueError("duplicate listener/trial rating: %s" % (key,))
            listener_trials.add(key)
            score = float(row["naturalness"])
            if not 1 <= score <= 5:
                raise ValueError("MOS rating must be in [1, 5]")
            values.setdefault(trials[trial]["system"], []).append(score)
    result = {}
    for system, scores in values.items():
        mean = statistics.mean(scores)
        stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0
        margin = 1.96 * stdev / math.sqrt(len(scores)) if len(scores) > 1 else 0.0
        result[system] = {
            "ratings": len(scores),
            "listeners": len({listener for listener, trial in listener_trials
                              if trials[trial]["system"] == system}),
            "mean": mean,
            "stdev": stdev,
            "ci95_low": max(1.0, mean - margin),
            "ci95_high": min(5.0, mean + margin),
        }
    return result
