"""Structurally prune F5-TTS FFNs from a profile and run the same RTF protocol."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car_tts.backend import F5Backend  # noqa: E402
from car_tts.benchmark import run_benchmark  # noqa: E402
from car_tts.experts import load_profile, make_masks, structurally_prune  # noqa: E402
from car_tts.io import atomic_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile")
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("--model", default="F5TTS_v1_Base")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default=None)
    parser.add_argument("--retain", type=float, default=0.8)
    parser.add_argument("--strategy", choices=("global", "equal"), default="global")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--nfe-step", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-state", default="", help="optional pruned state_dict path")
    args = parser.parse_args()
    if not 0 < args.retain <= 1:
        parser.error("--retain must be in (0, 1]")

    backend = F5Backend(args.model, args.checkpoint, args.device)
    profile = load_profile(args.profile)
    masks = make_masks(profile, args.retain, args.strategy)
    pruning = structurally_prune(backend.torch_model, profile, masks)
    output_dir = Path(args.output)
    result = run_benchmark(backend, args.manifest, output_dir, args.warmup, args.repeats,
                           args.seed, args.nfe_step)
    report = {"pruning": pruning, "retain_fraction": args.retain,
              "strategy": args.strategy, "benchmark": result}
    atomic_json(output_dir / "prune_benchmark.json", report)
    atomic_json(output_dir / "expert_masks.json", masks)
    if args.save_state:
        import torch

        torch.save(backend.torch_model.state_dict(), args.save_state)
    print(json.dumps({"aggregate_rtf": result["summary"]["aggregate_rtf"], **pruning}, indent=2))


if __name__ == "__main__":
    main()

