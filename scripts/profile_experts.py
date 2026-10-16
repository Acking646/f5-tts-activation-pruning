"""Profile clustered FFN expert activation on a fixed evaluation manifest."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car_tts.backend import F5Backend  # noqa: E402
from car_tts.experts import ExpertProfiler  # noqa: E402
from car_tts.io import atomic_json, read_jsonl  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("--model", default="F5TTS_v1_Base")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default=None)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--top-fraction", type=float, default=0.2)
    parser.add_argument("--nfe-step", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    backend = F5Backend(args.model, args.checkpoint, args.device)
    profiler = ExpertProfiler(backend.torch_model, args.experts, args.top_fraction, args.seed)
    output_dir = Path(args.output).parent / "profile_audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = list(read_jsonl(args.manifest))
    try:
        for case in cases:
            backend.synthesize(case["ref_audio"], case["ref_text"], case["text"],
                               output_dir / (str(case["id"]) + ".wav"), seed=args.seed,
                               nfe_step=args.nfe_step)
            profiler.commit_sample()
    finally:
        profiler.close()
    result = profiler.result()
    result["provenance"] = {"model": args.model, "checkpoint": args.checkpoint or "official_default",
                            "manifest": str(Path(args.manifest).resolve()), "samples": len(cases),
                            "seed": args.seed, "nfe_step": args.nfe_step}
    atomic_json(args.output, result)
    print(json.dumps({"output": args.output, "layers": len(result["layers"]),
                      "positions": result["total_positions"]}, indent=2))


if __name__ == "__main__":
    main()

