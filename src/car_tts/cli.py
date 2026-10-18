"""Command-line interface."""

import argparse
import json
from pathlib import Path

from .backend import create_backend
from .benchmark import format_summary, run_benchmark
from .dataset import prepare_dataset
from .intelligibility import WhisperTranscriber, evaluate_intelligibility
from .mos import aggregate_ratings, create_trial_manifest


def _backend_options(args):
    return {"model": args.model, "checkpoint": args.checkpoint, "vocab": args.vocab,
            "vocoder": args.vocoder, "device": args.device}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="car-tts")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="validate and split a real speech manifest")
    prepare.add_argument("manifest")
    prepare.add_argument("output")
    prepare.add_argument("--seed", type=int, default=42)

    infer = commands.add_parser("infer", help="synthesize one utterance")
    infer.add_argument("--text", required=True)
    infer.add_argument("--ref-audio", required=True)
    infer.add_argument("--ref-text", required=True)
    infer.add_argument("--output", required=True)

    benchmark = commands.add_parser("benchmark", help="measure end-to-end RTF")
    benchmark.add_argument("manifest")
    benchmark.add_argument("output")
    benchmark.add_argument("--warmup", type=int, default=1)
    benchmark.add_argument("--repeats", type=int, default=3)
    benchmark.add_argument("--nfe-step", type=int, default=32)

    mos_create = commands.add_parser("mos-create", help="create a randomized blind-test manifest")
    mos_create.add_argument("--system", action="append", required=True, help="NAME=manifest.jsonl")
    mos_create.add_argument("--output", required=True)
    mos_create.add_argument("--seed", type=int, default=42)

    mos_aggregate = commands.add_parser("mos-aggregate", help="aggregate naturalness ratings")
    mos_aggregate.add_argument("ratings")
    mos_aggregate.add_argument("trials")

    intelligibility = commands.add_parser(
        "intelligibility", help="transcribe generated speech and compute WER/CER"
    )
    intelligibility.add_argument("manifest")
    intelligibility.add_argument("--system", action="append", required=True, help="NAME=AUDIO_DIR")
    intelligibility.add_argument("--output", required=True)
    intelligibility.add_argument("--asr-model", default="openai/whisper-small")
    intelligibility.add_argument("--language", default=None, help="Whisper language code, e.g. en or zh")
    intelligibility.add_argument("--audio-pattern", default="{id}-r0.wav")
    intelligibility.add_argument("--device", default=None, help="cpu, cuda, or cuda:N")

    for subcommand in (infer, benchmark):
        subcommand.add_argument("--backend", choices=("f5tts", "mock"), default="f5tts")
        subcommand.add_argument("--model", default="F5TTS_v1_Base")
        subcommand.add_argument("--checkpoint", default="")
        subcommand.add_argument("--vocab", default="", help="local F5-TTS vocab.txt")
        subcommand.add_argument("--vocoder", default="", help="local Vocos directory")
        subcommand.add_argument("--device", default=None)
        subcommand.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_dataset(args.manifest, args.output, seed=args.seed)
    elif args.command == "infer":
        backend = create_backend(args.backend, **_backend_options(args))
        backend.synthesize(args.ref_audio, args.ref_text, args.text, args.output, seed=args.seed)
        result = {"output": str(Path(args.output).resolve())}
    elif args.command == "benchmark":
        backend = create_backend(args.backend, **_backend_options(args))
        result = run_benchmark(backend, args.manifest, args.output, warmup=args.warmup,
                               repeats=args.repeats, seed=args.seed, nfe_step=args.nfe_step)
        print(format_summary(result))
        return 0
    elif args.command == "mos-create":
        systems = dict(item.split("=", 1) for item in args.system)
        result = {"trials": create_trial_manifest(systems, args.output, args.seed)}
    elif args.command == "mos-aggregate":
        result = aggregate_ratings(args.ratings, args.trials)
    else:
        systems = dict(item.split("=", 1) for item in args.system)
        if len(systems) != len(args.system):
            parser.error("--system names must be unique")
        transcriber = WhisperTranscriber(args.asr_model, args.device, args.language)
        result = evaluate_intelligibility(
            args.manifest,
            systems,
            transcriber,
            args.output,
            args.audio_pattern,
            metadata={
                "asr_model": args.asr_model,
                "language": args.language or "auto",
                "device": args.device or "auto",
            },
        )
        result = {
            name: {"utterances": value["utterances"], "wer": value["wer"], "cer": value["cer"]}
            for name, value in result["systems"].items()
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
