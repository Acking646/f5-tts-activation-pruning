"""F5-TTS adapter plus a deterministic CI-only backend."""

from pathlib import Path
import math
import struct
import wave


class F5Backend:
    def __init__(self, model="F5TTS_v1_Base", checkpoint="", vocab="", vocoder="", device=None):
        try:
            from f5_tts.api import F5TTS
        except ImportError as exc:
            raise RuntimeError("Install the real runtime with: pip install -e .[tts]") from exc
        options = {"model": model}
        if checkpoint:
            options["ckpt_file"] = checkpoint
        if vocab:
            options["vocab_file"] = vocab
        if vocoder:
            options["vocoder_local_path"] = vocoder
        if device:
            options["device"] = device
        self.engine = F5TTS(**options)

    @property
    def torch_model(self):
        return self.engine.ema_model

    def synthesize(self, ref_audio, ref_text, text, output, seed=42, nfe_step=32, speed=1.0):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        self.engine.infer(
            ref_file=str(ref_audio),
            ref_text=ref_text,
            gen_text=text,
            file_wave=str(output),
            seed=seed,
            nfe_step=nfe_step,
            speed=speed,
            show_info=lambda _message: None,
        )
        return Path(output)


class MockBackend:
    """Generate a short sine WAV for tests; never use these results in a report."""

    def synthesize(self, ref_audio, ref_text, text, output, seed=42, nfe_step=32, speed=1.0):
        del ref_audio, ref_text, seed, nfe_step
        sample_rate = 16000
        duration = max(0.25, min(2.0, len(text) / 20.0)) / speed
        frames = int(duration * sample_rate)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            samples = [int(2500 * math.sin(2 * math.pi * 220 * index / sample_rate)) for index in range(frames)]
            handle.writeframes(struct.pack("<%dh" % len(samples), *samples))
        return Path(output)


def create_backend(name, **options):
    if name == "f5tts":
        return F5Backend(**options)
    if name == "mock":
        return MockBackend()
    raise ValueError("unknown backend: %s" % name)
