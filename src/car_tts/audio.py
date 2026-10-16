"""Small audio helpers that keep evaluation independent from heavy ML packages."""

from pathlib import Path
import wave


def wav_duration(path):
    """Return PCM WAV duration in seconds.

    F5-TTS writes PCM WAV files, so the standard library is sufficient for the
    benchmark. Other formats should be normalized during dataset preparation.
    """
    path = Path(path)
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        frames = handle.getnframes()
    if rate <= 0:
        raise ValueError("invalid sample rate in %s" % path)
    return frames / float(rate)


def wav_info(path):
    path = Path(path)
    with wave.open(str(path), "rb") as handle:
        return {
            "sample_rate": handle.getframerate(),
            "channels": handle.getnchannels(),
            "sample_width": handle.getsampwidth(),
            "frames": handle.getnframes(),
            "duration_s": handle.getnframes() / float(handle.getframerate()),
        }

