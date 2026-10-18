"""ASR-based WER/CER evaluation for generated speech."""

from pathlib import Path
import unicodedata

from .io import atomic_json, read_jsonl


def _strip_punctuation_and_symbols(text, replacement):
    result = []
    for character in unicodedata.normalize("NFKC", text).casefold():
        if unicodedata.category(character)[0] in ("P", "S"):
            result.append(replacement)
        else:
            result.append(character)
    return "".join(result)


def word_tokens(text):
    """Normalize a transcript and split it into words."""
    return _strip_punctuation_and_symbols(text, " ").split()


def character_tokens(text):
    """Normalize a transcript into non-space characters."""
    normalized = _strip_punctuation_and_symbols(text, "")
    return [character for character in normalized if not character.isspace()]


def edit_distance(reference, hypothesis):
    """Return Levenshtein distance between two token sequences."""
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_token in enumerate(reference, 1):
        current = [ref_index]
        for hyp_index, hyp_token in enumerate(hypothesis, 1):
            substitution = previous[hyp_index - 1] + (ref_token != hyp_token)
            insertion = current[hyp_index - 1] + 1
            deletion = previous[hyp_index] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def transcript_errors(reference, hypothesis, text_normalizer=None):
    if text_normalizer is not None:
        reference = text_normalizer(reference)
        hypothesis = text_normalizer(hypothesis)
    ref_words, hyp_words = word_tokens(reference), word_tokens(hypothesis)
    ref_characters, hyp_characters = character_tokens(reference), character_tokens(hypothesis)
    return {
        "word_errors": edit_distance(ref_words, hyp_words),
        "reference_words": len(ref_words),
        "character_errors": edit_distance(ref_characters, hyp_characters),
        "reference_characters": len(ref_characters),
    }


class WhisperTranscriber:
    """Whisper wrapper that decodes WAV without Transformers' torchcodec path."""

    def __init__(self, model, device=None, language=None):
        import soundfile
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

        self.torch = torch
        self.soundfile = soundfile
        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model, dtype=self.dtype)
        self.model.to(self.device)
        self.model.eval()
        self.sampling_rate = self.processor.feature_extractor.sampling_rate

        is_multilingual = getattr(self.model.generation_config, "is_multilingual", True)
        self.generate_kwargs = {}
        if is_multilingual:
            self.generate_kwargs["task"] = "transcribe"
            if language:
                self.generate_kwargs["language"] = language
        elif language and language.lower() not in ("en", "english"):
            raise ValueError("an English-only Whisper model cannot transcribe language %s" % language)

        english = not is_multilingual or (language and language.lower() in ("en", "english"))
        if english:
            self.normalize_text = self.processor.tokenizer.normalize
            self.normalizer_name = "whisper_english"
        else:
            self.normalize_text = self.processor.tokenizer.basic_normalize
            self.normalizer_name = "whisper_basic"

    def __call__(self, audio_path):
        from torchaudio.functional import resample

        audio, sampling_rate = self.soundfile.read(
            str(audio_path), dtype="float32", always_2d=True
        )
        waveform = self.torch.from_numpy(audio.mean(axis=1))
        if sampling_rate != self.sampling_rate:
            waveform = resample(waveform, sampling_rate, self.sampling_rate)
        inputs = self.processor(
            waveform.numpy(),
            sampling_rate=self.sampling_rate,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device, dtype=self.dtype)
        generate_kwargs = dict(self.generate_kwargs)
        if "attention_mask" in inputs:
            generate_kwargs["attention_mask"] = inputs.attention_mask.to(self.device)
        with self.torch.inference_mode():
            predicted_ids = self.model.generate(input_features, **generate_kwargs)
        return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


def evaluate_intelligibility(manifest, systems, transcribe, output, audio_pattern="{id}-r0.wav",
                             metadata=None, text_normalizer=None):
    """Transcribe matching system outputs and compute micro-averaged WER/CER."""
    cases = list(read_jsonl(manifest))
    if not cases:
        raise ValueError("manifest is empty")

    report = {
        "protocol": {
            "manifest": str(Path(manifest).resolve()),
            "audio_pattern": audio_pattern,
            "normalization": "Unicode NFKC, casefold, punctuation/symbol removal",
            **(metadata or {}),
        },
        "systems": {},
    }
    for system_name, directory in systems.items():
        directory = Path(directory)
        totals = {
            "word_errors": 0,
            "reference_words": 0,
            "character_errors": 0,
            "reference_characters": 0,
        }
        per_case = []
        for case in cases:
            case_id = str(case["id"])
            reference = str(case["text"])
            audio = directory / audio_pattern.format(id=case_id)
            if not audio.is_file():
                raise FileNotFoundError("missing audio for %s/%s: %s" %
                                        (system_name, case_id, audio))
            hypothesis = transcribe(audio)
            errors = transcript_errors(reference, hypothesis, text_normalizer)
            for key in totals:
                totals[key] += errors[key]
            per_case.append({
                "id": case_id,
                "audio": str(audio.resolve()),
                "reference": reference,
                "hypothesis": hypothesis,
                **errors,
            })

        if not totals["reference_words"] or not totals["reference_characters"]:
            raise ValueError("reference transcript has no scoreable words or characters")
        report["systems"][system_name] = {
            "utterances": len(per_case),
            **totals,
            "wer": totals["word_errors"] / totals["reference_words"],
            "cer": totals["character_errors"] / totals["reference_characters"],
            "per_case": per_case,
        }

    atomic_json(output, report)
    return report
