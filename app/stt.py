"""Ears: speech to text with faster-whisper (runs offline)."""
from faster_whisper import WhisperModel

from .config import pick_device


class SpeechToText:
    def __init__(self, cfg: dict):
        device = pick_device(cfg.get("device", "auto"))
        compute = cfg.get("compute_type", "auto")
        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"
        self.language = cfg.get("language") or None
        print(f"[stt] loading whisper '{cfg['model']}' on {device} ({compute})")
        self.model = WhisperModel(cfg["model"], device=device, compute_type=compute)

    def transcribe(self, audio_path: str) -> tuple[str, str]:
        """Returns (text, detected_language). Accepts webm/ogg/wav/mp3 — decoded internally."""
        segments, info = self.model.transcribe(
            audio_path,
            language=self.language,
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        return text, info.language
