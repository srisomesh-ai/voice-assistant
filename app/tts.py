"""Mouth: speaks in YOUR cloned voice using an IndicF5-family model (offline).

Zero-shot cloning: the model copies your voice from one 10–15 s reference clip.
No training run is needed to start; see README "Training on more of your voice".
"""
import io

import numpy as np
import soundfile as sf
from transformers import AutoModel

from .config import resolve


class CloneVoice:
    def __init__(self, cfg: dict):
        self.sample_rate = cfg.get("sample_rate", 24000)
        self.ref_audio = str(resolve(cfg["ref_audio"]))
        ref_text_path = resolve(cfg["ref_text_file"])
        if not resolve(cfg["ref_audio"]).exists():
            raise FileNotFoundError(
                f"Voice clip missing: {self.ref_audio}. Run: python scripts/prepare_voice.py <your-recording>"
            )
        if not ref_text_path.exists():
            raise FileNotFoundError(f"Transcript missing: {ref_text_path}")
        self.ref_text = ref_text_path.read_text(encoding="utf-8").strip()
        print(f"[tts] loading voice model '{cfg['model']}'")
        self.model = AutoModel.from_pretrained(cfg["model"], trust_remote_code=True)

    def synth(self, text: str) -> np.ndarray:
        audio = self.model(text, ref_audio_path=self.ref_audio, ref_text=self.ref_text)
        audio = np.asarray(audio)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        return audio.astype(np.float32)

    def synth_wav_bytes(self, text: str) -> bytes:
        buf = io.BytesIO()
        sf.write(buf, self.synth(text), samplerate=self.sample_rate, format="WAV")
        return buf.getvalue()
