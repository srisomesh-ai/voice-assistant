"""Quick test of your cloned voice without the server.

    python scripts/say.py "నమస్కారం, ఇది నా సొంత AI వాయిస్."
    python scripts/say.py "Hello, this is my own AI voice." --out test.wav
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import load_config  # noqa: E402
from app.tts import CloneVoice  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--out", default="say_test.wav")
    args = ap.parse_args()

    voice = CloneVoice(load_config()["tts"])
    t = time.time()
    Path(args.out).write_bytes(voice.synth_wav_bytes(args.text))
    print(f"Saved {args.out} in {time.time() - t:.1f}s — play it and compare with your real voice.")


if __name__ == "__main__":
    main()
