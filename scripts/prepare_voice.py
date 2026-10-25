"""Turn any recording of your voice into the reference clip the assistant clones.

Usage:
    python scripts/prepare_voice.py my_recording.m4a            # uses the first 15 s
    python scripts/prepare_voice.py my_recording.m4a --start 5  # skip first 5 s

What it does:
  1. Converts to clean 24 kHz mono WAV, removes silence at the edges,
     evens out loudness, cuts to max 15 seconds  -> voice/my_voice.wav
  2. Writes down what you said with Whisper           -> voice/my_voice.txt
     Open that .txt and fix any wrong words — an exact transcript makes the clone sound better.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import load_config  # noqa: E402
from app.stt import SpeechToText  # noqa: E402

OUT_WAV = ROOT / "voice" / "my_voice.wav"
OUT_TXT = ROOT / "voice" / "my_voice.txt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", help="any audio file: wav, mp3, m4a, ogg, webm...")
    ap.add_argument("--start", type=float, default=0.0, help="seconds to skip at the start")
    ap.add_argument("--length", type=float, default=15.0, help="clip length in seconds (max 15)")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found. Install it first (see README).")

    length = min(args.length, 15.0)
    OUT_WAV.parent.mkdir(exist_ok=True)
    filters = ",".join(
        [
            "highpass=f=70",
            "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.2",
            "areverse",
            "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.3",
            "areverse",
            "loudnorm=I=-18:TP=-2:LRA=11",
        ]
    )
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(args.start), "-i", args.recording,
        "-t", str(length + 2),  # a little extra; trimmed again below
        "-af", filters, "-ac", "1", "-ar", "24000", str(OUT_WAV),
    ]
    subprocess.run(cmd, check=True)
    # final exact cut to the max length
    tmp = OUT_WAV.with_suffix(".tmp.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(OUT_WAV),
         "-t", str(length), str(tmp)],
        check=True,
    )
    tmp.replace(OUT_WAV)
    print(f"Saved clip: {OUT_WAV}")

    print("Transcribing your clip...")
    text, lang = SpeechToText(load_config()["stt"]).transcribe(str(OUT_WAV))
    OUT_TXT.write_text(text + "\n", encoding="utf-8")
    print(f"Saved transcript ({lang}): {OUT_TXT}\n\n  {text}\n")
    print("Check the transcript above. If any word is wrong, fix it in voice/my_voice.txt.")


if __name__ == "__main__":
    main()
