"""Web server: phone/browser sends your voice, gets back text + audio in your cloned voice.

Run:  python -m app.server
"""
import base64
import json
import queue
import tempfile
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import ROOT, load_config
from .llm import Brain
from .stt import SpeechToText
from .tts import CloneVoice

CFG = load_config()
app = FastAPI(title="Somesh Voice Assistant")

ears = SpeechToText(CFG["stt"])
brain = Brain(CFG["llm"])
mouth = CloneVoice(CFG["tts"])
gpu_lock = threading.Lock()  # one conversation turn at a time on the GPU

_DONE = object()


def _line(obj: dict) -> bytes:
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


def _reply_stream(user_text: str):
    """NDJSON stream: {type:user} → many {type:say,text,audio} → {type:done}.

    The LLM writes sentences in a background thread while this thread turns each
    finished sentence into speech, so the first words play while the rest is
    still being thought up.
    """
    yield _line({"type": "user", "text": user_text})
    if not user_text:
        yield _line({"type": "done"})
        return

    sentences: queue.Queue = queue.Queue()

    def think():
        try:
            for s in brain.stream_sentences(user_text):
                sentences.put(s)
        except Exception as e:  # Ollama not running, model missing, etc.
            sentences.put(RuntimeError(f"Brain error: {e}"))
        finally:
            sentences.put(_DONE)

    threading.Thread(target=think, daemon=True).start()
    while True:
        item = sentences.get()
        if item is _DONE:
            break
        if isinstance(item, Exception):
            yield _line({"type": "error", "text": str(item)})
            break
        wav = mouth.synth_wav_bytes(item)
        yield _line({"type": "say", "text": item, "audio": base64.b64encode(wav).decode()})
    yield _line({"type": "done"})


def _locked(gen):
    with gpu_lock:
        yield from gen


@app.post("/api/talk")
async def talk(audio: UploadFile = File(...)):
    """Send a voice recording (webm/ogg/wav/mp3). Reply streams back as NDJSON."""
    suffix = Path(audio.filename or "rec.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        path = tmp.name

    def gen():
        try:
            text, lang = ears.transcribe(path)
        finally:
            Path(path).unlink(missing_ok=True)
        yield from _reply_stream(text)

    return StreamingResponse(_locked(gen()), media_type="application/x-ndjson")


@app.post("/api/chat")
async def chat(text: str = Form(...)):
    """Type instead of speaking; the reply still comes back in your voice."""
    return StreamingResponse(_locked(_reply_stream(text.strip())), media_type="application/x-ndjson")


@app.post("/api/reset")
async def reset():
    brain.reset()
    return {"ok": True}


@app.get("/api/health")
async def health():
    return {"ok": True, "stt": CFG["stt"]["model"], "llm": CFG["llm"]["model"], "tts": CFG["tts"]["model"]}


WEB = ROOT / "web"


@app.get("/")
async def index():
    page = WEB / "index.html"
    if page.exists():
        return FileResponse(page)
    return JSONResponse({"ok": True, "note": "Server running. Web page not added yet."})


app.mount("/static", StaticFiles(directory=WEB), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host=CFG["server"]["host"], port=CFG["server"]["port"])
