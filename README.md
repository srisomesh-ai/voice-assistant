# Somesh Voice Assistant

Your own talking AI assistant that replies **in your own voice**, in Telugu, English or both mixed.
Everything runs on your computer — **no paid APIs, no monthly bills.**

```
 You speak  ──►  Ears (Whisper)  ──►  Brain (local LLM via Ollama)  ──►  Mouth (your cloned voice)  ──►  You hear
```

| Part  | What runs | Cost |
|-------|-----------|------|
| Ears  | faster-whisper (speech → text) | free, offline |
| Brain | Gemma 3 through Ollama | free, offline |
| Mouth | Dhee-Indic-F5 / IndicF5 voice cloning from a 15-sec clip of you | free, offline |

> **Licence note:** the voice-clone models are licensed for **non-commercial use**. Perfect for your
> personal assistant. If you later want to sell it or use it for customer calls, we'll swap the voice model.

---

## 1. What computer you need

| Your machine | Settings to use in `config.yaml` | Speed |
|---|---|---|
| NVIDIA GPU, 8 GB+ VRAM | `stt.model: large-v3-turbo`, `llm.model: gemma3:12b` | natural, ~1–2 s to start speaking |
| NVIDIA GPU, 4–6 GB | `stt.model: medium`, `llm.model: gemma3:4b` | good |
| No GPU (normal laptop) | `stt.model: small`, `llm.model: gemma3:4b` | works, but slow (several seconds per sentence) |

## 2. Install (Windows — one time)

1. **Python 3.10** → https://www.python.org/downloads/ (tick "Add to PATH").
2. **ffmpeg** → open PowerShell: `winget install ffmpeg`
3. **Ollama** → https://ollama.com/download, then in PowerShell:
   ```
   ollama pull gemma3:4b
   ```
4. **This project**:
   ```
   git clone https://github.com/srisomesh-ai/voice-assistant.git
   cd voice-assistant
   python -m venv .venv
   .venv\Scripts\activate
   ```
5. **PyTorch** (pick one):
   - NVIDIA GPU: `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121`
   - No GPU: `pip install torch torchaudio`
6. **Everything else**:
   ```
   pip install -r requirements.txt
   ```

First run downloads the speech and voice models (a few GB) — only once.

## 3. Give it your voice

1. Record **15–20 seconds** of yourself on your phone (quiet room, natural tone, see `voice/README.md`).
2. Copy the file to the project folder and run:
   ```
   python scripts/prepare_voice.py my_recording.m4a
   ```
3. Open `voice/my_voice.txt` and correct any wrong words.
4. Test it:
   ```
   python scripts/say.py "నమస్కారం, ఇది నా సొంత AI వాయిస్."
   ```
   Play `say_test.wav`. Not happy? Record again — clip quality matters more than anything else.

## 4. Run it

```
python -m app.server
```

Open http://localhost:8000 on the same computer.

### Use it from your phone (free, secure link)

The phone browser only allows the microphone on **https**. A free Cloudflare quick tunnel gives you that:

```
winget install Cloudflare.cloudflared
cloudflared tunnel --url http://localhost:8000
```

It prints a link like `https://something.trycloudflare.com` — open it on your phone.
(Keep the computer on while you use it.)

## 5. Personalise

- `persona.md` — how it talks and what it knows about you. Edit freely.
- `config.yaml` — models, language, memory length.

## API (for your own apps later)

| Endpoint | Send | Get back (streamed, one JSON per line) |
|---|---|---|
| `POST /api/talk` | form field `audio` (webm/ogg/wav/mp3) | `{type:"user",text}`, then `{type:"say",text,audio(base64 wav)}` per sentence, then `{type:"done"}` |
| `POST /api/chat` | form field `text` | same as above |
| `POST /api/reset` | — | clears conversation memory |
| `GET /api/health` | — | which models are loaded |

## Training on more of your voice (phase 2)

Right now the assistant copies your voice from one short clip ("zero-shot cloning") — no training needed.
For an even closer match later, record **30–60 minutes** of yourself reading Telugu and English text,
and we fine-tune the voice model on those recordings (needs an 8 GB+ NVIDIA GPU, or a free Kaggle/Colab GPU session).
