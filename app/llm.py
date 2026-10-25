"""Brain: local LLM through Ollama's HTTP API (runs offline, free)."""
import json
import re
from typing import Iterator

import requests

from .config import resolve

# Sentence ends for English and Telugu (Telugu mostly uses '.', some text uses '।')
_SENTENCE_END = re.compile(r"([.!?।]+[\"')\]]?\s+|\n+)")
# Things that should never be read aloud
_MARKUP = re.compile(r"[*_#`>~|]|\[|\]|\(http\S*\)")
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U0001F000-\U0001F2FF\U00002600-\U000026FF]"
)


def clean_for_speech(text: str) -> str:
    text = _MARKUP.sub("", text)
    text = _EMOJI.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


class Brain:
    def __init__(self, cfg: dict):
        self.url = cfg["url"].rstrip("/") + "/api/chat"
        self.model = cfg["model"]
        self.temperature = cfg.get("temperature", 0.7)
        self.max_turns = cfg.get("max_history_turns", 10)
        persona_path = resolve(cfg.get("persona_file", "persona.md"))
        self.system = persona_path.read_text(encoding="utf-8") if persona_path.exists() else ""
        self.history: list[dict] = []

    def reset(self) -> None:
        self.history.clear()

    def _messages(self, user_text: str) -> list[dict]:
        keep = self.history[-self.max_turns * 2 :]
        msgs = [{"role": "system", "content": self.system}] if self.system else []
        return msgs + keep + [{"role": "user", "content": user_text}]

    def stream_tokens(self, user_text: str) -> Iterator[str]:
        body = {
            "model": self.model,
            "messages": self._messages(user_text),
            "stream": True,
            "options": {"temperature": self.temperature},
        }
        with requests.post(self.url, json=body, stream=True, timeout=300) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if chunk.get("done"):
                    break

    def stream_sentences(self, user_text: str, min_chars: int = 12) -> Iterator[str]:
        """Yields the reply one speakable sentence at a time, so speech can start early."""
        buf, full = "", ""
        for piece in self.stream_tokens(user_text):
            buf += piece
            full += piece
            while True:
                m = _SENTENCE_END.search(buf)
                if not m or m.end() < min_chars:
                    break
                sentence, buf = buf[: m.end()], buf[m.end() :]
                sentence = clean_for_speech(sentence)
                if sentence:
                    yield sentence
        tail = clean_for_speech(buf)
        if tail:
            yield tail
        self.history += [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": full.strip()},
        ]
