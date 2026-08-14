"""Sarvam speech-to-text.

Chosen over ElevenLabs because the corpus is Indic: Sarvam covers 22 Indian
languages, handles code-mixed Hinglish, and returns a detected language code we
can push straight into the retrieval language filter. Getting that filter right
matters more than transcript quality alone - a Hindi question searched against the
English shard retrieves nothing useful.
"""
from __future__ import annotations

import os
import time

import httpx

from harness.schemas import TranscriptResult

# Sarvam returns BCP-47-ish codes; retrieval keys on ISO-639-1.
_LANG = {"hi-IN": "hi", "gu-IN": "gu", "en-IN": "en", "en-US": "en"}


class SarvamSTT:
    def __init__(self, cfg: dict):
        sc = cfg["stt"]
        self.base_url = sc["base_url"]
        self.model = sc["model"]
        self.key = os.getenv("SARVAM_API_KEY", "").strip()

    def available(self) -> bool:
        return bool(self.key)

    def transcribe(self, audio: bytes, filename: str = "audio.webm",
                   timeout: float = 20.0) -> TranscriptResult:
        if not self.available():
            raise RuntimeError("SARVAM_API_KEY is not set")

        t0 = time.perf_counter()
        r = httpx.post(
            f"{self.base_url}/speech-to-text",
            headers={"api-subscription-key": self.key},
            files={"file": (filename, audio, "audio/webm")},
            data={"model": self.model},
            timeout=timeout,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        if r.status_code >= 400:
            raise RuntimeError(f"sarvam stt {r.status_code}: {r.text[:200]}")

        body = r.json()
        raw_lang = body.get("language_code") or ""
        return TranscriptResult(
            transcript=(body.get("transcript") or "").strip(),
            lang=_LANG.get(raw_lang, raw_lang.split("-")[0] or None),
            provider="sarvam",
            duration_ms=round(elapsed, 2),
        )
