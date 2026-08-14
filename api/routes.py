from __future__ import annotations

import time
from pathlib import Path

import yaml
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from harness.orchestrator import Orchestrator
from harness.schemas import AskRequest, ReasonCode
from stt.sarvam import SarvamSTT

ROOT = Path(__file__).resolve().parents[1]
router = APIRouter()

_orc: Orchestrator | None = None
_stt: SarvamSTT | None = None
_started = time.time()
_counters = {"ask": 0, "ask_voice": 0, "refused": 0, "by_path": {}}


def boot() -> Orchestrator:
    """Built once at startup, never per request. The embedder warm-up alone is
    seconds, and paying it on the first user request is how P100 gets ruined."""
    global _orc, _stt
    if _orc is None:
        _orc = Orchestrator()
        _stt = SarvamSTT(yaml.safe_load((ROOT / "config.yaml").read_text()))
    return _orc


def _record(answer) -> None:
    path = answer.path.value
    _counters["by_path"][path] = _counters["by_path"].get(path, 0) + 1
    if answer.refused:
        _counters["refused"] += 1


@router.post("/ask")
def ask(req: AskRequest):
    orc = boot()
    _counters["ask"] += 1
    answer = orc.ask(req.query, lang=req.lang, top_k=req.top_k,
                     use_cache=req.use_cache, allow_generative=req.allow_generative)
    _record(answer)
    return answer


@router.post("/ask-voice")
async def ask_voice(file: UploadFile = File(...), lang: str | None = Form(None),
                    use_cache: bool = Form(True)):
    orc = boot()
    _counters["ask_voice"] += 1

    audio = await file.read()
    if not audio:
        raise HTTPException(400, "empty audio upload")
    if not _stt.available():
        raise HTTPException(503, "SARVAM_API_KEY is not configured; use POST /ask with text")

    try:
        stt = _stt.transcribe(audio, filename=file.filename or "audio.webm")
    except RuntimeError as exc:
        raise HTTPException(502, f"speech-to-text failed: {exc}") from exc

    answer = orc.ask(stt.transcript, lang=lang or stt.lang, use_cache=use_cache)
    _record(answer)

    body = answer.model_dump()
    # STT is a network hop to a third party and is reported separately from the
    # core pipeline, which is what the 200ms target is scoped to.
    body["transcript"] = stt.transcript
    body["detected_lang"] = stt.lang
    body["timings"]["stt_ms"] = stt.duration_ms
    body["timings"]["wall_ms"] = round(stt.duration_ms + answer.timings.total_ms, 2)
    return body


@router.get("/health")
def health():
    orc = boot()
    return {**orc.health(), "uptime_s": round(time.time() - _started, 1),
            "stt_configured": _stt.available()}


@router.get("/metrics")
def metrics():
    orc = boot()
    return {
        "requests": dict(_counters),
        "cache": orc.cache.stats(),
        "providers": orc.providers.status(),
        "budgets_ms": orc.budgets,
        "reason_codes": [c.value for c in ReasonCode],
    }
