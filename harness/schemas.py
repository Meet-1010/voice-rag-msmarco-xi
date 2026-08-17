"""Pydantic contracts for every boundary in the pipeline."""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ReasonCode(str, Enum):
    UNSAFE_INPUT = "UNSAFE_INPUT"
    PROMPT_INJECTION = "PROMPT_INJECTION"
    EMPTY_INPUT = "EMPTY_INPUT"
    UNSUPPORTED_LANGUAGE = "UNSUPPORTED_LANGUAGE"
    OUT_OF_CORPUS = "OUT_OF_CORPUS"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    UNGROUNDED_OUTPUT = "UNGROUNDED_OUTPUT"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    ANSWERED_FROM_GENERAL_KNOWLEDGE = "ANSWERED_FROM_GENERAL_KNOWLEDGE"


class AnswerPath(str, Enum):
    CACHE = "cache"
    EXTRACTIVE = "extractive"
    GENERATIVE = "generative"
    # Answered from the model's own knowledge because the corpus had nothing.
    # Deliberately a separate path, never labelled "grounded", and always carries
    # OUT_OF_CORPUS so the provenance travels with the answer.
    GENERAL = "general"
    REFUSED = "refused"


class AskRequest(BaseModel):
    query: str = Field(..., max_length=2000)
    lang: str | None = Field(None, description="ISO-639-1; detected when omitted")
    top_k: int | None = Field(None, ge=1, le=50)
    use_cache: bool = True
    allow_generative: bool = True
    # When the corpus cannot answer: False refuses (strict RAG, what the task
    # asks us to demonstrate), True answers from model knowledge with the answer
    # explicitly marked ungrounded.
    allow_general: bool = True

    @field_validator("query")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()


class Citation(BaseModel):
    chunk_id: str
    passage_id: str
    text: str
    score: float
    lang: str


class StageTrace(BaseModel):
    stage: str
    duration_ms: float
    ok: bool = True
    error: str | None = None


class Timings(BaseModel):
    stages: dict[str, float]
    core_ms: float
    total_ms: float
    within_budget: bool


class Answer(BaseModel):
    answer: str
    citations: list[Citation] = []
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    path: AnswerPath
    lang: str | None = None
    grounded: bool = True
    refused: bool = False
    reason_code: ReasonCode | None = None
    provider: str | None = None
    timings: Timings
    trace: list[StageTrace] = []
    cache_similarity: float | None = None


class Refusal(BaseModel):
    """Refusals are answers too. Same envelope, so clients never branch on shape."""
    reason_code: ReasonCode
    message: str
    detail: str | None = None


class LLMAnswer(BaseModel):
    """What the LLM must return. Anything else triggers the repair loop."""
    answer: str
    citations: list[int] = []
    confidence: float = 0.5

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


class TranscriptResult(BaseModel):
    transcript: str
    lang: str | None = None
    provider: Literal["sarvam", "elevenlabs", "none"] = "sarvam"
    duration_ms: float = 0.0
