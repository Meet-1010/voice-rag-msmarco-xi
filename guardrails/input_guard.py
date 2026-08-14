"""First gate: is this even a question we should try to answer?

Runs before any embedding or retrieval, because rejecting here costs microseconds
and rejecting later costs the whole pipeline.
"""
from __future__ import annotations

import re

from harness.schemas import ReasonCode

# Injection attempts aimed at the answering prompt. Matched on the transcript
# because voice input is still user-controlled text by the time it reaches us.
_INJECTION = [
    re.compile(p, re.IGNORECASE) for p in (
        r"ignore\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+(instruction|prompt|direction|rule)",
        r"disregard\s+(all\s+|the\s+)?(previous|prior|above|system)",
        r"forget\s+(everything|all|your)\s+(you|instruction|rule|prompt)",
        r"you\s+are\s+now\s+(a|an|no longer)",
        r"(reveal|show|print|repeat|output)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instruction)",
        r"act\s+as\s+(if\s+you\s+are\s+)?(a|an)\s+\w+",
        r"\bDAN\b|\bjailbreak\b|developer\s+mode",
        r"pretend\s+(you|to\s+be)",
        r"new\s+(instruction|rule)s?\s*[:\-]",
        r"<\s*/?\s*(system|assistant|im_start|im_end)\s*>",
        r"निर्देश.*(अनदेखा|भूल)",
    )
]

_UNSAFE = [
    re.compile(p, re.IGNORECASE) for p in (
        r"how\s+(do\s+i|to)\s+(make|build|synthesi[sz]e)\s+.*(bomb|explosive|nerve\s+agent|meth)",
        r"how\s+(do\s+i|to)\s+(kill|murder|poison)\s+(someone|a\s+person|my)",
        r"(child|minor)\s+(porn|sexual)",
        r"credit\s+card\s+(number|dump)s?\s+(list|for\s+sale)",
    )
]

# Garbage transcripts: STT emits these when it hears noise.
_GARBAGE = re.compile(r"^[\W_]+$|^(uh|um|hmm|ah|er)[\s.,]*$", re.IGNORECASE)

_SCRIPT_RANGES = {"hi": (0x0900, 0x097F), "gu": (0x0A80, 0x0AFF)}


def detect_lang(text: str) -> str:
    """Script-based, not a model. The three languages we support use three distinct
    scripts, so counting codepoints is both exact and free.

    Anything outside those three scripts is counted as "other" rather than being
    left uncounted. Dropping it instead made every unsupported language fall
    through to the "en" default and sail past the language check.
    """
    counts = {"hi": 0, "gu": 0, "en": 0, "other": 0}
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        for lang, (lo, hi) in _SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                counts[lang] += 1
                break
        else:
            counts["en" if cp < 0x250 else "other"] += 1
    if not any(counts.values()):
        return "en"
    return max(counts.items(), key=lambda kv: kv[1])[0]


class InputGuard:
    def __init__(self, cfg: dict):
        gc = cfg["guardrails"]["input"]
        self.min_chars = gc["min_chars"]
        self.max_chars = gc["max_chars"]
        self.supported = set(gc["supported_langs"])

    def check(self, query: str, lang: str | None = None) -> tuple[ReasonCode | None, str | None, str]:
        """Return (reason_code or None, detail, resolved_lang)."""
        text = (query or "").strip()
        resolved = lang or detect_lang(text)

        if len(text) < self.min_chars or _GARBAGE.match(text):
            return ReasonCode.EMPTY_INPUT, "empty or unintelligible transcript", resolved
        if len(text) > self.max_chars:
            return ReasonCode.EMPTY_INPUT, f"query exceeds {self.max_chars} chars", resolved

        for pat in _UNSAFE:
            if pat.search(text):
                return ReasonCode.UNSAFE_INPUT, f"matched unsafe pattern {pat.pattern[:40]!r}", resolved
        for pat in _INJECTION:
            if pat.search(text):
                return ReasonCode.PROMPT_INJECTION, f"matched injection pattern {pat.pattern[:40]!r}", resolved

        if resolved not in self.supported:
            return ReasonCode.UNSUPPORTED_LANGUAGE, f"detected {resolved}", resolved
        return None, None, resolved
