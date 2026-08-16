"""Second gate: do we actually hold an answer to this?

Thresholds are per language, and that is not incidental. multilingual-e5 does not
produce comparable score distributions across scripts: measured on this index,
in-corpus English queries average 0.910 while Gujarati averages 0.869, yet
out-of-corpus queries sit at ~0.84 in every language. A single global threshold
therefore refuses far more genuine Gujarati questions than English ones while
being the most permissive exactly where the encoder is weakest.

Values come from guardrails/calibrate.py, which sweeps labelled in-corpus and
out-of-corpus sets per language and writes guardrails/thresholds.yaml.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from harness.schemas import ReasonCode

_SIDECAR = Path(__file__).resolve().parent / "thresholds.yaml"


class RelevanceGuard:
    def __init__(self, cfg: dict):
        rc = cfg["guardrails"]["relevance"]
        self.default = float(rc["min_top_score"])
        self.per_lang: dict[str, float] = {}
        self.min_margin = rc.get("min_margin", 0.0)
        self.source = "config.yaml"

        if _SIDECAR.exists():
            blob = yaml.safe_load(_SIDECAR.read_text(encoding="utf-8")) or {}
            rel = blob.get("relevance") or {}
            self.default = float(rel.get("default", self.default))
            self.per_lang = {k: float(v) for k, v in (rel.get("per_lang") or {}).items()}
            self.source = "thresholds.yaml"

    def threshold_for(self, lang: str | None) -> float:
        return self.per_lang.get(lang or "", self.default)

    def check(self, hits: list[dict], lang: str | None = None):
        """Return (reason_code or None, detail, top_score)."""
        if not hits:
            return ReasonCode.OUT_OF_CORPUS, "no candidates retrieved", 0.0

        # Threshold on raw cosine, never on the fused RRF score. RRF is a rank
        # statistic: its top value is roughly constant regardless of whether the
        # match is good, so thresholding it would accept everything.
        top = max(h.get("dense_score", 0.0) for h in hits)
        floor = self.threshold_for(lang)
        if top < floor:
            return (ReasonCode.OUT_OF_CORPUS,
                    f"top dense score {top:.3f} < {floor:.3f} for lang={lang}", top)

        # min_margin defaults to 0, i.e. off. The intuition that an in-corpus query
        # produces a peaked score distribution and an out-of-corpus one a flat
        # distribution does not hold here: over 400 in-corpus and 52 out-of-corpus
        # queries, top-1 score alone scores AUC 0.907 while the top1-minus-mean-of
        # -next-4 margin scores only 0.740, and combining them beats neither.
        if self.min_margin > 0 and len(hits) > 1:
            scores = sorted((h.get("dense_score", 0.0) for h in hits), reverse=True)
            if scores[0] - scores[1] < self.min_margin:
                return (ReasonCode.LOW_CONFIDENCE,
                        f"margin {scores[0]-scores[1]:.3f} < {self.min_margin:.3f}", top)
        return None, None, top
