"""Second gate: do we actually hold an answer to this?

Thresholds come from guardrails/calibrate.py, which sweeps them over labelled
in-corpus and out-of-corpus query sets. They are not tuned by feel.
"""
from __future__ import annotations

from harness.schemas import ReasonCode


class RelevanceGuard:
    def __init__(self, cfg: dict):
        rc = cfg["guardrails"]["relevance"]
        self.min_top_score = rc["min_top_score"]
        self.min_margin = rc.get("min_margin", 0.0)

    def check(self, hits: list[dict]) -> tuple[ReasonCode | None, str | None, float]:
        if not hits:
            return ReasonCode.OUT_OF_CORPUS, "no candidates retrieved", 0.0

        # Threshold on raw cosine, never on the fused RRF score. RRF is a rank
        # statistic: its top value is roughly constant regardless of whether the
        # match is good, so thresholding it would accept everything.
        top = max(h.get("dense_score", 0.0) for h in hits)
        if top < self.min_top_score:
            return (ReasonCode.OUT_OF_CORPUS,
                    f"top dense score {top:.3f} < {self.min_top_score:.3f}", top)

        if self.min_margin > 0 and len(hits) > 1:
            scores = sorted((h.get("dense_score", 0.0) for h in hits), reverse=True)
            if scores[0] - scores[1] < self.min_margin:
                return (ReasonCode.LOW_CONFIDENCE,
                        f"margin {scores[0]-scores[1]:.3f} < {self.min_margin:.3f}", top)
        return None, None, top
