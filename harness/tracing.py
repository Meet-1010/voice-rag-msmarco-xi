"""Per-stage span timing.

This is not observability decoration. The 200ms requirement is scoped to a subset
of the pipeline, so we need per-stage numbers to say honestly which stages fall
inside that budget and which do not. The trace is the evidence.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

# Stages that make up the "core" pipeline the latency target applies to. Network
# calls to third-party voice and LLM APIs are measured but excluded, and the README
# says so explicitly rather than burying it.
CORE_STAGES = {"guard_input", "cache", "embed", "retrieve", "rerank",
               "guard_relevance", "extract", "guard_grounding"}


class Trace:
    def __init__(self):
        self.spans: list[dict] = []
        self.t0 = time.perf_counter()

    @contextmanager
    def span(self, name: str):
        start = time.perf_counter()
        rec = {"stage": name, "duration_ms": 0.0, "ok": True}
        self.spans.append(rec)
        try:
            yield rec
        except Exception as exc:
            rec["ok"] = False
            rec["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            rec["duration_ms"] = round((time.perf_counter() - start) * 1000, 3)

    def mark(self, name: str, ms: float, **extra) -> None:
        self.spans.append({"stage": name, "duration_ms": round(ms, 3), "ok": True, **extra})

    @property
    def total_ms(self) -> float:
        return round((time.perf_counter() - self.t0) * 1000, 3)

    def core_ms(self) -> float:
        return round(sum(s["duration_ms"] for s in self.spans if s["stage"] in CORE_STAGES), 3)

    def by_stage(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for s in self.spans:
            out[s["stage"]] = round(out.get(s["stage"], 0.0) + s["duration_ms"], 3)
        return out

    def summary(self) -> dict:
        return {"stages": self.by_stage(), "core_ms": self.core_ms(), "total_ms": self.total_ms}
