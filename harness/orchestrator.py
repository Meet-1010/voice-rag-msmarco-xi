"""The pipeline state machine.

    guard_input -> cache -> embed -> retrieve -> rerank -> guard_relevance
                -> route(extractive | generative) -> guard_grounding -> respond

Written as an explicit sequence of guarded stages rather than a chain of function
calls so that every transition is traced, every stage has a timeout budget, and
every exit path produces the same response envelope. A refusal and an answer are
the same object with different fields set, so callers never branch on shape.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from guardrails.grounding_guard import GroundingGuard
from guardrails.input_guard import InputGuard
from guardrails.policy import message as refusal_message
from guardrails.relevance_guard import RelevanceGuard
from harness.prompts import REPAIR, SYSTEM, build_user_prompt
from harness.providers import ProviderChain, ProviderError, parse_json_answer
from harness.schemas import (Answer, AnswerPath, Citation, LLMAnswer, ReasonCode,
                             StageTrace, Timings)
from harness.tracing import Trace
from index.bm25 import BM25Index
from index.embedder import Embedder
from index.store import VectorStore
from retrieval.cache import SemanticCache
from retrieval.extractive import ExtractiveAnswerer
from retrieval.hybrid import HybridRetriever
from retrieval.rerank import CrossEncoderReranker

ROOT = Path(__file__).resolve().parents[1]


class Orchestrator:
    def __init__(self, config_path: str | Path | None = None):
        self.cfg = yaml.safe_load(Path(config_path or ROOT / "config.yaml").read_text())
        self.budgets = self.cfg["budgets_ms"]

        self.embedder = Embedder(self.cfg["embedder"])
        self.store = VectorStore(self.cfg, ROOT)
        bm_path = ROOT / ".artifacts" / "bm25.pkl"
        self.bm25 = BM25Index.load(bm_path) if bm_path.exists() else BM25Index()
        self.retriever = HybridRetriever(self.cfg, self.store, self.bm25, self.embedder, ROOT)
        self.reranker = CrossEncoderReranker(self.cfg)
        self.cache = SemanticCache(self.cfg, self.cfg["embedder"]["dim"])
        self.extractive = ExtractiveAnswerer(self.cfg, self.embedder)

        self.input_guard = InputGuard(self.cfg)
        self.relevance_guard = RelevanceGuard(self.cfg)
        self.grounding_guard = GroundingGuard(self.cfg, self.embedder)
        self.providers = ProviderChain(self.cfg)

        manifest = ROOT / ".artifacts" / "manifest.json"
        self.manifest = json.loads(manifest.read_text()) if manifest.exists() else {}

    # Tool-call surface. The orchestrator routes between these rather than
    # inlining them, which keeps each one independently testable and traceable.
    def tools(self) -> dict:
        return {
            "search_kb": self.retriever.search,
            "rerank": self.reranker.rerank,
            "answer_extractive": self.extractive.try_answer,
            "answer_generative": self._generate,
            "refuse": self._refuse,
        }

    def _timings(self, trace: Trace) -> Timings:
        core = trace.core_ms()
        return Timings(stages=trace.by_stage(), core_ms=core, total_ms=trace.total_ms,
                       within_budget=core <= self.budgets["core_total"])

    def _spans(self, trace: Trace) -> list[StageTrace]:
        return [StageTrace(**{k: v for k, v in s.items() if k in
                              {"stage", "duration_ms", "ok", "error"}}) for s in trace.spans]

    def _refuse(self, code: ReasonCode, lang: str | None, trace: Trace,
                detail: str | None = None, citations: list[Citation] | None = None) -> Answer:
        return Answer(
            answer=refusal_message(code, lang), citations=citations or [], confidence=0.0,
            path=AnswerPath.REFUSED, lang=lang, grounded=False, refused=True,
            reason_code=code, timings=self._timings(trace), trace=self._spans(trace),
        )

    @staticmethod
    def _citations(hits: list[dict], keep: list[int] | None = None) -> list[Citation]:
        chosen = hits
        if keep:
            picked = [hits[i - 1] for i in keep if 1 <= i <= len(hits)]
            chosen = picked or hits
        return [Citation(chunk_id=h["chunk_id"], passage_id=h["passage_id"],
                         text=(h.get("context") or h["text"])[:600],
                         score=round(float(h.get("dense_score", h.get("score", 0.0))), 4),
                         lang=h["lang"]) for h in chosen]

    def _generate(self, query: str, contexts: list[str], timeout_s: float) -> tuple[LLMAnswer, str]:
        user = build_user_prompt(query, contexts)
        raw, provider = self.providers.complete(SYSTEM, user, timeout_s)
        parsed = parse_json_answer(raw)

        if parsed is None:
            # One repair attempt. A second is not worth the latency; the extractive
            # fallback is already a decent answer.
            raw, provider = self.providers.complete(
                SYSTEM, f"{user}\n\n{REPAIR}\n\nYour previous reply was:\n{raw[:500]}", timeout_s)
            parsed = parse_json_answer(raw)
        if parsed is None:
            raise ProviderError("model did not return parseable JSON after repair")

        if isinstance(parsed.get("citations"), (int, str)):
            parsed["citations"] = [parsed["citations"]]
        parsed["citations"] = [int(c) for c in (parsed.get("citations") or [])
                               if str(c).strip().lstrip("-").isdigit()]
        return LLMAnswer(**parsed), provider

    def ask(self, query: str, lang: str | None = None, top_k: int | None = None,
            use_cache: bool = True, allow_generative: bool = True) -> Answer:
        trace = Trace()

        with trace.span("guard_input"):
            code, detail, lang = self.input_guard.check(query, lang)
        if code:
            return self._refuse(code, lang, trace, detail)

        with trace.span("embed"):
            q_vec = self.embedder.query(query)

        if use_cache:
            with trace.span("cache"):
                cached = self.cache.lookup(q_vec, lang)
            if cached:
                payload = cached["payload"]
                return Answer(
                    answer=payload["answer"],
                    citations=[Citation(**c) for c in payload["citations"]],
                    confidence=payload["confidence"], path=AnswerPath.CACHE, lang=lang,
                    grounded=True, provider=payload.get("provider"),
                    cache_similarity=round(cached["similarity"], 4),
                    timings=self._timings(trace), trace=self._spans(trace),
                )

        with trace.span("retrieve"):
            hits = self.retriever.search(query, q_vec, lang=lang, top_k=top_k)

        if self.reranker.applies_to(lang):
            with trace.span("rerank"):
                hits = self.reranker.rerank(query, hits, lang)

        with trace.span("guard_relevance"):
            code, detail, top_score = self.relevance_guard.check(hits, lang)
        if code:
            return self._refuse(code, lang, trace, detail)

        with trace.span("extract"):
            extracted = self.extractive.try_answer(query, q_vec, hits)

        if extracted:
            cites = self._citations([extracted["hit"]])
            answer = Answer(
                answer=extracted["answer"], citations=cites,
                confidence=extracted["confidence"], path=AnswerPath.EXTRACTIVE, lang=lang,
                grounded=True, provider="extractive",
                timings=self._timings(trace), trace=self._spans(trace),
            )
            # Extractive spans are copied out of the retrieved passage, so they are
            # grounded by construction. Re-checking would only add latency.
            if use_cache:
                self.cache.put(q_vec, query, self._cacheable(answer), lang)
            return answer

        contexts = [(h.get("context") or h["text"]) for h in hits[:5]]

        if not allow_generative or not self.providers.any_available():
            return self._degrade(query, q_vec, hits, contexts, lang, trace)

        try:
            with trace.span("generate"):
                llm, provider = self._generate(query, contexts, self.budgets["generate"] / 1000)
        except ProviderError:
            return self._degrade(query, q_vec, hits, contexts, lang, trace)

        if not llm.answer.strip():
            # The model followed instructions and declined; that is a real signal,
            # not a failure.
            return self._refuse(ReasonCode.OUT_OF_CORPUS, lang, trace, "model found no answer in context")

        with trace.span("guard_grounding"):
            text, dropped = self.grounding_guard.enforce_citations(llm.answer, contexts)
            code, metrics = self.grounding_guard.check(text, contexts)
        if code:
            return self._refuse(code, lang, trace, json.dumps(metrics),
                                citations=self._citations(hits, llm.citations))

        answer = Answer(
            answer=text, citations=self._citations(hits, llm.citations),
            confidence=llm.confidence, path=AnswerPath.GENERATIVE, lang=lang,
            grounded=True, provider=provider,
            timings=self._timings(trace), trace=self._spans(trace),
        )
        if use_cache:
            self.cache.put(q_vec, query, self._cacheable(answer), lang)
        return answer

    def _degrade(self, query, q_vec, hits, contexts, lang, trace) -> Answer:
        """No LLM reachable. Return the best retrieved span rather than a 500.

        This is what makes the system usable with zero API keys configured, and it
        is the terminal element of the provider chain rather than an error path.
        """
        with trace.span("extract"):
            fallback = self.extractive.try_answer(query, q_vec, hits)
        if fallback:
            return Answer(
                answer=fallback["answer"], citations=self._citations([fallback["hit"]]),
                confidence=fallback["confidence"] * 0.9, path=AnswerPath.EXTRACTIVE,
                lang=lang, grounded=True, provider="extractive-fallback",
                reason_code=ReasonCode.PROVIDER_UNAVAILABLE,
                timings=self._timings(trace), trace=self._spans(trace),
            )
        top = hits[0]
        return Answer(
            answer=(top.get("context") or top["text"])[:600],
            citations=self._citations(hits[:3]), confidence=0.35,
            path=AnswerPath.EXTRACTIVE, lang=lang, grounded=True,
            provider="passage-verbatim", reason_code=ReasonCode.PROVIDER_UNAVAILABLE,
            timings=self._timings(trace), trace=self._spans(trace),
        )

    @staticmethod
    def _cacheable(answer: Answer) -> dict:
        return {"answer": answer.answer,
                "citations": [c.model_dump() for c in answer.citations],
                "confidence": answer.confidence, "provider": answer.provider}

    def health(self) -> dict:
        return {
            "status": "ok",
            "indexed_points": self.store.count(),
            "bm25_languages": sorted(self.bm25.by_lang),
            "providers": self.providers.status(),
            "cache": self.cache.stats(),
            "manifest": self.manifest,
            "rerank_enabled": self.reranker.enabled,
        }
