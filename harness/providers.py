"""LLM provider chain: Groq -> Sarvam -> extractive.

The chain exists so the API never returns a 500 because someone else's service is
down. Each provider carries its own breaker, so one dead vendor does not slow down
requests that could have been served by the next one. The terminal fallback is
extractive answering, which needs no network at all - which also means the whole
system still answers with zero API keys configured.
"""
from __future__ import annotations

import json
import os
import re

import httpx

from .retry import CircuitBreaker, CircuitOpen, ProviderError, with_backoff

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class Provider:
    name = "base"

    def available(self) -> bool:
        raise NotImplementedError

    def complete(self, system: str, user: str, timeout: float) -> str:
        raise NotImplementedError


class GroqProvider(Provider):
    name = "groq"

    def __init__(self, cfg: dict):
        gc = cfg["llm"]["groq"]
        self.base_url = gc["base_url"]
        self.primary = gc["primary"]
        self.fast = gc["fast"]
        self.max_tokens = cfg["llm"]["max_tokens"]
        self.temperature = cfg["llm"]["temperature"]
        self.key = os.getenv("GROQ_API_KEY", "").strip()

    def available(self) -> bool:
        return bool(self.key)

    def complete(self, system: str, user: str, timeout: float, model: str | None = None) -> str:
        payload = {
            "model": model or self.primary,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        try:
            r = httpx.post(f"{self.base_url}/chat/completions", json=payload, timeout=timeout,
                           headers={"Authorization": f"Bearer {self.key}"})
        except httpx.HTTPError as exc:
            raise ProviderError(f"groq transport: {exc}") from exc
        if r.status_code == 429:
            raise ProviderError("groq rate limited")
        if r.status_code >= 400:
            raise ProviderError(f"groq {r.status_code}: {r.text[:200]}")
        return r.json()["choices"][0]["message"]["content"]


class SarvamProvider(Provider):
    name = "sarvam"

    def __init__(self, cfg: dict):
        sc = cfg["llm"]["sarvam"]
        self.base_url = sc["base_url"]
        self.model = sc["model"]
        self.max_tokens = cfg["llm"]["max_tokens"]
        self.temperature = cfg["llm"]["temperature"]
        self.key = os.getenv("SARVAM_API_KEY", "").strip()

    def available(self) -> bool:
        return bool(self.key)

    def complete(self, system: str, user: str, timeout: float) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        try:
            r = httpx.post(f"{self.base_url}/v1/chat/completions", json=payload, timeout=timeout,
                           headers={"api-subscription-key": self.key})
        except httpx.HTTPError as exc:
            raise ProviderError(f"sarvam transport: {exc}") from exc
        if r.status_code >= 400:
            raise ProviderError(f"sarvam {r.status_code}: {r.text[:200]}")
        return r.json()["choices"][0]["message"]["content"]


def parse_json_answer(raw: str) -> dict | None:
    """Models wrap JSON in prose or fences often enough that a bare json.loads is
    not worth attempting on its own."""
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


class ProviderChain:
    def __init__(self, cfg: dict):
        cb = cfg["circuit_breaker"]
        registry = {"groq": GroqProvider, "sarvam": SarvamProvider}
        self.providers: list[Provider] = []
        self.breakers: dict[str, CircuitBreaker] = {}
        for name in cfg["llm"]["provider_order"]:
            if name not in registry:
                continue
            p = registry[name](cfg)
            self.providers.append(p)
            self.breakers[p.name] = CircuitBreaker(cb["fail_threshold"], cb["reset_seconds"])

    def any_available(self) -> bool:
        return any(p.available() for p in self.providers)

    def status(self) -> dict:
        return {p.name: {"configured": p.available(), "circuit": self.breakers[p.name].state}
                for p in self.providers}

    def complete(self, system: str, user: str, timeout: float) -> tuple[str, str]:
        """Return (raw_text, provider_name). Raises ProviderError if the whole chain
        is exhausted, which the orchestrator turns into an extractive fallback."""
        errors = []
        for p in self.providers:
            if not p.available():
                continue
            breaker = self.breakers[p.name]
            if not breaker.allow():
                errors.append(f"{p.name}: circuit open")
                continue

            call = with_backoff()(p.complete)
            try:
                out = call(system, user, timeout)
                breaker.record_success()
                return out, p.name
            except (ProviderError, CircuitOpen) as exc:
                breaker.record_failure()
                errors.append(f"{p.name}: {exc}")
        raise ProviderError("; ".join(errors) or "no provider configured")
