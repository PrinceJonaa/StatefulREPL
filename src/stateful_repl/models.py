"""
Model Abstraction Layer — Phase 2.

Provides a unified interface (ModelAdapter) across LLM providers.
Concrete adapters: OpenAI, Anthropic, Local (Ollama/vLLM).

Usage:
    from stateful_repl.models import create_adapter

    adapter = create_adapter("openai", model="gpt-4o", api_key="sk-...")
    response = adapter.complete("What is 2+2?")
    logprobs = adapter.get_logprobs("The capital of France is Paris.")
    embedding = adapter.embed("hello world")
"""

from __future__ import annotations

import json
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


# ─────────────────────────────────────────────────────────
# Response types
# ─────────────────────────────────────────────────────────

@dataclass
class CompletionResponse:
    """Normalised response from any LLM provider."""

    text: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)  # prompt_tokens, completion_tokens, total_tokens
    logprobs: Optional[List[float]] = None
    raw: Optional[Dict[str, Any]] = None  # provider-specific payload

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)

    @property
    def cost_estimate(self) -> float:
        """Rough USD cost estimate (configurable per-model)."""
        rates = _MODEL_RATES.get(self.model, (0.0, 0.0))
        inp = self.usage.get("prompt_tokens", 0) * rates[0] / 1_000_000
        out = self.usage.get("completion_tokens", 0) * rates[1] / 1_000_000
        return inp + out


# Per-million-token rates: (input, output)
_MODEL_RATES: Dict[str, tuple] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-opus-4-20250514": (15.00, 75.00),
}


# ─────────────────────────────────────────────────────────
# Adapter Protocol
# ─────────────────────────────────────────────────────────

class ModelAdapter(ABC):
    """
    Unified interface for any LLM.

    Subclasses must implement complete(), get_logprobs(), embed().
    batch_complete() has a default sequential implementation.
    """

    model: str

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
        system: Optional[str] = None,
    ) -> CompletionResponse:
        ...

    @abstractmethod
    def get_logprobs(self, prompt: str, continuation: str) -> List[float]:
        """Return per-token log-probabilities for *continuation* given *prompt*."""
        ...

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Return embedding vector for *text*."""
        ...

    def batch_complete(
        self,
        prompts: List[str],
        **kwargs,
    ) -> List[CompletionResponse]:
        """Default: run sequentially. Override for provider-native batching."""
        return [self.complete(p, **kwargs) for p in prompts]


# ─────────────────────────────────────────────────────────
# OpenAI Adapter
# ─────────────────────────────────────────────────────────

class OpenAIAdapter(ModelAdapter):
    """Adapter for OpenAI API (gpt-4o, gpt-4o-mini, etc.)."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package required: pip install openai"
                )
            kwargs: Dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
        system: Optional[str] = None,
    ) -> CompletionResponse:
        client = self._get_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop or None,
        )
        choice = resp.choices[0]
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        }
        return CompletionResponse(
            text=choice.message.content or "",
            model=resp.model,
            usage=usage,
            raw=resp.model_dump() if hasattr(resp, "model_dump") else None,
        )

    def get_logprobs(self, prompt: str, continuation: str) -> List[float]:
        client = self._get_client()
        messages = [{"role": "user", "content": prompt + continuation}]
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=1,
            logprobs=True,
        )
        if resp.choices[0].logprobs and resp.choices[0].logprobs.content:
            return [t.logprob for t in resp.choices[0].logprobs.content]
        return []

    def embed(self, text: str) -> List[float]:
        client = self._get_client()
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return resp.data[0].embedding


# ─────────────────────────────────────────────────────────
# Anthropic Adapter
# ─────────────────────────────────────────────────────────

class AnthropicAdapter(ModelAdapter):
    """Adapter for Anthropic API (Claude models)."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package required: pip install anthropic"
                )
            self._client = Anthropic(api_key=self._api_key)
        return self._client

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
        system: Optional[str] = None,
    ) -> CompletionResponse:
        client = self._get_client()
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature > 0:
            kwargs["temperature"] = temperature
        if system:
            kwargs["system"] = system
        if stop:
            kwargs["stop_sequences"] = stop

        resp = client.messages.create(**kwargs)
        text = resp.content[0].text if resp.content else ""
        usage = {
            "prompt_tokens": resp.usage.input_tokens,
            "completion_tokens": resp.usage.output_tokens,
            "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens,
        }
        return CompletionResponse(
            text=text,
            model=resp.model,
            usage=usage,
            raw=resp.model_dump() if hasattr(resp, "model_dump") else None,
        )

    def get_logprobs(self, prompt: str, continuation: str) -> List[float]:
        # Anthropic doesn't expose logprobs; return empty
        return []

    def embed(self, text: str) -> List[float]:
        # Anthropic doesn't have an embedding API; raise or fallback
        raise NotImplementedError(
            "Anthropic does not provide embeddings. Use OpenAI or a local model."
        )


# ─────────────────────────────────────────────────────────
# Local / Ollama Adapter
# ─────────────────────────────────────────────────────────

class LocalAdapter(ModelAdapter):
    """
    Adapter for local models via Ollama or any OpenAI-compatible server.

    Default base_url: http://localhost:11434/v1 (Ollama)
    """

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
    ):
        self.model = model
        self._base_url = base_url
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package required for local adapter: pip install openai"
                )
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._client

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
        system: Optional[str] = None,
    ) -> CompletionResponse:
        client = self._get_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop or None,
        )
        choice = resp.choices[0]
        usage_data = {}
        if resp.usage:
            usage_data = {
                "prompt_tokens": resp.usage.prompt_tokens or 0,
                "completion_tokens": resp.usage.completion_tokens or 0,
                "total_tokens": resp.usage.total_tokens or 0,
            }
        return CompletionResponse(
            text=choice.message.content or "",
            model=self.model,
            usage=usage_data,
        )

    def get_logprobs(self, prompt: str, continuation: str) -> List[float]:
        return []  # Most local servers don't expose logprobs

    def embed(self, text: str) -> List[float]:
        client = self._get_client()
        resp = client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding


# ─────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────

_ADAPTERS = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "local": LocalAdapter,
    "ollama": LocalAdapter,
}


def create_adapter(provider: str, **kwargs) -> ModelAdapter:
    """
    Factory function to create a model adapter.

    provider: "openai", "anthropic", "local", or "ollama"
    kwargs: passed to the adapter constructor (model, api_key, base_url, etc.)
    """
    cls = _ADAPTERS.get(provider.lower())
    if cls is None:
        raise ValueError(
            f"Unknown provider: {provider!r}. "
            f"Available: {list(_ADAPTERS.keys())}"
        )
    return cls(**kwargs)
