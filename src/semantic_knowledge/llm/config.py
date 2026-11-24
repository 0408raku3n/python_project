"""Выбор адаптера и параметров генерации из окружения (без секретов в файлах)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Mapping

import httpx

from .errors import LLMConfigurationError
from .fake import ExpectedLLMCall, FakeLLMGateway
from .gateway import GenerationParams, LLMGateway
from .openai_gateway import OpenAILLMGateway


@dataclass(frozen=True)
class LLMSettings:
    provider: str = "fake"
    model: str = "fake"
    temperature: float = 0.0
    max_output_tokens: int = 2048
    timeout_seconds: float = 30.0
    openai_base_url: str = "https://api.openai.com/v1"

    def __post_init__(self) -> None:
        if self.provider not in ("fake", "openai"):
            raise LLMConfigurationError("LLM_PROVIDER должен быть fake или openai")
        self.generation_params()  # Валидация едина для окружения и явно заданных настроек.

    def generation_params(self) -> GenerationParams:
        return GenerationParams(
            model=self.model,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            timeout_seconds=self.timeout_seconds,
        )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> LLMSettings:
        env = os.environ if environ is None else environ
        provider = env.get("LLM_PROVIDER", "fake").strip().lower()
        model = env.get("LLM_MODEL", "fake" if provider == "fake" else "")
        try:
            temperature = float(env.get("LLM_TEMPERATURE", "0"))
            max_tokens = int(env.get("LLM_MAX_OUTPUT_TOKENS", "2048"))
            timeout = float(env.get("LLM_TIMEOUT_SECONDS", "30"))
        except ValueError as exc:
            raise LLMConfigurationError("Неверный числовой параметр LLM в окружении") from exc
        return cls(
            provider=provider,
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens,
            timeout_seconds=timeout,
            openai_base_url=env.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )


def create_llm_gateway(
    settings: LLMSettings | None = None,
    *,
    fake_calls: Iterable[ExpectedLLMCall] = (),
    http_client: httpx.Client | None = None,
) -> LLMGateway:
    """Единая точка композиции; прикладным компонентам нужен только LLMGateway."""
    chosen = settings if settings is not None else LLMSettings.from_env()
    if chosen.provider == "fake":
        return FakeLLMGateway(fake_calls)
    return OpenAILLMGateway(base_url=chosen.openai_base_url, client=http_client)
