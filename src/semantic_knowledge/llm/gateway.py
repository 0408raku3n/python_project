"""Независимый от поставщика контракт вызова LLM (docs/llm-contracts.md, §3)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping, Protocol, runtime_checkable

from jsonschema import Draft202012Validator, SchemaError

from .errors import LLMConfigurationError

OperationName = Literal[
    "intent.classify",
    "knowledge.extract",
    "answer.generate",
    "changes.explain",
]
MessageRole = Literal["system", "user"]


@dataclass(frozen=True)
class LLMMessage:
    role: MessageRole
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("system", "user"):
            raise LLMConfigurationError("Недопустимая роль сообщения LLM")
        if not isinstance(self.content, str) or not self.content.strip():
            raise LLMConfigurationError("Содержимое сообщения LLM должно быть непустым")


@dataclass(frozen=True)
class GenerationParams:
    model: str
    temperature: float = 0.0
    max_output_tokens: int = 2048
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise LLMConfigurationError("Не указана модель LLM")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(self.temperature)
            or not 0 <= self.temperature <= 2
        ):
            raise LLMConfigurationError("temperature должна быть числом от 0 до 2")
        if (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or self.max_output_tokens <= 0
        ):
            raise LLMConfigurationError("max_output_tokens должна быть положительным целым")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise LLMConfigurationError("timeout_seconds должна быть положительным числом")


@dataclass(frozen=True)
class LLMRequest:
    operation: OperationName
    messages: tuple[LLMMessage, ...]
    response_schema: Mapping[str, object]
    params: GenerationParams
    prompt_version: str

    def __post_init__(self) -> None:
        if self.operation not in (
            "intent.classify", "knowledge.extract", "answer.generate", "changes.explain"
        ):
            raise LLMConfigurationError("Неизвестная операция LLM")
        if not isinstance(self.messages, tuple) or not self.messages:
            raise LLMConfigurationError("Требуется хотя бы одно сообщение LLM")
        if not all(isinstance(message, LLMMessage) for message in self.messages):
            raise LLMConfigurationError("Неверный формат сообщений LLM")
        if not isinstance(self.response_schema, Mapping) or not self.response_schema:
            raise LLMConfigurationError("Не указана JSON-схема ответа LLM")
        try:
            Draft202012Validator.check_schema(self.response_schema)
        except SchemaError as exc:
            raise LLMConfigurationError("Некорректная JSON-схема запроса LLM") from exc
        if not isinstance(self.params, GenerationParams):
            raise LLMConfigurationError("Отсутствуют параметры генерации")
        if not isinstance(self.prompt_version, str) or not self.prompt_version.strip():
            raise LLMConfigurationError("Не указана версия промпта")


@dataclass(frozen=True)
class LLMResponse:
    """Сырой ответ; намеренно допускает пустой content для негативных тестов."""

    content: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None


@runtime_checkable
class LLMGateway(Protocol):
    """Один порт для реального провайдера и тестового адаптера."""

    def generate(self, request: LLMRequest) -> LLMResponse: ...
