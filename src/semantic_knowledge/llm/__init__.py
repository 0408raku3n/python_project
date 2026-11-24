"""Порт LLM, OpenAI-адаптер, сценарный тестовый провайдер и конфигурация."""

from .config import LLMSettings, create_llm_gateway
from .errors import (
    LLMConfigurationError,
    LLMError,
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from .fake import ExpectedLLMCall, FakeLLMGateway
from .gateway import (
    GenerationParams,
    LLMGateway,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    OperationName,
)
from .openai_gateway import OpenAILLMGateway
from .structured import generate_structured, parse_structured_response

__all__ = [
    "ExpectedLLMCall",
    "FakeLLMGateway",
    "GenerationParams",
    "LLMConfigurationError",
    "LLMError",
    "LLMGateway",
    "LLMInvalidResponseError",
    "LLMMessage",
    "LLMRateLimitError",
    "LLMRequest",
    "LLMResponse",
    "LLMSettings",
    "LLMTimeoutError",
    "LLMUnavailableError",
    "OpenAILLMGateway",
    "OperationName",
    "create_llm_gateway",
    "generate_structured",
    "parse_structured_response",
]
