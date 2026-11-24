"""OpenAI Chat Completions adapter (JSON mode + local JSON Schema validation).

No HTTP request is made at import or construction time. A caller may pass
httpx.Client with MockTransport to test the complete adapter offline.
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import urlsplit

import httpx

from .errors import (
    LLMConfigurationError,
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from .gateway import LLMRequest, LLMResponse

_JSON_SYSTEM_INSTRUCTION = "Return exactly one valid JSON object. No Markdown or other text."


class OpenAILLMGateway:
    """Адаптер OpenAI; ключ берётся только из окружения или внедряется извне."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        client: httpx.Client | None = None,
    ) -> None:
        token = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        if not isinstance(token, str) or not token.strip():
            raise LLMConfigurationError("Не задан OPENAI_API_KEY")
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise LLMConfigurationError("OPENAI_BASE_URL должен быть HTTPS URL без учётных данных")
        self._token = token
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._client = client

    def generate(self, request: LLMRequest) -> LLMResponse:
        # JSON mode does not support every keyword from the project's 2020-12
        # schemas (e.g. if/then/else). The full schema is checked locally.
        try:
            schema_text = json.dumps(
                request.response_schema, ensure_ascii=False, sort_keys=True
            )
        except (TypeError, ValueError) as exc:
            raise LLMConfigurationError("JSON-схема LLM не сериализуется") from exc
        payload: dict[str, object] = {
            "model": request.params.model,
            "messages": [
                {"role": "system", "content": _JSON_SYSTEM_INSTRUCTION},
                {
                    "role": "system",
                    "content": "The response must match this JSON Schema 2020-12: " + schema_text,
                },
                *(
                    {"role": message.role, "content": message.content}
                    for message in request.messages
                ),
            ],
            "temperature": request.params.temperature,
            "max_completion_tokens": request.params.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        try:
            if self._client is not None:
                response = self._client.post(
                    self._url,
                    json=payload,
                    headers=self._headers(),
                    timeout=request.params.timeout_seconds,
                )
            else:
                with httpx.Client(follow_redirects=False) as client:
                    response = client.post(
                        self._url,
                        json=payload,
                        headers=self._headers(),
                        timeout=request.params.timeout_seconds,
                    )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("Истекло время ожидания OpenAI") from exc
        except httpx.RequestError as exc:
            raise LLMUnavailableError("Не удалось подключиться к OpenAI") from exc

        if response.status_code in (408, 504):
            raise LLMTimeoutError("Тайм-аут OpenAI")
        if response.status_code == 429:
            raise LLMRateLimitError("Ограничение частоты запросов OpenAI")
        if response.status_code in (400, 401, 403, 404, 422):
            # Ошибка модели, авторизации или несовместимых параметров.
            # Не включать в исключение тело ответа или секреты.
            raise LLMConfigurationError(
                f"OpenAI отклонил конфигурацию (HTTP {response.status_code})"
            )
        if response.status_code >= 300:
            raise LLMUnavailableError(f"OpenAI недоступен (HTTP {response.status_code})")
        try:
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("not an object")
            choices = data["choices"]
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError("missing or ambiguous choices")
            choice = choices[0]
            content = choice["message"]["content"]
            finish_reason = choice["finish_reason"]
            model = data["model"]
            if not isinstance(content, str) or not isinstance(model, str) or not model.strip():
                raise ValueError("invalid model or content")
            if finish_reason is not None and not isinstance(finish_reason, str):
                raise ValueError("invalid finish reason")
            usage = data.get("usage") or {}
            if not isinstance(usage, dict):
                raise ValueError("invalid usage")
            input_tokens = _token_count(usage.get("prompt_tokens"))
            output_tokens = _token_count(usage.get("completion_tokens"))
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            raise LLMInvalidResponseError("OpenAI вернул некорректную структуру ответа") from exc
        return LLMResponse(
            content=content,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
        )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"}


def _token_count(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("invalid token count")
    return value
