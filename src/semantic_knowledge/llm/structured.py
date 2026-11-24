"""Строгая и единая для всех провайдеров проверка структурированного ответа."""

from __future__ import annotations

import json
from typing import Any

from jsonschema import Draft202012Validator, SchemaError, ValidationError

from .errors import LLMConfigurationError, LLMInvalidResponseError
from .gateway import LLMGateway, LLMRequest, LLMResponse


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("повторяющийся ключ JSON")
        obj[key] = value
    return obj


def _reject_constant(_: str) -> None:
    raise ValueError("нечисловая константа JSON")


def parse_structured_response(
    request: LLMRequest, response: LLMResponse
) -> dict[str, object]:
    """Проверить единственный JSON-объект по схеме 2020-12 без исправлений.

    Семантические проверки связей с текстом/графом остаются за обработчиком
    конкретной операции
    """
    if not isinstance(response.content, str) or not response.content.strip():
        raise LLMInvalidResponseError("LLM вернула пустой ответ")
    if response.finish_reason not in (None, "stop"):
        raise LLMInvalidResponseError("LLM завершила генерацию нештатно или с усечением")

    try:
        validator = Draft202012Validator(request.response_schema)
        Draft202012Validator.check_schema(request.response_schema)
    except SchemaError as exc:
        raise LLMConfigurationError("Некорректная JSON-схема запроса LLM") from exc

    try:
        data = json.loads(
            response.content,
            object_pairs_hook=_object_no_duplicates,
            parse_constant=_reject_constant,
        )
    except (TypeError, ValueError) as exc:
        raise LLMInvalidResponseError("Ответ LLM не является корректным JSON-объектом") from exc
    if not isinstance(data, dict):
        raise LLMInvalidResponseError("Ответ LLM должен быть одним JSON-объектом")

    try:
        validator.validate(data)
    except ValidationError as exc:
        # Не включаем содержимое ответа или исходного пользовательского текста.
        path = ".".join(str(part) for part in exc.absolute_path) or "<root>"
        raise LLMInvalidResponseError(f"Ответ LLM не соответствует схеме (поле {path})") from exc
    return data


def generate_structured(gateway: LLMGateway, request: LLMRequest) -> dict[str, object]:
    """Универсальный вызов и валидация; не зависит от выбранного API."""
    return parse_structured_response(request, gateway.generate(request))
