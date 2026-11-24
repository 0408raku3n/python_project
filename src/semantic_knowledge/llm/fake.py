"""Сценарный FakeLLMGateway"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

from .errors import LLMError
from .gateway import LLMRequest, LLMResponse, OperationName


@dataclass(frozen=True)
class ExpectedLLMCall:
    operation: OperationName
    response: LLMResponse | LLMError
    request: LLMRequest | None = None


class FakeLLMGateway:
    def __init__(self, expected_calls: Iterable[ExpectedLLMCall] = ()) -> None:
        self._pending: deque[ExpectedLLMCall] = deque(expected_calls)
        self.calls: list[LLMRequest] = []

    def enqueue(
        self,
        operation: OperationName,
        response: LLMResponse | LLMError,
        *,
        request: LLMRequest | None = None,
    ) -> None:
        """Добавить подготовленный ответ/ошибку и при желании точный запрос."""
        self._pending.append(ExpectedLLMCall(operation, response, request))

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        if not self._pending:
            raise AssertionError(f"Неожиданный вызов FakeLLMGateway: {request.operation}")
        expected = self._pending[0]
        if expected.operation != request.operation:
            raise AssertionError(
                f"Нарушен порядок FakeLLMGateway: ожидалось {expected.operation}, "
                f"получено {request.operation}"
            )
        if expected.request is not None and expected.request != request:
            raise AssertionError(f"Неверные параметры запроса FakeLLMGateway: {request.operation}")
        self._pending.popleft()
        if isinstance(expected.response, LLMError):
            raise expected.response
        return expected.response

    def assert_all_consumed(self) -> None:
        if self._pending:
            operations = [call.operation for call in self._pending]
            raise AssertionError(f"Неиспользованные ответы FakeLLMGateway: {operations}")

    @property
    def remaining(self) -> int:
        return len(self._pending)
