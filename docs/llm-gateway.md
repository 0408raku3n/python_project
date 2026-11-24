# Реализация LLMGateway 

Контракт определён в [`llm-contracts.md`](llm-contracts.md), §3. Публичные классы доступны из `semantic_knowledge.llm`; бизнес-логика должна принимать аргумент типа `LLMGateway` и **не импортировать** `OpenAILLMGateway` напрямую.

## Пример использования

```python
import json
from pathlib import Path
from semantic_knowledge.llm import (
    LLMMessage, LLMRequest, LLMSettings, create_llm_gateway, generate_structured,
)

settings = LLMSettings.from_env()
gateway = create_llm_gateway(settings)  # по умолчанию FakeLLMGateway
schema = json.loads(Path("docs/llm/schemas/intent-decision.schema.json").read_text())
request = LLMRequest(
    operation="intent.classify",
    messages=(
        LLMMessage(role="system", content="Верни JSON по указанной схеме."),
        LLMMessage(role="user", content='{"mode":"automatic","text":"Вопрос?"}'),
    ),
    response_schema=schema,
    params=settings.generation_params(),
    prompt_version="1",
)
# generate_structured(gateway, request) -> dict; для fake предварительно задайте ответы.
```

`generate(request)` возвращает `LLMResponse` с исходным текстом и метаданными. Общая функция `generate_structured(gateway, request)` разбирает **один** JSON-объект, отклоняет дубли ключей, `NaN`/`Infinity`, усечённый/пустой ответ и валидирует его против переданной JSON Schema 2020-12. Не выполняет семантических проверок соответствия источнику/графу.

## Конфигурация и секреты

Значения можно задать переменными окружения (образец `.env.example`):

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `LLM_PROVIDER` | `fake` | `fake` или `openai` |
| `LLM_MODEL` | `fake` для `fake`, обязательна для `openai` | Идентификатор модели |
| `LLM_TEMPERATURE` | `0` | Температура, 0–2 |
| `LLM_MAX_OUTPUT_TOKENS` | `2048` | Лимит ответа |
| `LLM_TIMEOUT_SECONDS` | `30` | Тайм-аут HTTP-запроса |
| `OPENAI_API_KEY` | нет | Секрет **только из окружения** при `openai` |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | HTTPS endpoint |

Пример настройки внешней LLM в терминале (не записывая секрет в репозиторий):

```bash
export LLM_PROVIDER=openai
export LLM_MODEL='имя-доступной-вам-модели'
read -rsp 'OpenAI API key: ' OPENAI_API_KEY; echo
export OPENAI_API_KEY
```

OpenAI-адаптер использует Chat Completions JSON mode и локальную проверку полной схемы: существующие схемы содержат `if/then/else`, которые не все режимы server-side structured output поддерживают. В запрос включаются отдельная системная инструкция с **самой JSON-схемой**, системные инструкции вызывающего кода, пользовательские сообщения, температура, лимит и тайм-аут. Схема передаётся как текст в системном сообщении (не как `response_format: json_schema`, поскольку используемые схемы не полностью совместимы с ограничениями этого режима). Если модель не поддерживает заданные параметры или JSON mode, её HTTP 400 преобразуется в `LLMConfigurationError` (параметры/модель нужно исправить). Ошибки и журналы не содержат ключи и сырые ответы.

`FakeLLMGateway([ExpectedLLMCall(operation, LLMResponse(...))])` получает очередь ожидаемых вызовов. Он сравнивает порядок операций, по желанию полный запрос, сохраняет `calls` и проверяет отсутствие оставшихся ответов через `assert_all_consumed()`. Вместо ответа может выдавать типизированную ошибку; никакой сети и ключа не требуется.

## Проверки

```bash
PYTHONPATH=src python -m pytest tests/unit/test_llm_gateway.py
PYTHONPATH=src python -m pytest
```

Тесты реального адаптера используют `httpx.MockTransport`, поэтому не совершают сетевых вызовов. Исключения: `LLMTimeoutError`, `LLMUnavailableError`, `LLMRateLimitError`, `LLMConfigurationError`, `LLMInvalidResponseError`.
