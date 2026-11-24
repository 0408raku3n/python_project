"""Контракт LLMGateway: все тесты полностью изолированы от сети."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import httpx
import pytest

from semantic_knowledge.llm import (
    ExpectedLLMCall,
    FakeLLMGateway,
    GenerationParams,
    LLMConfigurationError,
    LLMGateway,
    LLMInvalidResponseError,
    LLMMessage,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    LLMSettings,
    LLMTimeoutError,
    LLMUnavailableError,
    OpenAILLMGateway,
    create_llm_gateway,
    generate_structured,
    parse_structured_response,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEMAS = ROOT / "docs" / "llm" / "schemas"
EXAMPLES = ROOT / "examples" / "llm"
CONTRACT_EXAMPLES = [
    ("intent.classify", "intent-decision.schema.json", "intent-classification.json"),
    ("knowledge.extract", "extraction-result.schema.json", "knowledge-extraction.json"),
    ("answer.generate", "answer-result.schema.json", "answer-generation.json"),
    ("changes.explain", "change-explanation.schema.json", "change-explanation.json"),
]


def sample_request(
    operation: str = "intent.classify", *, schema: dict | None = None
) -> LLMRequest:
    return LLMRequest(
        operation=operation,  # type: ignore[arg-type]
        messages=(
            LLMMessage("system", "Отвечай только JSON"),
            LLMMessage("user", '{"text": "Что произошло?"}'),
        ),
        response_schema=schema or {"type": "object", "required": ["ok"]},
        params=GenerationParams(
            model="fake", temperature=0, max_output_tokens=256, timeout_seconds=2
        ),
        prompt_version="v1",
    )


def sample_response(content: str = '{"ok": true}', *, reason: str = "stop") -> LLMResponse:
    return LLMResponse(content=content, model="fake", finish_reason=reason)


def offline_gateway(
    handler: Callable[[httpx.Request], httpx.Response],
) -> tuple[OpenAILLMGateway, httpx.Client]:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAILLMGateway(api_key="offline-placeholder", client=client), client


def test_port_is_the_same_for_real_and_fake() -> None:
    fake = FakeLLMGateway()
    actual, client = offline_gateway(lambda _: httpx.Response(500))
    with client:
        assert isinstance(fake, LLMGateway)
        assert isinstance(actual, LLMGateway)


def test_fake_preserves_order_and_entire_call_history() -> None:
    first = sample_request()
    second = sample_request("knowledge.extract")
    fake = FakeLLMGateway(
        [
            ExpectedLLMCall(first.operation, sample_response(), request=first),
            ExpectedLLMCall(second.operation, sample_response('{"entities": []}')),
        ]
    )
    assert fake.generate(first) == sample_response()
    assert fake.generate(second).model == "fake"
    assert fake.calls == [first, second]
    assert fake.remaining == 0
    fake.assert_all_consumed()
    with pytest.raises(AssertionError, match="Неожиданный вызов"):
        fake.generate(first)


def test_fake_rejects_wrong_order_without_consuming_expectation() -> None:
    fake = FakeLLMGateway([ExpectedLLMCall("intent.classify", sample_response())])
    with pytest.raises(AssertionError, match="Нарушен порядок"):
        fake.generate(sample_request("answer.generate"))
    assert fake.remaining == 1
    with pytest.raises(AssertionError, match="Неиспользованные ответы"):
        fake.assert_all_consumed()
    fake.generate(sample_request())
    fake.assert_all_consumed()


def test_fake_validates_full_expected_request_and_can_enqueue() -> None:
    request = sample_request()
    fake = FakeLLMGateway()
    fake.enqueue("intent.classify", sample_response(), request=request)
    modified = LLMRequest(
        operation=request.operation,
        messages=(LLMMessage("user", "other"),),
        response_schema=request.response_schema,
        params=request.params,
        prompt_version=request.prompt_version,
    )
    with pytest.raises(AssertionError, match="Неверные параметры"):
        fake.generate(modified)
    assert fake.remaining == 1
    assert fake.generate(request).content == '{"ok": true}'


@pytest.mark.parametrize(
    "exception",
    [
        LLMTimeoutError("offline"),
        LLMUnavailableError("offline"),
        LLMRateLimitError("offline"),
    ],
)
def test_fake_can_emulate_typed_failures(exception: Exception) -> None:
    fake = FakeLLMGateway([ExpectedLLMCall("intent.classify", exception)])  # type: ignore[arg-type]
    with pytest.raises(type(exception)):
        fake.generate(sample_request())
    fake.assert_all_consumed()


@pytest.mark.parametrize("operation,schema_name,example_name", CONTRACT_EXAMPLES)
def test_all_four_real_contract_examples_validate(
    operation: str, schema_name: str, example_name: str
) -> None:
    schema = json.loads((SCHEMAS / schema_name).read_text(encoding="utf-8"))
    body = (EXAMPLES / example_name).read_text(encoding="utf-8")
    request = sample_request(operation, schema=schema)
    fake = FakeLLMGateway([ExpectedLLMCall(request.operation, sample_response(body))])
    assert generate_structured(fake, request) == json.loads(body)
    fake.assert_all_consumed()


@pytest.mark.parametrize(
    "raw,reason",
    [
        ("", "stop"),
        ("   ", "stop"),
        ("[]", "stop"),
        ('{"ok": true} {"ok": true}', "stop"),
        ("```json\n{\"ok\":true}\n```", "stop"),
        ('{"ok": true, "ok": false}', "stop"),
        ('{"ok": NaN}', "stop"),
        ('{"ok": Infinity}', "stop"),
        ('{"other": true}', "stop"),
        ("{}", "stop"),
        ('{"ok": true}', "length"),
        ('{"ok": true}', "content_filter"),
    ],
)
def test_invalid_or_truncated_structured_response_is_rejected(raw: str, reason: str) -> None:
    with pytest.raises(LLMInvalidResponseError) as exc:
        parse_structured_response(sample_request(), sample_response(raw, reason=reason))
    assert exc.value.code == "LLM_INVALID_RESPONSE"


def test_invalid_request_schema_is_configuration_error() -> None:
    with pytest.raises(LLMConfigurationError):
        sample_request(schema={"type": "not-a-type"})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": ""},
        {"model": "fake", "temperature": -1},
        {"model": "fake", "temperature": float("nan")},
        {"model": "fake", "temperature": True},
        {"model": "fake", "max_output_tokens": 0},
        {"model": "fake", "max_output_tokens": True},
        {"model": "fake", "timeout_seconds": float("inf")},
        {"model": "fake", "timeout_seconds": 0},
    ],
)
def test_invalid_generation_params_rejected_early(kwargs: dict) -> None:
    with pytest.raises(LLMConfigurationError):
        GenerationParams(**kwargs)


def test_invalid_request_fields_rejected_early() -> None:
    with pytest.raises(LLMConfigurationError):
        LLMMessage("tool", "hi")  # type: ignore[arg-type]
    with pytest.raises(LLMConfigurationError):
        LLMMessage("user", " ")
    with pytest.raises(LLMConfigurationError):
        sample_request(operation="unrecognized")


def test_configuration_selects_fake_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = LLMSettings.from_env({"LLM_PROVIDER": "fake"})
    assert settings.generation_params().model == "fake"
    fake = create_llm_gateway(
        settings, fake_calls=[ExpectedLLMCall("intent.classify", sample_response())]
    )
    assert isinstance(fake, FakeLLMGateway)
    assert fake.generate(sample_request()).model == "fake"


def test_config_from_environment_and_provider_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    env = {
        "LLM_PROVIDER": "openai",
        "LLM_MODEL": "demo-model",
        "LLM_TEMPERATURE": "0.25",
        "LLM_MAX_OUTPUT_TOKENS": "987",
        "LLM_TIMEOUT_SECONDS": "5.5",
    }
    settings = LLMSettings.from_env(env)
    assert settings.generation_params() == GenerationParams("demo-model", 0.25, 987, 5.5)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-placeholder")
    real, client = offline_gateway(lambda _: httpx.Response(500))
    with client:
        selected = create_llm_gateway(settings, http_client=client)
        assert isinstance(real, OpenAILLMGateway)
        assert isinstance(selected, OpenAILLMGateway)


@pytest.mark.parametrize(
    "env",
    [
        {"LLM_PROVIDER": "bad"},
        {"LLM_PROVIDER": "openai"},
        {"LLM_TEMPERATURE": "bogus"},
        {"LLM_MAX_OUTPUT_TOKENS": "not-an-int"},
        {"LLM_TIMEOUT_SECONDS": "-1"},
    ],
)
def test_invalid_environment_rejected(env: dict[str, str]) -> None:
    with pytest.raises(LLMConfigurationError):
        LLMSettings.from_env(env)


def test_missing_key_and_insecure_url_rejected_without_io(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMConfigurationError, match="OPENAI_API_KEY"):
        create_llm_gateway(LLMSettings(provider="openai", model="demo-model"))
    with pytest.raises(LLMConfigurationError, match="HTTPS"):
        OpenAILLMGateway(api_key="offline-placeholder", base_url="http://example.org/v1")
    with pytest.raises(LLMConfigurationError, match="HTTPS"):
        OpenAILLMGateway(api_key="offline-placeholder", base_url="https://user:pass@example.org")


def test_openai_transmits_roles_generation_params_and_reads_metadata() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.openai.com/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer offline-placeholder"
        assert request.extensions["timeout"]["read"] == 6.5
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "mock-model",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": '{"ok":true}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            },
        )

    gateway, client = offline_gateway(handler)
    original = sample_request()
    request = LLMRequest(
        operation=original.operation,
        messages=original.messages,
        response_schema=original.response_schema,
        params=GenerationParams("demo-model", 0.4, 123, 6.5),
        prompt_version="v1",
    )
    with client:
        answer = gateway.generate(request)
    assert answer == LLMResponse('{"ok":true}', "mock-model", 11, 7, "stop")
    assert seen == [
        {
            "model": "demo-model",
            "messages": [
                {
                    "role": "system",
                    "content": "Return exactly one valid JSON object. No Markdown or other text.",
                },
                {
                    "role": "system",
                    "content": (
                        "The response must match this JSON Schema 2020-12: "
                        + json.dumps(request.response_schema, ensure_ascii=False, sort_keys=True)
                    ),
                },
                {"role": "system", "content": "Отвечай только JSON"},
                {"role": "user", "content": '{"text": "Что произошло?"}'},
            ],
            "temperature": 0.4,
            "max_completion_tokens": 123,
            "response_format": {"type": "json_object"},
        }
    ]


@pytest.mark.parametrize(
    "status,error",
    [(408, LLMTimeoutError), (504, LLMTimeoutError), (429, LLMRateLimitError),
     (400, LLMConfigurationError), (401, LLMConfigurationError),
     (404, LLMConfigurationError), (302, LLMUnavailableError),
     (500, LLMUnavailableError), (503, LLMUnavailableError)],
)
def test_openai_maps_http_failures(status: int, error: type[Exception]) -> None:
    gateway, client = offline_gateway(lambda _: httpx.Response(status, text="do not disclose body"))
    with client, pytest.raises(error) as exc:
        gateway.generate(sample_request())
    assert "do not disclose body" not in str(exc.value)


@pytest.mark.parametrize(
    "exc,error",
    [
        (httpx.ConnectError("offline"), LLMUnavailableError),
        (httpx.ReadTimeout("offline"), LLMTimeoutError),
    ],
)
def test_openai_maps_transport_failures(exc: Exception, error: type[Exception]) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise exc

    gateway, client = offline_gateway(handler)
    with client, pytest.raises(error):
        gateway.generate(sample_request())


@pytest.mark.parametrize(
    "body",
    [
        "broken json",
        "[]",
        '{}',
        '{"model": "m", "choices": []}',
        '{"model": "m", "choices": [{"finish_reason": "stop", "message": {"content": null}}]}',
        (
            '{"model": "m", "choices": [{"finish_reason": "stop", '
            '"message": {"content": "{}"}}], "usage": {"prompt_tokens": true}}'
        ),
    ],
)
def test_openai_invalid_body_is_typed_error(body: str) -> None:
    gateway, client = offline_gateway(lambda _: httpx.Response(200, text=body))
    with client, pytest.raises(LLMInvalidResponseError):
        gateway.generate(sample_request())


def test_real_and_fake_use_the_same_structured_helper() -> None:
    request = sample_request()
    fake = FakeLLMGateway([ExpectedLLMCall(request.operation, sample_response())])

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "mock-model",
                "choices": [
                    {"message": {"content": '{"ok":true}'}, "finish_reason": "stop"}
                ],
            },
        )

    actual, client = offline_gateway(handler)
    with client:
        assert generate_structured(fake, request) == generate_structured(actual, request)
    fake.assert_all_consumed()
