"""Типизированные ошибки транспортного/форматного уровня LLM."""


class LLMError(Exception):
    code = "LLM_ERROR"


class LLMTimeoutError(LLMError):
    code = "LLM_TIMEOUT"


class LLMUnavailableError(LLMError):
    code = "LLM_UNAVAILABLE"


class LLMRateLimitError(LLMUnavailableError):
    code = "LLM_RATE_LIMIT"


class LLMConfigurationError(LLMError):
    code = "LLM_CONFIG_ERROR"


class LLMInvalidResponseError(LLMError):
    code = "LLM_INVALID_RESPONSE"
