"""OpenAI-compatible chat-completions provider.

Works with OpenAI, vLLM, Ollama, LM Studio, Groq, Together, DeepSeek —
anything that speaks ``POST /chat/completions``.

Security note
-------------
The API key is read from the environment or passed explicitly. It is never
logged, never included in exception messages, and never written into traces.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from ..errors import (
    MissingCredentialsError,
    ProviderResponseError,
    ProviderTransportError,
)
from ..types import (
    FinishReason,
    ToolCall,
    Usage,
    assistant_message,
)
from .base import GenerateRequest, LLMResponse

__all__ = ["OpenAICompatibleProvider", "DEFAULT_BASE_URL"]

DEFAULT_BASE_URL = "https://api.openai.com/v1"

_FINISH_REASON_MAP = {
    "stop": FinishReason.STOP,
    "tool_calls": FinishReason.TOOL_CALLS,
    "function_call": FinishReason.TOOL_CALLS,
    "length": FinishReason.LENGTH,
    "content_filter": FinishReason.ERROR,
}

#: 值得重试的 HTTP 状态码（限流与服务端抖动）
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class OpenAICompatibleProvider:
    """A thin, dependency-injectable client for chat-completions APIs."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        client: httpx.Client | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        backoff_base: float = 0.5,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self._max_retries = max(0, max_retries)
        self._backoff_base = backoff_base
        self._base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")

        if client is not None:
            # 依赖注入：测试用 MockTransport，生产用真实连接
            self._client = client
            self._owns_client = False
        else:
            resolved_key = api_key or os.environ.get(api_key_env)
            if not resolved_key and self._base_url == DEFAULT_BASE_URL:
                raise MissingCredentialsError(api_key_env)
            headers: dict[str, str] = {}
            if resolved_key:
                headers["Authorization"] = f"Bearer {resolved_key}"
            if extra_headers:
                headers.update(extra_headers)
            self._client = httpx.Client(
                base_url=self._base_url, timeout=timeout, headers=headers
            )
            self._owns_client = True

    # ------------------------------------------------------------ 生命周期

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OpenAICompatibleProvider:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------ 请求

    def generate(self, request: GenerateRequest) -> LLMResponse:
        payload = self._build_payload(request)
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.post("/chat/completions", json=payload)
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    raise ProviderTransportError(
                        f"Request to {self._base_url} failed: {type(exc).__name__}"
                    ) from exc
                self._sleep(attempt)
                continue

            if response.status_code in _RETRYABLE_STATUS and attempt < self._max_retries:
                last_error = None
                self._sleep(attempt)
                continue

            if response.status_code >= 400:
                raise ProviderTransportError(
                    f"HTTP {response.status_code} from chat-completions endpoint: "
                    f"{_safe_snippet(response.text)}"
                )

            try:
                data = response.json()
            except ValueError as exc:
                raise ProviderResponseError("Response body is not valid JSON") from exc

            return self._parse(data)

        raise ProviderTransportError(
            f"Exhausted {self._max_retries} retries without success"
        ) from last_error

    def _sleep(self, attempt: int) -> None:
        time.sleep(self._backoff_base * (2**attempt))

    def _build_payload(self, request: GenerateRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_api_payload() for m in request.messages],
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = [t.to_api_payload() for t in request.tools]
            payload["tool_choice"] = "auto"
        if request.extra:
            payload.update(request.extra)
        return payload

    # ------------------------------------------------------------ 解析

    def _parse(self, data: dict[str, Any]) -> LLMResponse:
        choices = data.get("choices")
        if not choices:
            raise ProviderResponseError("Response contained no 'choices' field")

        choice = choices[0]
        raw_message = choice.get("message") or {}
        tool_calls = self._parse_tool_calls(raw_message.get("tool_calls"))

        return LLMResponse(
            message=assistant_message(
                raw_message.get("content") or "", tool_calls=tool_calls
            ),
            usage=self._parse_usage(data.get("usage")),
            finish_reason=_FINISH_REASON_MAP.get(
                choice.get("finish_reason"), FinishReason.STOP
            ),
            raw=data,
        )

    @staticmethod
    def _parse_tool_calls(raw: Any) -> tuple[ToolCall, ...]:
        if not raw:
            return ()
        calls: list[ToolCall] = []
        for index, item in enumerate(raw):
            function = item.get("function") or {}
            name = function.get("name")
            if not name:
                continue  # 跳过残缺的调用，而不是让整轮崩溃
            calls.append(
                ToolCall(
                    id=item.get("id") or f"call_{index}",
                    name=name,
                    # ToolCall 的校验器负责把 JSON 字符串还原成 dict
                    arguments=function.get("arguments") or {},
                )
            )
        return tuple(calls)

    @staticmethod
    def _parse_usage(raw: Any) -> Usage:
        if not isinstance(raw, dict):
            return Usage()
        return Usage(
            prompt_tokens=int(raw.get("prompt_tokens") or 0),
            completion_tokens=int(raw.get("completion_tokens") or 0),
        )


def _safe_snippet(text: str, limit: int = 300) -> str:
    """Truncate an error body so logs stay readable and cheap."""
    cleaned = " ".join(text.split())
    return cleaned[:limit] + ("..." if len(cleaned) > limit else "")
