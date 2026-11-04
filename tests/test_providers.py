"""Tests for LLM providers — fully offline via httpx.MockTransport."""

from __future__ import annotations

import json

import httpx
import pytest

from conftest import add
from nexus_agents import ToolRegistry
from nexus_agents.errors import (
    MissingCredentialsError,
    ProviderResponseError,
    ProviderTransportError,
)
from nexus_agents.providers import (
    OpenAICompatibleProvider,
    ScriptedProvider,
    tool_call_message,
)
from nexus_agents.providers.base import GenerateRequest
from nexus_agents.types import FinishReason, Message, Role, Usage, assistant_message, user_message

BASE = "https://llm.test/v1"


def make_client(handler) -> httpx.Client:
    return httpx.Client(base_url=BASE, transport=httpx.MockTransport(handler))


def chat_payload(content: str | None = "hi", **extra) -> dict:
    message = {"role": "assistant", "content": content}
    message.update(extra)
    return {
        "id": "chatcmpl-1",
        "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }


def request(messages=None, tools=()):
    return GenerateRequest(
        messages=tuple(messages or [user_message("hello")]),
        tools=tools,
    )


# ================================================================ ScriptedProvider


def test_scripted_returns_responses_in_order():
    provider = ScriptedProvider(["a", "b"])
    assert provider.generate(request()).message.content == "a"
    assert provider.generate(request()).message.content == "b"


def test_scripted_records_every_request():
    provider = ScriptedProvider(["done"])
    provider.generate(request())
    assert provider.call_count == 1
    assert provider.requests[0].messages[0].content == "hello"


def test_scripted_repeats_last_by_default():
    provider = ScriptedProvider(["only"])
    for _ in range(3):
        assert provider.generate(request()).message.content == "only"
    assert provider.call_count == 3


def test_scripted_can_stop_repeating():
    provider = ScriptedProvider(["only"], repeat_last=False)
    assert provider.generate(request()).message.content == "only"
    assert provider.generate(request()).message.content == ""


def test_scripted_supports_callable_responses():
    def respond(req: GenerateRequest) -> Message:
        return assistant_message(f"saw {len(req.messages)} messages")

    provider = ScriptedProvider([respond])
    assert provider.generate(request()).message.content == "saw 1 messages"


def test_scripted_tool_call_helper_sets_finish_reason():
    provider = ScriptedProvider([tool_call_message("add", {"a": 1, "b": 2})])
    response = provider.generate(request())
    assert response.finish_reason is FinishReason.TOOL_CALLS
    assert response.message.tool_calls[0].name == "add"


def test_scripted_usage_is_configurable():
    provider = ScriptedProvider(["x"], usage=Usage(prompt_tokens=5, completion_tokens=6))
    assert provider.generate(request()).usage.total_tokens == 11


def test_scripted_reset_clears_state():
    provider = ScriptedProvider(["x"])
    provider.generate(request())
    provider.reset()
    assert provider.call_count == 0


# ================================================================ HTTP provider


def test_parses_plain_text_response():
    client = make_client(lambda r: httpx.Response(200, json=chat_payload("hello there")))
    provider = OpenAICompatibleProvider(model="test-model", client=client)
    response = provider.generate(request())

    assert response.message.content == "hello there"
    assert response.message.role is Role.ASSISTANT
    assert response.usage.total_tokens == 18
    assert response.finish_reason is FinishReason.STOP


def test_parses_tool_calls_with_json_string_arguments():
    payload = chat_payload(
        None,
        tool_calls=[
            {
                "id": "call_42",
                "type": "function",
                "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'},
            }
        ],
    )
    payload["choices"][0]["finish_reason"] = "tool_calls"

    client = make_client(lambda r: httpx.Response(200, json=payload))
    provider = OpenAICompatibleProvider(model="m", client=client)
    response = provider.generate(request())

    assert response.finish_reason is FinishReason.TOOL_CALLS
    call = response.message.tool_calls[0]
    assert call.id == "call_42"
    assert call.name == "add"
    assert call.arguments == {"a": 1, "b": 2}


def test_sends_tools_and_sampling_options():
    captured = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json=chat_payload("ok"))

    provider = OpenAICompatibleProvider(model="m", client=make_client(handler))
    provider.generate(
        GenerateRequest(
            messages=(user_message("hi"),),
            tools=ToolRegistry([add]).specs(),
            temperature=0.2,
            max_tokens=64,
        )
    )

    assert captured["body"]["model"] == "m"
    assert captured["body"]["temperature"] == 0.2
    assert captured["body"]["max_tokens"] == 64
    assert captured["body"]["tools"][0]["function"]["name"] == "add"
    assert captured["body"]["tool_choice"] == "auto"
    assert captured["body"]["messages"][0]["role"] == "user"


def test_omits_optional_fields_when_unset():
    captured = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json=chat_payload("ok"))

    provider = OpenAICompatibleProvider(model="m", client=make_client(handler))
    provider.generate(request())
    assert "temperature" not in captured["body"]
    assert "tools" not in captured["body"]


def test_http_error_raises_transport_error():
    client = make_client(lambda r: httpx.Response(401, json={"error": "bad key"}))
    provider = OpenAICompatibleProvider(model="m", client=client)
    with pytest.raises(ProviderTransportError) as exc:
        provider.generate(request())
    assert "401" in str(exc.value)
    assert "sk-" not in str(exc.value)  # 不能把密钥片段吐出来


def test_retries_on_server_error_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    attempts = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json=chat_payload("recovered"))

    provider = OpenAICompatibleProvider(model="m", client=make_client(handler), max_retries=2)
    response = provider.generate(request())

    assert response.message.content == "recovered"
    assert attempts["n"] == 3


def test_gives_up_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    client = make_client(lambda r: httpx.Response(500, text="always down"))
    provider = OpenAICompatibleProvider(model="m", client=client, max_retries=1)
    with pytest.raises(ProviderTransportError):
        provider.generate(request())


def test_network_failure_becomes_transport_error():
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    provider = OpenAICompatibleProvider(
        model="m", client=make_client(handler), max_retries=0
    )
    with pytest.raises(ProviderTransportError):
        provider.generate(request())


def test_malformed_response_raises_response_error():
    client = make_client(lambda r: httpx.Response(200, json={"choices": []}))
    provider = OpenAICompatibleProvider(model="m", client=client)
    with pytest.raises(ProviderResponseError):
        provider.generate(request())


def test_incomplete_tool_calls_are_skipped():
    payload = chat_payload(
        None,
        tool_calls=[{"id": "c0", "function": {"name": ""}}],  # 缺 name，应被忽略
    )
    client = make_client(lambda r: httpx.Response(200, json=payload))
    provider = OpenAICompatibleProvider(model="m", client=client)
    assert provider.generate(request()).message.tool_calls == ()


def test_missing_credentials_is_reported_clearly(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(MissingCredentialsError) as exc:
        OpenAICompatibleProvider(model="gpt-4o-mini")
    assert "OPENAI_API_KEY" in str(exc.value)


def test_local_endpoint_needs_no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = make_client(lambda r: httpx.Response(200, json=chat_payload("local")))
    # 自建 vLLM / Ollama 通常不需要鉴权，不应强制要 key
    provider = OpenAICompatibleProvider(
        model="local", base_url="http://localhost:8000/v1", client=client
    )
    assert provider.generate(request()).message.content == "local"


def test_context_manager_closes_owned_client():
    provider = OpenAICompatibleProvider(
        model="m", api_key="test-key", base_url=BASE
    )
    with provider as p:
        assert p is provider
    assert provider._client.is_closed is True
