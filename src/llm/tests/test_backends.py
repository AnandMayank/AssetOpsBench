"""Tests for backend selection and the OpenAI-compatible (TokenRouter) path."""

from __future__ import annotations

import sys
import types

import pytest

from llm import LiteLLMBackend, OpenAICompatBackend, is_openai_compat, make_backend


def _install_fake_openai(monkeypatch, captured: dict):
    """Install a stub ``openai`` module that records call kwargs."""

    def create(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content="hi")
                )
            ],
            usage=types.SimpleNamespace(prompt_tokens=3, completion_tokens=2),
        )

    class OpenAI:
        def __init__(self, base_url=None, api_key=None):
            captured["base_url"] = base_url
            captured["api_key"] = api_key
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=create)
            )

    class BadRequestError(Exception):
        pass

    fake = types.ModuleType("openai")
    fake.OpenAI = OpenAI
    fake.BadRequestError = BadRequestError
    monkeypatch.setitem(sys.modules, "openai", fake)


def test_is_openai_compat():
    assert is_openai_compat("tokenrouter/MiniMax-M3")
    assert not is_openai_compat("litellm_proxy/aws/claude-opus-4-6")
    assert not is_openai_compat("watsonx/meta-llama/llama-3-3-70b-instruct")


def test_make_backend_dispatch():
    assert isinstance(make_backend("tokenrouter/MiniMax-M3"), OpenAICompatBackend)
    assert isinstance(make_backend("litellm_proxy/aws/claude-opus-4-6"), LiteLLMBackend)
    assert isinstance(make_backend("watsonx/meta-llama/x"), LiteLLMBackend)


def test_unsupported_prefix_raises():
    with pytest.raises(ValueError):
        OpenAICompatBackend("gpt-4o")


def test_tokenrouter_strips_prefix_and_routes(monkeypatch):
    captured: dict = {}
    _install_fake_openai(monkeypatch, captured)
    monkeypatch.setenv("TOKENROUTER_BASE_URL", "https://api.tokenrouter.com/v1")
    monkeypatch.setenv("TOKENROUTER_API_KEY", "tr-key")

    result = make_backend("tokenrouter/MiniMax-M3").generate_with_usage("hello")

    assert captured["model"] == "MiniMax-M3"  # bare name, prefix stripped
    assert captured["base_url"] == "https://api.tokenrouter.com/v1"
    assert captured["api_key"] == "tr-key"
    assert result.text == "hi"
    assert (result.input_tokens, result.output_tokens) == (3, 2)


def test_model_id_property_keeps_full_string():
    assert OpenAICompatBackend("tokenrouter/MiniMax-M3").model_id == "tokenrouter/MiniMax-M3"


def _install_fake_litellm(monkeypatch, captured: dict):
    """Install a stub ``litellm`` module that records completion() kwargs."""

    def completion(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="hi"))],
            usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )

    fake = types.ModuleType("litellm")
    fake.completion = completion
    monkeypatch.setitem(sys.modules, "litellm", fake)


def test_together_ai_prefix_uses_together_api_key(monkeypatch):
    """P3.6: together_ai/* must NOT go through the LITELLM_API_KEY/
    LITELLM_BASE_URL else-branch -- it has its own credential (and no
    api_base override, unlike watsonx's optional WATSONX_URL)."""
    captured: dict = {}
    _install_fake_litellm(monkeypatch, captured)
    monkeypatch.setenv("TOGETHER_API_KEY", "together-key")
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)

    result = LiteLLMBackend("together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo").generate_with_usage("hi")

    assert captured["api_key"] == "together-key"
    assert "api_base" not in captured
    assert result.text == "hi"


def test_together_ai_missing_key_raises(monkeypatch):
    captured: dict = {}
    _install_fake_litellm(monkeypatch, captured)
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)

    with pytest.raises(KeyError):
        LiteLLMBackend("together_ai/some-model").generate_with_usage("hi")


def test_watsonx_prefix_still_uses_watsonx_creds(monkeypatch):
    """Regression: the together_ai/ branch must not disturb the existing
    watsonx/ credential path."""
    captured: dict = {}
    _install_fake_litellm(monkeypatch, captured)
    monkeypatch.setenv("WATSONX_APIKEY", "wx-key")
    monkeypatch.setenv("WATSONX_PROJECT_ID", "wx-project")
    monkeypatch.delenv("WATSONX_URL", raising=False)

    result = LiteLLMBackend("watsonx/meta-llama/llama-3-3-70b-instruct").generate_with_usage("hi")

    assert captured["api_key"] == "wx-key"
    assert captured["project_id"] == "wx-project"
    assert result.text == "hi"



def _install_fake_openai_with_fallback(monkeypatch, captured: dict):
    """Stub ``openai`` whose first call() rejects 'max_tokens' and
    'temperature', requiring both fallbacks OpenAICompatBackend now
    implements (verified live against openai/gpt-6-astra)."""
    calls: list = []

    class BadRequestError(Exception):
        pass

    def create(**kwargs):
        calls.append(dict(kwargs))
        if "max_tokens" in kwargs:
            raise BadRequestError(
                "Unsupported parameter: 'max_tokens' is not supported with "
                "this model. Use 'max_completion_tokens' instead.")
        if kwargs.get("temperature") is not None:
            raise BadRequestError(
                "Unsupported value: 'temperature' does not support 0 with "
                "this model. Only the default (1) value is supported.")
        captured.update(kwargs)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="hi"))],
            usage=types.SimpleNamespace(prompt_tokens=3, completion_tokens=2),
        )

    class OpenAI:
        def __init__(self, base_url=None, api_key=None):
            captured["base_url"] = base_url
            captured["api_key"] = api_key
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=create)
            )

    fake = types.ModuleType("openai")
    fake.OpenAI = OpenAI
    fake.BadRequestError = BadRequestError
    monkeypatch.setitem(sys.modules, "openai", fake)
    return calls


def test_openai_compat_falls_back_past_both_rejected_params(monkeypatch):
    """A model rejecting BOTH max_tokens and temperature=0 (verified live
    against openai/gpt-6-astra) must still succeed, retrying each
    rejected parameter exactly once, and must not silently retry forever."""
    captured: dict = {}
    calls = _install_fake_openai_with_fallback(monkeypatch, captured)
    monkeypatch.setenv("TOKENROUTER_BASE_URL", "https://api.tokenrouter.com/v1")
    monkeypatch.setenv("TOKENROUTER_API_KEY", "tr-key")

    result = OpenAICompatBackend("tokenrouter/openai/gpt-6-astra").generate_with_usage(
        "hello", temperature=0.0)

    assert result.text == "hi"
    assert len(calls) == 3  # max_tokens rejected, then temperature rejected, then success
    assert "max_completion_tokens" in captured
    assert "temperature" not in captured
