"""Tests for Headroom model wrapping helpers."""

from __future__ import annotations

import asyncio
import logging
import sys
from types import ModuleType

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from deepclaw import headroom as headroom_mod
from deepclaw.config import HeadroomConfig
from deepclaw.headroom import get_headroom_runtime_stats, wrap_model_with_headroom


class TestWrapModelWithHeadroom:
    def setup_method(self):
        headroom_mod._set_active_headroom_model(None)

    def test_returns_original_model_when_disabled(self):
        model = object()

        wrapped = wrap_model_with_headroom(model, HeadroomConfig(enabled=False))

        assert wrapped is model

    def test_wraps_resolved_model_when_enabled(self, monkeypatch):
        original_model = "openai:gpt-5"
        resolved_model = object()

        deepagents_models = ModuleType("deepagents._models")
        deepagents_models.resolve_model = lambda model: resolved_model

        headroom_integrations = ModuleType("headroom.integrations")

        class FakeHeadroomChatModel:
            def __init__(self, model):
                self.wrapped_model = model

        headroom_integrations.HeadroomChatModel = FakeHeadroomChatModel

        monkeypatch.setitem(sys.modules, "deepagents._models", deepagents_models)
        monkeypatch.setitem(sys.modules, "headroom.integrations", headroom_integrations)

        wrapped = wrap_model_with_headroom(original_model, HeadroomConfig(enabled=True))

        assert wrapped.wrapped_model is resolved_model

    def test_logs_savings_for_each_optimization(self, monkeypatch, caplog):
        resolved_model = object()

        deepagents_models = ModuleType("deepagents._models")
        deepagents_models.resolve_model = lambda model: resolved_model

        headroom_integrations = ModuleType("headroom.integrations")

        class FakeHeadroomChatModel:
            def __init__(self, model):
                self.wrapped_model = model

            def _optimize_messages(self, messages):
                metrics = type(
                    "Metrics",
                    (),
                    {
                        "tokens_before": 100,
                        "tokens_after": 40,
                        "tokens_saved": 60,
                        "savings_percent": 60.0,
                        "transforms_applied": ["router:smart_crusher"],
                    },
                )()
                return messages, metrics

        headroom_integrations.HeadroomChatModel = FakeHeadroomChatModel

        monkeypatch.setitem(sys.modules, "deepagents._models", deepagents_models)
        monkeypatch.setitem(sys.modules, "headroom.integrations", headroom_integrations)

        wrapped = wrap_model_with_headroom("openai:gpt-5", HeadroomConfig(enabled=True))

        with caplog.at_level(logging.INFO):
            optimized_messages, metrics = wrapped._optimize_messages(["hello"])

        assert optimized_messages == ["hello"]
        assert metrics.tokens_saved == 60
        assert "Headroom savings: 100 -> 40 tokens (60 saved, 60.0%)" in caplog.text

    def test_generate_uses_public_invoke_for_runnable_binding(self, monkeypatch):
        tool_message = AIMessage(
            content="",
            tool_calls=[
                {"name": "weather", "args": {"city": "SF"}, "id": "call_1", "type": "tool_call"}
            ],
        )

        class FakeRunnableBinding:
            def invoke(self, messages, **kwargs):
                return tool_message

            def _generate(self, messages, **kwargs):
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content="fake narration"))]
                )

        resolved_model = FakeRunnableBinding()

        deepagents_models = ModuleType("deepagents._models")
        deepagents_models.resolve_model = lambda model: resolved_model

        headroom_integrations = ModuleType("headroom.integrations")

        class FakeHeadroomChatModel:
            def __init__(self, model):
                self.wrapped_model = model

            def _optimize_messages(self, messages):
                metrics = type(
                    "Metrics",
                    (),
                    {
                        "tokens_before": 10,
                        "tokens_after": 10,
                        "tokens_saved": 0,
                        "savings_percent": 0.0,
                        "transforms_applied": [],
                    },
                )()
                return messages, metrics

        headroom_integrations.HeadroomChatModel = FakeHeadroomChatModel

        monkeypatch.setitem(sys.modules, "deepagents._models", deepagents_models)
        monkeypatch.setitem(sys.modules, "headroom.integrations", headroom_integrations)
        monkeypatch.setattr("deepclaw.headroom.RunnableBinding", FakeRunnableBinding)

        wrapped = wrap_model_with_headroom("openai:gpt-5", HeadroomConfig(enabled=True))
        result = wrapped._generate(["hello"])

        assert result.generations[0].message.tool_calls == tool_message.tool_calls
        assert result.generations[0].message.content == ""

    def test_stream_wraps_public_stream_chunks_for_runnable_binding(self, monkeypatch):
        class FakeRunnableBinding:
            def stream(self, messages, **kwargs):
                yield AIMessageChunk(content="public-stream")

            def _stream(self, messages, **kwargs):
                yield ChatGenerationChunk(message=AIMessageChunk(content="private-stream"))

        resolved_model = FakeRunnableBinding()

        deepagents_models = ModuleType("deepagents._models")
        deepagents_models.resolve_model = lambda model: resolved_model

        headroom_integrations = ModuleType("headroom.integrations")

        class FakeHeadroomChatModel:
            def __init__(self, model):
                self.wrapped_model = model

            def _optimize_messages(self, messages):
                metrics = type(
                    "Metrics",
                    (),
                    {
                        "tokens_before": 10,
                        "tokens_after": 10,
                        "tokens_saved": 0,
                        "savings_percent": 0.0,
                        "transforms_applied": [],
                    },
                )()
                return messages, metrics

        headroom_integrations.HeadroomChatModel = FakeHeadroomChatModel

        monkeypatch.setitem(sys.modules, "deepagents._models", deepagents_models)
        monkeypatch.setitem(sys.modules, "headroom.integrations", headroom_integrations)
        monkeypatch.setattr("deepclaw.headroom.RunnableBinding", FakeRunnableBinding)

        wrapped = wrap_model_with_headroom("openai:gpt-5", HeadroomConfig(enabled=True))

        chunks = list(wrapped._stream(["hello"]))

        assert len(chunks) == 1
        assert isinstance(chunks[0], ChatGenerationChunk)
        assert chunks[0].message.content == "public-stream"

    def test_astream_wraps_public_astream_chunks_for_runnable_binding(self, monkeypatch):
        class FakeRunnableBinding:
            async def astream(self, messages, **kwargs):
                yield AIMessageChunk(content="public-stream")

            async def _astream(self, messages, **kwargs):
                yield ChatGenerationChunk(message=AIMessageChunk(content="private-stream"))

        resolved_model = FakeRunnableBinding()

        deepagents_models = ModuleType("deepagents._models")
        deepagents_models.resolve_model = lambda model: resolved_model

        headroom_integrations = ModuleType("headroom.integrations")

        class FakeHeadroomChatModel:
            def __init__(self, model):
                self.wrapped_model = model

            def _optimize_messages(self, messages):
                metrics = type(
                    "Metrics",
                    (),
                    {
                        "tokens_before": 10,
                        "tokens_after": 10,
                        "tokens_saved": 0,
                        "savings_percent": 0.0,
                        "transforms_applied": [],
                    },
                )()
                return messages, metrics

        headroom_integrations.HeadroomChatModel = FakeHeadroomChatModel

        monkeypatch.setitem(sys.modules, "deepagents._models", deepagents_models)
        monkeypatch.setitem(sys.modules, "headroom.integrations", headroom_integrations)
        monkeypatch.setattr("deepclaw.headroom.RunnableBinding", FakeRunnableBinding)

        wrapped = wrap_model_with_headroom("openai:gpt-5", HeadroomConfig(enabled=True))

        async def _collect():
            return [chunk async for chunk in wrapped._astream(["hello"])]

        chunks = asyncio.run(_collect())
        assert len(chunks) == 1
        assert isinstance(chunks[0], ChatGenerationChunk)
        assert chunks[0].message.content == "public-stream"

    def test_bind_tools_preserves_deepclaw_wrapper_behavior(self, monkeypatch):
        tool_message = AIMessage(
            content="",
            tool_calls=[
                {"name": "weather", "args": {"city": "SF"}, "id": "call_1", "type": "tool_call"}
            ],
        )

        class FakeRunnableBinding:
            def invoke(self, messages, **kwargs):
                return tool_message

            def _generate(self, messages, **kwargs):
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content="fake narration"))]
                )

        class FakeBaseModel:
            def bind_tools(self, tools, **kwargs):
                return FakeRunnableBinding()

        resolved_model = FakeBaseModel()

        deepagents_models = ModuleType("deepagents._models")
        deepagents_models.resolve_model = lambda model: resolved_model

        headroom_integrations = ModuleType("headroom.integrations")

        class FakeHeadroomChatModel:
            def __init__(self, model):
                self.wrapped_model = model
                self.headroom_config = {"dummy": True}
                self.mode = "optimize"
                self.auto_detect_provider = True

            def _optimize_messages(self, messages):
                metrics = type(
                    "Metrics",
                    (),
                    {
                        "tokens_before": 10,
                        "tokens_after": 10,
                        "tokens_saved": 0,
                        "savings_percent": 0.0,
                        "transforms_applied": [],
                    },
                )()
                return messages, metrics

            def bind_tools(self, tools, **kwargs):
                return FakeHeadroomChatModel(self.wrapped_model.bind_tools(tools, **kwargs))

        headroom_integrations.HeadroomChatModel = FakeHeadroomChatModel

        monkeypatch.setitem(sys.modules, "deepagents._models", deepagents_models)
        monkeypatch.setitem(sys.modules, "headroom.integrations", headroom_integrations)
        monkeypatch.setattr("deepclaw.headroom.RunnableBinding", FakeRunnableBinding)

        wrapped = wrap_model_with_headroom("openai:gpt-5", HeadroomConfig(enabled=True))
        rebound = wrapped.bind_tools([object()])
        result = rebound._generate(["hello"])

        assert result.generations[0].message.tool_calls == tool_message.tool_calls

    def test_runtime_stats_return_inactive_defaults_when_no_model(self):
        stats = get_headroom_runtime_stats()

        assert stats["active"] is False
        assert stats["summary"]["total_requests"] == 0
        assert stats["recent"] == []

    def test_runtime_stats_include_summary_and_recent_metrics(self, monkeypatch):
        deepagents_models = ModuleType("deepagents._models")
        deepagents_models.resolve_model = lambda model: object()

        headroom_integrations = ModuleType("headroom.integrations")

        class FakeHeadroomChatModel:
            def __init__(self, model):
                self.wrapped_model = model
                self._metrics_history = [
                    type(
                        "Metrics",
                        (),
                        {
                            "tokens_before": 100,
                            "tokens_after": 40,
                            "tokens_saved": 60,
                            "savings_percent": 60.0,
                            "transforms_applied": ["router:smart_crusher"],
                        },
                    )(),
                    type(
                        "Metrics",
                        (),
                        {
                            "tokens_before": 80,
                            "tokens_after": 50,
                            "tokens_saved": 30,
                            "savings_percent": 37.5,
                            "transforms_applied": ["router:smart_crusher"],
                        },
                    )(),
                ]

            def get_savings_summary(self):
                return {
                    "total_requests": 2,
                    "total_tokens_saved": 90,
                    "average_savings_percent": 48.75,
                    "total_tokens_before": 180,
                    "total_tokens_after": 90,
                }

        headroom_integrations.HeadroomChatModel = FakeHeadroomChatModel

        monkeypatch.setitem(sys.modules, "deepagents._models", deepagents_models)
        monkeypatch.setitem(sys.modules, "headroom.integrations", headroom_integrations)

        wrap_model_with_headroom("openai:gpt-5", HeadroomConfig(enabled=True))
        stats = get_headroom_runtime_stats(recent_limit=1)

        assert stats["active"] is True
        assert stats["summary"]["total_tokens_saved"] == 90
        assert stats["wrapped_model_class"] == "object"
        assert stats["recent"] == [
            {
                "tokens_before": 80,
                "tokens_after": 50,
                "tokens_saved": 30,
                "savings_percent": 37.5,
                "transforms_applied": ["router:smart_crusher"],
            }
        ]
