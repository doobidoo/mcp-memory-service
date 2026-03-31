"""Tests for the LoCoMo LLM adapter module."""

import asyncio

import pytest

from locomo_llm import (
    MockAdapter,
    build_qa_prompt,
    create_adapter,
)


class TestBuildQAPrompt:
    def test_includes_question(self):
        prompt = build_qa_prompt("What is the capital?", ["Paris is the capital."])
        assert "What is the capital?" in prompt

    def test_includes_context(self):
        prompt = build_qa_prompt("Who wrote it?", ["Alice wrote the book.", "It was published in 2020."])
        assert "Alice wrote the book." in prompt
        assert "It was published in 2020." in prompt

    def test_instructs_concise_answer(self):
        prompt = build_qa_prompt("Any question?", ["Some context."])
        lower = prompt.lower()
        assert any(word in lower for word in ("concise", "short", "brief"))


class TestMockAdapter:
    def test_mock_returns_first_context(self):
        adapter = MockAdapter()
        result = asyncio.run(adapter.generate_answer("Q?", ["first line", "second line"]))
        assert result == "first line"

    def test_mock_returns_not_mentioned_when_empty(self):
        adapter = MockAdapter()
        result = asyncio.run(adapter.generate_answer("Q?", []))
        assert result == "not mentioned"


class TestCreateAdapter:
    def test_create_mock(self):
        adapter = create_adapter("mock")
        assert isinstance(adapter, MockAdapter)

    def test_create_mock_case_insensitive(self):
        adapter = create_adapter("Mock")
        assert isinstance(adapter, MockAdapter)

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM adapter"):
            create_adapter("gpt4")
