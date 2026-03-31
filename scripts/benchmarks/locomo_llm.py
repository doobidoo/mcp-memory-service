"""Pluggable LLM adapter module for LoCoMo QA benchmark evaluation."""

from __future__ import annotations

import asyncio
from typing import List, Protocol, runtime_checkable

QA_PROMPT_TEMPLATE = """Answer the following question based ONLY on the provided context.
Give a brief, concise answer. If the answer is not in the context, say "not mentioned".

Context:
{context}

Question: {question}

Answer:"""


def build_qa_prompt(question: str, context: List[str]) -> str:
    """Build QA prompt. Format context as bullet list."""
    bullet_list = "\n".join(f"- {item}" for item in context)
    return QA_PROMPT_TEMPLATE.format(context=bullet_list, question=question)


@runtime_checkable
class LLMAdapter(Protocol):
    async def generate_answer(self, question: str, context: List[str]) -> str:
        ...


class MockAdapter:
    """Returns first context line, or 'not mentioned' if empty."""

    async def generate_answer(self, question: str, context: List[str]) -> str:
        if not context:
            return "not mentioned"
        return context[0]


class ClaudeAdapter:
    """Uses anthropic SDK. Import anthropic lazily in __init__."""

    def __init__(self, model: str = "claude-sonnet-4-20250514") -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "anthropic package is required for ClaudeAdapter. "
                "Install it with: pip install anthropic"
            ) from e
        self.model = model
        self._anthropic = anthropic

    async def generate_answer(self, question: str, context: List[str]) -> str:
        if not hasattr(self, "_client"):
            self._client = self._anthropic.AsyncAnthropic()
        prompt = build_qa_prompt(question, context)
        message = await self._client.messages.create(
            model=self.model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()


class OllamaAdapter:
    """Uses aiohttp to call Ollama HTTP API. Import aiohttp lazily."""

    def __init__(
        self, model: str = "llama3", base_url: str = "http://localhost:11434"
    ) -> None:
        try:
            import aiohttp  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "aiohttp package is required for OllamaAdapter. "
                "Install it with: pip install aiohttp"
            ) from e
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._aiohttp = aiohttp

    async def generate_answer(self, question: str, context: List[str]) -> str:
        prompt = build_qa_prompt(question, context)
        url = f"{self.base_url}/api/generate"
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        if not hasattr(self, "_session") or self._session.closed:
            self._session = self._aiohttp.ClientSession()
        async with self._session.post(url, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data.get("response", "").strip()


def create_adapter(llm_name: str, model: str = "") -> LLMAdapter:
    """Factory. Supports 'claude', 'ollama', 'mock'. Raises ValueError for unknown."""
    name = llm_name.lower().strip()
    if name == "mock":
        return MockAdapter()
    if name == "claude":
        if model:
            return ClaudeAdapter(model=model)
        return ClaudeAdapter()
    if name == "ollama":
        if model:
            return OllamaAdapter(model=model)
        return OllamaAdapter()
    raise ValueError(
        f"Unknown LLM adapter: {llm_name!r}. Supported: 'claude', 'ollama', 'mock'."
    )
