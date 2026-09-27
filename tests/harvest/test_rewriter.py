"""Tests for harvest LLM rewriter — converts conversational text to standalone insights."""

import httpx
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from mcp_memory_service.harvest.rewriter import HarvestRewriter, RewriteResult


@pytest.fixture
def rewriter():
    """Rewriter with mocked LLM client."""
    return HarvestRewriter()


class TestRewriterOutput:
    """Rewriter should produce standalone insights from conversational text."""

    @pytest.mark.asyncio
    async def test_rewrite_produces_standalone_insight(self, rewriter):
        """LLM rewriter converts conversational text to standalone insight."""
        input_text = (
            "Boa pergunta. O problema era que o handler com asyncio.to_thread "
            "bloqueava o event loop. A solução foi usar asyncio.create_task."
        )
        with patch.object(rewriter, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = ("bug: asyncio.to_thread inside an async handler still blocks the event loop. Fix: use asyncio.create_task for background work.", "groq", "openai/gpt-oss-120b")
            result = await rewriter.rewrite(input_text, suggested_type="bug")

        assert result is not None
        assert isinstance(result, RewriteResult)
        assert "asyncio" in result.content
        assert "Boa pergunta" not in result.content
        assert len(result.content) < 200

    @pytest.mark.asyncio
    async def test_rewrite_returns_none_for_skip(self, rewriter):
        """LLM rewriter returns None when content has no clear insight."""
        input_text = "sim, pode fazer. Vamos em frente com isso."
        with patch.object(rewriter, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = ("SKIP", "groq", "openai/gpt-oss-120b")
            result = await rewriter.rewrite(input_text, suggested_type="context")

        assert result is None

    @pytest.mark.asyncio
    async def test_rewrite_preserves_suggested_type(self, rewriter):
        """Rewriter uses the type from LLM response, falling back to suggested_type."""
        input_text = "decidimos usar RRF em vez de weighted average porque tem melhor recall."
        with patch.object(rewriter, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = ("decision: Use RRF instead of weighted average for hybrid search — better recall in practice.", "groq", "openai/gpt-oss-120b")
            result = await rewriter.rewrite(input_text, suggested_type="decision")

        assert result is not None
        assert result.memory_type == "decision"

    @pytest.mark.asyncio
    async def test_rewrite_handles_llm_failure_gracefully(self, rewriter):
        """If LLM fails, rewriter returns None (don't store garbage)."""
        input_text = "The root cause was a missing index."
        with patch.object(rewriter, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("Groq rate limit exceeded")
            result = await rewriter.rewrite(input_text, suggested_type="bug")

        assert result is None

    @pytest.mark.asyncio
    async def test_rewrite_strips_markdown_and_filler(self, rewriter):
        """Rewriter output should not contain markdown formatting or conversational filler."""
        input_text = "## Debug\n\n**Resultado:** O bug era no parser de JSON que não tratava arrays vazios."
        with patch.object(rewriter, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = ("bug: JSON parser crashes on empty arrays — needs null check before iteration.", "groq", "openai/gpt-oss-120b")
            result = await rewriter.rewrite(input_text, suggested_type="bug")

        assert result is not None
        assert "##" not in result.content
        assert "**" not in result.content


class TestRewriterPrompt:
    """Verify the prompt construction is correct."""

    @pytest.mark.asyncio
    async def test_prompt_includes_input_text(self, rewriter):
        """The LLM prompt should include the input text."""
        input_text = "We decided to use PostgreSQL over MySQL."
        with patch.object(rewriter, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = ("decision: Use PostgreSQL over MySQL.", "groq", "openai/gpt-oss-120b")
            await rewriter.rewrite(input_text, suggested_type="decision")

        call_args = mock_llm.call_args[0][0]  # First positional arg = prompt
        assert "PostgreSQL" in call_args

    @pytest.mark.asyncio
    async def test_prompt_includes_type_hint(self, rewriter):
        """The LLM prompt should include the suggested type as context."""
        input_text = "The fix was adding a retry with exponential backoff."
        with patch.object(rewriter, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = ("bug: Fix connection failures with retry + exponential backoff.", "groq", "openai/gpt-oss-120b")
            await rewriter.rewrite(input_text, suggested_type="bug")

        call_args = mock_llm.call_args[0][0]
        assert "bug" in call_args.lower()


class TestRewriterParsing:
    """Test parsing of LLM response format."""

    def test_parse_typed_response(self, rewriter):
        """Parse 'TYPE: content' format correctly."""
        result = rewriter._parse_response("decision: Use RRF for hybrid search.", "decision")
        assert result is not None
        assert result.memory_type == "decision"
        assert result.content == "Use RRF for hybrid search."

    def test_parse_skip_response(self, rewriter):
        """Parse SKIP response."""
        result = rewriter._parse_response("SKIP", "bug")
        assert result is None

    def test_parse_skip_case_insensitive(self, rewriter):
        """SKIP detection is case-insensitive."""
        result = rewriter._parse_response("skip", "bug")
        assert result is None

    def test_parse_untyped_response_uses_suggested(self, rewriter):
        """If LLM doesn't include type prefix, use suggested_type."""
        result = rewriter._parse_response("Always use WAL mode for SQLite.", "convention")
        assert result is not None
        assert result.memory_type == "convention"
        assert result.content == "Always use WAL mode for SQLite."

    def test_parse_empty_response_returns_none(self, rewriter):
        """Empty LLM response returns None."""
        result = rewriter._parse_response("", "bug")
        assert result is None


class TestRewriterConfiguration:
    """Issue #178: a configured provider chain must count as configured.

    `_call_llm` prefers `HARVEST_LLM_PROVIDERS` and only falls back to Groq,
    but the harvester decided whether a rewriter existed at all by looking at
    `GROQ_API_KEY`. Deployments pointing at an OpenAI-compatible endpoint got
    no rewriting and no error.
    """

    def test_provider_chain_counts_as_configured(self, monkeypatch):
        monkeypatch.delenv('GROQ_API_KEY', raising=False)
        monkeypatch.setenv('HARVEST_LLM_PROVIDERS', 'litellm')
        monkeypatch.setenv('HARVEST_LLM_LITELLM_BASE_URL', 'http://litellm.svc:4000/v1')
        monkeypatch.setenv('HARVEST_LLM_LITELLM_MODEL', 'ollama/qwen2.5:7b-instruct')

        assert HarvestRewriter().is_configured is True

    def test_groq_key_alone_still_counts_as_configured(self, monkeypatch):
        monkeypatch.delenv('HARVEST_LLM_PROVIDERS', raising=False)
        monkeypatch.setenv('GROQ_API_KEY', 'gsk-test')

        assert HarvestRewriter().is_configured is True

    def test_no_provider_and_no_key_is_unconfigured(self, monkeypatch):
        monkeypatch.delenv('HARVEST_LLM_PROVIDERS', raising=False)
        monkeypatch.delenv('GROQ_API_KEY', raising=False)

        assert HarvestRewriter().is_configured is False

    def test_harvester_keeps_rewriter_when_only_provider_chain_is_set(self, monkeypatch):
        """The gate itself, not just the property."""
        from mcp_memory_service.harvest.harvester import SessionHarvester

        monkeypatch.delenv('GROQ_API_KEY', raising=False)
        monkeypatch.setenv('HARVEST_LLM_PROVIDERS', 'litellm')
        monkeypatch.setenv('HARVEST_LLM_LITELLM_BASE_URL', 'http://litellm.svc:4000/v1')
        monkeypatch.setenv('HARVEST_LLM_LITELLM_MODEL', 'ollama/qwen2.5:7b-instruct')

        harvester = SessionHarvester.__new__(SessionHarvester)
        rewriter = harvester._get_rewriter()

        assert rewriter is not None
        assert rewriter.is_configured is True


class TestRewriterProviderChainDeepDive:
    """Same treatment as test_classifier.py's TestClassifierProviderChain (#194):

    the #178 fix had no test pinning the credential-less-Groq-entry edge case
    or confirming the provider chain is what actually gets called, not just
    that is_configured reports True.
    """

    def test_keyless_groq_entry_does_not_count_as_configured(self, monkeypatch):
        """The legacy path synthesizes a Groq provider even with no key; it must not qualify."""
        monkeypatch.delenv('HARVEST_LLM_PROVIDERS', raising=False)
        monkeypatch.delenv('HARVEST_LLM_PROVIDER', raising=False)
        monkeypatch.delenv('GROQ_API_KEY', raising=False)

        rewriter = HarvestRewriter()

        assert rewriter._providers == []
        assert rewriter.is_configured is False

    @pytest.mark.asyncio
    async def test_rewrite_goes_through_the_provider_endpoint(self, monkeypatch):
        """A configured chain rewrites without ever touching the legacy Groq path."""
        monkeypatch.delenv('GROQ_API_KEY', raising=False)
        monkeypatch.setenv('HARVEST_LLM_PROVIDERS', 'local')
        monkeypatch.setenv('HARVEST_LLM_LOCAL_BASE_URL', 'http://localhost:11434/v1')
        monkeypatch.setenv('HARVEST_LLM_LOCAL_MODEL', 'qwen2.5-coder')

        rewriter = HarvestRewriter()
        assert [p.name for p in rewriter._providers] == ['local']

        with patch.object(
            rewriter, '_call_openai_compatible', new_callable=AsyncMock
        ) as mock_call, patch.object(rewriter, '_call_groq', new_callable=AsyncMock) as mock_groq:
            mock_call.return_value = "bug: fixed by adding a null check."
            result = await rewriter.rewrite("the fix was a null check", suggested_type="bug")

        assert result is not None
        assert mock_call.call_count == 1
        assert mock_call.call_args[0][0] == 'http://localhost:11434/v1'
        mock_groq.assert_not_called()

    @pytest.mark.asyncio
    async def test_provider_failure_falls_through_to_the_next_one(self, monkeypatch):
        monkeypatch.delenv('GROQ_API_KEY', raising=False)
        monkeypatch.setenv('HARVEST_LLM_PROVIDERS', 'broken,local')
        monkeypatch.setenv('HARVEST_LLM_BROKEN_BASE_URL', 'http://127.0.0.1:9/v1')
        monkeypatch.setenv('HARVEST_LLM_BROKEN_MODEL', 'nope')
        monkeypatch.setenv('HARVEST_LLM_LOCAL_BASE_URL', 'http://localhost:11434/v1')
        monkeypatch.setenv('HARVEST_LLM_LOCAL_MODEL', 'qwen2.5-coder')

        rewriter = HarvestRewriter()
        assert [p.name for p in rewriter._providers] == ['broken', 'local']

        with patch.object(
            rewriter, '_call_openai_compatible', new_callable=AsyncMock,
            side_effect=[Exception("connection refused"), "decision: use the fallback provider."],
        ) as mock_call:
            result = await rewriter.rewrite("we went with the fallback provider", suggested_type="decision")

        assert result is not None
        assert mock_call.call_count == 2

    @pytest.mark.asyncio
    async def test_call_timeout_reaches_the_http_client(self, monkeypatch):
        """_call_llm's timeout argument must reach _call_openai_compatible's
        actual httpx client, not stop at a hardcoded value (#321)."""
        monkeypatch.delenv('GROQ_API_KEY', raising=False)
        monkeypatch.setenv('HARVEST_LLM_PROVIDERS', 'local')
        monkeypatch.setenv('HARVEST_LLM_LOCAL_BASE_URL', 'http://localhost:11434/v1')
        monkeypatch.setenv('HARVEST_LLM_LOCAL_MODEL', 'qwen2.5-coder')

        rewriter = HarvestRewriter()

        captured = {}

        class _FakeAsyncClient:
            def __init__(self, timeout):
                captured['timeout'] = timeout

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def post(self, *args, **kwargs):
                class _Resp:
                    status_code = 200

                    def raise_for_status(self):
                        pass

                    def json(self):
                        return {"choices": [{"message": {"content": "decision: ok"}}]}

                return _Resp()

        monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

        await rewriter.rewrite("some text", suggested_type="decision")
        assert captured['timeout'] == HarvestRewriter._CALL_TIMEOUT_SINGLE

        captured.clear()
        await rewriter.rewrite_batch([{"content": "some text", "memory_type": "decision"}])
        assert captured['timeout'] == HarvestRewriter._CALL_TIMEOUT_BATCH

    @pytest.mark.asyncio
    async def test_wrapper_timeout_scales_with_provider_fallback_chain(self, monkeypatch):
        """rewrite_sync/rewrite_batch_sync's ThreadPoolExecutor timeout must
        grow with the number of fallback providers _call_llm may try in
        sequence, or a working provider later in the chain never gets a
        turn before the wrapper kills the call (#321).

        Must run inside a running loop (hence @pytest.mark.asyncio) so
        rewrite_sync/rewrite_batch_sync take the ThreadPoolExecutor branch
        (asyncio.get_running_loop() succeeds) instead of falling through to
        the direct asyncio.run() branch used when there's no loop at all.
        """
        monkeypatch.delenv('GROQ_API_KEY', raising=False)
        monkeypatch.setenv('HARVEST_LLM_PROVIDERS', 'a,b,c')
        for name in ('A', 'B', 'C'):
            monkeypatch.setenv(f'HARVEST_LLM_{name}_BASE_URL', f'http://{name.lower()}/v1')
            monkeypatch.setenv(f'HARVEST_LLM_{name}_MODEL', 'm')

        rewriter = HarvestRewriter()
        assert len(rewriter._providers) == 3

        with patch('concurrent.futures.ThreadPoolExecutor') as mock_pool_cls:
            mock_pool = MagicMock()
            mock_pool_cls.return_value.__enter__.return_value = mock_pool
            mock_future = MagicMock()

            def _submit(fn, coro):
                coro.close()  # never actually run; avoid an unawaited-coroutine warning
                return mock_future

            mock_pool.submit.side_effect = _submit
            mock_future.result.return_value = None

            rewriter.rewrite_sync("text")
            single_timeout = mock_future.result.call_args.kwargs['timeout']
            assert single_timeout == 3 * HarvestRewriter._CALL_TIMEOUT_SINGLE + HarvestRewriter._WRAPPER_MARGIN

            rewriter.rewrite_batch_sync([{"content": "text", "memory_type": "decision"}])
            batch_timeout = mock_future.result.call_args.kwargs['timeout']
            assert batch_timeout == 3 * HarvestRewriter._CALL_TIMEOUT_BATCH + HarvestRewriter._WRAPPER_MARGIN
