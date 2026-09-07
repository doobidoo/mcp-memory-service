"""LLM-based rewriter that converts conversational text to standalone insights."""

import asyncio
import concurrent.futures
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from ..compat import _sanitize_log_value

logger = logging.getLogger(__name__)

VALID_TYPES = {"decision", "bug", "convention", "learning", "context"}


@dataclass
class LLMProvider:
    """Configuration for a single LLM provider."""
    name: str
    base_url: str
    model: str
    api_key: str = ""


def load_llm_providers() -> list:
    """Load provider chain from env vars.

    Shared by HarvestRewriter and HarvestClassifier so both use the same
    provider-resolution rules (#178) instead of the classifier hardcoding
    GROQ_API_KEY only.
    """
    providers_str = os.environ.get("HARVEST_LLM_PROVIDERS", "")
    if not providers_str:
        # Legacy: single provider from HARVEST_LLM_PROVIDER
        provider = os.environ.get("HARVEST_LLM_PROVIDER", "groq")
        if provider == "groq":
            return [LLMProvider(
                name="groq",
                base_url="https://api.groq.com/openai/v1",
                model=os.environ.get("HARVEST_LLM_MODEL", "llama-3.3-70b-versatile"),
                api_key=os.environ.get("GROQ_API_KEY", ""),
            )]
        return []

    providers = []
    for name in providers_str.split(","):
        name = name.strip()
        prefix = f"HARVEST_LLM_{name.upper()}_"
        base_url = os.environ.get(f"{prefix}BASE_URL", "")
        model = os.environ.get(f"{prefix}MODEL", "")
        api_key = os.environ.get(f"{prefix}API_KEY", "")
        if base_url and model:
            providers.append(LLMProvider(name=name, base_url=base_url, model=model, api_key=api_key))
    return providers


def is_usable_provider(provider: LLMProvider) -> bool:
    """A provider is usable when it has an endpoint we can actually call.

    ``load_llm_providers`` synthesizes a Groq entry even with no credentials
    (the legacy single-provider path), so presence in the chain is not
    enough — the Groq entry needs a key. Self-hosted OpenAI-compatible
    endpoints legitimately run without one.
    """
    if not (provider.base_url and provider.model):
        return False
    if provider.name == "groq":
        return bool(provider.api_key)
    return True


REWRITE_PROMPT = """Given this excerpt from a coding session conversation:
---
{text}
---

Suggested type: {suggested_type}

Extract the key insight as a standalone fact in 1-2 sentences.

STRICT RULES — respond SKIP if ANY of these apply:
- It is a TEMPORAL FACT (true today but not in 30 days: PR status, pending reviews, current bugs being fixed)
- It is META-DISCUSSION about memory systems, harvest pipelines, bootstrap profiles, or AI tooling internals
- It is GENERIC wisdom without a concrete tool/command/pattern ("documentation is important", "architecture was wrong")
- It describes a PROBLEM without stating the SOLUTION or AVOIDANCE rule
- It repeats something already well-known ("use git", "write tests")

ACCEPT only if:
- Contains a concrete action, tool, command, file path, or behavioral rule
- Is TIMELESS (valid in 30 days, not just today)
- Is SPECIFIC to a project, technology, or workflow (not generic advice)
- Would help a developer avoid a mistake or follow a convention

Additional rules:
- Must be self-contained (understandable without the conversation)
- Remove conversational filler, markdown formatting, emojis
- If no clear insight exists, respond with exactly: SKIP
- IMPORTANT: Respond in the SAME LANGUAGE as the input text{locale_instruction}

Format: TYPE: content
Valid types: decision, bug, convention, learning, context

Examples:
- bug: asyncio.to_thread inside an async handler still blocks the event loop. Fix: use asyncio.create_task.
- convention: Nunca usar strReplace em MEMORY.md — usar insert (append-only, concorrência multi-sessão).
- decision: Usar RRF em vez de weighted average para hybrid search — melhor recall na prática.
- learning: SICAR não registra arrendatário — é domínio do SNCR (fundiário) e CAF (agricultura familiar).
- SKIP"""

LOCALE_INSTRUCTIONS = {
    "pt_BR": "\n- Se o texto está em português, responda em português.",
    "de": "\n- Wenn der Text auf Deutsch ist, antworten Sie auf Deutsch.",
    "en": "",
}

BATCH_PROMPT = """You will receive {n} memory excerpts. For EACH one, extract the key insight as 1-2 sentences, or say SKIP.

These are session checkpoints and notes — they contain temporal context (dates, session info) wrapping TIMELESS insights. Your job is to extract the timeless rule/convention/decision from inside the temporal wrapper.

STRICT RULES — respond SKIP ONLY if:
- The ENTIRE text is purely temporal with NO reusable insight (just "did X, did Y" without lessons)
- It is META-DISCUSSION about memory/harvest/bootstrap systems themselves
- It is GENERIC wisdom without concrete tool/command/pattern
- There is NO actionable rule, convention, or decision buried in the text

EXTRACT if you find ANY of these inside the text:
- A concrete convention or rule ("never do X", "always use Y")
- A technical decision with reasoning ("chose X because Y")
- A bug pattern with solution ("X causes Y, fix: Z")
- A workflow or process rule ("before doing X, always check Y")

Respond in the SAME LANGUAGE as each input text.

Format — one line per memory, numbered:
1. TYPE: insight
2. SKIP
3. TYPE: insight
...

Valid types: decision, bug, convention, learning, context

{memories}"""


@dataclass
class RewriteResult:
    """Result of LLM rewrite."""
    content: str
    memory_type: str


class HarvestRewriter:
    """Rewrites harvest candidates into standalone insights using an LLM.

    Respects HARVEST_LOCALE for multilingual output — responds in the same
    language as the input text.
    """

    # Per-call HTTP timeout budgets. Batch gets more headroom because one
    # call generates output for every item in the batch, not just one.
    # The ThreadPoolExecutor wrapper timeouts below must exceed these
    # (times the number of fallback providers, since _call_llm tries each
    # one in sequence) or the wrapper kills the call before the client's
    # own timeout — and therefore before a working fallback provider ever
    # gets a turn.
    _CALL_TIMEOUT_SINGLE = 30.0
    _CALL_TIMEOUT_BATCH = 60.0
    _WRAPPER_MARGIN = 10.0

    def __init__(self):
        # Filtered through is_usable_provider so a credential-less legacy Groq
        # entry (synthesized by load_llm_providers when HARVEST_LLM_PROVIDERS
        # is unset) never reaches _call_llm's provider loop — mirrors
        # classifier.py's _ensure_initialized (#194).
        self._providers = [p for p in load_llm_providers() if is_usable_provider(p)]
        # Legacy single-provider compat
        self._provider = os.environ.get("HARVEST_LLM_PROVIDER", "groq")
        self._model = os.environ.get("HARVEST_LLM_MODEL", "llama-3.3-70b-versatile")
        self._api_key = os.environ.get("GROQ_API_KEY", "")
        self._locale = os.environ.get("HARVEST_LOCALE", "en")
        self._locale_instruction = self._build_locale_instruction()

    @property
    def is_configured(self) -> bool:
        """True when this rewriter has somewhere to send a prompt.

        Either a usable provider from ``HARVEST_LLM_PROVIDERS``, or the legacy
        ``GROQ_API_KEY``. Callers used to check the Groq key alone, which
        disabled rewriting for every OpenAI-compatible endpoint (issue #178).
        """
        return any(is_usable_provider(p) for p in self._providers) or bool(self._api_key)

    def _build_locale_instruction(self) -> str:
        """Build locale instruction from HARVEST_LOCALE env var."""
        locales = [l.strip() for l in self._locale.split(",")]
        instructions = []
        for loc in locales:
            if loc in LOCALE_INSTRUCTIONS and LOCALE_INSTRUCTIONS[loc]:
                instructions.append(LOCALE_INSTRUCTIONS[loc])
        return "".join(instructions)

    async def rewrite(self, text: str, suggested_type: str = "observation", already_extracted: list = None) -> Optional[RewriteResult]:
        """Rewrite conversational text as a standalone insight.

        Returns None if LLM says SKIP or if an error occurs.
        Responds in the same language as the input text.
        """
        context_block = ""
        if already_extracted:
            items = "\n".join(f"- {x}" for x in already_extracted[-10:])
            context_block = f"\n\nInsights already extracted in this session (DO NOT repeat these concepts):\n{items}\n"

        prompt = REWRITE_PROMPT.format(
            text=text[:1000],
            suggested_type=suggested_type,
            locale_instruction=self._locale_instruction,
        ) + context_block

        try:
            response = await self._call_llm(prompt, timeout=self._CALL_TIMEOUT_SINGLE)
        except Exception as e:
            logger.warning("LLM rewrite failed: %s", _sanitize_log_value(str(e)))
            return None

        return self._parse_response(response, suggested_type)

    def rewrite_sync(self, text: str, suggested_type: str = "observation", already_extracted: list = None) -> Optional[RewriteResult]:
        """Synchronous wrapper for rewrite (works inside running event loop)."""
        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.rewrite(text, suggested_type, already_extracted))
                # Must exceed the actual per-request HTTP timeout times the
                # number of fallback providers _call_llm may try in
                # sequence, or this wrapper kills the call before a working
                # provider further down the chain ever gets a turn.
                wrapper_timeout = self._CALL_TIMEOUT_SINGLE * max(1, len(self._providers)) + self._WRAPPER_MARGIN
                return future.result(timeout=wrapper_timeout)
        except RuntimeError:
            # No running loop — safe to use asyncio.run directly
            return asyncio.run(self.rewrite(text, suggested_type, already_extracted))
        except Exception as e:
            logger.warning("LLM rewrite_sync failed: %s", _sanitize_log_value(str(e)))
            return None

    async def rewrite_batch(self, items: list) -> list:
        """Rewrite multiple items in a single LLM call. Returns list of Optional[RewriteResult].

        Args:
            items: list of dicts with 'content' and 'memory_type' keys.
        """
        if not items:
            return []

        mem_block = ""
        for i, item in enumerate(items, 1):
            mem_block += f"\n--- Memory {i} (type: {item['memory_type']}) ---\n{item['content'][:800]}\n"

        prompt = BATCH_PROMPT.format(n=len(items), memories=mem_block)

        try:
            response = await self._call_llm(prompt, timeout=self._CALL_TIMEOUT_BATCH)
        except Exception as e:
            logger.warning("Batch rewrite failed: %s", _sanitize_log_value(str(e)))
            return [None] * len(items)

        return self._parse_batch_response(response, items)

    def rewrite_batch_sync(self, items: list) -> list:
        """Synchronous wrapper for rewrite_batch."""
        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.rewrite_batch(items))
                # See rewrite_sync: must exceed the per-call timeout times
                # the fallback-provider count, not a magic number.
                wrapper_timeout = self._CALL_TIMEOUT_BATCH * max(1, len(self._providers)) + self._WRAPPER_MARGIN
                return future.result(timeout=wrapper_timeout)
        except RuntimeError:
            return asyncio.run(self.rewrite_batch(items))
        except Exception as e:
            logger.warning("Batch rewrite_sync failed: %s", _sanitize_log_value(str(e)))
            return [None] * len(items)

    def _parse_batch_response(self, response: str, items: list) -> list:
        """Parse numbered batch response into list of Optional[RewriteResult]."""
        results = [None] * len(items)
        if not response:
            return results

        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Match "N. TYPE: content" or "N. SKIP"
            match = re.match(r'^(\d+)\.\s*(.+)$', line)
            if not match:
                continue
            idx = int(match.group(1)) - 1
            content = match.group(2).strip()
            if idx < 0 or idx >= len(items):
                continue
            if content.upper().startswith("SKIP"):
                continue
            # Parse TYPE: content
            type_match = re.match(r'^(\w+):\s*(.+)$', content, re.DOTALL)
            if type_match:
                parsed_type = type_match.group(1).lower()
                insight = type_match.group(2).strip()
                if parsed_type in VALID_TYPES:
                    results[idx] = RewriteResult(content=insight, memory_type=parsed_type)
                else:
                    results[idx] = RewriteResult(content=content, memory_type=items[idx]['memory_type'])
            else:
                results[idx] = RewriteResult(content=content, memory_type=items[idx]['memory_type'])

        return results

    def _parse_response(self, response: str, suggested_type: str) -> Optional[RewriteResult]:
        """Parse LLM response into RewriteResult or None."""
        if not response or not response.strip():
            return None

        response = response.strip()

        # Check for SKIP
        if response.upper().startswith("SKIP"):
            return None

        # Try to parse "TYPE: content" format
        match = re.match(r'^(\w+):\s*(.+)$', response, re.DOTALL)
        if match:
            parsed_type = match.group(1).lower()
            content = match.group(2).strip()
            if parsed_type in VALID_TYPES:
                return RewriteResult(content=content, memory_type=parsed_type)
            # Unknown type — use content with suggested_type
            return RewriteResult(content=response, memory_type=suggested_type)

        # No type prefix — use full response with suggested_type
        return RewriteResult(content=response, memory_type=suggested_type)

    async def _call_llm(self, prompt: str, timeout: float) -> str:
        """Call LLM with provider fallback chain.

        ``timeout`` bounds each provider attempt individually — callers'
        ThreadPoolExecutor wrapper timeouts (rewrite_sync/rewrite_batch_sync)
        must account for this being applied once per provider in the chain.
        """
        if self._providers:
            for provider in self._providers:
                try:
                    return await self._call_openai_compatible(
                        provider.base_url, provider.model, provider.api_key, prompt, timeout
                    )
                except Exception as e:
                    err_str = str(e).lower()
                    if "rate limit" in err_str or "429" in err_str:
                        logger.warning("%s rate limited, trying next", _sanitize_log_value(provider.name))
                        continue
                    logger.warning(
                        "%s failed: %s, trying next",
                        _sanitize_log_value(provider.name),
                        _sanitize_log_value(str(e)),
                    )
                    continue
            raise RuntimeError("All LLM providers exhausted")
        # Legacy single-provider
        if self._provider == "groq":
            return await self._call_groq(prompt, timeout)
        raise ValueError(f"Unknown LLM provider: {self._provider}")

    async def _call_openai_compatible(self, base_url: str, model: str, api_key: str, prompt: str, timeout: float) -> str:
        """Call any OpenAI-compatible API (Groq, DeepSeek, Ollama, etc.)."""
        # httpx costs ~36ms to import and nothing else on this path pulls it in.
        # classifier.py imports this module at top level and harvest/__init__.py
        # imports classifier eagerly, so importing httpx at module level would
        # add that 36ms to every `import mcp_memory_service.harvest`, including
        # runs that never call an LLM. asyncio and concurrent.futures are already
        # in sys.modules by then, which is why those two sit at the top instead.
        import httpx  # inline import
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 200,
                },
            )
            if resp.status_code == 429:
                raise Exception(f"Rate limit exceeded: {resp.text[:100]}")
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"] or ""

    async def _call_groq(self, prompt: str, timeout: float) -> str:
        """Call Groq API."""
        try:
            from groq import AsyncGroq
        except ImportError:
            raise RuntimeError("groq package not installed. Run: pip install groq")

        client = AsyncGroq(api_key=self._api_key, timeout=timeout)
        response = await client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        return response.choices[0].message.content or ""
