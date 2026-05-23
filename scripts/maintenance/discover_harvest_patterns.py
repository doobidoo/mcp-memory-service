#!/usr/bin/env python3
"""
Pattern discovery for harvest locale plugins.

A one-shot CLI tool that analyzes low-yield harvest sessions — where regex
extraction found <3 matches from >50 messages — and proposes new regex
patterns using an optional LLM.

Usage:
    python scripts/maintenance/discover_harvest_patterns.py session.jsonl
    python scripts/maintenance/discover_harvest_patterns.py session.jsonl --locale de
    python scripts/maintenance/discover_harvest_patterns.py session.jsonl --output /tmp/patterns.yaml

Environment:
    GROQ_API_KEY           For Groq LLM (default, fastest)
    OPENAI_API_KEY         For OpenAI-compatible API (fallback)

Output:
    YAML file at patterns/auto_generated/{locale}.yaml by default, with
    proposed patterns per memory type and usage counters.
"""

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mcp_memory_service.harvest.parser import TranscriptParser, ParsedMessage
from src.mcp_memory_service.harvest.extractor import PatternExtractor
from src.mcp_memory_service.harvest.patterns import PATTERNS_DIR
from src.mcp_memory_service.harvest.models import HARVEST_TYPES
logger = logging.getLogger(__name__)
LOW_YIELD_MAX_MATCHES = 3
LOW_YIELD_MIN_MESSAGES = 50


DISCOVERY_PROMPT_TEMPLATE = """\
You are analyzing a {locale} coding session transcript for a harvest pattern extractor.
The extractor uses regex patterns to identify useful memories from chat messages:
- decision: architectural choices, technology selections, trade-offs
- bug: root causes, fixes, regressions, errors with causes
- convention: coding rules, standards, patterns, always/never practices
- learning: discoveries, insights, lessons, TIL moments
- context: current state, progress, next steps, blockers

These messages were NOT captured by the existing regex patterns.
Identify which contain genuine decisions, bugs, conventions, learnings, or context.

For each message you identify, propose a regex pattern that would match it and
similar phrases. The pattern MUST be a proper regex, NOT a verbatim message copy.

Guidelines for GOOD regex patterns:
1. Use non-capturing groups (?:...) for alternatives (e.g., \\b(?:decided|chose|opted)\\b)
2. Use word boundaries \\b for whole-word matching
3. Include alternatives for common phrasings of the same concept
4. Be specific enough to avoid false positives, broad enough to generalize
5. For non-English locales, include both native and English alternatives

GOOD example pattern: \\b(?:ich denke|ich glaube|meiner Meinung nach)\\b.*\\bsollten\\b
BAD example pattern (DO NOT DO THIS): ich denke wir sollten das nochmal testen

Respond ONLY with a JSON object:
{{
  "patterns": {{
    "decision": [
      {{"pattern": "...", "confidence": 0.7, "example": "matching text", "source_message": "message that triggered it"}}
    ],
    "bug": [],
    "convention": [],
    "learning": [],
    "context": []
  }}
}}

Unmatched messages (limit 50):
{unmatched_text}
"""


@dataclass
class ProposedPattern:
    pattern: str
    confidence: float = 0.7
    example: str = ""
    source_message: str = ""

    def to_yaml_entry(self, indent: int = 4) -> str:
        pad = " " * indent
        escaped = self.pattern.replace("'", "''")
        return (
            f"{pad}- pattern: '{escaped}'\n"
            f"{pad}  confidence: {self.confidence}\n"
        )


@dataclass
class DiscoveryResult:
    session_file: str
    locale: str
    total_messages: int
    candidates_found: int
    is_low_yield: bool
    patterns: Dict[str, List[ProposedPattern]] = field(default_factory=dict)
    llm_used: Optional[str] = None

    def to_yaml(self) -> str:
        lines = [
            f"# Auto-generated patterns from {self.session_file}",
            f"# Locale: {self.locale}",
            f"# Total messages: {self.total_messages}",
            f"# Candidates found: {self.candidates_found}",
            f"# LLM: {self.llm_used or 'none'}",
            f"language: {self.locale}",
            "version: 1",
            "",
            "patterns:",
        ]
        for mem_type in HARVEST_TYPES:
            pats = self.patterns.get(mem_type, [])
            if not pats:
                continue
            lines.append(f"  {mem_type}:")
            for p in pats:
                lines.append(p.to_yaml_entry(indent=4))
            lines.append("")
        return "\n".join(lines)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover harvest patterns from low-yield sessions"
    )
    parser.add_argument(
        "session_file",
        type=str,
        help="Path to JSONL session transcript file",
    )
    parser.add_argument(
        "--locale",
        type=str,
        default=None,
        help="Locale code (e.g., en, de, pt_BR). Auto-detected from patterns if omitted.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output YAML path. Default: patterns/auto_generated/{locale}.yaml",
    )
    parser.add_argument(
        "--llm",
        type=str,
        choices=["groq", "openai", "auto"],
        default="auto",
        help="LLM provider to use (default: auto-detect from env vars)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print proposed patterns to stdout without writing",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s [%(name)s] %(message)s",
        stream=sys.stderr,
    )


def detect_locale(messages: List[ParsedMessage]) -> str:
    """Heuristic locale detection from message content."""
    de_signals = [
        r'\b(?:und|die|der|das|ist|nicht|wir|ich|ein|eine)\b',
        r'\b(?:werden|wurde|haben|hat|wird|kann|muss|soll)\b',
    ]
    pt_signals = [
        r'\b(?:não|uma|para|com|dos|das|mais|como|vai|foi|ser|era)\b',
        r'\b(?:isso|aquele|você|ele|ela|nosso|dela|dele)\b',
    ]

    sample = " ".join(m.text.lower() for m in messages[:100])
    de_count = sum(len(re.findall(sig, sample)) for sig in de_signals)
    pt_count = sum(len(re.findall(sig, sample)) for sig in pt_signals)

    if de_count > pt_count and de_count > 10:
        return "de"
    if pt_count > de_count and pt_count > 10:
        return "pt_BR"
    return "en"


def create_llm_client(llm_choice: str):
    """Create an LLM client based on available env vars."""
    groq_key = os.environ.get("GROQ_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if llm_choice == "auto":
        if groq_key:
            llm_choice = "groq"
        elif openai_key:
            llm_choice = "openai"
        else:
            return None

    if llm_choice == "groq" and groq_key:
        try:
            from groq import Groq
            return _GroqClient(Groq(api_key=groq_key))
        except ImportError:
            logger.warning("groq package not installed, try: pip install groq")

    if llm_choice == "openai" and openai_key:
        try:
            from openai import OpenAI
            return _OpenAIClient(OpenAI(api_key=openai_key))
        except ImportError:
            logger.warning("openai package not installed, try: pip install openai")

    return None


class _GroqClient:
    def __init__(self, client):
        self._client = client

    def generate(self, prompt: str, system: str = "") -> Optional[str]:
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            model_override = os.environ.get("GROQ_MODEL")
            models = [model_override] if model_override else ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
            for model in models:
                try:
                    resp = self._client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=2000,
                        temperature=0.1,
                        response_format={"type": "json_object"},
                    )
                    return resp.choices[0].message.content
                except Exception as e:
                    logger.debug(f"Groq model {model} failed: {e}")
                    continue
            return None
        except Exception as e:
            logger.error(f"Groq call failed: {e}")
            return None


class _OpenAIClient:
    def __init__(self, client):
        self._client = client

    def generate(self, prompt: str, system: str = "") -> Optional[str]:
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            resp = self._client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                messages=messages,
                max_tokens=2000,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI call failed: {e}")
            return None


def parse_llm_response(text: str) -> Dict[str, List[ProposedPattern]]:
    """Parse LLM JSON response into structured patterns."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace == -1 or last_brace <= first_brace:
        logger.warning("No JSON found in LLM response")
        return {t: [] for t in HARVEST_TYPES}
    try:
        data = json.loads(text[first_brace:last_brace + 1])
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM JSON: {e}")
        return {t: [] for t in HARVEST_TYPES}

    RE_HAS_REGEX = re.compile(r'[\[(){}|?*+^$.\\]|\\[bBAzZ]|\\[wWsSdD]')
    raw_patterns = data.get("patterns", {})
    result: Dict[str, List[ProposedPattern]] = {}
    for mem_type in HARVEST_TYPES:
        entries = raw_patterns.get(mem_type, [])
        result[mem_type] = []
        for entry in entries:
            if not isinstance(entry, dict) or "pattern" not in entry:
                continue
            pattern = entry["pattern"]
            if not RE_HAS_REGEX.search(pattern):
                logger.warning(f"Rejected plain-text pattern (not a regex): {pattern[:60]}")
                continue
            try:
                re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                logger.warning(f"Invalid regex from LLM: {pattern}: {e}")
                continue
            result[mem_type].append(ProposedPattern(
                pattern=entry["pattern"],
                confidence=float(entry.get("confidence", 0.7)),
                example=str(entry.get("example", "")),
                source_message=str(entry.get("source_message", "")),
            ))
    return result


def run_discovery(
    session_path: Path,
    locale: str,
    messages: List[ParsedMessage],
    candidates: List,
    unmatched_texts: List[str],
    llm_client,
) -> DiscoveryResult:
    """Run the discovery pipeline: low-yield check → LLM → structured output."""
    is_low_yield = (
        len(candidates) < LOW_YIELD_MAX_MATCHES
        and len(messages) >= LOW_YIELD_MIN_MESSAGES
    )

    result = DiscoveryResult(
        session_file=session_path.name,
        locale=locale,
        total_messages=len(messages),
        candidates_found=len(candidates),
        is_low_yield=is_low_yield,
    )

    if not is_low_yield:
        logger.info(
            f"Session not low-yield: {len(candidates)} candidates "
            f"from {len(messages)} messages (threshold: <{LOW_YIELD_MAX_MATCHES} "
            f"from >={LOW_YIELD_MIN_MESSAGES})"
        )
        return result

    if not llm_client:
        logger.warning(
            "Low-yield session detected but no LLM configured. "
            "Set GROQ_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY."
        )
        return result

    prompt = DISCOVERY_PROMPT_TEMPLATE.format(
        locale=locale,
        unmatched_text="\n---\n".join(unmatched_texts[:50]),
    )

    logger.info(f"Sending {min(len(unmatched_texts), 50)} unmatched messages to LLM...")
    response = llm_client.generate(
        prompt=prompt,
        system=(
            "You are a regex pattern discovery assistant for a coding session "
            "harvest system. You analyze chat messages and propose regex patterns "
            "that capture decisions, bugs, conventions, learnings, and context. "
            "Be precise: prefer specific patterns over overly broad ones."
        ),
    )

    if not response:
        logger.warning("LLM returned no response")
        return result

    result.patterns = parse_llm_response(response)
    result.llm_used = type(llm_client).__name__.lstrip("_").replace("Client", "").lower()

    total = sum(len(v) for v in result.patterns.values())
    logger.info(f"LLM proposed {total} patterns across {sum(1 for v in result.patterns.values() if v)} memory types")

    return result


def write_output(result: DiscoveryResult, output_path: Path, dry_run: bool) -> None:
    """Write discovered patterns to YAML."""
    if dry_run:
        print(result.to_yaml())
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.to_yaml(), encoding="utf-8")
    logger.info(f"Patterns written to {output_path}")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    session_path = Path(args.session_file)
    if not session_path.exists():
        logger.error(f"Session file not found: {session_path}")
        return 1

    logger.info(f"Parsing session: {session_path}")
    transcript_parser = TranscriptParser()
    messages = transcript_parser.parse_file(session_path)

    if not messages:
        logger.error("No messages parsed from session file")
        return 1

    logger.info(f"Parsed {len(messages)} messages")

    locale = args.locale or detect_locale(messages)
    logger.info(f"Using locale: {locale}")

    logger.info("Running pattern extraction...")
    extractor = PatternExtractor(locale=locale)
    candidates = []
    unmatched_texts = []

    for msg in messages:
        extracted = extractor.extract(msg)
        if extracted:
            candidates.extend(extracted)
        else:
            unmatched_texts.append(msg.text)

    by_type = {}
    for c in candidates:
        by_type[c.memory_type] = by_type.get(c.memory_type, 0) + 1

    logger.info(
        f"Pattern extractor: {len(candidates)} candidates from {len(messages)} messages "
        f"({', '.join(f'{k}={v}' for k, v in sorted(by_type.items()))})"
    )

    llm_client = create_llm_client(args.llm)

    result = run_discovery(
        session_path=session_path,
        locale=locale,
        messages=messages,
        candidates=candidates,
        unmatched_texts=unmatched_texts,
        llm_client=llm_client,
    )

    total_proposed = sum(len(v) for v in result.patterns.values())
    if total_proposed == 0:
        if result.is_low_yield:
            logger.info("No patterns discovered — LLM may not be available or found no candidates")
        else:
            logger.info("Session yield is sufficient; no discovery needed")
        return 0

    output_path = args.output
    if not output_path:
        output_path = str(PATTERNS_DIR / "auto_generated" / f"{locale}.yaml")

    write_output(result, Path(output_path), args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
