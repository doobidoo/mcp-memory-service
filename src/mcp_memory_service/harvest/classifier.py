"""LLM-based classification layer for harvest candidates.

Phase 2 enhancement: validates regex-extracted candidates using LLM,
improving precision from ~47% to ≥80% by filtering false positives,
refining content, and deduplicating similar candidates.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional

from .models import HarvestCandidate

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a memory classifier for an AI coding assistant. "
    "You evaluate candidate memories extracted from coding session transcripts. "
    "Be strict — only approve candidates that are genuinely useful as standalone memories."
)

CLASSIFY_PROMPT_TEMPLATE = """Evaluate this candidate memory extracted from a coding session transcript.

CANDIDATE:
- Type: {memory_type}
- Content: {content}
- Regex confidence: {confidence}

CONTEXT (surrounding messages):
{context}

Answer these questions (respond ONLY with the JSON object, no other text):
{{
  "keep": true/false,
  "reason": "one-line explanation",
  "refined_content": "improved standalone version of the content (or null if keep=false)",
  "memory_type": "{memory_type}" or corrected type,
  "confidence": 0.0-1.0
}}

REJECTION CRITERIA (keep=false if ANY apply):
- Fragment: doesn't make sense without surrounding conversation
- Conversation noise: describes what's happening rather than capturing an insight
- Tool/system output: skill definitions, system reminders, CI output
- Too generic: "we should test this" without specifics
- Duplicate of another candidate (check context)

APPROVAL CRITERIA (keep=true if ALL apply):
- Self-contained: makes sense as a standalone memory
- Actionable or informative: captures a decision, bug root cause, convention, or learning
- Specific: contains concrete details (names, versions, reasons)
"""

DEDUP_PROMPT_TEMPLATE = """Given these candidate memories, identify duplicates or near-duplicates.
Return a JSON array of indices to KEEP (0-based). Remove duplicates, keeping the highest quality version.

CANDIDATES:
{candidates_text}

Respond ONLY with a JSON array of indices to keep, e.g. [0, 2, 4]
"""


@dataclass
class ClassificationResult:
    """Result of LLM classification for a single candidate."""
    keep: bool
    reason: str
    refined_content: Optional[str] = None
    memory_type: Optional[str] = None
    confidence: float = 0.0
    original: Optional[HarvestCandidate] = None


class HarvestClassifier:
    """LLM-based classifier for harvest candidates.

    Uses Groq API (fast, cheap) with fallback to skip classification
    if no LLM is available. Integrates with existing GroqAgentBridge.
    """

    def __init__(self, groq_api_key: Optional[str] = None):
        self._groq_bridge = None
        self._api_key = groq_api_key or os.environ.get("GROQ_API_KEY")

    def _ensure_initialized(self):
        """Lazy-init Groq bridge."""
        if self._groq_bridge is not None:
            return True

        if not self._api_key:
            logger.warning("No GROQ_API_KEY — LLM classification unavailable")
            return False

        try:
            from groq import Groq
            self._groq_bridge = _GroqClassifierBridge(api_key=self._api_key)
            logger.info("Harvest classifier: Groq bridge initialized")
            return True
        except ImportError:
            logger.warning("groq package not installed — LLM classification unavailable")
            return False
        except Exception as e:
            logger.warning(f"Failed to init Groq bridge for harvest classifier: {e}")
            return False

    def classify(
        self,
        candidates: List[HarvestCandidate],
        context_messages: Optional[List[str]] = None,
    ) -> List[HarvestCandidate]:
        """Classify candidates using LLM, return filtered and refined list.

        Args:
            candidates: Regex-extracted candidates to validate.
            context_messages: Optional surrounding messages for context.

        Returns:
            Filtered list of validated candidates with refined content/confidence.
        """
        if not candidates:
            return []

        if not self._ensure_initialized():
            logger.info("LLM unavailable — returning regex candidates unfiltered")
            return candidates

        context = "\n".join(context_messages[-6:]) if context_messages else "(no context)"

        # Step 1: Classify each candidate
        classified = []
        for candidate in candidates:
            result = self._classify_single(candidate, context)
            if result and result.keep:
                classified.append(self._apply_result(candidate, result))

        # Step 2: Deduplicate if multiple candidates remain
        if len(classified) > 1:
            classified = self._deduplicate(classified)

        return classified

    def _classify_single(
        self, candidate: HarvestCandidate, context: str
    ) -> Optional[ClassificationResult]:
        """Classify a single candidate via LLM."""
        prompt = CLASSIFY_PROMPT_TEMPLATE.format(
            memory_type=candidate.memory_type,
            content=candidate.content,
            confidence=candidate.confidence,
            context=context[:2000],
        )

        models = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
        for model in models:
            try:
                result = self._groq_bridge.call_model(
                    prompt=prompt,
                    model=model,
                    max_tokens=300,
                    temperature=0.1,
                    system_message=SYSTEM_PROMPT,
                )
                if result["status"] != "success":
                    if "429" in str(result.get("error", "")):
                        logger.warning(f"Rate limit on {model}, trying next")
                        continue
                    logger.warning(f"Groq error on {model}: {result.get('error')}")
                    continue

                return self._parse_classification(result["response"])
            except Exception as e:
                logger.warning(f"Classification failed with {model}: {e}")
                continue

        logger.warning("All LLM models failed — keeping candidate unfiltered")
        return ClassificationResult(keep=True, reason="LLM unavailable", confidence=candidate.confidence)

    def _parse_classification(self, response: str) -> ClassificationResult:
        """Parse LLM JSON response into ClassificationResult."""
        # Extract JSON from response (handle markdown code blocks)
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Find outermost JSON object: first '{' to last '}'
            first_brace = text.find("{")
            last_brace = text.rfind("}")
            if first_brace != -1 and last_brace > first_brace:
                try:
                    data = json.loads(text[first_brace:last_brace + 1])
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse LLM response: {text[:200]}")
                    return ClassificationResult(keep=True, reason="parse error — keeping", confidence=0.5)
            else:
                logger.warning(f"Could not parse LLM response: {text[:200]}")
                return ClassificationResult(keep=True, reason="parse error — keeping", confidence=0.5)

        return ClassificationResult(
            keep=bool(data.get("keep", False)),
            reason=data.get("reason", ""),
            refined_content=data.get("refined_content"),
            memory_type=data.get("memory_type"),
            confidence=float(data.get("confidence", 0.5)),
        )

    def _apply_result(
        self, candidate: HarvestCandidate, result: ClassificationResult
    ) -> HarvestCandidate:
        """Apply classification result to candidate."""
        if result.refined_content:
            candidate.content = result.refined_content
        if result.memory_type:
            candidate.memory_type = result.memory_type
            # Replace harvest: tag but preserve other tags
            candidate.tags = [t for t in candidate.tags if not t.startswith("harvest:")]
            candidate.tags.append(f"harvest:{result.memory_type}")
        candidate.confidence = result.confidence
        candidate.tags.append("llm-verified")
        return candidate

    def _deduplicate(self, candidates: List[HarvestCandidate]) -> List[HarvestCandidate]:
        """Remove near-duplicate candidates using LLM."""
        if len(candidates) <= 1:
            return candidates

        candidates_text = "\n".join(
            f"[{i}] ({c.memory_type}, conf={c.confidence:.2f}): {c.content[:200]}"
            for i, c in enumerate(candidates)
        )

        prompt = DEDUP_PROMPT_TEMPLATE.format(candidates_text=candidates_text)

        try:
            result = self._groq_bridge.call_model(
                prompt=prompt,
                model="llama-3.1-8b-instant",
                max_tokens=100,
                temperature=0.0,
                system_message="You deduplicate memories. Respond only with a JSON array of indices.",
            )
            if result["status"] == "success":
                text = result["response"].strip()
                match = re.search(r'\[[\d,\s]*\]', text)
                if match:
                    keep_indices = json.loads(match.group())
                    return [candidates[i] for i in keep_indices if i < len(candidates)]
        except Exception as e:
            logger.warning(f"Deduplication failed: {e}")

        return candidates


class _GroqClassifierBridge:
    """Thin wrapper around Groq SDK for classifier use.

    Avoids sys.path manipulation by importing groq directly.
    Same call_model interface as GroqAgentBridge for compatibility.
    """

    def __init__(self, api_key: str):
        from groq import Groq
        self._client = Groq(api_key=api_key)

    def call_model(self, prompt: str, model: str = "llama-3.1-8b-instant",
                   max_tokens: int = 300, temperature: float = 0.1,
                   system_message: Optional[str] = None) -> dict:
        """Call Groq model and return result dict."""
        try:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})

            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return {
                "status": "success",
                "response": response.choices[0].message.content,
                "model": model,
                "tokens_used": response.usage.total_tokens,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
