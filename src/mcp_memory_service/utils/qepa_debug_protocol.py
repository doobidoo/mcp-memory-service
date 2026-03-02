from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, List


@dataclass
class QEPADebugRecord:
    """
    Minimal, reusable container for a single failing run in Q/E/P/A form.

    Q = query (what the user or upstream agent asked)
    E = environment (retriever config, backends, tags, run metadata, etc.)
    P = prompt (what was actually sent to the LLM)
    A = answer (the LLM output that we want to debug)

    This is intentionally generic. It can be used with any "failure map" or
    "global debug card" style workflow, including but not limited to WFGY.
    """

    query: str
    environment: Optional[Mapping[str, Any]] = None
    prompt: Optional[str] = None
    answer: Optional[str] = None
    meta: Optional[Mapping[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a plain dict that is easy to log or serialize.
        Keys are Q, E, P, A, plus optional meta.
        """
        payload: Dict[str, Any] = {"Q": self.query}

        if self.environment:
            payload["E"] = dict(self.environment)

        if self.prompt is not None:
            payload["P"] = self.prompt

        if self.answer is not None:
            payload["A"] = self.answer

        if self.meta:
            payload["meta"] = dict(self.meta)

        return payload

    def to_markdown(self) -> str:
        """
        Render the record as a markdown block that can be pasted into an LLM
        conversation as structured context.
        """
        lines: List[str] = []

        # Q
        lines.append("### Q · query")
        q_text = (self.query or "").strip() or "(empty)"
        lines.append(q_text)

        # E
        if self.environment:
            lines.append("")
            lines.append("### E · environment")
            for key, value in self.environment.items():
                lines.append(f"- **{key}**: {value}")

        # P
        if self.prompt:
            lines.append("")
            lines.append("### P · prompt")
            lines.append("```text")
            lines.append(self.prompt)
            lines.append("```")

        # A
        if self.answer:
            lines.append("")
            lines.append("### A · answer")
            lines.append("```text")
            lines.append(self.answer)
            lines.append("```")

        # meta
        if self.meta:
            lines.append("")
            lines.append("### meta")
            for key, value in self.meta.items():
                lines.append(f"- **{key}**: {value}")

        return "\n".join(lines)


def build_debug_prompt(
    card_instructions: str,
    record: QEPADebugRecord,
    *,
    extra_instructions: Optional[str] = None,
) -> str:
    """
    Combine a short "failure map" instruction block with a Q/E/P/A record
    to produce an LLM ready prompt.
    """
    parts: List[str] = []

    card_block = (card_instructions or "").strip()
    if card_block:
        parts.append(card_block)
        parts.append("")

    parts.append("You are given a structured record of one failing run in Q/E/P/A form.")
    parts.append("Use it together with your failure map to classify the failure mode")
    parts.append("and suggest the smallest structural change you would try first.")
    parts.append("")
    parts.append(record.to_markdown())

    if extra_instructions:
        extra_block = extra_instructions.strip()
        if extra_block:
            parts.append("")
            parts.append(extra_block)

    return "\n".join(parts)
