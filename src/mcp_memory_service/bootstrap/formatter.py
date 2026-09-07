"""Bootstrap profile formatter — plugin system for multi-agent support."""

import os
import re
from datetime import datetime, timezone
from typing import Dict, Optional


class BootstrapFormatter:
    """Base class for bootstrap profile formatters."""
    name: str = "base"

    def format(self, avoidances: list, preferences: list,
               conventions: list, meta: dict, *,
               beliefs: Optional[list] = None, task_context: Optional[list] = None) -> str:
        raise NotImplementedError


class ClaudeFormatter(BootstrapFormatter):
    """Default upstream format — markdown sections."""
    name = "claude"

    def format(self, avoidances: list, preferences: list,
               conventions: list, meta: dict, *,
               beliefs: Optional[list] = None, task_context: Optional[list] = None) -> str:
        now = meta.get("generated_at", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        sections = [f"=== BEHAVIORAL PROFILE (v1, generated {now}) ===\n"]

        if meta.get("fresh_start"):
            sections.append("⚠️ FRESH START SESSION: This session ignores learned rules to validate baseline behavior.\n")

        if avoidances:
            sections.append("## Avoidances (from errors — DO NOT ignore)")
            sections.extend(avoidances[:10])
            sections.append("")

        if preferences:
            sections.append("## Preferences (user-stated)")
            sections.extend(preferences[:10])
            sections.append("")

        if conventions:
            sections.append("## Conventions (learned from decisions)")
            sections.extend(conventions[:10])
            sections.append("")

        if beliefs:
            sections.append("## Beliefs (derived from repeated observations)")
            sections.extend(beliefs[:10])
            sections.append("")

        if task_context:
            sections.append("## Task Context (relevant to current task)")
            sections.extend(task_context[:5])
            sections.append("")

        sections.append("## Meta-instruction")
        sections.append("Rules with confidence < 0.7 are advisory. If user behavior contradicts them,")
        sections.append("note the contradiction in your session report.")
        sections.append("=== END PROFILE ===")

        return "\n".join(sections)


class KiroFormatter(BootstrapFormatter):
    """Kiro CLI format — enforcement language, steering-compatible."""
    name = "kiro"

    # Locale strings — externalized for i18n
    LOCALE = {
        "title": "# Bootstrap Profile (auto-generated)\n",
        "avoidances_header": "## Avoidances (past errors)",
        "avoidances_prefix": "❌ NEVER:",
        "conventions_header": "## Conventions (consolidated decisions)",
        "conventions_prefix": "✅ ALWAYS:",
        "preferences_header": "## Preferences (learned)",
        "preferences_prefix": "📌",
        "beliefs_header": "## Beliefs (derived knowledge)",
        "beliefs_prefix": "🧠",
        "task_header": "## Task Context",
        "task_prefix": "📎",
    }

    LOCALE_PT_BR = {
        "title": "# Bootstrap Profile (auto-generated)\n",
        "avoidances_header": "## Regras de Evitação (erros passados)",
        "avoidances_prefix": "❌ NUNCA:",
        "conventions_header": "## Convenções (decisões consolidadas)",
        "conventions_prefix": "✅ SEMPRE:",
        "preferences_header": "## Preferências (aprendidas)",
        "preferences_prefix": "📌",
        "beliefs_header": "## Crenças (conhecimento derivado)",
        "beliefs_prefix": "🧠",
        "task_header": "## Contexto da Tarefa",
        "task_prefix": "📎",
    }

    def __init__(self, locale: str = "en"):
        self._strings = self.LOCALE_PT_BR if locale == "pt_BR" else self.LOCALE

    def _clean(self, text: str) -> str:
        """Remove leading '- ' and confidence suffix."""
        text = re.sub(r'^-\s*[⚠️]*\s*', '', text)
        text = re.sub(r'\s*\[confidence:.*?\]\s*$', '', text)
        return text.strip()

    def format(self, avoidances: list, preferences: list,
               conventions: list, meta: dict, *,
               beliefs: Optional[list] = None, task_context: Optional[list] = None) -> str:
        s = self._strings
        lines = [s["title"]]

        if avoidances:
            lines.append(s["avoidances_header"])
            for a in avoidances[:10]:
                lines.append(f"{s['avoidances_prefix']} {self._clean(a)}")
            lines.append("")

        if conventions:
            lines.append(s["conventions_header"])
            for c in conventions[:10]:
                lines.append(f"{s['conventions_prefix']} {self._clean(c)}")
            lines.append("")

        if preferences:
            lines.append(s["preferences_header"])
            for p in preferences[:10]:
                lines.append(f"{s['preferences_prefix']} {self._clean(p)}")
            lines.append("")

        if beliefs:
            lines.append(s["beliefs_header"])
            for b in (beliefs or [])[:10]:
                lines.append(f"{s['beliefs_prefix']} {self._clean(b)}")
            lines.append("")

        if task_context:
            lines.append(s["task_header"])
            for t in (task_context or [])[:5]:
                lines.append(f"{s['task_prefix']} {self._clean(t)}")
            lines.append("")

        return "\n".join(lines)


_FORMATTERS: Dict[str, BootstrapFormatter] = {
    "claude": ClaudeFormatter(),
    "kiro": KiroFormatter(locale=os.environ.get("HARVEST_LOCALE", "en").split(",")[-1]),
}


def get_formatter(name: str) -> BootstrapFormatter:
    """Get formatter by name. Returns claude (default) if unknown."""
    return _FORMATTERS.get(name, _FORMATTERS["claude"])


def get_formatter_for_agent(agent_id: str) -> BootstrapFormatter:
    """Get formatter for a specific agent_id, checking env var overrides."""
    # Check agent-specific override: MCP_BOOTSTRAP_FORMATTER_{agent_id}
    formatter_name = os.environ.get(f"MCP_BOOTSTRAP_FORMATTER_{agent_id}")
    if not formatter_name:
        # Fall back to global default
        formatter_name = os.environ.get("MCP_BOOTSTRAP_FORMATTER", "claude")
    return get_formatter(formatter_name)
