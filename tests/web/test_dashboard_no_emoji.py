"""The dashboard uses inline SVG icons and plain text, not emoji."""

import json
import re
from pathlib import Path

import pytest

STATIC_DIR = Path(__file__).parents[2] / "src" / "mcp_memory_service" / "web" / "static"
# Pictographs, symbols and dingbat emoji, variation selector, ZWJ and emoji-only HTML entities.
# The text stars U+2605/U+2606 are typographic glyphs, not emoji, and stay allowed.
EMOJI_RE = re.compile(
    r"(?![★☆])[\U0001F000-\U0001FAFF☀-⛿✅⭐️‍]|&#(?:1[0-9]{5}|65039);"
)
# The language picker shows a flag per locale; that is content, not an icon.
ALLOWED_KEYS = {"meta.language.flag"}


def _emoji_lines(text):
    return [(n, line.strip()[:80]) for n, line in enumerate(text.splitlines(), 1) if EMOJI_RE.search(line)]


@pytest.mark.parametrize("locale", sorted(p.name for p in (STATIC_DIR / "i18n").glob("*.json")))
def test_locale_strings_have_no_emoji(locale):
    data = json.loads((STATIC_DIR / "i18n" / locale).read_text(encoding="utf-8"))

    offenders = {k: v for k, v in data.items() if k not in ALLOWED_KEYS and isinstance(v, str) and EMOJI_RE.search(v)}

    assert offenders == {}


def test_index_html_has_no_emoji():
    assert _emoji_lines((STATIC_DIR / "index.html").read_text(encoding="utf-8")) == []


def test_app_js_renders_no_emoji():
    # console.* lines and the flag fallback for locales without one are not rendered as UI icons.
    lines = [
        line for line in (STATIC_DIR / "app.js").read_text(encoding="utf-8").splitlines()
        if not re.match(r"\s*console\.", line) and "flag" not in line
    ]

    assert _emoji_lines("\n".join(lines)) == []
