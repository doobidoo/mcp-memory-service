"""The dashboard stylesheet stays free of hue gradients, catch-all transitions and heavy weights."""

import re
from pathlib import Path

CSS = re.sub(
    r"/\*.*?\*/",
    "",
    (Path(__file__).parents[2] / "src" / "mcp_memory_service" / "web" / "static" / "style.css").read_text(
        encoding="utf-8"
    ),
    flags=re.S,
)
RULE_RE = re.compile(r"([^{}]+)\{([^{}]*)\}")


def _rules():
    for match in RULE_RE.finditer(CSS):
        yield match.group(1).strip(), match.group(2)


def test_no_gradients():
    assert [sel for sel, body in _rules() if "gradient(" in body] == []


def test_transitions_name_their_properties():
    # Graph nodes and links are SVG content and keep their own transitions.
    offenders = [
        sel for sel, body in _rules()
        if re.search(r"transition:\s*all\b", body) and not sel.startswith((".graph-node", ".graph-link"))
    ]
    assert offenders == []


def test_no_hover_lift_or_zoom():
    offenders = [
        sel for sel, body in _rules()
        if ":hover" in sel and re.search(r"transform:\s*(translateY\(-|scale\(1\.)", body)
    ]
    assert offenders == []


def test_weights_stay_at_or_below_600():
    offenders = [
        sel for sel, body in _rules()
        if not sel.startswith("@font-face") and re.search(r"font-weight:\s*(700|800|900|bold)\b", body)
    ]
    assert offenders == []
