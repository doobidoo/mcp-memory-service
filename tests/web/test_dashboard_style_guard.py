"""The dashboard stylesheet stays free of hue gradients, catch-all transitions and heavy weights."""

import re
from pathlib import Path

import pytest

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


TIMING_WORDS = {"ease", "ease-in", "ease-out", "ease-in-out", "linear", "step-start", "step-end", "infinite"}


def _split_top_level(value):
    """Split a comma list, ignoring commas inside parentheses (cubic-bezier, var)."""
    parts, depth, buf = [], 0, ""
    for ch in value:
        depth += (ch == "(") - (ch == ")")
        if ch == "," and depth == 0:
            parts.append(buf)
            buf = ""
        else:
            buf += ch
    return parts + [buf]


def _names_a_property(part):
    for token in re.findall(r"[\w-]+\([^)]*\)|[^\s]+", part.strip()):
        if token == "all":
            return False
        if re.match(r"^[\d.]+m?s$|^(var|cubic-bezier|steps)\(", token) or token in TIMING_WORDS:
            continue
        return True
    return False


def catch_all_transition(body):
    """True when a rule animates every property, explicitly or by omitting the property name."""
    if re.search(r"transition-property:\s*all\b", body):
        return True
    has_property_list = re.search(r"transition-property:\s*(?!all\b)[\w-]", body) is not None
    for value in re.findall(r"(?<![\w-])transition:\s*([^;]+)", body):
        if value.strip() in ("none", "initial", "unset", "inherit"):
            continue
        if not all(_names_a_property(part) for part in _split_top_level(value)) and not has_property_list:
            return True
    return False


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("transition: all 0.2s ease;", True),
        ("transition: 0.2s ease;", True),
        ("transition: var(--transition-base);", True),
        ("transition-property: all; transition-duration: 0.2s;", True),
        ("transition: color 0.2s, all 0.3s;", True),
        ("transition: var(--transition-base); transition-property: color, opacity;", False),
        ("transition: opacity var(--transition-base), transform var(--transition-base);", False),
        ("transition: background-color 0.2s cubic-bezier(0.22, 1, 0.36, 1);", False),
        ("transition: none;", False),
        ("color: red;", False),
    ],
)
def test_catch_all_detection(body, expected):
    assert catch_all_transition(body) is expected


def test_transitions_name_their_properties():
    # Graph nodes and links are SVG content and keep their own transitions.
    offenders = [
        sel for sel, body in _rules()
        if catch_all_transition(body) and not sel.startswith((".graph-node", ".graph-link"))
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
