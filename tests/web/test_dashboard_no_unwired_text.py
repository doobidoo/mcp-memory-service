"""The dashboard's index.html carries no hardcoded English UI text (#1928).

Every visible text node, ``placeholder`` and ``aria-label`` that says something
in English must be wired to the i18n catalogs through a ``data-i18n*``
attribute — on the element itself or on an ancestor that replaces its whole
content (``data-i18n-html``). Otherwise the string stays English after a
language switch, which is how the #1806 slices found their leftovers.

The allowlist below is short and each entry says why it is there. Technical
literals (HTTP methods, endpoint paths) are the same in every language, and two
fallback strings are replaced by app.js on load.
"""

import re
from html.parser import HTMLParser
from pathlib import Path

INDEX_HTML = (
    Path(__file__).resolve().parents[2] / "src" / "mcp_memory_service" / "web" / "static" / "index.html"
)

# A run of two or more letters: numbers, punctuation and single-letter units do not count.
WORD = re.compile(r"[A-Za-z]{2,}")

# HTTP methods and /api/... endpoint paths in the API reference section: identical
# in every language, so they are not localized.
TECHNICAL_LITERAL = re.compile(r"^(?:(?:GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)|/api/\S*)$")

# Fallback text that app.js replaces on load; the HTML value is only shown
# before the catalog applies.
RUNTIME_WIRED = {
    # searchModeText: this.t('search.modeLive', ...) / this.t('search.modeManual', ...)
    "Live Search",
    # modal.settings.server.reconnectingIn split around the {seconds} placeholder
    "Reconnecting in",
}

VOID_ELEMENTS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}

# The data-i18n* attribute that wires each localized attribute.
ATTRIBUTE_WIRING = {
    "placeholder": "data-i18n-placeholder",
    "title": "data-i18n-title",
    "aria-label": "data-i18n-aria",
}


class _UnwiredTextCollector(HTMLParser):
    """Collect visible text nodes and localization attributes that carry English words."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.unwired = []

    def _wired_ancestry(self) -> bool:
        return any(
            key.startswith("data-i18n")
            for _, attrs in self.stack
            for key in attrs
        )

    def handle_starttag(self, tag, attrs):
        attr_map = dict(attrs)
        self.stack.append((tag, attr_map))
        if tag in VOID_ELEMENTS:
            self.stack.pop()
        for attribute, wiring in ATTRIBUTE_WIRING.items():
            value = attr_map.get(attribute)
            if value and WORD.search(value) and wiring not in attr_map:
                self.unwired.append(f'<{tag} {attribute}="{value}"> (needs {wiring})')

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        while self.stack and self.stack[-1][0] != tag:
            self.stack.pop()
        if self.stack:
            self.stack.pop()

    def handle_data(self, data):
        text = " ".join(data.split())
        if not text or not WORD.search(text):
            return
        if not self.stack:
            return
        tag, attrs = self.stack[-1]
        if tag in ("script", "style"):
            return
        if self._wired_ancestry():
            return
        if text in RUNTIME_WIRED:
            return
        if all(TECHNICAL_LITERAL.match(token) for token in text.split()):
            return
        line = self.getpos()[0]
        self.unwired.append(f"line {line}: <{tag}> {text!r}")


def test_index_html_has_no_unwired_english_text():
    collector = _UnwiredTextCollector()
    collector.feed(INDEX_HTML.read_text(encoding="utf-8"))

    assert collector.unwired == [
        # Nothing should appear here: wire new UI text with data-i18n* attributes,
        # and extend the allowlists above only for technical literals (with the
        # reason in a comment) or text that app.js replaces on load.
    ]
