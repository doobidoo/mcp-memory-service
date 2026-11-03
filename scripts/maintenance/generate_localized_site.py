#!/usr/bin/env python3
"""Render introductory language pages without adding a deployment build step."""

import argparse
import html
import json
import re
from pathlib import Path
from string import Template

BASE_URL = "https://mcpmemory.services"
ROOT = Path(__file__).resolve().parents[2]
SOURCES = Path("scripts/maintenance/site_locales")
AUTOMATIC_KEYS = {"canonical", "alternates", "language_links"}
GENERATED_MARKER = "<!-- Generated localized introduction. -->"


def validate_catalog(catalog, path, required):
    """Reject incomplete catalogs and unsafe language paths before rendering."""
    if not isinstance(catalog, dict) or set(catalog) != required:
        raise ValueError(f"{path.name}: missing or unexpected translation keys")
    if not all(isinstance(value, str) and value.strip() for value in catalog.values()):
        raise ValueError(f"{path.name}: translations must be non-empty strings")
    locale = catalog["locale"]
    if not re.fullmatch(r"[a-z]{2}(?:-[a-z]{2})?", locale) or locale == "en":
        raise ValueError(f"{path.name}: invalid or reserved locale")
    if path.stem != locale:
        raise ValueError(f"{path.name}: filename must match locale")


def load_catalogs(root):
    """Require complete, plain-text catalogs with safe language identifiers."""
    template = Template((root / SOURCES / "landing.html").read_text(encoding="utf-8"))
    keys = {
        match.group("named") or match.group("braced")
        for match in template.pattern.finditer(template.template)
    } - {None}
    required = (keys - AUTOMATIC_KEYS) | {"language_name"}
    catalogs = []
    for path in sorted((root / SOURCES).glob("*.json")):
        catalog = json.loads(path.read_text(encoding="utf-8"))
        validate_catalog(catalog, path, required)
        catalogs.append(catalog)
    if not catalogs:
        raise ValueError("At least one localized catalog is required")
    return template, catalogs


def alternates(languages):
    """Use reciprocal alternates and the existing English page as fallback."""
    links = [
        f'<link rel="alternate" hreflang="{locale}" href="{BASE_URL}{path}">'
        for locale, _, path in languages
    ]
    links.append(f'<link rel="alternate" hreflang="x-default" href="{BASE_URL}/">')
    return "\n".join(links)


def language_links(languages, current):
    """Keep explicit language links usable without JavaScript on mobile too."""
    links = []
    for locale, name, path in languages:
        active = ' aria-current="page"' if locale == current else ""
        links.append(
            f'<a href="{path}" lang="{locale}" hreflang="{locale}"{active}>'
            f"{html.escape(name)}</a>"
        )
    return "\n".join(links)


def replace_block(source, name, content):
    """Replace one owned block, refusing missing or repeated markers."""
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError(f"Expected one {name} block")
    before, rest = source.split(start)
    _, after = rest.split(end)
    return f"{before}{start}\n{content}\n{end}{after}"


def generated_files(root):
    """Render catalogs and update only the owned home/sitemap sections."""
    template, catalogs = load_catalogs(root)
    languages = [("en", "English", "/")] + [
        (c["locale"], c["language_name"], f'/{c["locale"]}/') for c in catalogs
    ]
    alternate_links = alternates(languages)
    outputs = {}
    for catalog in catalogs:
        locale = catalog["locale"]
        values = {key: html.escape(value, quote=True) for key, value in catalog.items()}
        values.update(
            canonical=f"{BASE_URL}/{locale}/",
            alternates=alternate_links,
            language_links=language_links(languages, locale),
        )
        outputs[Path(f"site/{locale}/index.html")] = template.substitute(values)

    home = (root / "site/index.html").read_text(encoding="utf-8")
    home = replace_block(home, "locale-alternates", alternate_links)
    outputs[Path("site/index.html")] = replace_block(
        home, "locale-navigation", language_links(languages, "en")
    )
    entries = "\n".join(
        f"  <url><loc>{BASE_URL}{path}</loc><changefreq>monthly</changefreq></url>"
        for locale, _, path in languages
        if locale != "en"
    )
    outputs[Path("site/sitemap.xml")] = replace_block(
        (root / "site/sitemap.xml").read_text(encoding="utf-8"), "locale-pages", entries
    )
    return outputs


def reject_retired_pages(root, outputs):
    """Reject obsolete owned HTML without deleting or touching unrelated pages."""
    for path in (root / "site").glob("*/index.html"):
        if path.relative_to(root) in outputs:
            continue
        if GENERATED_MARKER in path.read_text(encoding="utf-8"):
            raise ValueError(f"Retired localized page: remove {path.relative_to(root)}")


def main():
    """Check committed output or explicitly regenerate it for a content change."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args()
    stale = []
    outputs = generated_files(ROOT)
    reject_retired_pages(ROOT, outputs)
    for relative, content in outputs.items():
        path = ROOT / relative
        if path.exists() and path.read_text(encoding="utf-8") == content:
            continue
        stale.append(str(relative))
        if args.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    if args.check and stale:
        parser.exit(1, "Localized site out of date: " + ", ".join(stale) + "\n")
    print("Localized site: " + ("up to date" if args.check else "generated"))


if __name__ == "__main__":
    main()
