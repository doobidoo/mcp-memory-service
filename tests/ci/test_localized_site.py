"""Protect language-page generation, escaping, and reciprocal search metadata."""

import html
import importlib.util
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "generate_localized_site", ROOT / "scripts/maintenance/generate_localized_site.py"
)
site = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(site)


class LocalizedSiteTests(unittest.TestCase):
    """Exercise the real catalog/template and isolated invalid inputs."""

    def test_committed_pages_match_sources(self):
        for path, expected in site.generated_files(ROOT).items():
            with self.subTest(path=path):
                self.assertEqual((ROOT / path).read_text(), expected)

    def test_reciprocal_alternates_and_self_canonical(self):
        pages = site.generated_files(ROOT)
        _, catalogs = site.load_catalogs(ROOT)
        languages = [("en", "/")] + [
            (c["locale"], f'/{c["locale"]}/') for c in catalogs
        ]
        for locale, url in languages:
            path = "site/index.html" if locale == "en" else f"site/{locale}/index.html"
            page = pages[Path(path)]
            self.assertIn(f'<html lang="{locale}">', page)
            self.assertIn(f'rel="canonical" href="{site.BASE_URL}{url}"', page)
            for lang, target in languages + [("x-default", "/")]:
                self.assertIn(f'hreflang="{lang}" href="{site.BASE_URL}{target}"', page)
            self.assertIn(f'hreflang="{locale}" aria-current="page"', page)

    def test_sitemap_contains_language_page(self):
        xml = site.generated_files(ROOT)[Path("site/sitemap.xml")]
        tree = ElementTree.fromstring(xml)
        locations = [e.text for e in tree.findall("{*}url/{*}loc")]
        _, catalogs = site.load_catalogs(ROOT)
        for catalog in catalogs:
            self.assertEqual(
                locations.count(f'{site.BASE_URL}/{catalog["locale"]}/'), 1
            )

    def test_missing_or_duplicated_markers_fail(self):
        for source in ["", "<!-- test:start -->" * 2 + "<!-- test:end -->"]:
            with self.assertRaises(ValueError):
                site.replace_block(source, "test", "content")

    def catalog_fixture(self):
        """Create a disposable copy so failures cannot change real catalogs."""
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        shutil.copytree(ROOT / site.SOURCES, root / site.SOURCES)
        (root / "site").mkdir()
        for name in ["index.html", "sitemap.xml"]:
            shutil.copyfile(ROOT / "site" / name, root / "site" / name)
        path = root / site.SOURCES / "de.json"
        return root, path, json.loads(path.read_text())

    def test_incomplete_catalog_fails(self):
        root, path, catalog = self.catalog_fixture()
        del catalog["headline"]
        path.write_text(json.dumps(catalog))
        with self.assertRaisesRegex(ValueError, "translation keys"):
            site.generated_files(root)

    def test_unsafe_or_mismatched_locale_fails(self):
        root, path, catalog = self.catalog_fixture()
        for locale in ["../x", "en", "es"]:
            catalog["locale"] = locale
            path.write_text(json.dumps(catalog))
            with self.assertRaises(ValueError):
                site.generated_files(root)

    def test_translations_are_escaped(self):
        root, path, catalog = self.catalog_fixture()
        catalog["headline"] = '<script>alert("x")</script>'
        path.write_text(json.dumps(catalog))
        page = site.generated_files(root)[Path("site/de/index.html")]
        self.assertIn("&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;", page)
        self.assertNotIn('<script>alert("x")</script>', page)

    def test_structured_data_is_valid_json_from_raw_text(self):
        root, path, catalog = self.catalog_fixture()
        catalog["title"] = 'Größe & "Zitat" </script><b>'
        path.write_text(json.dumps(catalog))
        page = site.generated_files(root)[Path("site/de/index.html")]
        blocks = re.findall(
            r'<script type="application/ld\+json">\n(.*?)\n</script>', page, re.S
        )
        self.assertEqual(len(blocks), 1)
        data = json.loads(blocks[0])
        self.assertEqual(data["name"], catalog["title"])
        self.assertEqual(data["url"], "https://mcpmemory.services/de/")
        self.assertEqual(data["inLanguage"], "de")
        self.assertNotIn("&amp;", blocks[0])

    def test_new_language_updates_all_alternates(self):
        root, _, catalog = self.catalog_fixture()
        catalog.update(locale="zz", language_name="Zz", og_locale="zz_ZZ")
        (root / site.SOURCES / "zz.json").write_text(json.dumps(catalog))
        pages = site.generated_files(root)
        for path, page in pages.items():
            if path.suffix != ".html":
                continue
            self.assertIn('hreflang="zz" href="https://mcpmemory.services/zz/"', page)

    def test_setup_button_links_to_existing_document(self):
        page = site.generated_files(ROOT)[Path("site/de/index.html")]
        self.assertIn(
            'href="https://github.com/doobidoo/mcp-memory-service/blob/main/docs/setup-guide.md"',
            page,
        )
        self.assertTrue((ROOT / "docs/setup-guide.md").is_file())

    def test_install_command_includes_local_semantic_search(self):
        pages = site.generated_files(ROOT)
        for path, page in pages.items():
            if path in {Path("site/index.html"), Path("site/sitemap.xml")}:
                continue
            with self.subTest(path=path):
                self.assertIn(
                    "<code>pip install 'mcp-memory-service[sqlite]'</code>",
                    html.unescape(page),
                    "The quickstart needs ONNX dependencies for semantic search",
                )

    def test_removed_catalog_rejects_retired_output(self):
        root, _, catalog = self.catalog_fixture()
        catalog.update(locale="es", language_name="Español", og_locale="es_ES")
        path = root / site.SOURCES / "es.json"
        path.write_text(json.dumps(catalog))
        page = site.generated_files(root)[Path("site/es/index.html")]
        retired = root / "site/es/index.html"
        retired.parent.mkdir()
        retired.write_text(page)
        path.unlink()
        for mode in ["--check", "--write"]:
            with self.subTest(mode=mode):
                with (
                    patch.object(site, "ROOT", root),
                    patch.object(sys, "argv", ["generate_localized_site.py", mode]),
                ):
                    with self.assertRaisesRegex(ValueError, "Retired localized page"):
                        site.main()
        self.assertTrue(retired.is_file())

    def test_existing_non_generated_pages_are_preserved(self):
        root, _, _ = self.catalog_fixture()
        page = root / "site/demo/index.html"
        page.parent.mkdir()
        page.write_text("<!DOCTYPE html><title>Existing demo</title>")
        site.reject_retired_pages(root, site.generated_files(root))
        self.assertTrue(page.is_file())


if __name__ == "__main__":
    unittest.main()
