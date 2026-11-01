# Reaching developers in more languages

The pilot adds a German introduction to mcpmemory.services. English remains at `/`;
German lives at `/de/`. Spanish is the provisional second language, subject to
audience signals and an available fluent reviewer. Dashboard language catalogs are
separate: a translated dashboard alone does not create a searchable entry page.

## Content brief

Write for developers who want their AI client to retain project knowledge between
sessions. Explain the problem, show a concrete example, then give the reader a
working installation path. Each introduction covers:

- Persistent project decisions, semantic retrieval and local hosting.
- A small store/recall example, explicitly labeled as example client prompts.
- Python installation, Claude Code configuration and a first verified memory.
- Client support, local-storage boundaries, optional automation and costs.
- Links to the English documentation, browser demo and GitHub project.

Adapt wording to the audience; keep commands, product names and factual boundaries
intact. Do not imply that connecting MCP automatically saves every conversation or
that the AI client's own model requests stay local. Avoid version numbers and
volatile benchmark/test counts in introductions so releases need not rewrite every
translation.

The initial German search phrases are hypotheses, not measured keywords:
"dauerhaftes Gedächtnis für KI-Assistenten", "Claude Code Projektwissen speichern"
and "MCP Memory Server selbst hosten". A fluent reviewer should check how developers
actually describe the problem. Use these ideas naturally in useful prose, without
keyword lists. Search Console query data should guide later revisions.

## Maintaining language pages

The shared HTML template and plain-text catalogs are in
`scripts/maintenance/site_locales/`. CSS is shared through
`site/assets/localized.css` and the existing site styles. Deployment still uploads
the committed `site/` directory directly; no runtime translation or new build
service is needed.

To update or add a language:

1. Copy `de.json` to a locale-named catalog, such as `es.json`. Translate every
   value except `locale` and `og_locale`, which are language metadata. Set these
   explicitly, and set `language_name` to the native language name.
2. Keep every key from the reference catalog. Values are plain text: HTML is
   escaped by the generator. All visible labels, examples and metadata need review.
3. Run the generator and checks from the repository root:

   ```bash
   python3 scripts/maintenance/generate_localized_site.py --write
   python3 scripts/maintenance/generate_localized_site.py --check
   python3 -m unittest discover -s tests/ci -p test_localized_site.py -v
   ```

4. Commit the catalog and generated HTML, root-page language links and sitemap.
   The generator updates reciprocal `hreflang` links automatically. Each page has
   its own canonical URL; the existing English root is the default language.
5. Check narrow and wide screens, light and dark modes, keyboard navigation,
   installation commands, local asset links and readable accented characters.
6. Have a fluent speaker review the complete page before publication. Record the
   reviewer in the PR; do not publish placeholder or partially translated pages.

Language links remain visible on mobile. Readers choose their language explicitly;
there is no redirect based on browser language or location. The Localized Site CI
workflow checks committed output on changes to the site, template, catalogs,
generator, tests or workflow.

When retiring or renaming a language, remove its old generated HTML explicitly
alongside the catalog change. Both generator modes reject leftover owned pages
before writing anything, so obsolete translations cannot silently ship. Existing
pages such as the demo are not treated as generated language output.

## Measurement and rollout

`site/index.html` documents zone-level Cloudflare Web Analytics injection. Do not
add another beacon. This repository does not establish whether collection is
currently active or whether Search Console is configured. Verify those in the
existing accounts before claiming a measured lift.

A public HTML fetch on 2026-11-01 returned the English home page with HTTP 200 but
without a Cloudflare Insights beacon URL. That sample does not prove collection is
active or inactive for every visitor. Verify the actual browser network traffic
and account configuration before using traffic reports as a baseline.

Capture a 28-day baseline before the first localized page goes live:

| Measure | Source | Initial status |
| --- | --- | --- |
| Search impressions, clicks, queries and countries | Search Console | Export needed; account access not available in this session |
| Visits by landing page, referrer and country | Existing Cloudflare analytics | Collection and export need verification |
| Outbound clicks to GitHub and installation docs | An explicitly configured event/report | Not established; do not infer from page views |

Use dated exports in the maintainer's reporting location, outside the repository.
Compare equal-length periods and record publication dates and promotions. Country
is an imperfect proxy for language. Repository stars and package downloads cannot
be attributed to a language page from aggregate totals alone.

After publication, review indexing and early traffic after four weeks, then assess
expansion after eight to twelve weeks. Look for relevant non-brand search queries,
search clicks and engaged entry-page traffic. A low-volume result calls for further
observation and content review, rather than a claim of success or failure.

For each pilot language, prepare one short announcement with a localized problem,
one useful example and a link to that language page. Use consistent campaign
parameters, for example `/de/?utm_source=discord&utm_medium=community&utm_campaign=language-pilot`.
Verify the analytics tool can report those parameters before relying on campaign
attribution. Introduce the project in relevant developer communities in accordance
with their posting rules; the existing project Discord mainly reaches current
community members. Add README language links when the pages are published.

## Spanish contributor brief

Prepare the Spanish introduction after the foundation is merged and a fluent
reviewer is available. Scope: `es.json`, generated `/es/` output, language metadata
and sitemap updates, plus the README language link. Reuse the shared layout and
commands; translate and review all content, including SEO and social descriptions.
Suggested wording such as "memoria persistente para asistentes de IA" needs fluent
review and is not a measured search-volume claim. Installation and client support
must stay consistent with the reference introduction.

Publish the contributor issue only when that scope is ready to claim. Use
`enhancement`, `localization`, `help wanted`, `good first issue` and `auto-assign`;
name the language reviewer. Acceptance requires generation/check commands above,
mobile/desktop verification and fluent review. Keep community announcements as a
separate publication step after the page is live.

Reference: [Google's guidance for multilingual sites](https://developers.google.com/search/docs/specialty/international/managing-multi-regional-sites).
