# Diagrams

The two interactive diagrams on mcpmemory.services are generated from the
specifications in this directory with the [Archify](https://github.com/tt-a1i/archify)
skill (MIT, version 3.0.1 when these were made). The `.json` files are the source;
the HTML pages under `site/` are the output.

| Diagram | Specification | Page |
|---|---|---|
| Architecture: three ways into the service | [`architecture.architecture.json`](architecture.architecture.json) | [`site/architecture/index.html`](../../site/architecture/index.html) |
| Release and mirror topology | [`deployment.architecture.json`](deployment.architecture.json) | [`site/deployment/index.html`](../../site/deployment/index.html) |

The architecture diagram is written from the source at the commit pinned in
`meta.repository.revision`: every component carries the file and line range that
supports it, and the pages link those ranges on GitHub. The deployment diagram
describes where the project is hosted and mirrored; it is a hand-authored snapshot
that is redrawn when the topology changes.

## Regenerate

Run from the repository root, with the Archify skill installed:

```bash
A=~/.claude/skills/archify/bin/archify.mjs   # wherever the skill lives
node "$A" finalize architecture docs/diagrams/architecture.architecture.json site/architecture/index.html --repo-root . --quality showcase --json
node "$A" finalize architecture docs/diagrams/deployment.architecture.json   site/deployment/index.html   --repo-root . --quality showcase --json
```

All four gates (validate, deliver, check, browser-check) must pass.

- Before regenerating the architecture diagram, set `meta.repository.revision` to the
  commit you checked it against and re-check the `sources` line ranges; they drift as
  the code changes. Archify verifies them against the committed bytes at that commit,
  so commit the source changes first.
- `finalize` writes a `*.delivery.json` receipt next to the HTML. It contains local
  paths: delete it before committing.
- The pages embed their font. `site/_headers` gives `/architecture/*` and
  `/deployment/*` a policy with `font-src 'self' data:` for that reason.
- Archify has no option for custom colors or fonts. `meta.visual_preset` picks one of
  its four built-in presets; these pages use `blueprint`, the one that sits closest to
  the navy and cyan of the landing page.

## Thumbnails on the landing page

The link tiles on the landing page (architecture, deployment, live demo, setup
guide) show a screenshot of the page they open. They are generated from the real
pages, not drawn by hand:

```bash
node scripts/maintenance/site_thumbnails.mjs            # all four
node scripts/maintenance/site_thumbnails.mjs demo       # one of them
```

It needs Node 22 or newer and a Chromium or Chrome (`CHROME_BIN` picks one), and
writes `site/assets/images/thumb-<name>.webp` at 1200x675. The demo run waits for the
model to load and runs one search, so it takes about twenty seconds. Run it again
after a visible change to one of the four pages, and after regenerating a diagram.

The same script draws the Open Graph card, the picture shown when a link to the site is
shared (`node scripts/maintenance/site_thumbnails.mjs og`). Its source is
[`scripts/maintenance/og_card.html`](../../scripts/maintenance/og_card.html), the output
`site/assets/images/og-home.jpg` at 1200x630. Social networks cache the image by URL, so
when the card changes, give the file a new name and update the `og:image` tags too.
