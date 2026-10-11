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
