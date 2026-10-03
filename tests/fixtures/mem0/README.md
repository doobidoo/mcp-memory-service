# mem0 export fixtures

`oss_qdrant_export.json` is an unmodified artifact produced by mem0's own
`scripts/oss-to-platform-migrate.sh --export-only` path.

Provenance:

- upstream repository: https://github.com/mem0ai/mem0
- upstream commit: `94c3fe9f238f3dbf29c9ce98643bd71eb13077cd` (Python SDK 2.2.1)
- producer: `scripts/oss-to-platform-migrate.sh`
- source store: Python OSS hosted Qdrant
- generated: 2026-10-01 from the Qdrant response records recorded in
  `tests/test_oss_to_platform_migrate.py` at the same commit
- artifact sha256: `41e6216440af67126950ca8af8c4c2917bb4e179b7af3bcbbe06c9a430c274f9`

The file is included so parser and importer tests run against a real exporter
artifact rather than a fixture written by this project.
