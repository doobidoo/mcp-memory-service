# Changelog fragments

One file per pull request. At release time `scripts/release/collect_changelog.py`
merges them into the `## [Unreleased]` section of `CHANGELOG.md` and deletes them.

Fragments exist so that two pull requests never edit the same lines of the same
file. Before this directory, every change had to add its entry at the top of
`CHANGELOG.md`, which conflicts with every other open branch — and the result was
that almost nobody did: of the 19 pull requests merged between v11.12.0 and
v11.13.0, exactly one had left an entry, and the other 18 were reconstructed from
commit messages days later (#1273).

## Naming

    <number>.<category>.md

`<number>` is the PR number, or the issue number if you do not have a PR number
yet — it only has to be unique. `<category>` is one of:

| category   | CHANGELOG section | for |
|------------|-------------------|-----|
| `added`    | `### Added`       | new behaviour, new configuration, new endpoints |
| `fixed`    | `### Fixed`       | bugs |
| `removed`  | `### Removed`     | deletions, retired features |
| `internal` | `### Internal`    | CI, tooling, docs, dependency bumps, refactors with no user-visible effect |

Example: `changelog.d/1273.internal.md`

## Content

One entry, written the way the existing CHANGELOG reads: a bolded claim, then what
was actually wrong and what changed. Cite the PR and, where there is one, the issue
it closes. Credit external contributors by handle.

```markdown
- **Vectorize rejected tag-filtered retrieval above 16 results (#1259, massimiliano1991).**
  Every Cloudflare query sets `returnMetadata="all"`, for which Vectorize caps `topK`
  at 50, but `retrieve()` passed the sqlite-vec KNN ceiling of 4096 instead. Both call
  sites are clamped now.
```

Write what a reader six months from now needs: the symptom, not the patch. "Fixed a
bug in retrieve()" tells them nothing they could not have guessed from the commit.

A change that genuinely needs no entry — a typo fix, a comment — can be exempted by a
maintainer with the `skip-changelog` label. The label only takes effect on the next
event the pull request emits, so push a commit or update the branch from main after
labelling; a re-run replays the old payload and changes nothing (#1266).
