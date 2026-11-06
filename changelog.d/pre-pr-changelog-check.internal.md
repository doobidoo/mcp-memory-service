- **`pre_pr_check.sh` now runs the changelog fragment check.** Check 6.11 calls
  `scripts/ci/check_changelog_entry.sh --staged origin/main`, the same script CI's
  `changelog.yml` runs, but against the index, so a missing fragment or one without a
  `- ` list item fails locally instead of in CI after the push.
