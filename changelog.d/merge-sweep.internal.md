- **`/merge-sweep` command for merge rounds over open PRs.**
  The prompt that worked for clearing cheap-to-merge PRs now lives in
  `.claude/commands/merge-sweep.md`: it checks CI per head SHA, open review threads,
  conflicts and Dependabot bounds, classifies each PR, and merges nothing until the
  maintainer approves the table.
