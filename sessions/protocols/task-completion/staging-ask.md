### Step 2: Review Unstaged Changes

Check for any unstaged changes:
```bash
git status
```

If changes exist, present them using this format:

```markdown
[DECISION: Staging Changes]
Found the following unstaged changes:

Modified:
- file1.py: [brief description if known]
- file2.js: [brief description]

Untracked:
- newfile.md
- temp/

How would you like to stage these changes?
- ALL: Stage all changes (git add -A)
- SELECTIVE: Review and select specific files
- REVIEW: Show diffs before deciding

Your choice:
```

Based on user preference, stage appropriately.