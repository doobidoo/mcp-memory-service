**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`

---
name: forge-global-noop-roundtrip
description: No-op Wish → Review roundtrip without domain specifics

---

# Forge Global No-Op Roundtrip

## Steps
1) Create placeholder wish
```
mcp__genie__run agent="wish" prompt="Create placeholder wish with no code changes"
```
2) Review placeholder wish
```
mcp__genie__run agent="review" prompt="Review placeholder wish"
```

## Success Criteria
- Wish document written under `.genie/wishes/<slug>/`
- Review runs without code-specific dependencies

## Evidence
- Save outputs to `.genie/qa/evidence/forge-noop-<timestamp>.txt`

