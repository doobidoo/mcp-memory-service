## ⚠️ CRITICAL: NEVER SELECTIVELY STAGE FILES ⚠️

**YOU MUST ALWAYS USE `git add -A` TO STAGE ALL CHANGES**

Every change made on the task branch is part of the complete implementation. Leaving ANY changes behind means:
- The codebase becomes inconsistent  
- Features are broken or partially missing on main
- The task is incomplete

**NEVER DO:**
- `git add specific-file.txt` - Selective staging
- `git add src/` - Partial directory staging
- Manually picking which changes to include

**ALWAYS DO:**
- `git add -A` - Stages ALL changes including deletions
- `git add .` - Stages all in current directory (if at repo root)