> - Check git status first for uncommitted work in super-repo AND all submodules
>   - Address EVERY file shown in `git status`, not just expected files
>   - Common missed files: CLAUDE.md, sessions/state files, test outputs
>   - Either commit ALL changes or explicitly discuss with user
> - Checkout the branch in the super-repo
> - For each affected submodule, navigate to it and checkout the matching branch
> - Only pull from remote if the remote branch exists