### Super-repo Submodule Management

**CRITICAL: Create matching branches in ALL affected submodules**
- Check the task frontmatter for the submodules list
- For each submodule listed:
  - Navigate to that submodule directory and verify repo (.git exists)
  - Check for uncommitted changes first and, if any, ask user how to handle them
  - If not on {default_branch}, checkout {default_branch} and pull latest (do not destroy uncommitted changes)
  - Create a branch with the same name as the task branch
  - Return to the parent directory

Example: If working on tt1-login-ux-flow affecting io_web and io_user_model, create tt1-login-ux-flow branches in both submodules.

**Branch Discipline Rules:**
- Task frontmatter must list ALL submodules that might be edited
- All listed submodules MUST have matching task branches
- Before editing any file, verify the submodule is on the correct branch
- If a module needs to be added mid-task, create its branch immediately