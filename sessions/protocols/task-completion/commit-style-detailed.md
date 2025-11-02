Use branch-prefix format with comprehensive body:
- Summary line: `branch-prefix: brief description` (50 chars max)
- Blank line
- Body explaining what changed and why (wrap at 72 chars)
- List key changes with bullet points
- Example:
  ```
  feature: complete sessions migration
  
  Migrated local hook configuration to use cc-sessions package
  exclusively. This eliminates dual maintenance and ensures all
  customizations are preserved through configuration.
  
  - Created protocol startup-load command for task loading
  - Removed modified add pattern (impractical)  
  - Fixed protocol/todo mismatches
  - Updated all import paths for symlinked setup
  ```