### Step 3: Commit and Merge

1. Stage all changes with `git add -A`
2. Commit with descriptive message
   
   {commit_style_guidance}
   
3. {merge_instruction}

   If asking about merge:
   ```markdown
   [DECISION: Merge to {default_branch}]
   Would you like to merge this branch to {default_branch}?
   - YES: Merge to {default_branch}
   - NO: Keep changes on feature branch

   Your choice:
   ```

4. {push_instruction}

   If asking about push:
   ```markdown
   [DECISION: Push to Remote]
   Would you like to push the changes to remote?
   - YES: Push to origin
   - NO: Keep changes local only

   Your choice:
   ```