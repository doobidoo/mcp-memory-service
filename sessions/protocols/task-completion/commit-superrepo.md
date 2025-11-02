### Step 3: Commit and Merge (Super-repo Structure)

**CRITICAL ORDER**: Process from deepest submodules to super-repo

#### A. Deepest Submodules First (Depth 2+)
For any submodules within submodules:
1. Navigate to each modified deep submodule
2. Stage all changes with `git add -A`
3. Commit all changes with descriptive message
   
   {commit_style_guidance}
   
4. {merge_instruction}

   If asking about merge:
   ```markdown
   [DECISION: Merge Submodule to {default_branch}]
   Would you like to merge this submodule branch to {default_branch}?
   - YES: Merge to {default_branch}
   - NO: Keep changes on feature branch

   Your choice:
   ```

5. {push_instruction}

   If asking about push:
   ```markdown
   [DECISION: Push Submodule to Remote]
   Would you like to push the submodule changes to remote?
   - YES: Push to origin
   - NO: Keep changes local only

   Your choice:
   ```

#### B. Direct Submodules (Depth 1)
For all modified direct submodules:
1. Navigate to each modified submodule
2. Stage all changes with `git add -A`
3. Commit all changes with descriptive message
   
   {commit_style_guidance}
   
4. {merge_instruction}

   If asking about merge:
   ```markdown
   [DECISION: Merge Submodule to {default_branch}]
   Would you like to merge this submodule branch to {default_branch}?
   - YES: Merge to {default_branch}
   - NO: Keep changes on feature branch

   Your choice:
   ```

5. {push_instruction}

   If asking about push:
   ```markdown
   [DECISION: Push Submodule to Remote]
   Would you like to push the submodule changes to remote?
   - YES: Push to origin
   - NO: Keep changes local only

   Your choice:
   ```

#### C. Super-repo (Root)
After ALL submodules are committed and merged:
1. Return to super-repo root
2. Stage all changes with `git add -A`
3. Commit all changes with descriptive message

   {commit_style_guidance}

4. {merge_instruction}

   If asking about merge:
   ```markdown
   [DECISION: Merge to {default_branch}]
   Would you like to merge this branch to {default_branch}?
   - YES: Merge to {default_branch}
   - NO: Keep changes on feature branch

   Your choice:
   ```

5. {push_instruction}

   If asking about push:
   ```markdown
   [DECISION: Push to Remote]
   Would you like to push the changes to remote?
   - YES: Push to origin
   - NO: Keep changes local only

   Your choice:
   ```