# Task Completion Protocol

When a task meets its success criteria:

{todos}

{git_add_warning}

## 1. Pre-Completion Checks

Verify before proceeding:

```markdown
[STATUS: Pre-Completion Checks]
✓ All success criteria checked off in task file
✓ No unaddressed work remaining

Ready to proceed with task completion.
```

If any checks fail, stop and address the remaining work first.

{directory_completion_check}

## 2-4. Run Completion Agents

Delegate to specialized agents in this order:
```
1. code-review agent - Review all implemented code for security/quality
   Include: Changed files, task context, implementation approach, complete explanation of threat model (avoids ridiculous suggestions like "Critical: command injection vulnerability" when the environment is the shell and the command writer is the user who already has access to their own shell and can run dangerous commands if they truly wanted to)
   **IMPORTANT**: After code-review completes, report findings using this format:

```markdown
[FINDINGS: Code Review]
The code review agent has completed its analysis:

Critical Issues:
□ [None found / Description of critical issues]

Warnings:
□ [Description of any warnings]

Suggestions:
□ [Optional improvements identified]

Would you like to address any of these findings before completing the task?
- YES: We'll fix the issues first
- NO: Proceed with task completion

Your choice:
```

   - Wait for user confirmation before proceeding
   
2. service-documentation agent - Update CLAUDE.md files 
   Include: List of services modified during task
   
3. logging agent - Finalize task documentation
   Include: Task completion summary, final status
```

## 5. Update Index Files

Before archiving the task:
1. Check all index files in `sessions/tasks/indexes/`
2. For each index that contains this task:
   - Move the task entry from the appropriate priority section under "Active Tasks"
   - Add it to the "Completed Tasks" section
   - Keep the same format: `` `[task-filename]` - [brief description]``
3. If no indexes contain the task, skip this step

## 6. Task Archival

After updating indexes:
```bash
# Update task file 'status' to 'completed' (do not add any fields)
# Move to done/ directory
mv sessions/tasks/[priority]-[task-name].md sessions/tasks/done/
# or for directories:
mv sessions/tasks/[priority]-[task-name]/ sessions/tasks/done/
```

## 4. Git Operations (Commit & Merge)

**NOTE**: Do not commit until the task file is marked complete and moved to done/. This ensures the completed task file is included in its final location.

{staging_instructions}

{commit_instructions}

## Important Notes

- NEVER skip the agent steps - they maintain system integrity
- Task files in done/ serve as historical record
- Completed experiments should document learnings even if code is discarded
- If task is abandoned incomplete, document why in task file before archiving
