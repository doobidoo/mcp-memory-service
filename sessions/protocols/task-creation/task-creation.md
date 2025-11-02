# Task Creation Protocol

{todos}

## Creating a Task
Follow these numbered steps to complete each todo above:

### 1: Create task file from template with appropriate priority, type, and structure

#### First, determine task priority
All task files MUST include a priority prefix before the task type:

- `h-` → High priority
- `m-` → Medium priority  
- `l-` → Low priority
- `?-` → Investigate (task may be obsolete, speculative priority)

Examples:
- `h-fix-auth-redirect.md`
- `m-implement-oauth.md`
- `l-docs-api-reference.md`
- `?-research-old-feature.md`

#### Then, choose task type prefix based on the primary goal (comes after priority):

- `implement-` → New functionality (creates feature/ branch)
- `fix-` → Bug fixes, corrections (creates fix/ branch)  
- `refactor-` → Code improvements (creates feature/ branch)
- `research-` → Investigation only (no branch needed)
- `experiment-` → Proof of concepts (creates experiment/ branch)
- `migrate-` → Moving/updating systems (creates feature/ branch)
- `test-` → Adding tests (creates feature/ branch)
- `docs-` → Documentation (creates feature/ branch)

Combine: `[priority]-[type]-[descriptive-name]`

#### Next, decide if task needs file or directory structure

**Use a FILE when**:
- Single focused goal
- Estimated < 3 days work
- No obvious subtasks at creation time
- Examples:
  - `h-fix-auth-redirect.md`
  - `m-research-mcp-features.md`
  - `l-refactor-redis-client.md`

**Use a DIRECTORY when**:
- Multiple distinct phases
- Needs clear subtasks from the start
- Estimated > 3 days work
- Examples:
  - `h-implement-auth/` (magic links + OAuth + sessions)
  - `m-migrate-to-postgres/` (schema + data + cutover)
  - `l-test-all-services/` (per-service test files)

#### For directory tasks, confirm with user

If you determine the task needs directory structure, explicitly confirm:

```markdown
[DECISION: Directory Task Structure]
This task appears complex enough to require directory structure with subtasks.

Using a directory means:
- Creating subtasks will be the first step after task creation
- All work will be done iteratively on the same task branch
- You'll plan and spec out subtasks comprehensively before implementation
- Individual subtask commits won't merge to main until all subtasks complete

Would you like to use directory structure for this task? (yes/no)
```

Only proceed with directory structure if user confirms. If they say no, use a file instead.

#### Propose the task naming to user

Before creating the file, present a structured proposal:

```markdown
[PROPOSAL: Task Name]
Priority: [h/m/l/?]
Type: [implement/fix/refactor/research/etc]
Name: [priority]-[type]-[descriptive-name]
Full path: sessions/tasks/[priority]-[type]-[descriptive-name].md

Structure: [FILE/DIRECTORY]
Rationale: [why file vs directory]

Approve this task naming?
```

#### Finally, create the task file
Once approved, create the file:

For file:
```bash
cp sessions/tasks/TEMPLATE.md sessions/tasks/[priority]-[task-name].md
```
For directory:
```bash
mkdir sessions/tasks/[priority]-[task-name]
cp sessions/tasks/TEMPLATE.md sessions/tasks/[priority]-[task-name]/README.md
```

Then fill out task frontmatter
  - name: Must match filename (including priority prefix)
  - branch: Based on task type (or 'none' for research)
  - status: Start as 'pending'
  - created: Today's date{submodules_field}

### 2: Ask user about task success and propose success criteria

First, ask the user about their vision and propose specific success criteria based on the task:

```markdown
[QUESTION: Task Success]
Based on your requirements, I propose the following success criteria:

□ [Specific measurable criterion based on user input]
□ [Additional criterion I've identified]
□ [Another criterion for completeness]

Would you like to adjust or add to these criteria? What else might need to be true in order for this task to be complete/successful?
```

Once approved, write a clear description of what we're solving/building in Problem/Goal section and record the success criteria with checkboxes in the text file.

### 3: Run context-gathering agent or mark complete

Present the decision to the user:

```markdown
[DECISION: Context Gathering]
Would you like me to run the context-gathering agent now to create a comprehensive context manifest?

- YES: I'll run the agent to analyze the codebase and create context
- NO: We'll skip this for now (must be done during task startup)

Your choice:
```

  - If yes: Use context-gathering agent on sessions/tasks/[priority]-[task-name].md
  - If no: Mark this step complete and continue
  - Context manifest MUST be complete before work begins (if not now, during task startup)

### 4: Update service index files if applicable
  - Check if task relates to any task indexes (sessions/tasks/indexes)
  - If not, present a structured decision:

```markdown
[DECISION: Task Index]
I didn't find any task indexes that fit this task.
Would you like me to create a new index category for this type of task?

- YES: Create new index file
- NO: No index needed

Your choice:
```

  - **If creating a new index**:
    1. Copy the index template: `cp cc-sessions/cc_sessions/templates/INDEX_TEMPLATE.md sessions/tasks/indexes/[index-name].md`
    2. Fill out the frontmatter:
       - `index`: Short identifier (e.g., `auth-oauth`, `mcp`, `user-model`)
       - `name`: Human-readable name (e.g., "Authentication & OAuth")
       - `description`: Brief description of what tasks belong in this index
    3. Add the new task to the appropriate priority section
  - **If using existing index**:
    - Add task to relevant index files under appropriate priority section
    - Use format: `` `[task-filename]` - [brief description]``
    - For directory tasks, append `/` to the filename
  - Skip if no relevant index exists and user declines to create one

### 5: Commit the new task file
- Stage the task file and any updated index files
- Commit with descriptive message about the new task


## Task Evolution

If a file task needs subtasks during work:
1. Create directory with same name
2. Move original file to directory as README.md
3. Add subtask files
4. Update active task reference if needed
  - ex:
  ```json
  {{ "task": "some-task-dir/README.md" }}
  ```
  - ex:
  ```json
  {{ "task": "some-task-dir/some-subtask.md" }}
  ```

## Important Note on Context

The context-gathering agent is responsible for creating a complete, self-contained context manifest. This replaces the deprecated patterns system. If the context proves insufficient during implementation, the agent's prompt should be improved rather than adding workarounds or modifying the context manifest manually.

## Protocol Completion

Once the task file has been created and the context-gathering agent has populated the context manifest:

1. Inform the user that the task has been successfully created
2. Show the task file path: `sessions/tasks/[priority]-[task-name].md`
3. **DO NOT start working on the task** - The task has been created but will remain in 'pending' status
4. The task will not be started now unless the user explicitly asks to begin work on it
5. If the user wants to start the task, they should use the task startup protocol

This completes the task creation protocol. The task is now ready to be started at a future time.
