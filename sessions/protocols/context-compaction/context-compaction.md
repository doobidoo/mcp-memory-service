# Context Compaction Protocol

When context window reaches high capacity, you will be instructed to perform these steps:

```markdown
[STATUS: Context Compaction]
Context window above 85% capacity
Initiating maintenance agents for context preservation...
```

{todos}

## 1. Run Maintenance Agents

Before compacting, delegate to agents:

1. **logging agent** - Update work logs in task file
   ```markdown
   [RUNNING: Logging Agent]
   Consolidating work logs and updating task documentation...
   ```
   - Automatically receives full conversation context
   - Logs work progress and updates task status

   After completion:
   ```markdown
   ✓ Complete
   ```

2. **context-refinement agent** - Check for discoveries/drift
   ```markdown
   [RUNNING: Context-Refinement Agent]
   Checking for new discoveries or context drift...
   ```
   - Reads transcript files automatically
   - Will update context ONLY if changes found
   - Skip if task is complete

   After completion:
   ```markdown
   ✓ Complete - [No updates needed / Context manifest updated]
   ```

3. **service-documentation agent** - Update CLAUDE.md files
   ```markdown
   [RUNNING: Service-Documentation Agent]
   Updating service documentation if needed...
   ```
   - Only if service interfaces changed significantly
   - Include list of modified services

   After completion:
   ```markdown
   ✓ Complete - [No updates needed / Updated X service files]
   ```

## 2. Completion Summary

After all agents complete, report status:

```markdown
[COMPLETE: Context Compaction]
✓ Work logs consolidated
✓ Context manifest [updated/current]
✓ Service documentation [updated/current]

Ready to continue with fresh context window.
```

## Note on Context Refinement

The context-refinement agent is speculative - it will only update the context manifest if genuine drift or new discoveries occurred. This prevents unnecessary updates while ensuring important findings are captured.
