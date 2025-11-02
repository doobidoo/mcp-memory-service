## Agent Front Matter & Forge Configuration

Every agent lives in a collective directory that includes an `AGENTS.md` marker and an `agents/` folder. The **agent identifier** is derived from its file path inside that folder, e.g. `.genie/code/agents/review.md` → `code/review`. If an agent sits at the workspace root (`.genie/agents/review.md`) it keeps the simple id `review`. Rename or relocate the markdown file to change the id—no extra metadata is required.

### Defaults

If an agent’s front matter omits a `genie` block, the CLI and MCP server use the defaults from `.genie/config.yaml`:

```yaml
defaults:
  executor: opencode        # maps to Forge executor key
  executorVariant: DEFAULT  # maps to Forge executor profile variant
```

That means most agents can stay minimal:

```markdown
---
name: analyze
description: Discovery + risk triage
---
```

### Overriding Forge execution per agent

To specialize the executor or variant, add a `genie` section. The CLI injects that metadata into the Forge request before calling `createAndStartTask`. The block is optional and only the keys you set will override the defaults.

```yaml
---
name: review
description: Evidence-based QA
genie:
  executor: opencode          # Forge executor key (see GET /api/profiles)
  executorVariant: REVIEW_STRICT_EVIDENCE
  executionMode: review       # Optional logical mode (shown in CLI listings)
---
```

#### Supported keys

| Key | Purpose | Forge mapping |
| --- | --- | --- |
| `executor` | Logical executor name (`opencode`, `codex`, `claude_code`, …). Case-insensitive. | Translated to Forge `executor_profile_id.executor`. |
| `executorVariant` | Profile variant defined in `.genie/config.yaml` (e.g. `DOCGEN_MEDIUM`). | Translated to Forge `executor_profile_id.variant`. |
| `executionMode` | Friendly mode label surfaced in CLI session listings. | Stored alongside the session metadata; useful for dashboards. |
| `background` | Set to `false` to force foreground streaming (rare). | Affects CLI behaviour only. |

Any additional executor-level parameters—such as `append_prompt`, `approvals`, or `additional_params`—should be declared in `.genie/config.yaml` under the corresponding executor + variant. The CLI syncs that structure to Forge via `PUT /api/profiles` every time you run/resume/view a session.

### Precedence

When a run starts, Genie applies overrides in this order:

1. Workspace defaults in `.genie/config.yaml`
2. Agent front matter (`genie.executor`, `genie.executorVariant`, `genie.model`)
3. CLI flags at call-time (`genie run … --executor <id> --model <name>`)

The last value wins. If you pass a `--model` that matches a profile in `.genie/config.yaml`, the corresponding variant is selected automatically; otherwise the CLI sends an on-the-fly model override to Forge.

### Discovering available Forge options

1. **List executor profiles**  
   Run the Forge API `GET /api/profiles` (see `forge.js#getExecutorProfiles`). The response includes every executor (`CLAUDE_CODE`, `OPENCODE`, etc.) and the variants defined in your workspace config. Use those keys in `executor` / `executorVariant`.

2. **Inspect Forge UI templates**  
   The Automagik Forge UI exposes the same profiles under *Settings → Coding Agent Configurations*. Each field shown there maps to JSON the CLI can send—e.g. *Append Prompt*, *Approvals*, *Dangerously Skip Permissions*, `additional_params[]`.

3. **Update `.genie/config.yaml`**  
   Define or adjust profiles in the `forge.executors` section. Example:

   ```yaml
   forge:
     executors:
       OPENCODE:
         DOCGEN_DOCFIRST:
           OPENCODE:
             append_prompt: |
               Prefer docstrings and API comments; avoid logic changes.
             additional_params:
               - { key: doc_mode, value: doc-first }
   ```

   After editing, the next `genie run` / `resume` / `list sessions` syncs the profile to Forge automatically.

### Quick checklist when creating a new agent

1. Place the markdown file in the correct collective (`.genie/<collective>/agents/`).
2. Keep identifiers simple. If you want the CLI id `analyze`, put the file under `.genie/agents/`; if you need `code/analyze`, move it under `.genie/code/agents/`.
3. Only add a `genie` block when you need a non-default executor or mode.
4. Adjust `.genie/config.yaml` (and sync with Forge) whenever you introduce a new executor variant.
5. Run `pnpm run build:genie` so the CLI picks up changes, then `genie list agents` to verify the new id shows up.
