**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`

---
name: Forge API Integration Guidance
description: Canonical rules for synchronising Genie agent metadata with Automagik Forge (executor profiles, templates, sessions)
category: integrations
genie:
  background: false
---

# Forge API – Genie Integration Skill

This skill captures the working knowledge required to mirror Genie’s `.md` agent definitions into Automagik Forge reliably. Treat it as the definitive playbook—update it whenever Forge semantics evolve.

## 1. Executor Profiles (`/api/profiles`)
- **Endpoint:** `GET /api/profiles` returns an object with `executors` mapping executor keys to profile variants.
- **Update constraints:**
  - Forge rejects top-level strings; the payload MUST be `{"executors": {...}}`.
  - Variants are stored under upper-case keys (`DEFAULT`, `QA_CHECKLIST`, etc.). Store everything upper-case to avoid mismatches.
  - Valid knob names (observed 2025‑10‑20): `append_prompt`, `model`, `model_reasoning_effort`, `sandbox`, `additional_params`, `allow_all_tools`, `dangerously_skip_permissions`, `dangerously_allow_all`, `plan`, `approvals`, `force`, `yolo`.
  - `append_prompt` exists even when the UI omits it; populate it explicitly when we need prompt suffixes.
  - Example (adds `QA_CHECKLIST` variant for `OPENCODE`):
    ```json
    PUT /api/profiles
    {
      "executors": {
        "OPENCODE": {
          "DEFAULT": { "OPENCODE": { "append_prompt": null } },
          "QA_CHECKLIST": {
            "OPENCODE": {
              "append_prompt": "## QA Automation Checklist Mode",
              "additional_params": [
                { "key": "playbook", "value": "qa-automation-checklist" },
                { "key": "evidence_mode", "value": "strict" }
              ]
            }
          }
        },
        "...": {}
      }
    }
    ```
- **CLI impact:** Agents can specify a variant via front-matter (`genie.executorProfile: QA_CHECKLIST`). `genie run --executor opencode` will push `{ executor: "OPENCODE", variant: "QA_CHECKLIST" }` to Forge.

## 2. Task Templates (`/api/templates`)
- Templates are simple `{template_name, title, description, project_id}` records. Description is free-form markdown/plain text.
- Use them to surface Genie instructions inside Forge’s UI; they do not control execution or models.
- Example sync:
  ```ts
  const templateBody = fs.readFileSync('.genie/create/agents/wish.md', 'utf8');
  await forge.createTaskTemplate({
    template_name: 'genie-wish-qa-codex',
    title: 'Genie Wish: QA Codex Automation Checklist',
    description: templateBody,
    project_id: null
  });
  ```
- Remember: this only mirrors content. Execution still depends on executor profiles / Genie front-matter.

## 3. Sessions
- Forge session creation expects `{ executor_profile_id: { executor, variant } }`. `variant` must match one of the profile keys (defaults to `DEFAULT`).
- Genie session metadata stores both `executor` and `executorVariant`; ensure we set both when forging sessions (fallbacks removed).

## 4. Best Practices & Lessons
- 🔁 **Roundtrip test before mutating profiles:** Slam the existing `profiles.content` into `PUT /api/profiles` to verify format, then mutate.
- 🪪 **Keep history:** Save every API interaction log in `.genie/qa/evidence/forge-api-report-YYYYMMDDHHMM.md`.
- 📜 **Front-matter contract:** Every agent that declares `genie.executor` SHOULD also declare the matching Forge variant if it is not `DEFAULT`.
- 🧩 **Future work:** Consider scripted export/import (CLI verb) to sync collectives → Forge templates & profile variants automatically.
