# Permission Hook Opt-in Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the `permission-request.js` hook an explicit user opt-in instead of being silently installed with all other hooks, addressing issue #503.

**Architecture:** Three coordinated changes: (1) remove the hook from the automatic copy list in `install_basic_hooks()`, (2) add an interactive prompt + CLI flag in `main()` that explains the global effect and lets the user decide, (3) update docs and config template to reflect `enabled: false` by default.

**Tech Stack:** Python 3 (installer), JSON (config), Markdown (docs)

---

### Task 1: Remove permission-request.js from automatic copy list

**Files:**
- Modify: `claude-hooks/install_hooks.py:721-727`

**Step 1: Open the file and locate `install_basic_hooks()`**

The `core_files` list on line 721 contains `"permission-request.js"`. This causes it to be copied unconditionally as part of every basic install.

**Step 2: Remove `permission-request.js` from `core_files`**

Change:
```python
core_files = [
    "session-start.js",
    "session-end.js",
    "memory-retrieval.js",
    "topic-change.js",
    "permission-request.js"
]
```

To:
```python
core_files = [
    "session-start.js",
    "session-end.js",
    "memory-retrieval.js",
    "topic-change.js",
]
```

**Step 3: Verify no other place in `install_basic_hooks()` copies the file**

Search for any other reference to `permission-request` in that method – there are none (confirmed in code review).

**Step 4: Commit**

```bash
git add claude-hooks/install_hooks.py
git commit -m "fix: remove permission-request.js from automatic install list

Addresses #503 - the hook should be opt-in, not silently installed.
File still exists in repo for users who want it."
```

---

### Task 2: Add `install_permission_hook()` method to HookInstaller

**Files:**
- Modify: `claude-hooks/install_hooks.py` (add new method after `install_basic_hooks()`)

**Step 1: Find the end of `install_basic_hooks()` method**

Look for the closing `return True` / `return False` around line ~800. The new method goes right after it.

**Step 2: Add the new method**

Insert after `install_basic_hooks()`:

```python
def install_permission_hook(self) -> bool:
    """Copy permission-request.js to the hooks directory."""
    self.info("Installing permission-request hook...")
    try:
        (self.claude_hooks_dir / "core").mkdir(parents=True, exist_ok=True)
        src = self.script_dir / "core" / "permission-request.js"
        dst = self.claude_hooks_dir / "core" / "permission-request.js"
        if src.exists():
            shutil.copy2(src, dst)
            self.success("permission-request.js installed")
            return True
        else:
            self.error("permission-request.js not found in source directory")
            return False
    except Exception as e:
        self.error(f"Failed to install permission hook: {e}")
        return False
```

**Step 3: Commit**

```bash
git add claude-hooks/install_hooks.py
git commit -m "feat: add install_permission_hook() method for opt-in installation"
```

---

### Task 3: Add `--permission-hook` / `--no-permission-hook` CLI flags

**Files:**
- Modify: `claude-hooks/install_hooks.py:1380-1396` (argparse block in `main()`)

**Step 1: Locate the argparse block in `main()`**

It starts around line 1380 with `parser.add_argument('--basic', ...)`.

**Step 2: Add the two new flags**

After the existing `parser.add_argument('--force', ...)` line, add:

```python
parser.add_argument('--permission-hook', action='store_true', default=None,
                    dest='permission_hook',
                    help='Install the permission-request hook (opt-in, global effect on ALL MCP servers)')
parser.add_argument('--no-permission-hook', action='store_false',
                    dest='permission_hook',
                    help='Skip the permission-request hook installation')
```

Note: Both flags write to `args.permission_hook`. When neither flag is given, `args.permission_hook` is `None` (triggers interactive prompt later).

**Step 3: Commit**

```bash
git add claude-hooks/install_hooks.py
git commit -m "feat: add --permission-hook / --no-permission-hook CLI flags"
```

---

### Task 4: Add interactive opt-in prompt in `main()`

**Files:**
- Modify: `claude-hooks/install_hooks.py` (in `main()`, after the "Determine what to install" block around line 1461)

**Step 1: Locate the "Determine what to install" block**

Around line 1460–1464:
```python
install_all = not (args.basic or args.natural_triggers or args.auto_capture) or args.all
install_basic = args.basic or install_all
install_natural_triggers = args.natural_triggers or install_all
install_auto_capture = args.auto_capture or install_all
```

**Step 2: Add the permission hook decision logic directly after that block**

```python
# Permission hook: explicit opt-in required (issue #503)
if args.permission_hook is True:
    install_permission_hook = True
    installer.info("Permission hook: enabled via --permission-hook flag")
elif args.permission_hook is False:
    install_permission_hook = False
    installer.info("Permission hook: skipped via --no-permission-hook flag")
else:
    # Interactive prompt - default is NO
    installer.header("Optional: Permission Request Hook")
    installer.info("")
    installer.info("This hook auto-approves safe MCP tool calls (read-only operations like")
    installer.info("get, list, retrieve, search) and prompts for destructive ones")
    installer.info("(delete, write, update, etc.), reducing repetitive confirmation dialogs.")
    installer.info("")
    installer.warn("GLOBAL EFFECT: This hook applies to ALL MCP servers on your system,")
    installer.warn("not just the memory service. It will affect every MCP server you use")
    installer.warn("(browser automation, code-context, Context7, and any future servers).")
    installer.info("")
    installer.info("Why it ships with mcp-memory-service:")
    installer.info("  Memory operations are frequent and repetitive by design. This hook")
    installer.info("  was developed alongside the memory service and is tested against its")
    installer.info("  tool naming conventions. A standalone Gist version is also available:")
    installer.info("  https://gist.github.com/doobidoo/fa84d31c0819a9faace345ca227b268f")
    installer.info("")
    try:
        answer = input("  Install permission-request hook? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = ""
    install_permission_hook = answer in ("y", "yes")
    if install_permission_hook:
        installer.success("Permission hook will be installed")
    else:
        installer.info("Permission hook skipped (install later with --permission-hook)")
```

**Step 3: Commit**

```bash
git add claude-hooks/install_hooks.py
git commit -m "feat: add interactive opt-in prompt for permission-request hook

Shows global effect warning and rationale before asking user.
Default answer is N (no install). Supports non-interactive via flags."
```

---

### Task 5: Wire opt-in decision into `configure_claude_settings()`

**Files:**
- Modify: `claude-hooks/install_hooks.py:972` (`configure_claude_settings` signature)
- Modify: `claude-hooks/install_hooks.py:1029-1044` (PreToolUse block inside the method)
- Modify: `claude-hooks/install_hooks.py:1508-1516` (call sites in `main()`)

**Step 1: Update `configure_claude_settings()` signature**

Change:
```python
def configure_claude_settings(self, install_mid_conversation: bool = False, install_auto_capture: bool = False) -> bool:
```
To:
```python
def configure_claude_settings(self, install_mid_conversation: bool = False, install_auto_capture: bool = False, install_permission_hook: bool = False) -> bool:
```

**Step 2: Wrap the PreToolUse block with the new flag**

The existing code around line 1029 is:
```python
# Add PreToolUse hook for MCP permission auto-approval (v8.73.0+)
permission_request_script = self.claude_hooks_dir / 'core' / 'permission-request.js'
if permission_request_script.exists():
    hook_config["hooks"]["PreToolUse"] = [...]
    self.success("Added PreToolUse hook for MCP permission auto-approval")
```

Change to:
```python
# Add PreToolUse hook for MCP permission auto-approval (opt-in only, issue #503)
if install_permission_hook:
    permission_request_script = self.claude_hooks_dir / 'core' / 'permission-request.js'
    if permission_request_script.exists():
        hook_config["hooks"]["PreToolUse"] = [
            {
                "matcher": "mcp__",
                "hooks": [
                    {
                        "type": "command",
                        "command": f'node "{self.claude_hooks_dir}/core/permission-request.js"',
                        "timeout": 5
                    }
                ]
            }
        ]
        self.success("Added PreToolUse hook for MCP permission auto-approval")
    else:
        self.warn("permission-request.js not found, skipping PreToolUse hook")
```

**Step 3: Update the call site in `main()`**

Change:
```python
if not installer.configure_claude_settings(install_mid_conversation=install_natural_triggers,
                                          install_auto_capture=install_auto_capture):
```
To:
```python
if not installer.configure_claude_settings(install_mid_conversation=install_natural_triggers,
                                          install_auto_capture=install_auto_capture,
                                          install_permission_hook=install_permission_hook):
```

**Step 4: Wire `install_permission_hook` into the actual file copy**

After the existing `if install_auto_capture:` block in `main()` (around line 1503), add:

```python
if install_permission_hook:
    if not installer.install_permission_hook():
        overall_success = False
```

**Step 5: Commit**

```bash
git add claude-hooks/install_hooks.py
git commit -m "feat: wire permission hook opt-in through configure_claude_settings()

PreToolUse hook only registered when user explicitly opts in.
install_permission_hook flag propagated through entire install flow."
```

---

### Task 6: Update dry-run output and installation summary

**Files:**
- Modify: `claude-hooks/install_hooks.py:1471-1486` (dry-run block)
- Modify: `claude-hooks/install_hooks.py:1524-1536` (success summary block)

**Step 1: Add permission hook to dry-run output**

In the `if args.dry_run:` block, after the `if install_auto_capture:` section, add:

```python
if install_permission_hook:
    installer.info("  - Permission Request Hook (global: affects ALL MCP servers)")
else:
    installer.info("  - Permission Request Hook: SKIPPED (opt-in, use --permission-hook)")
```

**Step 2: Add permission hook to success summary**

In the success summary block, add a conditional line:

```python
if install_permission_hook:
    installer.info("  ✅ Permission Request Hook (auto-approves safe MCP operations)")
else:
    installer.info("  ℹ  Permission Request Hook not installed (run with --permission-hook to add)")
```

**Step 3: Commit**

```bash
git add claude-hooks/install_hooks.py
git commit -m "fix: update dry-run output and summary to reflect permission hook opt-in"
```

---

### Task 7: Update `config.template.json` – default `enabled: false`

**Files:**
- Modify: `claude-hooks/config.template.json:72-78`

**Step 1: Change `permissionRequest.enabled` to `false`**

Change:
```json
"permissionRequest": {
  "enabled": true,
  "autoApprove": true,
  "customSafePatterns": [],
  "customDestructivePatterns": [],
  "logDecisions": false
}
```
To:
```json
"permissionRequest": {
  "enabled": false,
  "autoApprove": true,
  "customSafePatterns": [],
  "customDestructivePatterns": [],
  "logDecisions": false
}
```

**Step 2: Commit**

```bash
git add claude-hooks/config.template.json
git commit -m "fix: set permissionRequest.enabled default to false in config template"
```

---

### Task 8: Update `README-PERMISSION-REQUEST.md` – add rationale section

**Files:**
- Modify: `claude-hooks/README-PERMISSION-REQUEST.md`

**Step 1: Add a "Why is this in mcp-memory-service?" section**

Add after the "Overview" section and before "Features":

```markdown
## Why is this in mcp-memory-service?

This hook was developed alongside the memory service because memory operations
are frequent and repetitive — retrieving, searching, and listing memories would
otherwise generate constant permission prompts.

The hook is tested against the memory service's tool naming conventions and ships
here for convenience. However, it works universally across **all MCP servers**,
which is also why installation is **opt-in** (see below).

A standalone version is available as a GitHub Gist:
https://gist.github.com/doobidoo/fa84d31c0819a9faace345ca227b268f

## Opt-in Installation

This hook is **not installed automatically**. During `install_hooks.py` you will
be prompted with an explanation of its global effect and asked to confirm.

To install non-interactively:
```bash
python install_hooks.py --permission-hook      # install it
python install_hooks.py --no-permission-hook   # skip it explicitly
```
```

**Step 2: Commit**

```bash
git add claude-hooks/README-PERMISSION-REQUEST.md
git commit -m "docs: add opt-in rationale and global-effect warning to README-PERMISSION-REQUEST"
```

---

### Task 9: Update `claude-hooks/README.md` – mention opt-in

**Files:**
- Modify: `claude-hooks/README.md`

**Step 1: Find the section that describes the permission hook**

Search for `permission` in the file and locate where the hook is listed as a component.

**Step 2: Add an "(opt-in)" note**

Wherever the permission hook is listed (likely in a feature table or bullet list), append `(opt-in)` and a brief note that it affects all MCP servers.

Example diff (adapt to actual content):
```
- permission-request.js — Auto-approve safe MCP operations  →
- permission-request.js — Auto-approve safe MCP operations **(opt-in, global effect)**
```

**Step 3: Commit**

```bash
git add claude-hooks/README.md
git commit -m "docs: mark permission-request hook as opt-in in main README"
```

---

### Task 10: Manual smoke test

**No code changes – verification only.**

**Step 1: Run installer in dry-run mode and confirm permission prompt is shown**

```bash
cd /Users/hkr/GitHub/mcp-memory-service/claude-hooks
python install_hooks.py --dry-run
```

Expected: Dry-run output mentions "Permission Request Hook: SKIPPED" (default).

**Step 2: Run with `--permission-hook` flag in dry-run**

```bash
python install_hooks.py --dry-run --permission-hook
```

Expected: Output shows "Permission Request Hook (global: affects ALL MCP servers)".

**Step 3: Run with `--no-permission-hook` flag in dry-run**

```bash
python install_hooks.py --dry-run --no-permission-hook
```

Expected: Output shows hook as SKIPPED.

**Step 4: Verify interactive path works (simulated)**

```bash
echo "n" | python install_hooks.py --dry-run 2>&1 | grep -i permission
echo "y" | python install_hooks.py --dry-run 2>&1 | grep -i permission
```

Expected: `n` path shows SKIPPED, `y` path shows it would be installed.

---

### Task 11: Create GitHub release via agent

**Step 1: Invoke github-release-manager agent**

After all commits are verified, invoke the `github-release-manager` agent to:
- Determine version bump (patch: `v10.17.15`)
- Update `CHANGELOG.md` with the opt-in change referencing issue #503
- Update `pyproject.toml` and `_version.py`
- Create PR and release

**Do NOT manually bump versions** – use the agent per CLAUDE.md release protocol.
