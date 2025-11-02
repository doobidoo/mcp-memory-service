**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`

---
name: qa/cli-commands
description: CLI command interface testing via npx automagik-genie
parent: qa
---

# QA Workflow • CLI Commands

## Test Scenario
Validate CLI command interface, argument parsing, help system, and error handling.

## Prerequisites
```bash
# Install package globally or use npx
npm install -g automagik-genie@next
# OR use npx automagik-genie for all commands
```

## Test Suite

### Version Display
**Command:**
```bash
npx automagik-genie --version
```

**Expected Evidence:**
- Version number matches package.json
- Format: vX.Y.Z or vX.Y.Z-rc.N
- No errors

**Verification:**
- [ ] Version displays correctly
- [ ] Matches package.json version
- [ ] Clean output

---

### Help System Completeness
**Command:**
```bash
npx automagik-genie --help
```

**Expected Evidence:**
- All commands listed with descriptions
- Usage examples provided
- Options documented

**Verification:**
- [ ] Help text complete
- [ ] All commands documented
- [ ] Examples clear

---

### Help System Accuracy
**Command:**
```bash
# Test each command's help
npx automagik-genie run --help
npx automagik-genie list --help
npx automagik-genie view --help
```

**Expected Evidence:**
- Help text matches actual command behavior
- Options accurately described
- No outdated information

**Verification:**
- [ ] Help matches behavior
- [ ] Options accurate
- [ ] No misleading information

---

### Valid Argument Parsing
**Command:**
```bash
npx automagik-genie run --agent genie --prompt "test prompt"
```

**Expected Evidence:**
- Command executes correctly
- Arguments parsed as expected
- Agent launches with prompt

**Verification:**
- [ ] Arguments parsed correctly
- [ ] Command executes
- [ ] No parsing errors

---

### Invalid Argument Handling
**Command:**
```bash
npx automagik-genie run --invalid-flag value
```

**Expected Evidence:**
- Clear error message
- Usage hint provided
- Lists valid options

**Verification:**
- [ ] Error message clear
- [ ] Suggests correction
- [ ] No crash

---

### Missing Required Argument
**Command:**
```bash
npx automagik-genie run --agent genie
# Missing --prompt
```

**Expected Evidence:**
- Error identifies missing --prompt
- Shows required arguments
- Example usage provided

**Verification:**
- [ ] Missing argument identified
- [ ] Error clear and helpful
- [ ] User knows what to add

---

### Boolean Flag Handling
**Command:**
```bash
npx automagik-genie view --session-id <id> --full
```

**Expected Evidence:**
- --full flag parsed as boolean true
- Full transcript returned
- Flag toggles behavior correctly

**Verification:**
- [ ] Boolean flag works
- [ ] Behavior changes correctly
- [ ] No value required

---

### Value Flag Parsing
**Command:**
```bash
npx automagik-genie run --agent genie --prompt "test" --model opus
```

**Expected Evidence:**
- --model flag value parsed
- Agent uses specified model
- Value applied correctly

**Verification:**
- [ ] Value flag parsed
- [ ] Value used correctly
- [ ] Format validation works

---

### Command Aliases (if supported)
**Command:**
```bash
# Test short forms if available
npx automagik-genie r --agent genie --prompt "test"
# vs
npx automagik-genie run --agent genie --prompt "test"
```

**Expected Evidence:**
- Aliases work identically to full commands
- No behavior difference
- Both forms documented

**Verification:**
- [ ] Aliases functional
- [ ] Behavior identical
- [ ] Documented in help

---

### Special Character Handling
**Command:**
```bash
npx automagik-genie run --agent genie --prompt "test with 'quotes' and \"double quotes\""
```

**Expected Evidence:**
- Special characters handled correctly
- Prompt delivered intact
- No parsing corruption

**Verification:**
- [ ] Special chars preserved
- [ ] No escaping issues
- [ ] Prompt delivered correctly

---

### Long String Arguments
**Command:**
```bash
npx automagik-genie run --agent genie --prompt "$(cat long-prompt.txt)"
```

**Expected Evidence:**
- Long strings accepted
- No truncation
- Full content delivered

**Verification:**
- [ ] Long strings work
- [ ] No truncation
- [ ] Content integrity maintained

---

### Command Chaining (if supported)
**Command:**
```bash
npx automagik-genie run --agent genie --prompt "test" && npx automagik-genie list
```

**Expected Evidence:**
- Commands execute sequentially
- Second command waits for first
- Both complete successfully

**Verification:**
- [ ] Chaining works
- [ ] Sequential execution
- [ ] Both commands succeed

---

## Error Scenarios

### Unknown Command
**Command:**
```bash
npx automagik-genie unknown-command
```

**Expected Evidence:**
- "Unknown command" error
- Lists available commands
- Helpful suggestion

**Verification:**
- [ ] Error clear
- [ ] Commands listed
- [ ] User guided to solution

---

### Empty Prompt
**Command:**
```bash
npx automagik-genie run --agent genie --prompt ""
```

**Expected Evidence:**
- Error or prompt for input
- Clear message about empty prompt
- No crash or hang

**Verification:**
- [ ] Empty prompt handled
- [ ] Clear error message
- [ ] System stable

---

### Invalid Agent Name
**Command:**
```bash
npx automagik-genie run --agent nonexistent --prompt "test"
```

**Expected Evidence:**
- Error lists available agents
- Clear message
- No crash

**Verification:**
- [ ] Error helpful
- [ ] Agent list shown
- [ ] Graceful failure

---

## Execution Notes

**Run this workflow via:**
```
mcp__genie__run with agent="qa/cli-commands" and prompt="Execute all CLI command tests"
```

**Evidence capture:**
- Save terminal outputs to `.genie/qa/evidence/cli-commands-<timestamp>.txt`
- Screenshot any UI elements if applicable
- Document exit codes for error cases
