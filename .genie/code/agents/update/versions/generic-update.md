# Generic Update Guide
**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`
**Purpose:** Guidance for users upgrading from older versions without specific transition guides
**Applies to:** Any version transition not covered by specific guides

---

## 📊 General Update Process

You've successfully updated to a newer version of Genie! Since your previous version is significantly older, a specific transition guide isn't available. This generic guide will help you through the update.

---

## 💾 Your Backup

Your previous configuration has been safely backed up at:
```
.genie/backups/{BACKUP_ID}/
├── genie/           # Entire .genie directory
└── docs/            # Root documentation files
    ├── AGENTS.md    (if existed)
    └── CLAUDE.md    (if existed)
```

**All your previous work is safe.** You can reference these files at any time.

---

## 🎯 What Happened During Update

1. **Backup Created** - Your old configuration was saved
2. **Fresh Install** - New framework templates were installed
3. **Version Updated** - `.genie/state/version.json` now reflects new version

**Important:** The update installed clean templates. If you had customizations in:
- `AGENTS.md` - Framework behavioral patterns
- `CLAUDE.md` - Project-specific patterns
- `.genie/agents/` - Custom agent files

These are now in your backup, NOT in the current installation.

---

## ✅ Post-Update Checklist

### 1. Verify Installation
```bash
# Check new version
cat .genie/state/version.json

# List available agents
genie list

# Test basic agent
genie run plan "test prompt"
```

**Expected:** All commands work without errors

### 2. Review Your Backup

If you had customizations, review what you had:

**Check AGENTS.md customizations:**
```bash
# View your old AGENTS.md
cat .genie/backups/{BACKUP_ID}/docs/AGENTS.md
```

**Check CLAUDE.md customizations:**
```bash
# View your old CLAUDE.md
cat .genie/backups/{BACKUP_ID}/docs/CLAUDE.md
```

**Check custom agents:**
```bash
# List your old agents
ls -la .genie/backups/{BACKUP_ID}/genie/agents/
```

### 3. Decide What to Preserve

**For each customization you want to keep:**
1. Open the current (new) file
2. Open the backup file side-by-side
3. Manually copy over your important customizations
4. Save and test

**Tips:**
- Don't copy entire files (framework has evolved)
- Copy specific sections or entries that are YOUR work
- Test after each change

### 4. Test Your Workflow

After merging customizations:
```bash
# Test agents
genie run plan "create feature X"
genie run wish "implement feature Y"
genie run forge "execute task Z"

# Test background execution
genie run genie "analyze project" --background

# Check MCP integration
genie status
```

---

## 🚨 Common Issues

### Issue: Agent doesn't work as expected
**Possible causes:**
- Framework changed agent structure
- Customization conflicts with new version
- Missing dependencies

**Solutions:**
1. Check CHANGELOG.md for breaking changes
2. Review new agent templates in `.genie/agents/`
3. Test without customizations first
4. Incrementally add back customizations

### Issue: MCP connection errors
**Possible causes:**
- Configuration changed between versions
- MCP server needs restart

**Solutions:**
```bash
# Reconfigure MCP
genie init --provider [codex|claude-code]

# Restart your code editor
# (to pick up new MCP configuration)
```

### Issue: Custom agents not found
**Possible causes:**
- Agent directory structure changed
- Custom agents not migrated

**Solutions:**
1. Check backup: `.genie/backups/{BACKUP_ID}/genie/agents/`
2. Review new agent structure in `.genie/agents/`
3. Re-create custom agents following new structure
4. Test: `genie run {custom-agent} "test"`

---

## 📚 Where to Find Help

### 1. Check the Changelog
```bash
cat CHANGELOG.md
```
Look for breaking changes and new features between your old version and current version.

### 2. Review Documentation
```bash
cat README.md
```
Updated documentation reflects current version capabilities.

### 3. Compare Agent Templates
```bash
# Current agents
ls -la .genie/agents/

# Your old agents (backup)
ls -la .genie/backups/{BACKUP_ID}/genie/agents/

# Compare structure
diff -r .genie/agents/ .genie/backups/{BACKUP_ID}/genie/agents/
```

### 4. GitHub Issues
If you encounter problems:
- Check: https://github.com/namastexlabs/automagik-genie/issues
- Report: Create new issue with version details

---

## 🔄 Rollback Instructions

If the new version isn't working and you need to revert:

```bash
# List available backups
ls -la .genie/backups/

# Rollback to your previous version
genie rollback --id {BACKUP_ID}

# Or rollback to latest backup
genie rollback --latest
```

**Note:** Rollback restores:
- Entire `.genie/` directory
- Root `AGENTS.md` file (if existed)
- Root `CLAUDE.md` file (if existed)

After rollback, you'll be back to your previous working state.

---

## 🎯 General Migration Strategy

For major version jumps:

1. **Backup verified** ✅
2. **New version tested** (basic workflows)
3. **Review customizations** (from backup)
4. **Selective migration** (important parts only)
5. **Incremental testing** (after each change)
6. **Document what you preserved** (for future updates)

---

## 📖 Understanding Update Philosophy

**Genie's update approach:**
- ✅ **Clean install** - Fresh framework templates
- ✅ **Safe backup** - Your work is preserved
- ✅ **Manual merge** - You control what to keep
- ❌ **No auto-merge** - No automated file merging

**Why this approach?**
- Prevents merge conflicts
- Ensures framework integrity
- Gives you full control
- Safer than automated merging

**Your responsibility:**
- Review backup if you had customizations
- Manually re-apply important changes
- Test after migration

---

## 🎓 Best Practices

### For Future Updates

1. **Document your customizations** - Keep notes on what you changed
2. **Minimize customizations** - Only customize what's necessary
3. **Regular updates** - Don't skip too many versions
4. **Test updates** - Try in test project first if possible

### Customization Strategy

- **AGENTS.md**: Add learning entries, don't modify core structure
- **CLAUDE.md**: Add project-specific patterns, don't remove framework refs
- **Custom agents**: Store in clearly marked sections/files
- **Version control**: Track your customizations in git

---

## ✅ Verification Complete

When you've completed migration:

- [ ] New version tested and working
- [ ] Backup reviewed for customizations
- [ ] Important customizations manually merged
- [ ] All workflows tested (plan, wish, forge, review)
- [ ] MCP integration verified
- [ ] Documentation updated (if needed)

**You're ready to use the new version! 🧞**

---

## 📞 Need More Help?

For version-specific guidance:
1. Check `.genie/agents/update/versions/` for your version
2. Read CHANGELOG.md for detailed version history
3. Visit GitHub Issues for known problems
4. Create issue if you found a bug

**Update successful! Welcome to the latest Genie! 🎉**
