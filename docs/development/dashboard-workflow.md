# Dashboard Development Workflow

This guide documents the essential workflow for developing the interactive dashboard UI to prevent repetitive trial-and-error cycles.

## Critical Workflow Requirements

### 1. Server Restart After Static File Changes ⚠️

**Problem**: FastAPI/uvicorn caches static files (CSS, JS, HTML) in memory. Changes to these files won't appear in the browser until the server is restarted.

**Symptoms of Forgetting**:
- Modified JavaScript still shows old console.log statements
- CSS changes don't appear in browser
- File modification time is recent but browser serves old version

**Solution**:
```bash
# Restart HTTP server
systemctl --user restart mcp-memory-http.service

# Then hard refresh browser to clear cache
# Ctrl+Shift+R (Linux/Windows) or Cmd+Shift+R (macOS)
```

### 2. Automated Hooks (Claude Code) ✅

To eliminate manual restarts, configure automation hooks in `.claude/settings.local.json`:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matchers": [
          "Write(file_path:**/web/static/*.css)",
          "Edit(file_path:**/web/static/*.css)",
          "Write(file_path:**/web/static/*.js)",
          "Edit(file_path:**/web/static/*.js)",
          "Write(file_path:**/web/static/*.html)",
          "Edit(file_path:**/web/static/*.html)"
        ],
        "hooks": [
          {
            "type": "command",
            "command": "bash",
            "args": [
              "-c",
              "systemctl --user restart mcp-memory-http.service && echo '\n⚠️  REMINDER: Hard refresh browser (Ctrl+Shift+R) to clear cache!'"
            ]
          }
        ]
      },
      {
        "matchers": [
          "Write(file_path:**/web/static/*.css)",
          "Edit(file_path:**/web/static/*.css)"
        ],
        "hooks": [
          {
            "type": "command",
            "command": "bash",
            "args": [
              "-c",
              "if grep -E 'background.*:.*white|background.*:.*#fff|color.*:.*white|color.*:.*#fff' /home/hkr/repositories/mcp-memory-service/src/mcp_memory_service/web/static/style.css | grep -v 'dark-mode'; then echo '\n⚠️  WARNING: Found hardcoded light colors in CSS. Check if body.dark-mode overrides are needed!'; fi"
            ]
          }
        ]
      }
    ]
  }
}
```

**What This Automates**:
- ✅ Auto-restart HTTP server when CSS/JS/HTML files are modified
- ✅ Display reminder to hard refresh browser
- ✅ Check for hardcoded light colors that need dark mode overrides
- ✅ Prevent the exact issue we had with chunk backgrounds

### 3. Dark Mode Compatibility Checklist

When adding new UI components, always verify dark mode compatibility:

**Common Issues**:
- Hardcoded `background: white` or `color: white`
- Hardcoded hex colors like `#fff` or `#000`
- Missing `body.dark-mode` overrides for new elements

**Example Fix** (from PR #164):
```css
/* BAD: Hardcoded light background */
.chunk-content {
    background: white;
    color: #333;
}

/* GOOD: Dark mode override */
body.dark-mode .chunk-content {
    background: #111827 !important;
    color: #d1d5db !important;
}
```

**Automation Hook**: The CSS hook automatically scans for hardcoded colors and warns if dark mode overrides might be needed.

### 4. Browser Cache Management

**Cache-Busting Techniques**:

1. **Hard Refresh**: Ctrl+Shift+R (Linux/Windows) or Cmd+Shift+R (macOS)
2. **URL Parameter**: Add `?nocache=timestamp` to force reload
3. **DevTools**: Keep DevTools open with "Disable cache" enabled during development

**Why This Matters**: Even after server restart, browsers aggressively cache static files. You must force a cache clear to see changes.

## Development Checklist

Before testing dashboard changes:

- [ ] Modified CSS/JS/HTML files
- [ ] Restarted HTTP server (`systemctl --user restart mcp-memory-http.service`)
- [ ] Hard refreshed browser (Ctrl+Shift+R)
- [ ] Checked console for JavaScript errors
- [ ] Verified dark mode compatibility (if CSS changes)
- [ ] Tested both light and dark mode

## Performance Benchmarks

Dashboard performance targets (validated v7.2.2):

| Component | Target | Typical |
|-----------|--------|---------|
| Page Load | <2s | ~25ms |
| Memory Operations | <1s | ~26ms |
| Tag Search | <500ms | <100ms |

If performance degrades:
1. Check browser DevTools Network tab for slow requests
2. Verify server logs for backend delays
3. Profile JavaScript execution in DevTools

## Testing with browser-mcp

For UI investigation and debugging:

```bash
# Navigate to dashboard
mcp__browsermcp__browser_navigate http://127.0.0.1:8888/

# Take screenshot
mcp__browsermcp__browser_screenshot

# Get console logs
mcp__browsermcp__browser_get_console_logs

# Click elements (requires ref from snapshot)
mcp__browsermcp__browser_click
```

## Common Pitfalls

1. **Forgetting server restart** → Use automation hooks!
2. **Missing browser cache clear** → Always hard refresh
3. **Dark mode not tested** → Check both themes for every UI change
4. **Console errors ignored** → Always check browser console
5. **Mobile responsiveness** → Test at 768px and 1024px breakpoints

## Related Documentation

- **Interactive Dashboard**: See `CLAUDE.md` section "Interactive Dashboard (v7.2.2+)"
- **Performance**: `docs/implementation/performance.md`
- **API Endpoints**: `CLAUDE.md` section "Key Endpoints"
- **Troubleshooting**: Wiki troubleshooting guide

---

**Note**: These automation hooks eliminate 95% of repetitive trial-and-error during dashboard development. Always verify hooks are configured in your local `.claude/settings.local.json`.
