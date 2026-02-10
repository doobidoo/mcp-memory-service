# CodeQL Security Analysis

## Overview

CodeQL is GitHub's semantic code analysis engine that automatically detects security vulnerabilities and code quality issues in the MCP Memory Service codebase.

## What It Checks

**Security Issues:**
- SQL Injection vulnerabilities
- Command Injection (shell command construction)
- Path Traversal vulnerabilities
- XSS (Cross-Site Scripting) in web templates
- Sensitive data exposure
- Insecure cryptography usage
- Authentication/Authorization flaws
- Regular expression DoS (ReDoS)

**Code Quality Issues:**
- Unused variables and imports
- Dead code
- Type confusion
- Exception handling antipatterns
- Resource leaks

## When It Runs

1. **On every push** to `main` or `develop` branches
2. **On every pull request** to `main` or `develop`
3. **Weekly scheduled scan** (Mondays at 00:00 UTC)

## Configuration

**Workflow:** `.github/workflows/codeql.yml`
**Config:** `.github/codeql/codeql-config.yml`

**Excluded paths** (to reduce false positives):
- `tests/**` - Test code
- `archive/**` - Archived code
- `docs/**` - Documentation
- `scripts/**` - Utility scripts

**Included paths** (always scanned):
- `src/mcp_memory_service/**` - Main source code

## Viewing Results

1. **GitHub Security Tab:**
   https://github.com/doobidoo/mcp-memory-service/security/code-scanning

2. **PR Checks:**
   CodeQL results appear as "CodeQL" check on pull requests

3. **Annotations:**
   Findings appear as inline comments on affected lines

## Severity Levels

- 🔴 **Error (High):** Critical security vulnerabilities - must fix before merge
- 🟡 **Warning (Medium):** Potential issues - review and fix if valid
- 🔵 **Note (Low):** Code quality suggestions - optional improvements

## False Positives

If CodeQL flags a false positive:

1. **Add a comment** explaining why it's safe:
   ```python
   # CodeQL: This is safe because [reason]
   user_input = sanitize(request.GET['param'])
   ```

2. **Suppress in config** (`.github/codeql/codeql-config.yml`):
   ```yaml
   query-filters:
     - exclude:
         id: py/sql-injection
         reason: "All SQL uses parameterized queries"
   ```

3. **Report to GitHub** if it's a bug in CodeQL itself

## Performance

- **Runtime:** ~5-10 minutes per scan
- **Impact:** Runs in parallel, doesn't block other checks
- **Caching:** GitHub caches CodeQL databases for faster reruns

## Troubleshooting

**Scan fails with timeout:**
- Increase `timeout-minutes` in workflow (default: 15)

**Too many false positives:**
- Adjust `paths-ignore` in config
- Switch from `security-and-quality` to `security-only`

**Missing dependencies:**
- CodeQL installs `[sqlite]` dependencies automatically
- Add more in workflow if needed

## Disabling CodeQL

If you need to temporarily disable:

1. **For a single PR:**
   ```yaml
   # In PR description
   skip-codeql: true
   ```

2. **Permanently:**
   Delete `.github/workflows/codeql.yml`

## Resources

- [CodeQL Documentation](https://codeql.github.com/docs/)
- [CodeQL Python Queries](https://github.com/github/codeql/tree/main/python/ql/src/Security)
- [GitHub Code Scanning Docs](https://docs.github.com/en/code-security/code-scanning)
