# Release Checklist

This checklist ensures that critical bugs like the HTTP-MCP bridge issues are caught before release.

## Pre-Release Testing

### ✅ Core Functionality Tests
- [ ] **Health Check Endpoints**
  - [ ] `/api/health` returns 200 with healthy status
  - [ ] `/health` returns 404 (wrong endpoint)
  - [ ] Health check works through MCP bridge
  - [ ] Health check works with Claude Desktop

- [ ] **Memory Storage Operations**  
  - [ ] Store memory returns HTTP 200 with `success: true`
  - [ ] Duplicate detection returns HTTP 200 with `success: false`
  - [ ] Invalid requests return appropriate error codes
  - [ ] All operations work through MCP bridge

- [ ] **API Endpoint Consistency**
  - [ ] All endpoints use `/api/` prefix correctly
  - [ ] URL construction doesn't break base paths
  - [ ] Bridge correctly appends paths to base URL

### ✅ HTTP-MCP Bridge Specific Tests
- [ ] **Status Code Handling**
  - [ ] Bridge accepts HTTP 200 responses (not just 201)
  - [ ] Bridge checks `success` field for actual result
  - [ ] Bridge handles both success and failure in 200 responses
  
- [ ] **URL Construction**
  - [ ] Bridge preserves `/api` base path in URLs
  - [ ] `new URL()` calls don't replace existing paths
  - [ ] All API calls reach correct endpoints

- [ ] **MCP Protocol Compliance**
  - [ ] `initialize` method works
  - [ ] `tools/list` returns all tools
  - [ ] `tools/call` executes correctly
  - [ ] Error responses are properly formatted

### ✅ End-to-End Testing
- [ ] **Claude Desktop Integration**
  - [ ] Memory storage through Claude Desktop works
  - [ ] Memory retrieval through Claude Desktop works
  - [ ] Health checks show healthy status
  - [ ] No "unhealthy" false positives

- [ ] **Remote Server Testing**
  - [ ] Bridge connects to remote server correctly
  - [ ] Authentication works with API keys
  - [ ] All operations work across network
  - [ ] SSL certificates are handled properly

### ✅ Contract Validation
- [ ] **API Response Formats**
  - [ ] Memory storage responses match documented format
  - [ ] Health responses match documented format
  - [ ] Error responses match documented format
  - [ ] Search responses match documented format

- [ ] **Backward Compatibility**
  - [ ] Existing configurations continue to work
  - [ ] No breaking changes to client interfaces
  - [ ] Bridge supports both HTTP 200 and 201 responses

## Automated Testing Requirements

### ✅ Unit Tests
- [ ] HTTP-MCP bridge unit tests pass
- [ ] Mock server responses are realistic
- [ ] All edge cases are covered
- [ ] Error conditions are tested

### ✅ Integration Tests  
- [ ] Bridge-server integration tests pass
- [ ] Contract tests validate API behavior
- [ ] End-to-end MCP protocol tests pass
- [ ] Real server connectivity tests pass

### ✅ CI/CD Pipeline
- [ ] Bridge tests run on every commit
- [ ] Tests block merges if failing
- [ ] Contract validation passes
- [ ] Multiple Node.js versions tested

## Manual Testing Checklist

### ✅ Critical User Paths
1. **Claude Desktop User**:
   - [ ] Install and configure Claude Desktop with MCP Memory Service
   - [ ] Store a memory using Claude Desktop
   - [ ] Retrieve memories using Claude Desktop  
   - [ ] Verify health check shows healthy status
   - [ ] Confirm no "unhealthy" warnings appear

2. **Remote Server User**:
   - [ ] Configure bridge to connect to remote server
   - [ ] Test memory operations work correctly
   - [ ] Verify all API endpoints are reachable
   - [ ] Confirm authentication works

3. **API Consumer**:
   - [ ] Test direct HTTP API calls work
   - [ ] Verify response formats match documentation
   - [ ] Test error conditions return expected responses

### ✅ Platform Testing
- [ ] **Windows**: Bridge works with Windows Claude Desktop
- [ ] **macOS**: Bridge works with macOS Claude Desktop  
- [ ] **Linux**: Bridge works with Linux installations

## Code Quality Checks

### ✅ Code Review Requirements
- [ ] All HTTP status code assumptions documented
- [ ] URL construction logic reviewed
- [ ] Error handling covers all scenarios
- [ ] No hardcoded endpoints or assumptions

### ✅ Documentation Updates
- [ ] API contract documentation updated
- [ ] Bridge usage documentation updated
- [ ] Troubleshooting guides updated
- [ ] Breaking changes documented

## Release Process

### ✅ Version Management (3-File Procedure)
- [ ] **Update `src/mcp_memory_service/__init__.py`**
  - [ ] Update `__version__` string (e.g., `"8.17.0"`)
  - [ ] Verify version format follows semantic versioning (MAJOR.MINOR.PATCH)

- [ ] **Update `pyproject.toml`**
  - [ ] Update `version` field in `[project]` section
  - [ ] Ensure version matches `__init__.py` exactly

- [ ] **Lock dependencies**
  - [ ] Run `uv lock` to update `uv.lock` file
  - [ ] Commit all three files together in version bump commit

- [ ] **Semantic Versioning Rules**
  - [ ] MAJOR: Breaking changes (API changes, removed features)
  - [ ] MINOR: New features (backward compatible)
  - [ ] PATCH: Bug fixes (no API changes)

### ✅ CHANGELOG Quality Gates
- [ ] **Format Validation**
  - [ ] Follows [Keep a Changelog](https://keepachangelog.com/) format
  - [ ] Version header includes date: `## [8.17.0] - 2025-11-04`
  - [ ] Changes categorized: Added/Changed/Fixed/Removed/Deprecated/Security

- [ ] **Content Requirements**
  - [ ] All user-facing changes documented
  - [ ] Breaking changes clearly marked with **BREAKING**
  - [ ] Performance improvements include metrics (e.g., "50% faster")
  - [ ] Bug fixes reference issue numbers (e.g., "Fixes #123")
  - [ ] Technical details for maintainers in appropriate sections

- [ ] **Migration Guidance** (if breaking changes)
  - [ ] Before/after code examples provided
  - [ ] Environment variable changes documented
  - [ ] Database migration scripts linked
  - [ ] Deprecation timeline specified

### ✅ GitHub Workflow Verification
- [ ] **All Workflows Pass** (check Actions tab)
  - [ ] Docker Publish workflow (builds multi-platform images)
  - [ ] Publish and Test workflow (PyPI publish + installation tests)
  - [ ] HTTP-MCP Bridge Tests (validates MCP protocol compliance)
  - [ ] Platform Tests (macOS/Windows/Linux matrix)

- [ ] **Docker Images Built**
  - [ ] `mcp-memory-service:latest` tag updated
  - [ ] `mcp-memory-service:v8.x.x` version tag created
  - [ ] Multi-platform images (linux/amd64, linux/arm64)

- [ ] **PyPI Package Published**
  - [ ] Package available at https://pypi.org/project/mcp-memory-service/
  - [ ] Installation test passes: `pip install mcp-memory-service==8.x.x`

### ✅ Git Tag and Release
- [ ] **Create annotated Git tag**
  ```bash
  git tag -a v8.x.x -m "Release v8.x.x: Brief description"
  ```
  - [ ] Tag follows `vMAJOR.MINOR.PATCH` format
  - [ ] Tag message summarizes key changes

- [ ] **Push tag to remote**
  ```bash
  git push origin v8.x.x
  ```
  - [ ] Tag triggers release workflows

- [ ] **Create GitHub Release**
  - [ ] Title: `vx.x.x - Short Description`
  - [ ] Body: Copy relevant CHANGELOG section
  - [ ] Mark as pre-release if RC version
  - [ ] Attach any release artifacts (if applicable)

### ✅ Post-Release Issue Closure
- [ ] **Review Fixed Issues**
  - [ ] Search for issues closed by commits in this release
  - [ ] Verify each issue is actually resolved

- [ ] **Close Issues with Context**
  ```markdown
  Resolved in v8.x.x via #PR_NUMBER

  [Link to CHANGELOG entry]
  [Link to relevant Wiki page if applicable]

  Thank you for reporting this issue!
  ```
  - [ ] Include PR link for traceability
  - [ ] Reference CHANGELOG section
  - [ ] Tag issues with `released` label

- [ ] **Update Related Documentation**
  - [ ] Wiki pages updated with new features/fixes
  - [ ] Troubleshooting guides reflect resolved issues
  - [ ] FAQ updated if new common questions emerged

### ✅ Communication
- [ ] Release notes highlight critical fixes
- [ ] Breaking changes clearly documented
- [ ] Migration guide provided if needed
- [ ] Users notified of important changes

## Post-Release Monitoring

### ✅ Health Monitoring
- [ ] Monitor for increased error rates
- [ ] Watch for "unhealthy" status reports
- [ ] Track Claude Desktop connectivity issues
- [ ] Monitor API endpoint usage patterns

### ✅ User Feedback
- [ ] Monitor GitHub issues for reports
- [ ] Check community discussions for problems
- [ ] Respond to user reports quickly
- [ ] Document common issues and solutions

---

## Lessons from HTTP-MCP Bridge Bug

**Critical Mistakes to Avoid:**
1. **Never assume status codes** - Always test against actual server responses
2. **Test critical components** - If users depend on it, it needs comprehensive tests
3. **Validate URL construction** - `new URL()` behavior with base paths is tricky
4. **Document actual behavior** - API contracts must match reality, not hopes
5. **Test end-to-end flows** - Unit tests alone miss integration problems

**Required for Every Release:**
- [ ] HTTP-MCP bridge tested with real server
- [ ] All assumptions about server behavior validated
- [ ] Critical user paths manually tested
- [ ] API contracts verified against implementation

**Emergency Response Plan:**
- If critical bugs are found in production:
  1. Create hotfix branch immediately
  2. Write failing test that reproduces the bug
  3. Fix bug and verify test passes
  4. Release hotfix within 24 hours
  5. Post-mortem to prevent similar issues

---

## Rollback Procedure

### ✅ Emergency Rollback (if release breaks production)

**When to Rollback:**
- Critical functionality broken (storage, retrieval, MCP protocol)
- Data corruption risk identified
- Security vulnerability introduced
- Widespread user-reported failures

**Rollback Steps:**

1. **Immediate Actions**
   - [ ] Create GitHub issue documenting the problem
   - [ ] Tag issue with `critical`, `rollback-needed`
   - [ ] Notify users via GitHub Discussions/Release notes

2. **Docker Rollback**
   ```bash
   # Tag previous version as latest
   git checkout vPREVIOUS_VERSION
   docker build -f tools/docker/Dockerfile -t mcp-memory-service:latest .
   docker push mcp-memory-service:latest
   ```
   - [ ] Verify previous Docker image works
   - [ ] Update documentation to reference previous version

3. **PyPI Rollback** (yank bad version)
   ```bash
   # Yank the broken version (keeps it available but discourages use)
   pip install twine
   twine yank mcp-memory-service==8.x.x
   ```
   - [ ] Yank version on PyPI
   - [ ] Publish notice in release notes

4. **Git Tag Management**
   - [ ] Keep the bad tag for history (don't delete)
   - [ ] Create new hotfix tag (e.g., `v8.x.x+1`) with fix
   - [ ] Mark GitHub Release as "This release has known issues - use v8.x.x-1 instead"

5. **User Communication**
   - [ ] Post issue explaining problem and rollback
   - [ ] Update README with rollback instructions
   - [ ] Pin issue to repository
   - [ ] Post in Discussions with migration path

6. **Post-Rollback Analysis**
   - [ ] Document what went wrong in post-mortem
   - [ ] Add regression test to prevent recurrence
   - [ ] Update this checklist with lessons learned
   - [ ] Review release testing procedures

**Recovery Timeline:**
- Hour 1: Identify issue, create GitHub issue, begin rollback
- Hour 2-4: Complete rollback, verify previous version works
- Hour 4-24: Investigate root cause, prepare hotfix
- Day 2: Release hotfix with comprehensive tests
- Week 1: Post-mortem, update testing procedures

---

This checklist must be completed for every release to prevent critical bugs from reaching users.