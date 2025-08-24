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

### ✅ Version Management
- [ ] Version numbers updated in all files
- [ ] Changelog includes all critical fixes
- [ ] Git tags created with release notes
- [ ] Semantic versioning rules followed

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

This checklist must be completed for every release to prevent critical bugs from reaching users.