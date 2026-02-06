# Authentication Implementation Testing Guide

## Overview
This document outlines testing steps for the dashboard authentication handling implementation (Issue #414).

## Implementation Summary

### Files Modified
1. **src/mcp_memory_service/web/static/app.js**
   - Added `authState` property to constructor
   - Added `detectAuthRequirement()` method
   - Modified `init()` to detect auth before loading data
   - Updated `apiCall()` to include authentication headers
   - Added `handleAuthFailure()` method
   - Added `showAuthModal()` method
   - Added `authenticateWithApiKey()` method
   - Added `setupAuthListeners()` method

2. **src/mcp_memory_service/web/static/index.html**
   - Added authentication modal HTML with API key and OAuth options

3. **src/mcp_memory_service/web/static/style.css**
   - Added authentication modal styles (light and dark mode)

## Testing Checklist

### 1. Anonymous Access Mode (No Authentication Required)
- [ ] Set `MCP_ALLOW_ANONYMOUS_ACCESS=true` in environment
- [ ] Start HTTP server: `python scripts/server/run_http_server.py`
- [ ] Open dashboard in browser: http://localhost:8000
- [ ] Verify: Dashboard loads without showing auth modal
- [ ] Verify: All operations work (add memory, search, etc.)

### 2. API Key Authentication (Required)
- [ ] Remove or set `MCP_ALLOW_ANONYMOUS_ACCESS=false`
- [ ] Set `MCP_API_KEY=test-api-key-12345` in environment
- [ ] Restart HTTP server
- [ ] Open dashboard in browser: http://localhost:8000
- [ ] Verify: Auth modal appears automatically
- [ ] Test invalid key:
  - [ ] Enter wrong API key
  - [ ] Click "Authenticate"
  - [ ] Verify: Error toast appears
  - [ ] Verify: Auth modal remains open
- [ ] Test valid key:
  - [ ] Enter `test-api-key-12345`
  - [ ] Click "Authenticate"
  - [ ] Verify: Success toast appears
  - [ ] Verify: Auth modal closes
  - [ ] Verify: Dashboard data loads
  - [ ] Verify: All operations work

### 3. API Key Persistence
- [ ] After successful authentication, refresh the page
- [ ] Verify: Auth modal does not appear (key stored in localStorage)
- [ ] Verify: Dashboard loads immediately
- [ ] Open browser DevTools > Application > Local Storage
- [ ] Verify: `mcp_api_key` is present with correct value

### 4. 401 Handling During Operations
- [ ] Authenticate successfully
- [ ] In DevTools console, run: `localStorage.removeItem('mcp_api_key')`
- [ ] Try to add a new memory or perform search
- [ ] Verify: Auth modal reappears
- [ ] Re-authenticate
- [ ] Verify: Operation can be retried

### 5. Enter Key Support
- [ ] Trigger auth modal
- [ ] Type API key in input field
- [ ] Press Enter key (don't click button)
- [ ] Verify: Authentication attempt is made

### 6. OAuth Button (Redirect Test)
- [ ] Trigger auth modal
- [ ] Click "Login with OAuth" button
- [ ] Verify: Browser redirects to `/oauth/authorize`
- [ ] (OAuth flow testing requires OAuth server setup - out of scope for this test)

### 7. Dark Mode Compatibility
- [ ] Open dashboard
- [ ] Click Settings (gear icon)
- [ ] Change theme to Dark
- [ ] Trigger auth modal (clear localStorage or use unauthenticated mode)
- [ ] Verify: Auth modal uses dark theme styles
- [ ] Verify: Text is readable
- [ ] Verify: Input fields use dark background

### 8. Mobile Responsiveness
- [ ] Open DevTools > Toggle device toolbar
- [ ] Select mobile viewport (e.g., iPhone 12)
- [ ] Trigger auth modal
- [ ] Verify: Modal is responsive and usable on mobile
- [ ] Verify: Input fields are properly sized
- [ ] Verify: Buttons are touch-friendly

### 9. Accessibility
- [ ] Open auth modal
- [ ] Press Tab key repeatedly
- [ ] Verify: Focus moves through interactive elements in logical order
- [ ] Verify: API key input can be focused and typed in
- [ ] Verify: Buttons can be activated with Enter/Space

### 10. Error Scenarios
- [ ] Network failure simulation:
  - [ ] Open DevTools > Network tab
  - [ ] Set throttling to "Offline"
  - [ ] Try to authenticate
  - [ ] Verify: Error is handled gracefully
  - [ ] Verify: User sees appropriate error message

## Manual Test Commands

```bash
# Test 1: Anonymous access
export MCP_ALLOW_ANONYMOUS_ACCESS=true
python scripts/server/run_http_server.py

# Test 2: API key required
unset MCP_ALLOW_ANONYMOUS_ACCESS
export MCP_API_KEY=test-api-key-12345
python scripts/server/run_http_server.py

# Test 3: Check localStorage in browser console
localStorage.getItem('mcp_api_key')

# Test 4: Clear stored credentials
localStorage.removeItem('mcp_api_key')
localStorage.removeItem('mcp_oauth_token')
```

## Expected Outcomes

### Successful Implementation
- ✅ Auth modal appears when authentication is required
- ✅ Users can authenticate with API key
- ✅ Credentials persist across page refreshes
- ✅ 401 errors trigger re-authentication
- ✅ All UI elements are styled consistently
- ✅ Dark mode works correctly
- ✅ No JavaScript errors in console

### Known Limitations
- OAuth flow requires full OAuth server setup (tested separately)
- OAuth token refresh not implemented (future enhancement)
- Multi-factor authentication not supported

## Regression Testing

After implementation, verify these existing features still work:
- [ ] Memory search (semantic, tag-based, time-based)
- [ ] Add/edit/delete memories
- [ ] Settings modal
- [ ] Document upload
- [ ] Graph visualization
- [ ] Dark mode toggle
- [ ] Language switching (i18n)

## Performance Considerations

- Auth detection adds one extra `/api/health` call on page load
- LocalStorage read/write operations are synchronous but negligible
- No performance degradation expected for normal operations

## Security Notes

- API keys stored in localStorage (client-side only)
- Not encrypted (acceptable for local development)
- Production deployments should use HTTPS
- OAuth tokens expire according to server configuration

## Next Steps (Future Enhancements)

1. Add OAuth token refresh logic
2. Implement "Remember me" checkbox
3. Add logout button to clear credentials
4. Support multiple authentication providers
5. Add session timeout handling
6. Implement two-factor authentication
