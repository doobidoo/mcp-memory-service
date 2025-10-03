# Cloudflare Authentication Troubleshooting

This guide helps resolve common Cloudflare authentication issues with the MCP Memory Service.

## Overview

Cloudflare API tokens come in different types with varying scopes and verification methods. Understanding these differences is crucial for proper authentication.

## Token Types and Verification

### Account-Scoped Tokens (Recommended)

**What they are:** Tokens with specific permissions limited to a particular Cloudflare account.

**Required Permissions:**
- `D1 Database:Edit` - For D1 database operations
- `Vectorize:Edit` - For vector index operations

**Verification Endpoint:**
```bash
curl "https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/tokens/verify" \
     -H "Authorization: Bearer {YOUR_TOKEN}"
```

**Success Response:**
```json
{
  "result": {
    "id": "token_id_here",
    "status": "active",
    "expires_on": "2026-04-30T23:59:59Z"
  },
  "success": true,
  "errors": [],
  "messages": [
    {
      "code": 10000,
      "message": "This API Token is valid and active"
    }
  ]
}
```

### Global Tokens (Legacy)

**What they are:** Tokens with broader permissions across all accounts.

**Verification Endpoint:**
```bash
curl "https://api.cloudflare.com/client/v4/user/tokens/verify" \
     -H "Authorization: Bearer {YOUR_TOKEN}"
```

## Common Error Messages

### "Invalid API Token" (Error 1000)

**Cause:** Using the wrong verification endpoint for your token type.

**Solution:**
1. If using account-scoped token, use the account-specific endpoint
2. If using global token, use the user endpoint
3. Check token expiration date
4. Verify token permissions

**Example:**
```bash
# ❌ Wrong: Testing account-scoped token with user endpoint
curl "https://api.cloudflare.com/client/v4/user/tokens/verify" \
     -H "Authorization: Bearer account_scoped_token"
# Returns: {"success":false,"errors":[{"code":1000,"message":"Invalid API Token"}]}

# ✅ Correct: Testing account-scoped token with account endpoint
curl "https://api.cloudflare.com/client/v4/accounts/your_account_id/tokens/verify" \
     -H "Authorization: Bearer account_scoped_token"
# Returns: {"success":true,...}
```

### "401 Unauthorized" During Operations

**Cause:** Token lacks required permissions for specific operations.

**Solution:**
1. Verify token has `D1 Database:Edit` permission
2. Verify token has `Vectorize:Edit` permission
3. Check if token has expired
4. Ensure account ID matches token scope

### "Client error '401 Unauthorized'" in MCP Service

**Cause:** Environment variables may not be properly loaded or token is invalid.

**Debugging Steps:**
1. Check environment variable loading:
   ```bash
   python scripts/validation/diagnose_backend_config.py
   ```

2. Test token manually:
   ```bash
   curl "https://api.cloudflare.com/client/v4/accounts/$CLOUDFLARE_ACCOUNT_ID/tokens/verify" \
        -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN"
   ```

3. Test D1 database access:
   ```bash
   curl -X POST "https://api.cloudflare.com/client/v4/accounts/$CLOUDFLARE_ACCOUNT_ID/d1/database/$CLOUDFLARE_D1_DATABASE_ID/query" \
        -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
        -H "Content-Type: application/json" \
        -d '{"sql": "SELECT name FROM sqlite_master WHERE type='"'"'table'"'"';"}'
   ```

## Token Creation Guide

### Creating Account-Scoped Tokens

1. Go to [Cloudflare Dashboard](https://dash.cloudflare.com/profile/api-tokens)
2. Click "Create Token"
3. Use "Custom token" template
4. Set permissions:
   - `Account` → `Cloudflare D1:Edit`
   - `Account` → `Vectorize:Edit`
5. Set account resources to your specific account
6. Add client IP restrictions (optional but recommended)
7. Set expiration date
8. Create and copy the token immediately

### Token Security Best Practices

- ✅ Use account-scoped tokens with minimal required permissions
- ✅ Set expiration dates (e.g., 1 year maximum)
- ✅ Add IP restrictions when possible
- ✅ Store tokens securely (environment variables, not in code)
- ✅ Rotate tokens regularly
- ❌ Never commit tokens to version control
- ❌ Don't use global tokens unless absolutely necessary

## Environment Configuration

### Required Variables

```bash
# Account-scoped token (recommended)
CLOUDFLARE_API_TOKEN=your_account_scoped_token_here
CLOUDFLARE_ACCOUNT_ID=your_account_id_here
CLOUDFLARE_D1_DATABASE_ID=your_d1_database_id_here
CLOUDFLARE_VECTORIZE_INDEX=mcp-memory-index
```

### Validation Command

```bash
# Test all configuration
python scripts/validation/diagnose_backend_config.py

# Quick token test
curl "https://api.cloudflare.com/client/v4/accounts/$CLOUDFLARE_ACCOUNT_ID/tokens/verify" \
     -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN"
```

## Troubleshooting Checklist

- [ ] Token is account-scoped and has correct permissions
- [ ] Using correct verification endpoint (`/accounts/{id}/tokens/verify`)
- [ ] Environment variables are loaded correctly
- [ ] Account ID matches token scope
- [ ] Token has not expired
- [ ] D1 database ID is correct
- [ ] Vectorize index exists
- [ ] MCP service has been restarted after configuration changes

## Getting Help

If you're still experiencing issues:

1. Run the diagnostic script: `python scripts/validation/diagnose_backend_config.py`
2. Check the [GitHub Issues](https://github.com/doobidoo/mcp-memory-service/issues)
3. Review the main [README.md](../../README.md) for setup instructions
4. Check the [CLAUDE.md](../../CLAUDE.md) for Claude Code specific guidance