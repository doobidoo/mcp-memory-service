# Cloudflare API Token Configuration for MCP Memory Service

This guide helps you create and configure a Cloudflare API token with the correct permissions for the MCP Memory Service Cloudflare backend.

## Required API Token Permissions

To use the Cloudflare backend, your API token must have these permissions:

### Essential Permissions

#### D1 Database
- **Permission**: `Cloudflare D1:Edit`
- **Purpose**: Storing memory metadata, tags, and relationships
- **Required**: Yes

#### Vectorize Index
- **Permission**: `AI Gateway:Edit` or `Vectorize:Edit`
- **Purpose**: Storing and querying memory embeddings
- **Required**: Yes

#### Workers AI
- **Permission**: `AI Gateway:Read` or `Workers AI:Read`
- **Purpose**: Generating embeddings using Cloudflare's AI models
- **Model Used**: `@cf/baai/bge-base-en-v1.5`
- **Required**: Yes

#### Account Access
- **Permission**: `Account:Read`
- **Purpose**: Basic account-level operations
- **Required**: Yes

#### R2 Storage (Optional)
- **Permission**: `R2:Edit`
- **Purpose**: Large content storage (files > 1MB)
- **Required**: Only if using R2 for large content storage

## Token Creation Steps

1. **Navigate to Cloudflare Dashboard**
   - Go to: https://dash.cloudflare.com/profile/api-tokens

2. **Create Custom Token**
   - Click "Create Token" > "Custom token"

3. **Configure Token Permissions**
   - **Token name**: `MCP Memory Service Token` (or similar)
   - **Permissions**: Add all required permissions listed above
   - **Account resources**: Include your Cloudflare account
   - **Zone resources**: Include required zones (or all zones)
   - **IP address filtering**: Leave blank for maximum compatibility
   - **TTL**: Set appropriate expiration date

4. **Save and Copy Token**
   - Click "Continue to summary" > "Create Token"
   - **Important**: Copy the token immediately - it won't be shown again

## Environment Configuration

Add the token to your environment configuration:

### Option 1: Project .env File
```bash
# Add to .env file in project root
MCP_MEMORY_STORAGE_BACKEND=cloudflare
CLOUDFLARE_API_TOKEN=your_new_token_here
CLOUDFLARE_ACCOUNT_ID=your_account_id
CLOUDFLARE_D1_DATABASE_ID=your_d1_database_id
CLOUDFLARE_VECTORIZE_INDEX=your_vectorize_index_name
```

### Option 2: Claude Desktop Configuration
```json
{
  "mcpServers": {
    "memory": {
      "command": "uv",
      "args": [
        "--directory", "path/to/mcp-memory-service",
        "run", "python", "-m", "mcp_memory_service.server"
      ],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "cloudflare",
        "CLOUDFLARE_API_TOKEN": "your_new_token_here",
        "CLOUDFLARE_ACCOUNT_ID": "your_account_id",
        "CLOUDFLARE_D1_DATABASE_ID": "your_d1_database_id",
        "CLOUDFLARE_VECTORIZE_INDEX": "your_vectorize_index_name"
      }
    }
  }
}
```

## Verification

Test your token configuration:

```bash
# Navigate to project directory
cd path/to/mcp-memory-service

# Test the configuration
uv run python -c "
import asyncio
from src.mcp_memory_service.storage.cloudflare import CloudflareStorage
import os

async def test():
    storage = CloudflareStorage(
        api_token=os.getenv('CLOUDFLARE_API_TOKEN'),
        account_id=os.getenv('CLOUDFLARE_ACCOUNT_ID'),
        vectorize_index=os.getenv('CLOUDFLARE_VECTORIZE_INDEX'),
        d1_database_id=os.getenv('CLOUDFLARE_D1_DATABASE_ID')
    )
    await storage.initialize()
    print('Token configuration successful!')

asyncio.run(test())
"
```

## Common Authentication Issues

### Error Codes and Solutions

#### Error 9109: Location Restriction
- **Symptom**: "Cannot use the access token from location: [IP]"
- **Cause**: Token has IP address restrictions
- **Solution**: Remove IP restrictions or add current IP to allowlist

#### Error 7403: Insufficient Permissions
- **Symptom**: "The given account is not valid or is not authorized"
- **Cause**: Token lacks required service permissions
- **Solution**: Add missing permissions (D1, Vectorize, Workers AI)

#### Error 10000: Authentication Error
- **Symptom**: "Authentication error" for specific services
- **Cause**: Token missing permissions for specific services
- **Solution**: Verify all required permissions are granted

#### Error 1000: Invalid API Token
- **Symptom**: "Invalid API Token"
- **Cause**: Token may be malformed or expired
- **Solution**: Create a new token or check token format

### Google SSO Accounts

If you use Google SSO for Cloudflare:

1. **Set Account Password**
   - Go to **My Profile** → **Authentication**
   - Click **"Set Password"** to add a password to your account
   - Use this password when prompted during token creation

2. **Alternative: Global API Key**
   - Go to **My Profile** → **API Tokens**
   - Scroll to **"Global API Key"** section
   - Use Global API Key + email for authentication

## Security Best Practices

1. **Minimal Permissions**: Only grant permissions required for your use case
2. **Token Rotation**: Regularly rotate API tokens (e.g., every 90 days)
3. **Environment Variables**: Never commit tokens to version control
4. **IP Restrictions**: Use IP restrictions in production environments
5. **Monitoring**: Monitor token usage in Cloudflare dashboard
6. **Expiration**: Set reasonable TTL for tokens

## Troubleshooting Steps

If authentication continues to fail:

1. **Verify Configuration**
   - Check all environment variables are set correctly
   - Confirm resource IDs (account, database, index) are accurate

2. **Test Individual Services**
   - Test account access first
   - Then test each service (D1, Vectorize, Workers AI) individually

3. **Check Cloudflare Logs**
   - Review API usage logs in Cloudflare dashboard
   - Look for specific error messages and timestamps

4. **Validate Permissions**
   - Double-check all required permissions are selected
   - Ensure permissions include both read and write access where needed

5. **Network Issues**
   - Verify network connectivity to Cloudflare APIs
   - Check if corporate firewall blocks API access

For additional help, see the [Cloudflare Setup Guide](../cloudflare-setup.md) or the main [troubleshooting documentation](./README.md).
