# LM Studio Compatibility Guide

## Issue Description

When using MCP Memory Service with LM Studio or Claude Desktop, you may encounter errors when operations are cancelled or timeout:

### Error Types

1. **Validation Error (LM Studio)**:
```
pydantic_core._pydantic_core.ValidationError: 5 validation errors for ClientNotification
ProgressNotification.method
  Input should be 'notifications/progress' [type=literal_error, input_value='notifications/cancelled', input_type=str]
```

2. **Timeout Error (Claude Desktop)**:
```
Message from client: {"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":0,"reason":"McpError: MCP error -32001: Request timed out"}}
Server transport closed unexpectedly, this is likely due to the process exiting early.
```

These occur because:
- LM Studio and Claude Desktop send non-standard `notifications/cancelled` messages
- These messages aren't part of the official MCP (Model Context Protocol) specification
- Timeouts can cause the server to exit prematurely on Windows systems

## Solution

The MCP Memory Service now includes an automatic compatibility patch that handles LM Studio's non-standard notifications. This patch is applied automatically when the server starts.

### How It Works

1. **Automatic Detection**: The server detects when clients send `notifications/cancelled` messages
2. **Graceful Handling**: Instead of crashing, the server handles these gracefully:
   - Logs the cancellation reason (including timeouts)
   - Converts to harmless notifications that don't cause validation errors
   - Continues operation normally
3. **Platform Optimizations**:
   - **Windows**: Extended timeouts (30s vs 15s) due to security software interference
   - **Cross-platform**: Enhanced signal handling for graceful shutdowns

### What You Need to Do

**Nothing!** The compatibility patch is applied automatically when you start the MCP Memory Service.

### Verifying the Fix

You can verify the patch is working by checking the server logs. You should see:

```
Applied enhanced LM Studio/Claude Desktop compatibility patch for notifications/cancelled
```

When operations are cancelled or timeout, you'll see:

```
Intercepted cancelled notification (ID: 0): McpError: MCP error -32001: Request timed out
Operation timeout detected: McpError: MCP error -32001: Request timed out
```

Instead of a crash, the server will continue running.

## Technical Details

The compatibility layer is implemented in `src/mcp_memory_service/lm_studio_compat.py` and:

1. **Notification Patching**: Monkey-patches the MCP library's `ClientNotification.model_validate` method
2. **Timeout Detection**: Identifies and logs timeout scenarios vs regular cancellations
3. **Graceful Substitution**: Converts `notifications/cancelled` to valid `InitializedNotification` objects
4. **Platform Optimization**: Uses extended timeouts on Windows (30s vs 15s)
5. **Signal Handling**: Adds Windows-specific signal handlers for graceful shutdowns
6. **Alternative Patching**: Fallback approach modifies the session receive loop if needed

## Windows-Specific Improvements

- **Extended Timeouts**: 30-second timeout for storage initialization (vs 15s on other platforms)
- **Security Software Compatibility**: Accounts for Windows Defender and antivirus delays
- **Signal Handling**: Enhanced SIGTERM/SIGINT handling for clean shutdowns
- **Timeout Recovery**: Better recovery from initialization timeouts

## Limitations

- **Workaround Nature**: This addresses non-standard client behavior, not a server issue
- **Cancelled Operations**: Operations aren't truly cancelled server-side, just client notifications are handled
- **Timeout Recovery**: While timeouts are handled gracefully, the original operation may still complete

## Future Improvements

Ideally, this should be fixed in one of two ways:

1. **LM Studio Update**: LM Studio should follow the MCP specification and not send non-standard notifications
2. **MCP Library Update**: The MCP library could be updated to handle vendor-specific extensions gracefully

## Troubleshooting

If you still experience issues:

1. Ensure you're using the latest version of MCP Memory Service
2. Check that the patch is being applied (look for the log message)
3. Report the issue with full error logs to the repository

## Related Issues

- This is a known compatibility issue between LM Studio and the MCP protocol
- Similar issues may occur with other non-standard MCP clients
- The patch specifically handles LM Studio's behavior and may need updates for other clients
