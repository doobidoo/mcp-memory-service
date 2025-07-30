# Remote Memory Development Archive

This directory contains development artifacts from the remote memory bridge implementation.

## Contents

### Bridge Implementations
- `http-mcp-bridge.js` - Original bridge implementation (less robust)
- `simple-http-bridge.js` - Basic test bridge for initial connectivity
- `dummy-memory-server.js` - Mock memory server for testing
- `disable-cert-check.js` - Utility for disabling certificate validation

### Test Files
- `test_*.js` - Various JavaScript test scripts
- `test_*.sh` - Shell script tests for different scenarios
- `simple-memory-test.js` - Basic memory operation testing

### Development Data
- `init_request.json` - Sample initialization request for testing
- `remote-bridge/` - Incomplete experiment with alternative bridge structure

## Production Files

The production-ready remote memory bridge is located at:
- `examples/http-mcp-bridge-robust.js` - Production bridge with retry logic and error handling
- `examples/README-remote-memory.md` - User documentation

## Usage

These files are kept for reference and debugging but are not intended for production use. 
For setting up remote memory connections, use the production files in the main examples directory.

## History

These files were created during the July 30, 2025 development session where we successfully 
implemented remote memory connectivity between Claude Desktop and a remote MCP Memory Service.