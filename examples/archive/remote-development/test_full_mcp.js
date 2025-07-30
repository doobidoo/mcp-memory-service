#!/usr/bin/env node
/**
 * Test the full MCP protocol flow
 */
const { spawn } = require('child_process');

async function testMCPFlow() {
    console.log('Testing full MCP protocol flow...');
    
    // Set up environment
    const env = {
        ...process.env,
        MCP_MEMORY_HTTP_ENDPOINT: 'https://memory.local/api',
        MCP_MEMORY_API_KEY: 'mcp-0b1ccbde2197a08dcb12d41af4044be6',
        NODE_TLS_REJECT_UNAUTHORIZED: '0'
    };
    
    // Start the bridge
    const bridge = spawn('node', ['./examples/http-mcp-bridge-robust.js'], {
        env: env,
        stdio: ['pipe', 'pipe', 'pipe']
    });
    
    let responses = [];
    bridge.stdout.on('data', (data) => {
        const lines = data.toString().split('\n').filter(line => line.trim());
        lines.forEach(line => {
            try {
                const response = JSON.parse(line);
                responses.push(response);
                console.log('Response:', JSON.stringify(response, null, 2));
            } catch (e) {
                // Ignore non-JSON output
            }
        });
    });
    
    bridge.stderr.on('data', (data) => {
        console.log('Bridge log:', data.toString());
    });
    
    // Wait for bridge to start
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Send initialize request
    console.log('\n=== Sending initialize request ===');
    const initRequest = {
        jsonrpc: "2.0",
        id: 1,
        method: "initialize",
        params: {
            protocolVersion: "2024-11-05",
            capabilities: {},
            clientInfo: { name: "test-client", version: "1.0.0" }
        }
    };
    bridge.stdin.write(JSON.stringify(initRequest) + '\n');
    
    // Wait for response
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Send tools/list request
    console.log('\n=== Sending tools/list request ===');
    const toolsRequest = {
        jsonrpc: "2.0",
        id: 2,
        method: "tools/list",
        params: {}
    };
    bridge.stdin.write(JSON.stringify(toolsRequest) + '\n');
    
    // Wait for response
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Cleanup
    bridge.kill();
    
    console.log('\n=== Test completed ===');
    if (responses.length === 0) {
        console.log('No responses received - there may be an issue');
    }
}

testMCPFlow().catch(console.error);