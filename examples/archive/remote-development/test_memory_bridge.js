#!/usr/bin/env node
/**
 * Test script for MCP Memory Bridge
 */

const { spawn } = require('child_process');
const readline = require('readline');

// Path to the bridge script
const bridgePath = '/Users/hkr/Documents/GitHub/mcp-memory-service/examples/simple-http-bridge.js';

// Environment variables for the bridge
const env = {
  ...process.env,
  'MCP_MEMORY_HTTP_ENDPOINT': 'https://memory.local/api',
  'MCP_MEMORY_API_KEY': 'mcp-0b1ccbde2197a08dcb12d41af4044be6',
  'NODE_TLS_REJECT_UNAUTHORIZED': '0'
};

console.log('Starting MCP Memory Bridge test...');
console.log(`Bridge: ${bridgePath}`);
console.log(`Endpoint: ${env.MCP_MEMORY_HTTP_ENDPOINT}`);
console.log(`API Key: ${env.MCP_MEMORY_API_KEY ? '[SET]' : '[NOT SET]'}`);

// Start the bridge process
const bridge = spawn('node', [bridgePath], { 
  env,
  stdio: ['pipe', 'pipe', 'pipe']
});

// Set up readline interface for stdin
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  prompt: 'MCP> '
});

// Forward stdout from the bridge to our stdout
bridge.stdout.on('data', (data) => {
  const responses = data.toString().trim().split('\n');
  for (const response of responses) {
    if (response) {
      try {
        const parsed = JSON.parse(response);
        console.log('\nResponse:', JSON.stringify(parsed, null, 2));
        rl.prompt();
      } catch (error) {
        console.log('\nInvalid JSON:', response);
        rl.prompt();
      }
    }
  }
});

// Forward stderr from the bridge to our stderr with timestamps
bridge.stderr.on('data', (data) => {
  const lines = data.toString().trim().split('\n');
  for (const line of lines) {
    if (line) {
      console.error(`[${new Date().toISOString()}] ${line}`);
    }
  }
});

// Handle bridge process exit
bridge.on('exit', (code) => {
  console.log(`Bridge process exited with code ${code}`);
  rl.close();
  process.exit(code);
});

// Handle errors
bridge.on('error', (error) => {
  console.error(`Bridge process error: ${error.message}`);
  rl.close();
  process.exit(1);
});

// Test requests
const requests = [
  {
    name: 'initialize',
    request: {
      jsonrpc: '2.0',
      id: 1,
      method: 'initialize',
      params: {
        protocolVersion: '2025-06-18',
        capabilities: {},
        clientInfo: {
          name: 'mcp-test-client',
          version: '1.0.0'
        }
      }
    }
  },
  {
    name: 'check_database_health',
    request: {
      jsonrpc: '2.0',
      id: 2,
      method: 'check_database_health',
      params: {}
    }
  },
  {
    name: 'store_memory',
    request: {
      jsonrpc: '2.0',
      id: 3,
      method: 'store_memory',
      params: {
        content: 'Test memory content',
        metadata: {
          tags: ['test', 'memory'],
          type: 'note'
        }
      }
    }
  },
  {
    name: 'retrieve_memory',
    request: {
      jsonrpc: '2.0',
      id: 4,
      method: 'retrieve_memory',
      params: {
        query: 'test',
        n_results: 5
      }
    }
  }
];

// Print the available test requests
console.log('\nAvailable test requests:');
requests.forEach((req, index) => {
  console.log(`${index + 1}. ${req.name}`);
});

rl.prompt();

// Handle user input
rl.on('line', (line) => {
  const input = line.trim();
  
  if (input === 'exit' || input === 'quit') {
    console.log('Exiting...');
    bridge.kill();
    rl.close();
    return;
  }
  
  if (input === 'help') {
    console.log('\nCommands:');
    console.log('- 1-4: Send predefined test requests');
    console.log('- raw <json>: Send raw JSON-RPC request');
    console.log('- exit/quit: Exit the test');
    rl.prompt();
    return;
  }
  
  if (input.startsWith('raw ')) {
    try {
      const json = input.substring(4);
      const request = JSON.parse(json);
      console.log(`Sending raw request: ${JSON.stringify(request)}`);
      bridge.stdin.write(JSON.stringify(request) + '\n');
    } catch (error) {
      console.error(`Error parsing JSON: ${error.message}`);
      rl.prompt();
    }
    return;
  }
  
  const index = parseInt(input, 10) - 1;
  if (index >= 0 && index < requests.length) {
    const req = requests[index];
    console.log(`Sending ${req.name} request...`);
    bridge.stdin.write(JSON.stringify(req.request) + '\n');
  } else {
    console.log('Invalid command. Type "help" for available commands.');
    rl.prompt();
  }
});

rl.on('close', () => {
  console.log('Test finished');
  bridge.kill();
  process.exit(0);
});