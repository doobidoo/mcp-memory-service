#!/usr/bin/env node
/**
 * Dummy Memory Server - Minimal MCP Server
 */

// Keep track of the next message ID
let nextMessageId = 1;

// Dummy memory store
const memories = [];

/**
 * Process incoming JSON-RPC requests
 * This implements the bare minimum to satisfy Claude's requirements
 */
function processRequest(request) {
  const { method, params, id } = request;
  console.error(`Received request: ${method}`);

  let result;
  try {
    switch (method) {
      case 'initialize':
        result = {
          capabilities: {
            operations: [
              "store_memory",
              "retrieve_memory",
              "search_by_tag",
              "delete_memory",
              "check_database_health"
            ]
          },
          serverInfo: {
            name: "dummy-memory-server",
            version: "1.0.0"
          }
        };
        break;

      case 'store_memory':
        // Add memory with generated ID
        const memory = {
          id: `mem_${nextMessageId++}`,
          content: params.content,
          metadata: params.metadata || {}
        };
        memories.push(memory);
        result = { success: true, message: 'Memory stored successfully' };
        break;

      case 'retrieve_memory':
        // Find memories that contain the query string
        const matches = memories.filter(memory => 
          memory.content.toLowerCase().includes(params.query.toLowerCase())
        ).slice(0, params.n_results || 5);
        
        result = {
          memories: matches.map(memory => ({
            content: memory.content,
            metadata: memory.metadata
          }))
        };
        break;

      case 'search_by_tag':
        // Find memories that have any of the requested tags
        const tagMatches = memories.filter(memory => {
          const memoryTags = memory.metadata?.tags || [];
          const searchTags = Array.isArray(params.tags) ? params.tags : [params.tags];
          return searchTags.some(tag => memoryTags.includes(tag));
        });
        
        result = {
          memories: tagMatches.map(memory => ({
            content: memory.content,
            metadata: memory.metadata
          }))
        };
        break;

      case 'delete_memory':
        // Remove memory with matching content hash
        const initialLength = memories.length;
        const contentToDelete = params.content_hash;
        const newMemories = memories.filter(memory => memory.id !== contentToDelete);
        memories.length = 0;
        memories.push(...newMemories);
        
        result = {
          success: initialLength !== memories.length,
          message: initialLength !== memories.length ? 
            'Memory deleted successfully' : 
            'Memory not found'
        };
        break;

      case 'check_database_health':
        result = {
          status: 'healthy',
          backend: 'in-memory',
          statistics: {
            memory_count: memories.length,
            uptime_seconds: Math.floor((Date.now() - startTime) / 1000)
          }
        };
        break;

      case 'shutdown':
        // Acknowledge shutdown but don't actually shut down
        result = {};
        break;

      default:
        throw new Error(`Unknown method: ${method}`);
    }

    return {
      jsonrpc: "2.0",
      id: id,
      result: result
    };
  } catch (error) {
    console.error(`Error processing request: ${error.message}`);
    return {
      jsonrpc: "2.0",
      id: id,
      error: {
        code: -32000,
        message: error.message
      }
    };
  }
}

// Record start time for uptime calculation
const startTime = Date.now();
console.error(`Dummy Memory Server starting at ${new Date().toISOString()}`);

// Read from stdin and write to stdout
let buffer = '';
process.stdin.on('data', (chunk) => {
  buffer += chunk.toString();
  
  // Process complete messages (delimited by newlines)
  let newlineIndex;
  while ((newlineIndex = buffer.indexOf('\n')) !== -1) {
    const line = buffer.slice(0, newlineIndex);
    buffer = buffer.slice(newlineIndex + 1);
    
    if (line.trim()) {
      try {
        const request = JSON.parse(line);
        const response = processRequest(request);
        console.log(JSON.stringify(response));
      } catch (error) {
        console.error(`Error: ${error.message}`);
        console.log(JSON.stringify({
          jsonrpc: "2.0",
          id: null,
          error: {
            code: -32700,
            message: "Parse error"
          }
        }));
      }
    }
  }
});

// Handle signals but don't exit
process.on('SIGINT', () => {
  console.error('SIGINT received, but continuing to run');
});

process.on('SIGTERM', () => {
  console.error('SIGTERM received, but continuing to run');
});

// Keep the process alive
setInterval(() => {}, 10000);