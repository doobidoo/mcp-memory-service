/// <reference types="@cloudflare/workers-types" />

export interface Env {
  DB: D1Database;
  AI: any; // Workers AI binding
  DEBUG?: string; // Environment variable to control debug output
}

// Debug logger utility that only logs when DEBUG env var is set to a truthy value
function logger(env: Env, message: string, ...args: any[]) {
  // Only log if DEBUG environment variable is set to a truthy value
  if (env.DEBUG && ['1', 'true', 'yes'].includes(env.DEBUG.toLowerCase())) {
    console.error(`[DEBUG] ${message}`, ...args);
  }
}

interface MemoryRow {
  id: string;
  content: string;
  embedding: string;
  tags: string | null;
  metadata: string | null;
  created_at?: number;
}

interface MemoryResult {
  id: string;
  content: string;
  similarity: number;
  tags: any[];
  metadata: any;
}

export default {
  // Add debug logging for all requests
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    logger(env, `Received ${request.method} request to ${request.url}`);
    const { pathname } = new URL(request.url);

    if (pathname === "/mcp") {
      // Check if it's a GET request (SSE connection) or POST request (method call)
      logger(env, `Handling MCP request: ${request.method}`);
      if (request.method === "GET") {
        const { readable, writable } = new TransformStream();
        const writer = writable.getWriter();

        const headers = new Headers({
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          "Connection": "keep-alive",
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type"
        });

        // Send an initial message to establish the connection as a JSON-RPC notification
        // JSON-RPC notifications don't have an id field but must have a method
        const initialMessage = {
          jsonrpc: "2.0",
          method: "connection_established",
          params: {
            status: "connected",
            server: "mcp-memory-cloudflare",
            version: "1.0.0"
          }
        };
        
        // Start the SSE connection with an initial message
        ctx.waitUntil((async () => {
          logger(env, "Sending initial connection_established message");
          await sendSSE(writer, initialMessage);
          await handleMCP(request, env, writer);
        })());

        return new Response(readable, { headers });
      } else if (request.method === "POST") {
        // Handle direct JSON-RPC calls via POST
        return handleMethodCall(request, env);
      } else if (request.method === "OPTIONS") {
        // Handle CORS preflight requests
        return new Response(null, {
          headers: {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "86400"
          }
        });
      } else {
        // Method not allowed
        return new Response(JSON.stringify({
          jsonrpc: "2.0",
          id: null,
          error: {
            code: -32600,
            message: "Invalid Request: Method not allowed"
          }
        }), {
          headers: { "Content-Type": "application/json" },
          status: 405
        });
      }
    }

    if (pathname === "/list_tools") {
      return new Response(JSON.stringify({
        tools: [
          { name: "store_memory", description: "Store new information with optional tags" },
          { name: "retrieve_memory", description: "Find relevant memories based on query" },
          { name: "recall_memory", description: "Retrieve memories using natural language time expressions" },
          { name: "search_by_tag", description: "Search memories by tags" },
          { name: "delete_memory", description: "Delete a specific memory by its hash" },
          { name: "delete_by_tag", description: "Delete all memories with a specific tag" },
          { name: "cleanup_duplicates", description: "Find and remove duplicate entries" },
          { name: "get_embedding", description: "Get raw embedding vector for content" },
          { name: "check_embedding_model", description: "Check if embedding model is loaded and working" },
          { name: "debug_retrieve", description: "Retrieve memories with debug information" },
          { name: "exact_match_retrieve", description: "Retrieve memories using exact content match" },
          { name: "check_database_health", description: "Check database health and get statistics" },
          { name: "recall_by_timeframe", description: "Retrieve memories within a specific timeframe" },
          { name: "delete_by_timeframe", description: "Delete memories within a specific timeframe" },
          { name: "delete_before_date", description: "Delete memories before a specific date" }
        ]
      }, null, 2), {
        headers: { "Content-Type": "application/json" }
      });
    }

    return new Response("Not found", { status: 404 });
  },
};

async function handleDeleteMemory(args: any, env: Env, writer: WritableStreamDefaultWriter) {
  logger(env, "handleDeleteMemory called", args);
  try {
    const contentHash = args.content_hash;
    if (!contentHash) {
      await sendSSE(writer, { error: "Missing content_hash" });
      return;
    }

    logger(env, `Deleting memory with hash ${contentHash}`);
    await env.DB.prepare("DELETE FROM memories WHERE id = ?").bind(contentHash).run();
    await sendSSE(writer, { result: `Deleted memory with hash ${contentHash}` });
  } catch (e) {
    await sendSSE(writer, { error: e instanceof Error ? e.message : String(e) });
  }
}

async function handleDeleteByTag(args: any, env: Env, writer: WritableStreamDefaultWriter) {
  logger(env, "handleDeleteByTag called", args);
  try {
    const tag = args.tag;
    if (!tag) {
      await sendSSE(writer, { error: "Missing tag" });
      return;
    }

    const { results } = await env.DB.prepare("SELECT id, tags FROM memories").all();
    const rows = results as unknown as MemoryRow[];

    let deleteCount = 0;
    for (const row of rows) {
      const memoryTags = JSON.parse(row.tags ?? "[]");
      if (memoryTags.includes(tag)) {
        await env.DB.prepare("DELETE FROM memories WHERE id = ?").bind(row.id).run();
        deleteCount++;
      }
    }

    await sendSSE(writer, { result: `Deleted ${deleteCount} memories with tag ${tag}` });
  } catch (e) {
    await sendSSE(writer, { error: e instanceof Error ? e.message : String(e) });
  }
}

async function handleCheckDatabaseHealth(args: any, env: Env, writer: WritableStreamDefaultWriter) {
  logger(env, "handleCheckDatabaseHealth called");
  try {
    const { results } = await env.DB.prepare("SELECT COUNT(*) as count FROM memories").all();
    const count = results[0]?.count ?? 0;
    await sendSSE(writer, { status: "healthy", total_memories: count });
  } catch (e) {
    await sendSSE(writer, { status: "error", error: e instanceof Error ? e.message : String(e) });
  }
}

async function handleRecallMemory(args: any, env: Env, writer: WritableStreamDefaultWriter) {
  logger(env, "handleRecallMemory called", args);
  try {
    const query = args.query;
    const nResults = args.n_results || 5;
    if (!query) {
      await sendSSE(writer, { error: "Missing query" });
      return;
    }

    const queryEmbedding = await generateEmbedding(query, env);

    const { results } = await env.DB.prepare("SELECT id, content, embedding, tags, metadata FROM memories").all();
    const rows = results as unknown as MemoryRow[];

    const scored: MemoryResult[] = [];
    for (const row of rows) {
      const embedding = JSON.parse(row.embedding);
      const sim = cosineSimilarity(queryEmbedding, embedding);
      scored.push({
        id: row.id,
        content: row.content,
        similarity: sim,
        tags: JSON.parse(row.tags ?? "[]"),
        metadata: JSON.parse(row.metadata ?? "{}")
      });
    }

    scored.sort((a, b) => b.similarity - a.similarity);
    const top = scored.slice(0, nResults);

    await sendSSE(writer, { results: top });
  } catch (e) {
    await sendSSE(writer, { error: e instanceof Error ? e.message : String(e) });
  }
}

async function handleCleanupDuplicates(args: any, env: Env, writer: WritableStreamDefaultWriter) {
  logger(env, "handleCleanupDuplicates called");
  try {
    const { results } = await env.DB.prepare("SELECT id, content FROM memories").all();
    const rows = results as unknown as MemoryRow[];

    const seen = new Map<string, string>();
    let deleteCount = 0;

    for (const row of rows) {
      const hash = row.id;
      if (seen.has(hash)) {
        await env.DB.prepare("DELETE FROM memories WHERE id = ?").bind(row.id).run();
        deleteCount++;
      } else {
        seen.set(hash, row.content);
      }
    }

    await sendSSE(writer, { result: `Deleted ${deleteCount} duplicate memories` });
  } catch (e) {
    await sendSSE(writer, { error: e instanceof Error ? e.message : String(e) });
  }
}

async function handleDeleteByTimeframe(args: any, env: Env, writer: WritableStreamDefaultWriter) {
  logger(env, "handleDeleteByTimeframe called", args);
  try {
    const startDateStr = args.start_date;
    const endDateStr = args.end_date || args.start_date;
    const tag = args.tag;

    if (!startDateStr) {
      await sendSSE(writer, { error: "Missing start_date" });
      return;
    }

    const startTimestamp = new Date(startDateStr).getTime() / 1000;
    const endTimestamp = new Date(endDateStr).getTime() / 1000 + 86399; // end of day

    const { results } = await env.DB.prepare(
      "SELECT id, tags, created_at FROM memories WHERE created_at BETWEEN ? AND ?"
    ).bind(startTimestamp, endTimestamp).all();

    const rows = results as unknown as MemoryRow[];

    let deleteCount = 0;
    for (const row of rows) {
      const memoryTags = JSON.parse(row.tags ?? "[]");
      if (!tag || memoryTags.includes(tag)) {
        await env.DB.prepare("DELETE FROM memories WHERE id = ?").bind(row.id).run();
        deleteCount++;
      }
    }

    await sendSSE(writer, { deleted: deleteCount });
  } catch (e) {
    await sendSSE(writer, { error: e instanceof Error ? e.message : String(e) });
  }
}

async function handleGetEmbedding(args: any, env: Env, writer: WritableStreamDefaultWriter) {
  logger(env, "handleGetEmbedding called", args);
  try {
    const content = args.content;
    if (!content) {
      await sendSSE(writer, { error: "Missing content" });
      return;
    }

    const embedding = await generateEmbedding(content, env);
    await sendSSE(writer, { embedding });
  } catch (e) {
    await sendSSE(writer, { error: e instanceof Error ? e.message : String(e) });
  }
}

async function handleDeleteBeforeDate(args: any, env: Env, writer: WritableStreamDefaultWriter) {
  logger(env, "handleDeleteBeforeDate called", args);
  try {
    const beforeDateStr = args.before_date;
    const tag = args.tag;

    if (!beforeDateStr) {
      await sendSSE(writer, { error: "Missing before_date" });
      return;
    }

    const beforeTimestamp = new Date(beforeDateStr).getTime() / 1000;

    const { results } = await env.DB.prepare(
      "SELECT id, tags, created_at FROM memories WHERE created_at < ?"
    ).bind(beforeTimestamp).all();

    const rows = results as unknown as MemoryRow[];

    let deleteCount = 0;
    for (const row of rows) {
      const memoryTags = JSON.parse(row.tags ?? "[]");
      if (!tag || memoryTags.includes(tag)) {
        await env.DB.prepare("DELETE FROM memories WHERE id = ?").bind(row.id).run();
        deleteCount++;
      }
    }

    await sendSSE(writer, { deleted: deleteCount });
  } catch (e) {
    await sendSSE(writer, { error: e instanceof Error ? e.message : String(e) });
  }
}

async function handleRecallByTimeframe(args: any, env: Env, writer: WritableStreamDefaultWriter) {
  logger(env, "handleRecallByTimeframe called", args);
  try {
    const startDateStr = args.start_date;
    const endDateStr = args.end_date || args.start_date;
    const nResults = args.n_results || 5;

    if (!startDateStr) {
      await sendSSE(writer, { error: "Missing start_date" });
      return;
    }

    const startTimestamp = new Date(startDateStr).getTime() / 1000;
    const endTimestamp = new Date(endDateStr).getTime() / 1000 + 86399; // end of day

    const { results } = await env.DB.prepare(
      "SELECT id, content, tags, metadata, created_at FROM memories WHERE created_at BETWEEN ? AND ? ORDER BY created_at DESC LIMIT ?"
    ).bind(startTimestamp, endTimestamp, nResults).all();

    await sendSSE(writer, { results });
  } catch (e) {
    await sendSSE(writer, { error: e instanceof Error ? e.message : String(e) });
  }
}

async function handleExactMatchRetrieve(args: any, env: Env, writer: WritableStreamDefaultWriter) {
  logger(env, "handleExactMatchRetrieve called", args);
  try {
    const content = args.content;
    if (!content) {
      await sendSSE(writer, { error: "Missing content" });
      return;
    }

    const { results } = await env.DB.prepare("SELECT id, content, tags, metadata FROM memories WHERE content = ?").bind(content).all();
    const rows = results as unknown as MemoryRow[];

    await sendSSE(writer, { results: rows });
  } catch (e) {
    await sendSSE(writer, { error: e instanceof Error ? e.message : String(e) });
  }
}

async function handleCheckEmbeddingModel(args: any, env: Env, writer: WritableStreamDefaultWriter) {
  logger(env, "handleCheckEmbeddingModel called");
  try {
    await generateEmbedding("test", env);
    await sendSSE(writer, { status: "Embedding model is working" });
  } catch (e) {
    await sendSSE(writer, { status: "Embedding model check failed", error: e instanceof Error ? e.message : String(e) });
  }
}

async function handleSearchByTag(args: any, env: Env, writer: WritableStreamDefaultWriter) {
  logger(env, "handleSearchByTag called", args);
  try {
    const tags = args.tags;
    if (!tags || !Array.isArray(tags) || tags.length === 0) {
      await sendSSE(writer, { error: "Missing or invalid tags" });
      return;
    }

    const { results } = await env.DB.prepare("SELECT id, content, tags, metadata FROM memories").all();
    const rows = results as unknown as MemoryRow[];

    const matches = rows.filter(row => {
      const memoryTags = JSON.parse(row.tags ?? "[]");
      return memoryTags.some((t: string) => tags.includes(t));
    });

    await sendSSE(writer, { results: matches });
  } catch (e) {
    await sendSSE(writer, { error: e instanceof Error ? e.message : String(e) });
  }
}

interface DebugMemoryResult {
  memory: {
    content: string;
    content_hash: string;
    tags: any[];
    metadata: any;
  };
  debug_info: {
    raw_similarity: number;
    raw_distance: number;
    memory_id: string;
  };
}

async function handleDebugRetrieve(args: any, env: Env, writer: WritableStreamDefaultWriter) {
  logger(env, "handleDebugRetrieve called", args);
  try {
    const query = args.query;
    const nResults = args.n_results || 5;
    const similarityThreshold = args.similarity_threshold || 0.0;
    
    if (!query) {
      await sendSSE(writer, { error: "Missing query" });
      return;
    }
    
    const queryEmbedding = await generateEmbedding(query, env);
    
    const { results } = await env.DB.prepare("SELECT id, content, embedding, tags, metadata FROM memories").all();
    const rows = results as unknown as MemoryRow[];
    
    // Calculate similarity and filter by threshold
    const scored: DebugMemoryResult[] = [];
    for (const row of rows) {
      const embedding = JSON.parse(row.embedding);
      const similarity = cosineSimilarity(queryEmbedding, embedding);
      
      // Only include memories with similarity >= threshold
      if (similarity >= similarityThreshold) {
        scored.push({
          memory: {
            content: row.content,
            content_hash: row.id,
            tags: JSON.parse(row.tags ?? "[]"),
            metadata: JSON.parse(row.metadata ?? "{}")
          },
          debug_info: {
            raw_similarity: similarity,
            raw_distance: 1 - similarity,
            memory_id: row.id
          }
        });
      }
    }
    
    // Sort by similarity descending
    scored.sort((a, b) => b.debug_info.raw_similarity - a.debug_info.raw_similarity);
    
    // Limit to n_results
    const top = scored.slice(0, nResults);
    
    if (top.length === 0) {
      await sendSSE(writer, { result: "No matching memories found" });
      return;
    }
    
    // Format the results
    const formattedResults = top.map((result, i) => {
      const memoryInfo = [
        `Memory ${i+1}:`,
        `Content: ${result.memory.content}`,
        `Hash: ${result.memory.content_hash}`,
        `Raw Similarity Score: ${result.debug_info.raw_similarity.toFixed(4)}`,
        `Raw Distance: ${result.debug_info.raw_distance.toFixed(4)}`,
        `Memory ID: ${result.debug_info.memory_id}`
      ];
      
      if (result.memory.tags && result.memory.tags.length > 0) {
        memoryInfo.push(`Tags: ${result.memory.tags.join(', ')}`);
      }
      
      memoryInfo.push("---");
      return memoryInfo.join("\n");
    });
    
    await sendSSE(writer, {
      result: "Found the following memories:\n\n" + formattedResults.join("\n"),
      debug_results: top
    });
  } catch (e) {
    await sendSSE(writer, { error: `Error in debug retrieve: ${e instanceof Error ? e.message : String(e)}` });
  }
}

async function handleStoreMemory(args: any, env: Env, writer: WritableStreamDefaultWriter) {
  logger(env, "handleStoreMemory called", args);
  try {
    const content = args.content;
    const metadata = args.metadata || {};
    if (!content) {
      await sendSSE(writer, { error: "Missing content" });
      return;
    }

    const embedding = await generateEmbedding(content, env);

    const encoder = new TextEncoder();
    const data = encoder.encode(content + JSON.stringify(metadata));
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const contentHash = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');

    const tags = metadata.tags || [];
    const memoryType = metadata.type || "";
    const createdAt = Math.floor(Date.now() / 1000);

    await env.DB.prepare(
      `INSERT OR IGNORE INTO memories (id, content, embedding, tags, memory_type, metadata, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?)`
    ).bind(
      contentHash,
      content,
      JSON.stringify(embedding),
      JSON.stringify(tags),
      memoryType,
      JSON.stringify(metadata),
      createdAt
    ).run();

    await sendSSE(writer, { result: "Memory stored", id: contentHash });
  } catch (e) {
    await sendSSE(writer, { error: e instanceof Error ? e.message : String(e) });
  }
}

async function handleRetrieveMemory(args: any, env: Env, writer: WritableStreamDefaultWriter) {
  logger(env, "handleRetrieveMemory called", args);
  try {
    const query = args.query;
    const nResults = args.n_results || 5;
    if (!query) {
      await sendSSE(writer, { error: "Missing query" });
      return;
    }

    const queryEmbedding = await generateEmbedding(query, env);

    const { results } = await env.DB.prepare("SELECT id, content, embedding, tags, metadata FROM memories").all();
    const rows = results as unknown as MemoryRow[];

    const scored: MemoryResult[] = [];
    for (const row of rows) {
      const embedding = JSON.parse(row.embedding);
      const sim = cosineSimilarity(queryEmbedding, embedding);
      scored.push({
        id: row.id,
        content: row.content,
        similarity: sim,
        tags: JSON.parse(row.tags ?? "[]"),
        metadata: JSON.parse(row.metadata ?? "{}")
      });
    }

    scored.sort((a, b) => b.similarity - a.similarity);
    const top = scored.slice(0, nResults);

    await sendSSE(writer, { results: top });
  } catch (e) {
    await sendSSE(writer, { error: e instanceof Error ? e.message : String(e) });
  }
}

async function handleMCP(request: Request, env: Env, writer: WritableStreamDefaultWriter) {
  // For SSE connections (GET requests), we don't expect a request body
  // Instead, we set up an event listener for messages sent via POST requests
  
  logger(env, "SSE connection established");
  
  // Keep the connection open and wait for messages
  try {
    // Send heartbeat messages at regular intervals to keep the connection alive
    const heartbeatInterval = 15000; // 15 seconds
    
    // Use a loop to continuously send heartbeats rather than a single long timeout
    while (true) {
      // Wait for the heartbeat interval
      await new Promise(resolve => setTimeout(resolve, heartbeatInterval));
      
      // Send a heartbeat message to keep the connection alive
      logger(env, "Sending heartbeat");
      await sendSSE(writer, {
        jsonrpc: "2.0",
        method: "heartbeat",
        params: {
          timestamp: new Date().toISOString()
        }
      });
    }
  } catch (e) {
    // Handle any errors that occur during the connection
    logger(env, "Error in SSE connection", e);
    await sendSSE(writer, {
      jsonrpc: "2.0",
      id: null,
      error: {
        code: -32000,
        message: "Server error: " + (e instanceof Error ? e.message : String(e))
      }
    });
  } finally {
    // Close the writer when done
    logger(env, "Closing SSE connection");
    writer.close();
  }
}

// This function handles JSON-RPC method calls via POST requests
async function handleMethodCall(request: Request, env: Env): Promise<Response> {
  logger(env, "Handling method call via POST");
  try {
    const body = await request.json() as any;
    logger(env, `Method call: ${body.method}`, body.params);
    const jsonrpc = body.jsonrpc || "2.0";
    const id = body.id;
    const method = body.method;
    const args = body.params || {};

    // Validate JSON-RPC 2.0 request
    if (jsonrpc !== "2.0") {
      return new Response(JSON.stringify({
        jsonrpc: "2.0",
        id: id,
        error: {
          code: -32600,
          message: "Invalid Request: Not a valid JSON-RPC 2.0 request"
        }
      }), {
        headers: { "Content-Type": "application/json" }
      });
    }

    if (!method) {
      return new Response(JSON.stringify({
        jsonrpc: "2.0",
        id: id,
        error: {
          code: -32600,
          message: "Invalid Request: Method not specified"
        }
      }), {
        headers: { "Content-Type": "application/json" }
      });
    }

    // Process the method call
    let result;
    
    if (method === "store_memory") {
      result = await processStoreMemory(args, env);
    } else if (method === "retrieve_memory") {
      result = await processRetrieveMemory(args, env);
    } else if (method === "recall_memory") {
      result = await processRecallMemory(args, env);
    } else if (method === "recall_by_timeframe") {
      result = await processRecallByTimeframe(args, env);
    } else if (method === "exact_match_retrieve") {
      result = await processExactMatchRetrieve(args, env);
    } else if (method === "search_by_tag") {
      result = await processSearchByTag(args, env);
    } else if (method === "delete_memory") {
      result = await processDeleteMemory(args, env);
    } else if (method === "delete_by_tag") {
      result = await processDeleteByTag(args, env);
    } else if (method === "cleanup_duplicates") {
      result = await processCleanupDuplicates(args, env);
    } else if (method === "get_embedding") {
      result = await processGetEmbedding(args, env);
    } else if (method === "check_embedding_model") {
      result = await processCheckEmbeddingModel(args, env);
    } else if (method === "debug_retrieve") {
      result = await processDebugRetrieve(args, env);
    } else if (method === "check_database_health") {
      result = await processCheckDatabaseHealth(args, env);
    } else if (method === "delete_by_timeframe") {
      result = await processDeleteByTimeframe(args, env);
    } else if (method === "delete_before_date") {
      result = await processDeleteBeforeDate(args, env);
    } else {
      return new Response(JSON.stringify({
        jsonrpc: "2.0",
        id: id,
        error: {
          code: -32601,
          message: "Method not found"
        }
      }), {
        headers: { "Content-Type": "application/json" }
      });
    }

    // Return the result
    return new Response(JSON.stringify({
      jsonrpc: "2.0",
      id: id,
      result: result
    }), {
      headers: { "Content-Type": "application/json" }
    });
  } catch (e) {
    return new Response(JSON.stringify({
      jsonrpc: "2.0",
      id: null, // We don't know the ID if we couldn't parse the request
      error: {
        code: -32700,
        message: "Parse error: " + (e instanceof Error ? e.message : String(e))
      }
    }), {
      headers: { "Content-Type": "application/json" },
      status: 400
    });
  }
}

async function sendSSE(writer: WritableStreamDefaultWriter, data: any) {
  // No debug logging here to avoid excessive logs for heartbeats
  // Ensure we're sending a valid JSON-RPC 2.0 message
  let formattedData = data;
  
  // If it's already a valid JSON-RPC message, use it as is
  if (data.jsonrpc === "2.0" && (
      // Valid notification (has method, no id)
      (data.method && data.id === undefined) ||
      // Valid request (has method and id)
      (data.method && data.id !== undefined) ||
      // Valid response (has id and result)
      (data.id !== undefined && data.result !== undefined) ||
      // Valid error (has id and error)
      (data.id !== undefined && data.error !== undefined)
  )) {
    formattedData = data;
  }
  // Otherwise, format it as a JSON-RPC notification
  else {
    formattedData = {
      jsonrpc: "2.0",
      method: "notification",
      params: data
    };
  }
  
  const payload = `data: ${JSON.stringify(formattedData)}\n\n`;
  await writer.write(new TextEncoder().encode(payload));
}

async function generateEmbedding(text: string, env: Env): Promise<number[]> {
  logger(env, "Generating embedding");
  const model = "@cf/baai/bge-base-en-v1.5";
  const response = await env.AI.run(model, { text });
  logger(env, "Embedding generated successfully");
  if (!response || !response.data || !response.data[0]) {
    logger(env, "Failed to generate embedding", response);
    throw new Error("Failed to generate embedding");
  }
  return response.data[0];
}

function cosineSimilarity(a: number[], b: number[]): number {
  const dot = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
  const normB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
  return dot / (normA * normB);
}

// Process functions for each method
// These functions handle the actual business logic and return results
// They are called by handleMethodCall for POST requests

async function processStoreMemory(args: any, env: Env): Promise<any> {
  logger(env, "processStoreMemory called", args);
  const content = args.content;
  const metadata = args.metadata || {};
  if (!content) {
    throw new Error("Missing content");
  }

  const embedding = await generateEmbedding(content, env);

  const encoder = new TextEncoder();
  const data = encoder.encode(content + JSON.stringify(metadata));
  const hashBuffer = await crypto.subtle.digest('SHA-256', data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  const contentHash = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');

  const tags = metadata.tags || [];
  const memoryType = metadata.type || "";
  const createdAt = Math.floor(Date.now() / 1000);

  await env.DB.prepare(
    `INSERT OR IGNORE INTO memories (id, content, embedding, tags, memory_type, metadata, created_at)
     VALUES (?, ?, ?, ?, ?, ?, ?)`
  ).bind(
    contentHash,
    content,
    JSON.stringify(embedding),
    JSON.stringify(tags),
    memoryType,
    JSON.stringify(metadata),
    createdAt
  ).run();

  return { result: "Memory stored", id: contentHash };
}

async function processRetrieveMemory(args: any, env: Env): Promise<any> {
  logger(env, "processRetrieveMemory called", args);
  const query = args.query;
  const nResults = args.n_results || 5;
  if (!query) {
    throw new Error("Missing query");
  }

  const queryEmbedding = await generateEmbedding(query, env);

  const { results } = await env.DB.prepare("SELECT id, content, embedding, tags, metadata FROM memories").all();
  const rows = results as unknown as MemoryRow[];

  const scored: MemoryResult[] = [];
  for (const row of rows) {
    const embedding = JSON.parse(row.embedding);
    const sim = cosineSimilarity(queryEmbedding, embedding);
    scored.push({
      id: row.id,
      content: row.content,
      similarity: sim,
      tags: JSON.parse(row.tags ?? "[]"),
      metadata: JSON.parse(row.metadata ?? "{}")
    });
  }

  scored.sort((a, b) => b.similarity - a.similarity);
  const top = scored.slice(0, nResults);

  return { results: top };
}

async function processRecallMemory(args: any, env: Env): Promise<any> {
  logger(env, "processRecallMemory called", args);
  const query = args.query;
  const nResults = args.n_results || 5;
  if (!query) {
    throw new Error("Missing query");
  }

  const queryEmbedding = await generateEmbedding(query, env);

  const { results } = await env.DB.prepare("SELECT id, content, embedding, tags, metadata FROM memories").all();
  const rows = results as unknown as MemoryRow[];

  const scored: MemoryResult[] = [];
  for (const row of rows) {
    const embedding = JSON.parse(row.embedding);
    const sim = cosineSimilarity(queryEmbedding, embedding);
    scored.push({
      id: row.id,
      content: row.content,
      similarity: sim,
      tags: JSON.parse(row.tags ?? "[]"),
      metadata: JSON.parse(row.metadata ?? "{}")
    });
  }

  scored.sort((a, b) => b.similarity - a.similarity);
  const top = scored.slice(0, nResults);

  return { results: top };
}

async function processRecallByTimeframe(args: any, env: Env): Promise<any> {
  logger(env, "processRecallByTimeframe called", args);
  const startDateStr = args.start_date;
  const endDateStr = args.end_date || args.start_date;
  const nResults = args.n_results || 5;

  if (!startDateStr) {
    throw new Error("Missing start_date");
  }

  const startTimestamp = new Date(startDateStr).getTime() / 1000;
  const endTimestamp = new Date(endDateStr).getTime() / 1000 + 86399; // end of day

  const { results } = await env.DB.prepare(
    "SELECT id, content, tags, metadata, created_at FROM memories WHERE created_at BETWEEN ? AND ? ORDER BY created_at DESC LIMIT ?"
  ).bind(startTimestamp, endTimestamp, nResults).all();

  return { results };
}

async function processExactMatchRetrieve(args: any, env: Env): Promise<any> {
  logger(env, "processExactMatchRetrieve called", args);
  const content = args.content;
  if (!content) {
    throw new Error("Missing content");
  }

  const { results } = await env.DB.prepare("SELECT id, content, tags, metadata FROM memories WHERE content = ?").bind(content).all();
  const rows = results as unknown as MemoryRow[];

  return { results: rows };
}

async function processSearchByTag(args: any, env: Env): Promise<any> {
  logger(env, "processSearchByTag called", args);
  const tags = args.tags;
  if (!tags || !Array.isArray(tags) || tags.length === 0) {
    throw new Error("Missing or invalid tags");
  }

  const { results } = await env.DB.prepare("SELECT id, content, tags, metadata FROM memories").all();
  const rows = results as unknown as MemoryRow[];

  const matches = rows.filter(row => {
    const memoryTags = JSON.parse(row.tags ?? "[]");
    return memoryTags.some((t: string) => tags.includes(t));
  });

  return { results: matches };
}

async function processDeleteMemory(args: any, env: Env): Promise<any> {
  logger(env, "processDeleteMemory called", args);
  const contentHash = args.content_hash;
  if (!contentHash) {
    throw new Error("Missing content_hash");
  }

  await env.DB.prepare("DELETE FROM memories WHERE id = ?").bind(contentHash).run();
  return { result: `Deleted memory with hash ${contentHash}` };
}

async function processDeleteByTag(args: any, env: Env): Promise<any> {
  logger(env, "processDeleteByTag called", args);
  const tag = args.tag;
  if (!tag) {
    throw new Error("Missing tag");
  }

  const { results } = await env.DB.prepare("SELECT id, tags FROM memories").all();
  const rows = results as unknown as MemoryRow[];

  let deleteCount = 0;
  for (const row of rows) {
    const memoryTags = JSON.parse(row.tags ?? "[]");
    if (memoryTags.includes(tag)) {
      await env.DB.prepare("DELETE FROM memories WHERE id = ?").bind(row.id).run();
      deleteCount++;
    }
  }

  return { result: `Deleted ${deleteCount} memories with tag ${tag}` };
}

async function processCleanupDuplicates(args: any, env: Env): Promise<any> {
  logger(env, "processCleanupDuplicates called");
  const { results } = await env.DB.prepare("SELECT id, content FROM memories").all();
  const rows = results as unknown as MemoryRow[];

  const seen = new Map<string, string>();
  let deleteCount = 0;

  for (const row of rows) {
    const hash = row.id;
    if (seen.has(hash)) {
      await env.DB.prepare("DELETE FROM memories WHERE id = ?").bind(row.id).run();
      deleteCount++;
    } else {
      seen.set(hash, row.content);
    }
  }

  return { result: `Deleted ${deleteCount} duplicate memories` };
}

async function processGetEmbedding(args: any, env: Env): Promise<any> {
  logger(env, "processGetEmbedding called", args);
  const content = args.content;
  if (!content) {
    throw new Error("Missing content");
  }

  const embedding = await generateEmbedding(content, env);
  return { embedding };
}

async function processCheckEmbeddingModel(args: any, env: Env): Promise<any> {
  logger(env, "processCheckEmbeddingModel called");
  await generateEmbedding("test", env);
  return { status: "Embedding model is working" };
}

async function processDebugRetrieve(args: any, env: Env): Promise<any> {
  logger(env, "processDebugRetrieve called", args);
  const query = args.query;
  const nResults = args.n_results || 5;
  const similarityThreshold = args.similarity_threshold || 0.0;
  
  if (!query) {
    throw new Error("Missing query");
  }
  
  const queryEmbedding = await generateEmbedding(query, env);
  
  const { results } = await env.DB.prepare("SELECT id, content, embedding, tags, metadata FROM memories").all();
  const rows = results as unknown as MemoryRow[];
  
  // Calculate similarity and filter by threshold
  const scored: DebugMemoryResult[] = [];
  for (const row of rows) {
    const embedding = JSON.parse(row.embedding);
    const similarity = cosineSimilarity(queryEmbedding, embedding);
    
    // Only include memories with similarity >= threshold
    if (similarity >= similarityThreshold) {
      scored.push({
        memory: {
          content: row.content,
          content_hash: row.id,
          tags: JSON.parse(row.tags ?? "[]"),
          metadata: JSON.parse(row.metadata ?? "{}")
        },
        debug_info: {
          raw_similarity: similarity,
          raw_distance: 1 - similarity,
          memory_id: row.id
        }
      });
    }
  }
  
  // Sort by similarity descending
  scored.sort((a, b) => b.debug_info.raw_similarity - a.debug_info.raw_similarity);
  
  // Limit to n_results
  const top = scored.slice(0, nResults);
  
  if (top.length === 0) {
    return { result: "No matching memories found" };
  }
  
  // Format the results
  const formattedResults = top.map((result, i) => {
    const memoryInfo = [
      `Memory ${i+1}:`,
      `Content: ${result.memory.content}`,
      `Hash: ${result.memory.content_hash}`,
      `Raw Similarity Score: ${result.debug_info.raw_similarity.toFixed(4)}`,
      `Raw Distance: ${result.debug_info.raw_distance.toFixed(4)}`,
      `Memory ID: ${result.debug_info.memory_id}`
    ];
    
    if (result.memory.tags && result.memory.tags.length > 0) {
      memoryInfo.push(`Tags: ${result.memory.tags.join(', ')}`);
    }
    
    memoryInfo.push("---");
    return memoryInfo.join("\n");
  });
  
  return {
    result: "Found the following memories:\n\n" + formattedResults.join("\n"),
    debug_results: top
  };
}

async function processCheckDatabaseHealth(args: any, env: Env): Promise<any> {
  logger(env, "processCheckDatabaseHealth called");
  const { results } = await env.DB.prepare("SELECT COUNT(*) as count FROM memories").all();
  const count = results[0]?.count ?? 0;
  return { status: "healthy", total_memories: count };
}

async function processDeleteByTimeframe(args: any, env: Env): Promise<any> {
  logger(env, "processDeleteByTimeframe called", args);
  const startDateStr = args.start_date;
  const endDateStr = args.end_date || args.start_date;
  const tag = args.tag;

  if (!startDateStr) {
    throw new Error("Missing start_date");
  }

  const startTimestamp = new Date(startDateStr).getTime() / 1000;
  const endTimestamp = new Date(endDateStr).getTime() / 1000 + 86399; // end of day

  const { results } = await env.DB.prepare(
    "SELECT id, tags, created_at FROM memories WHERE created_at BETWEEN ? AND ?"
  ).bind(startTimestamp, endTimestamp).all();

  const rows = results as unknown as MemoryRow[];

  let deleteCount = 0;
  for (const row of rows) {
    const memoryTags = JSON.parse(row.tags ?? "[]");
    if (!tag || memoryTags.includes(tag)) {
      await env.DB.prepare("DELETE FROM memories WHERE id = ?").bind(row.id).run();
      deleteCount++;
    }
  }

  return { deleted: deleteCount };
}

async function processDeleteBeforeDate(args: any, env: Env): Promise<any> {
  logger(env, "processDeleteBeforeDate called", args);
  const beforeDateStr = args.before_date;
  const tag = args.tag;

  if (!beforeDateStr) {
    throw new Error("Missing before_date");
  }

  const beforeTimestamp = new Date(beforeDateStr).getTime() / 1000;

  const { results } = await env.DB.prepare(
    "SELECT id, tags, created_at FROM memories WHERE created_at < ?"
  ).bind(beforeTimestamp).all();

  const rows = results as unknown as MemoryRow[];

  let deleteCount = 0;
  for (const row of rows) {
    const memoryTags = JSON.parse(row.tags ?? "[]");
    if (!tag || memoryTags.includes(tag)) {
      await env.DB.prepare("DELETE FROM memories WHERE id = ?").bind(row.id).run();
      deleteCount++;
    }
  }

  return { deleted: deleteCount };
}