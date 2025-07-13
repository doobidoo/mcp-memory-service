#!/usr/bin/env python3
"""
Simple health check utility for MCP Memory Service.
"""
import asyncio
import json
import os
import sys
from typing import Dict, Any

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import MCP client
try:
    from mcp import Client
except ImportError:
    print("MCP client not found. Install with: pip install mcp")
    sys.exit(1)

async def health_check() -> Dict[str, Any]:
    """Run a health check on the memory service."""
    try:
        # Connect to memory service
        client = Client("memory")
        
        # Wait for connection
        await client.initialize()
        print("Connected to memory service!")
        
        # Get available tools
        tools = await client.list_tools()
        print(f"Found {len(tools)} tools")
        
        # Check if health check tool is available
        health_check_tool = None
        for tool in tools:
            if tool.name == "check_database_health":
                health_check_tool = tool
                break
        
        if not health_check_tool:
            print("ERROR: Health check tool not found")
            return {"error": "Health check tool not available"}
        
        # Run health check
        print("\nRunning database health check...")
        result = await client.call_tool("check_database_health", {})
        
        # Parse the result (it's returned as a text string)
        if result and result[0].text:
            # Extract the JSON part from the response
            text = result[0].text
            json_start = text.find('{')
            if json_start >= 0:
                try:
                    health_data = json.loads(text[json_start:])
                    return health_data
                except json.JSONDecodeError:
                    print(f"ERROR: Failed to parse health check result: {text}")
                    return {"error": "Failed to parse health check result"}
            else:
                return {"error": "No JSON data found in response", "raw": text}
        else:
            return {"error": "Empty response from health check"}
    
    except Exception as e:
        print(f"ERROR: Health check failed: {str(e)}")
        return {"error": str(e)}
    finally:
        # Disconnect client
        await client.shutdown()

async def main():
    """Main function."""
    print("=== MCP Memory Service Health Check ===\n")
    
    # Check if memory service is configured
    result = await health_check()
    
    # Print results
    print("\n=== Health Check Results ===")
    print(json.dumps(result, indent=2))
    
    # Check for validation status
    if "validation" in result:
        status = result["validation"].get("status")
        if status == "healthy":
            print("\n✅ Database is HEALTHY")
        else:
            print(f"\n❌ Database is UNHEALTHY: {result['validation'].get('message', 'No message')}")
    
    # Check for statistics
    if "statistics" in result and result["statistics"].get("status") != "error":
        stats = result["statistics"]
        print(f"\nTotal memories: {stats.get('total_memories', 'unknown')}")
        print(f"Backend: {stats.get('backend', 'unknown')}")
    
    # Check for performance
    if "performance" in result:
        perf = result["performance"].get("server", {})
        print(f"\nTotal queries: {perf.get('total_queries', 0)}")
        print(f"Avg query time: {perf.get('average_query_time_ms', 0)} ms")

if __name__ == "__main__":
    asyncio.run(main())