# Advanced Hybrid Search - Real-World Usage Examples

## API Usage Examples

### Example 1: Project Troubleshooting Scenario

**Scenario**: Developer needs to find all information about deployment issues in Project Alpha

**REST API Call:**
```bash
curl -X POST "https://localhost:8443/api/search/advanced" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "query": "Project Alpha database timeout deployment error",
    "search_mode": "hybrid",
    "n_results": 20,
    "consolidate_related": true,
    "include_context": true,
    "filters": {
      "memory_types": ["task", "decision", "note", "reference"],
      "tags": ["project-alpha", "deployment", "database"],
      "time_range": "last 2 weeks",
      "metadata_filters": {
        "priority": ["high", "critical"],
        "status": ["in-progress", "completed", "failed"]
      }
    },
    "ranking_options": {
      "semantic_weight": 0.5,
      "keyword_weight": 0.4,
      "recency_weight": 0.1,
      "boost_exact_matches": true
    }
  }'
```

**Response:**
```json
{
  "results": [
    {
      "primary_memory": {
        "content": "Project Alpha production deployment failed at 2024-01-15 10:30 AM. Database connection timeout after 15 seconds. Error: Connection pool exhausted. Impact: 500+ users affected. Rolling back to previous version.",
        "content_hash": "pa_deploy_error_20240115",
        "tags": ["project-alpha", "deployment", "database", "production", "error"],
        "memory_type": "task",
        "created_at_iso": "2024-01-15T10:35:00Z",
        "metadata": {
          "priority": "critical",
          "status": "in-progress",
          "project_id": "alpha-001",
          "environment": "production",
          "impact_level": "high"
        }
      },
      "similarity_score": 0.98,
      "relevance_reason": "Exact match: 'Project Alpha', 'database timeout', 'deployment error' + high semantic similarity",
      "consolidation": {
        "related_memories": [
          {
            "content": "DECISION: Increase database connection timeout from 15s to 45s for Project Alpha. Approved by Tech Lead. Implementation scheduled for next deployment window.",
            "content_hash": "pa_timeout_decision_20240115",
            "relationship": "solution",
            "similarity_score": 0.89,
            "memory_type": "decision",
            "relevance_reason": "Direct solution to the identified problem"
          },
          {
            "content": "Project Alpha database configuration: Connection pool size: 20, Timeout: 15s, Retry attempts: 3. Performance baseline established.",
            "content_hash": "pa_db_config_baseline",
            "relationship": "context",
            "similarity_score": 0.85,
            "memory_type": "reference",
            "relevance_reason": "Configuration context for troubleshooting"
          },
          {
            "content": "Post-deployment monitoring shows Project Alpha database connections stabilized after timeout increase. No further timeout errors in 48 hours.",
            "content_hash": "pa_monitor_success_20240117",
            "relationship": "follow_up",
            "similarity_score": 0.82,
            "memory_type": "note",
            "relevance_reason": "Follow-up results and validation"
          }
        ],
        "topic_cluster": "project-alpha-database-deployment",
        "consolidation_summary": "Database timeout deployment issue resolved by increasing connection timeout from 15s to 45s. Monitoring confirms successful resolution.",
        "timeline": [
          {
            "date": "2024-01-15T10:30:00Z",
            "event": "Deployment failure detected",
            "type": "problem"
          },
          {
            "date": "2024-01-15T14:00:00Z", 
            "event": "Solution decided and approved",
            "type": "solution"
          },
          {
            "date": "2024-01-16T09:00:00Z",
            "event": "Fix implemented and deployed",
            "type": "implementation"
          },
          {
            "date": "2024-01-17T10:00:00Z",
            "event": "Success validated through monitoring",
            "type": "validation"
          }
        ]
      }
    }
  ],
  "consolidated_topics": [
    {
      "topic": "Project Alpha Database Issues",
      "memory_count": 8,
      "key_themes": ["timeout", "connection pool", "performance", "monitoring"],
      "timeline": "2024-01-10 to 2024-01-18",
      "status": "resolved"
    },
    {
      "topic": "Project Alpha Deployment Process",
      "memory_count": 15,
      "key_themes": ["rollback procedures", "deployment windows", "approval process"],
      "timeline": "2024-01-01 to present",
      "status": "ongoing"
    }
  ],
  "search_intelligence": {
    "query_analysis": {
      "intent": "troubleshooting",
      "entities": ["Project Alpha", "database timeout", "deployment error"],
      "confidence": 0.94,
      "suggested_filters": ["infrastructure", "database", "production"],
      "query_type": "problem_resolution"
    },
    "recommendations": [
      "Search for 'Project Alpha monitoring dashboard' for real-time metrics",
      "Consider searching 'database performance optimization' for preventive measures",
      "Review memories tagged with 'post-mortem' for similar incident analysis"
    ],
    "related_searches": [
      "Project Alpha performance metrics",
      "database connection pool tuning",
      "deployment rollback procedures"
    ]
  },
  "performance_metrics": {
    "total_processing_time_ms": 87,
    "semantic_search_time_ms": 34,
    "keyword_search_time_ms": 12,
    "consolidation_time_ms": 28,
    "relationship_mapping_time_ms": 13
  }
}
```

### Example 2: Knowledge Discovery Scenario

**Scenario**: Product manager wants to understand all decisions made about user authentication

**REST API Call:**
```bash
curl -X POST "https://localhost:8443/api/search/advanced" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "query": "user authentication security decisions",
    "search_mode": "auto",
    "n_results": 25,
    "consolidate_related": true,
    "include_context": true,
    "filters": {
      "memory_types": ["decision", "note"],
      "time_range": "last 6 months",
      "metadata_filters": {
        "category": ["security", "architecture", "user-experience"]
      }
    }
  }'
```

## MCP API Usage Examples

### Example 1: MCP Tool via HTTP Bridge

**Request:**
```http
POST /api/mcp/tools/call
Content-Type: application/json

{
  "tool_name": "advanced_memory_search",
  "arguments": {
    "query": "API rate limiting implementation discussion",
    "search_mode": "hybrid",
    "consolidate_related": true,
    "max_results": 15,
    "filters": {
      "memory_types": ["decision", "task", "reference"],
      "tags": ["api", "rate-limiting", "performance"],
      "time_range": "last month"
    }
  }
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "search_results": [
      {
        "primary_content": "DECISION: Implement token bucket rate limiting for public API endpoints. Limit: 1000 requests/hour per API key. Burst capacity: 100 requests. Approved by architecture team.",
        "content_hash": "api_rate_limit_decision_001",
        "relevance_score": 0.95,
        "memory_type": "decision",
        "tags": ["api", "rate-limiting", "architecture", "approved"],
        "created_at": "2024-01-10T14:30:00Z",
        "consolidation": {
          "related_content": [
            {
              "content": "Research: Token bucket vs sliding window rate limiting algorithms. Token bucket provides better burst handling for API scenarios.",
              "relationship": "background_research",
              "memory_type": "reference"
            },
            {
              "content": "TASK: Implement rate limiting middleware in Express.js API server. Use redis for distributed rate limit storage. Due: 2024-01-20",
              "relationship": "implementation_task", 
              "memory_type": "task",
              "status": "completed"
            }
          ],
          "topic_summary": "API rate limiting decision with token bucket algorithm, researched and implemented successfully"
        }
      }
    ],
    "total_found": 8,
    "consolidated_topics": [
      {
        "topic": "API Security & Performance",
        "memory_count": 12,
        "key_themes": ["rate limiting", "authentication", "caching", "monitoring"]
      }
    ],
    "processing_time_ms": 45
  }
}
```

### Example 2: Direct MCP Server Call (Claude Desktop)

**MCP Tool Definition:**
```json
{
  "name": "advanced_memory_search",
  "description": "Search memories using hybrid semantic + keyword search with automatic content consolidation",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Search query combining keywords and concepts"
      },
      "search_mode": {
        "type": "string",
        "enum": ["hybrid", "semantic", "keyword", "auto"],
        "default": "hybrid",
        "description": "Search strategy to use"
      },
      "consolidate_related": {
        "type": "boolean", 
        "default": true,
        "description": "Automatically group related memories and provide context"
      },
      "max_results": {
        "type": "integer",
        "default": 10,
        "minimum": 1,
        "maximum": 50,
        "description": "Maximum number of results to return"
      },
      "filters": {
        "type": "object",
        "properties": {
          "memory_types": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Filter by memory types: task, decision, note, reference"
          },
          "tags": {
            "type": "array", 
            "items": {"type": "string"},
            "description": "Filter by specific tags"
          },
          "time_range": {
            "type": "string",
            "description": "Natural language time filter: 'last week', 'this month', etc."
          },
          "metadata_filters": {
            "type": "object",
            "description": "Filter by metadata fields and values"
          }
        }
      }
    },
    "required": ["query"]
  }
}
```

**Example Usage in Claude Desktop:**

**User Query:** "Find all discussions about microservices architecture decisions"

**MCP Tool Call:**
```json
{
  "tool": "advanced_memory_search",
  "arguments": {
    "query": "microservices architecture decisions discussion",
    "search_mode": "hybrid",
    "consolidate_related": true,
    "max_results": 20,
    "filters": {
      "memory_types": ["decision", "note", "reference"],
      "tags": ["microservices", "architecture"],
      "time_range": "last 3 months"
    }
  }
}
```

**Claude's Response Based on Tool Result:**

"I found 15 memories related to microservices architecture decisions from the last 3 months. Here's what I discovered:

## Key Architecture Decisions

**Primary Decision (Jan 15, 2024):**
*DECISION: Adopt microservices architecture for the new customer platform. Services: User Management, Order Processing, Payment Gateway, Notification Service. Each service will have its own database and API. Approved by architecture committee.*

**Related Context:**
- **Research Background**: Comparison study of monolith vs microservices showed 40% better scalability for our use case
- **Implementation Plan**: 6-month phased migration starting with User Management service
- **Success Metrics**: Service response time < 200ms, 99.9% uptime target

## Consolidated Topics Found:

1. **Service Communication** (8 memories)
   - Event-driven architecture using RabbitMQ
   - API gateway implementation with Kong
   - Service discovery with Consul

2. **Data Management** (4 memories)  
   - Database per service pattern
   - Event sourcing for audit trails
   - Data consistency strategies

3. **DevOps & Deployment** (3 memories)
   - Docker containerization approach
   - Kubernetes orchestration setup
   - CI/CD pipeline modifications

The search found strong relationships between the architecture decisions and their implementation details, providing a complete picture of your microservices adoption journey."

### Example 3: Project Consolidation MCP Tool

**MCP Tool Definition:**
```json
{
  "name": "consolidate_project_memories",
  "description": "Automatically gather and organize all memories related to a specific project",
  "inputSchema": {
    "type": "object", 
    "properties": {
      "project_identifier": {
        "type": "string",
        "description": "Project name or identifier (e.g., 'Project Alpha', 'mobile-app-v2')"
      },
      "consolidation_depth": {
        "type": "string",
        "enum": ["shallow", "deep", "comprehensive"],
        "default": "deep",
        "description": "How extensively to search for related content"
      },
      "include_timeline": {
        "type": "boolean",
        "default": true,
        "description": "Generate chronological timeline of project events"
      }
    },
    "required": ["project_identifier"]
  }
}
```

**Usage Example:**
```json
{
  "tool": "consolidate_project_memories", 
  "arguments": {
    "project_identifier": "mobile app redesign",
    "consolidation_depth": "comprehensive",
    "include_timeline": true
  }
}
```

**Tool Response:**
```json
{
  "project_overview": {
    "name": "Mobile App Redesign",
    "total_memories": 47,
    "date_range": "2023-11-01 to 2024-01-20",
    "status": "in_progress",
    "key_stakeholders": ["Product Team", "UX Design", "Mobile Dev Team"]
  },
  "timeline": [
    {
      "date": "2023-11-01",
      "event": "Project kickoff and requirements gathering",
      "type": "milestone",
      "memories": 3
    },
    {
      "date": "2023-11-15", 
      "event": "UX wireframes and user research completed",
      "type": "deliverable",
      "memories": 8
    },
    {
      "date": "2023-12-01",
      "event": "Technical architecture decisions finalized", 
      "type": "decision",
      "memories": 5
    }
  ],
  "key_decisions": [
    {
      "content": "DECISION: Use React Native for cross-platform development. Allows 80% code sharing between iOS/Android. Team already familiar with React.",
      "impact": "high",
      "date": "2023-11-20"
    }
  ],
  "outstanding_issues": [
    {
      "content": "ISSUE: Performance concerns with large image galleries in React Native. Need optimization strategy.",
      "priority": "high", 
      "status": "open",
      "assigned_to": "Mobile Dev Team"
    }
  ],
  "related_projects": [
    {
      "name": "API v2 Migration",
      "relationship": "dependency",
      "status": "completed"
    }
  ]
}
```

## Claude Code Integration Examples

### Example 1: Claude Code Slash Command

```bash
# Enhanced memory search with consolidation
claude /memory-search-advanced "database performance optimization" --consolidate --filters="tags:database,performance;type:decision,reference"

# Quick project overview
claude /memory-project-overview "Project Beta" --timeline --issues

# Intelligent search with auto-suggestions
claude /memory-smart-search "user feedback login problems" --auto-expand --suggest-actions
```

### Example 2: Claude Code Hook Integration

**Session Hook Usage:**
```javascript
// .claude/hooks/memory-enhanced-search.js
module.exports = {
  name: "enhanced-memory-search",
  description: "Automatically use hybrid search for memory queries",
  trigger: "before_memory_search",
  
  async execute(context) {
    // Automatically enhance memory searches with hybrid mode
    if (context.tool === "retrieve_memory") {
      context.arguments.search_mode = "hybrid";
      context.arguments.consolidate_related = true;
      context.arguments.include_context = true;
    }
    return context;
  }
};
```

These examples demonstrate how the Advanced Hybrid Search enhancement provides rich, contextual, and intelligent search capabilities through both REST API and MCP interfaces, making it easy for users to find and understand related information in their memory store.