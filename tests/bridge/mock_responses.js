/**
 * Mock Server Responses for Testing
 * 
 * These responses match the ACTUAL behavior of the MCP Memory Service API,
 * not what we might assume or hope it returns.
 */

const mockResponses = {
    // Health endpoint responses
    health: {
        healthy: {
            status: 200, // NOT 204 or other codes
            body: {
                status: 'healthy',
                version: '6.6.1',
                timestamp: '2025-08-24T12:00:00Z',
                uptime_seconds: 3600,
                storage_type: 'sqlite_vec',
                statistics: {
                    total_memories: 100,
                    total_tags: 25
                }
            }
        },
        unhealthy: {
            status: 503,
            body: {
                status: 'unhealthy',
                error: 'Database connection failed'
            }
        }
    },
    
    // Memory storage responses - CRITICAL: Server returns 200, not 201!
    memories: {
        createSuccess: {
            status: 200, // ACTUAL: Returns 200, not 201 for creation!
            body: {
                success: true, // Key field for determining actual success
                message: 'Memory stored successfully',
                content_hash: 'abc123def456',
                memory: {
                    content: 'Test memory content',
                    content_hash: 'abc123def456',
                    tags: ['test', 'source:test-client'],
                    memory_type: 'note',
                    metadata: {
                        hostname: 'test-client'
                    },
                    created_at: 1756054456.123,
                    created_at_iso: '2025-08-24T12:00:00.123Z',
                    updated_at: 1756054456.123,
                    updated_at_iso: '2025-08-24T12:00:00.123Z'
                }
            }
        },
        duplicate: {
            status: 200, // SAME status code as success!
            body: {
                success: false, // This field determines it's a duplicate
                message: 'Duplicate content detected',
                content_hash: 'abc123def456',
                memory: null
            }
        },
        invalidRequest: {
            status: 400,
            body: {
                detail: 'Invalid request: content is required'
            }
        },
        unauthorized: {
            status: 401,
            body: {
                detail: 'Invalid API key'
            }
        },
        serverError: {
            status: 500,
            body: {
                detail: 'Internal server error'
            }
        }
    },
    
    // Memory retrieval/search responses
    search: {
        withResults: {
            status: 200,
            body: {
                results: [
                    {
                        memory: {
                            content: 'Matching memory content',
                            content_hash: 'hash1',
                            tags: ['test', 'search'],
                            memory_type: 'note',
                            created_at_iso: '2025-08-24T11:00:00Z',
                            metadata: {}
                        },
                        relevance_score: 0.95
                    },
                    {
                        memory: {
                            content: 'Another matching memory',
                            content_hash: 'hash2',
                            tags: ['test'],
                            memory_type: 'reference',
                            created_at_iso: '2025-08-24T10:00:00Z',
                            metadata: {}
                        },
                        relevance_score: 0.87
                    }
                ]
            }
        },
        empty: {
            status: 200,
            body: {
                results: []
            }
        }
    },
    
    // Tag search responses
    tagSearch: {
        withResults: {
            status: 200,
            body: {
                memories: [
                    {
                        content: 'Memory with specific tag',
                        content_hash: 'tag_hash1',
                        tags: ['specific-tag', 'other-tag'],
                        memory_type: 'note',
                        created_at_iso: '2025-08-24T09:00:00Z'
                    }
                ]
            }
        },
        empty: {
            status: 200,
            body: {
                memories: []
            }
        }
    },
    
    // Delete memory responses
    deleteMemory: {
        success: {
            status: 200,
            body: {
                success: true,
                message: 'Memory deleted successfully'
            }
        },
        notFound: {
            status: 404,
            body: {
                detail: 'Memory not found'
            }
        }
    },
    
    // Edge cases and error conditions
    edgeCases: {
        // When the /api path is missing (404 because endpoint wrong)
        missingApiPath: {
            status: 404,
            body: {
                detail: 'Not Found'
            }
        },
        // Network timeout
        timeout: {
            error: new Error('ETIMEDOUT')
        },
        // Connection refused
        connectionRefused: {
            error: new Error('ECONNREFUSED')
        },
        // Invalid JSON response
        invalidJson: {
            status: 200,
            body: 'This is not JSON', // String instead of object
            raw: true
        },
        // HTML error page instead of JSON
        htmlError: {
            status: 500,
            body: '<html><body>500 Internal Server Error</body></html>',
            raw: true,
            contentType: 'text/html'
        }
    }
};

/**
 * Helper function to create a mock HTTP response object
 */
function createMockResponse(mockData) {
    if (mockData.error) {
        throw mockData.error;
    }
    
    return {
        statusCode: mockData.status,
        headers: {
            'content-type': mockData.contentType || 'application/json'
        },
        on: (event, callback) => {
            if (event === 'data') {
                const data = mockData.raw ? 
                    mockData.body : 
                    JSON.stringify(mockData.body);
                callback(Buffer.from(data));
            } else if (event === 'end') {
                callback();
            }
        }
    };
}

/**
 * Helper to create a mock request object
 */
function createMockRequest() {
    const req = {
        on: (event, callback) => {
            if (event === 'error') {
                // Store error handler
                req.errorHandler = callback;
            } else if (event === 'timeout') {
                req.timeoutHandler = callback;
            }
        },
        write: () => {},
        end: () => {},
        destroy: () => {},
        setTimeout: () => {}
    };
    return req;
}

module.exports = {
    mockResponses,
    createMockResponse,
    createMockRequest
};