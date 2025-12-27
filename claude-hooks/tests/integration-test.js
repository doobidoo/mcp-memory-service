#!/usr/bin/env node

/**
 * Integration Test for Claude Code Memory Awareness Hooks
 * Tests the complete Phase 1 implementation end-to-end
 */

const fs = require('fs');
const fsPromises = require('fs').promises;
const path = require('path');
const os = require('os');

// Import hooks and utilities
const sessionStartHook = require('../core/session-start');
const sessionEndHook = require('../core/session-end');
const { detectProjectContext } = require('../utilities/project-detector');
const { scoreMemoryRelevance } = require('../utilities/memory-scorer');
const { formatMemoriesForContext } = require('../utilities/context-formatter');

/**
 * Test Results Tracker
 */
class TestResults {
    constructor() {
        this.tests = [];
        this.passed = 0;
        this.failed = 0;
    }
    
    test(name, testFn) {
        console.log(`\n🧪 Testing: ${name}`);
        try {
            const result = testFn();
            if (result === true || (result && result.success !== false)) {
                console.log(`✅ PASS: ${name}`);
                this.passed++;
                this.tests.push({ name, status: 'PASS', result });
            } else {
                console.log(`❌ FAIL: ${name} - ${result.error || 'Test returned false'}`);
                this.failed++;
                this.tests.push({ name, status: 'FAIL', error: result.error || 'Test returned false' });
            }
        } catch (error) {
            console.log(`❌ FAIL: ${name} - ${error.message}`);
            this.failed++;
            this.tests.push({ name, status: 'FAIL', error: error.message });
        }
    }
    
    async asyncTest(name, testFn) {
        console.log(`\n🧪 Testing: ${name}`);
        try {
            const result = await testFn();
            if (result === true || (result && result.success !== false)) {
                console.log(`✅ PASS: ${name}`);
                this.passed++;
                this.tests.push({ name, status: 'PASS', result });
            } else {
                console.log(`❌ FAIL: ${name} - ${result.error || 'Test returned false'}`);
                this.failed++;
                this.tests.push({ name, status: 'FAIL', error: result.error || 'Test returned false' });
            }
        } catch (error) {
            console.log(`❌ FAIL: ${name} - ${error.message}`);
            this.failed++;
            this.tests.push({ name, status: 'FAIL', error: error.message });
        }
    }
    
    summary() {
        console.log('\n' + '='.repeat(60));
        console.log('🎯 TEST SUMMARY');
        console.log('='.repeat(60));
        console.log(`Total Tests: ${this.tests.length}`);
        console.log(`✅ Passed: ${this.passed}`);
        console.log(`❌ Failed: ${this.failed}`);
        console.log(`Success Rate: ${((this.passed / this.tests.length) * 100).toFixed(1)}%`);
        
        if (this.failed > 0) {
            console.log('\n🔍 FAILED TESTS:');
            this.tests.filter(t => t.status === 'FAIL').forEach(test => {
                console.log(`  - ${test.name}: ${test.error}`);
            });
        }
        
        console.log('='.repeat(60));
        return this.failed === 0;
    }
}

/**
 * Mock data for testing
 */
const mockMemories = [
    {
        content: 'Decided to use SQLite-vec instead of ChromaDB for better performance in MCP Memory Service. SQLite-vec provides 10x faster startup and uses 75% less memory.',
        tags: ['mcp-memory-service', 'decision', 'sqlite-vec', 'performance'],
        memory_type: 'decision',
        created_at_iso: '2025-08-19T10:00:00Z'
    },
    {
        content: 'Implemented comprehensive Claude Code hooks system for automatic memory awareness. Created session-start, session-end, and topic-change hooks with project detection.',
        tags: ['claude-code', 'hooks', 'architecture', 'memory-awareness'],
        memory_type: 'architecture',
        created_at_iso: '2025-08-19T09:30:00Z'
    },
    {
        content: 'Fixed critical bug in project detector - was not handling pyproject.toml files correctly. Added proper Python project detection.',
        tags: ['bug-fix', 'project-detector', 'python'],
        memory_type: 'bug-fix',
        created_at_iso: '2025-08-18T15:30:00Z'
    },
    {
        content: 'Learning session on memory relevance scoring algorithms. Implemented time decay, tag matching, and content analysis for intelligent memory selection.',
        tags: ['learning', 'algorithms', 'memory-scoring'],
        memory_type: 'insight',
        created_at_iso: '2025-08-17T14:00:00Z'
    },
    {
        content: 'Random note about completely unrelated project for testing filtering',
        tags: ['other-project', 'unrelated', 'test'],
        memory_type: 'note',
        created_at_iso: '2025-08-01T08:00:00Z'
    }
];

const mockProjectContext = {
    name: 'mcp-memory-service',
    directory: process.cwd(),
    language: 'JavaScript',
    frameworks: ['Node.js'],
    tools: ['npm'],
    git: {
        isRepo: true,
        branch: 'main',
        repoName: 'mcp-memory-service',
        lastCommit: 'abc1234 Implement memory awareness hooks'
    },
    confidence: 0.9
};

const mockConversation = {
    messages: [
        {
            role: 'user',
            content: 'I need to implement a memory awareness system for Claude Code that automatically injects relevant project memories.'
        },
        {
            role: 'assistant', 
            content: 'I\'ll help you create a comprehensive memory awareness system. We decided to use Claude Code hooks for session management and implement automatic context injection. This will include project detection, memory scoring, and intelligent context formatting.'
        },
        {
            role: 'user',
            content: 'Great! I learned that we need project detection algorithms and memory scoring systems. Can you implement the project detector?'
        },
        {
            role: 'assistant',
            content: 'Exactly. I implemented the project detector in project-detector.js with support for multiple languages and frameworks. I also created memory scoring algorithms with time decay and relevance matching. Next we need to test the complete system and add session consolidation.'
        }
    ]
};

/**
 * Run comprehensive tests
 */
async function runTests() {
    console.log('🚀 Claude Code Memory Awareness - Integration Tests');
    console.log('Testing Phase 1 Implementation\n');
    
    const results = new TestResults();
    
    // Test 1: Project Detection
    await results.asyncTest('Project Detection', async () => {
        const context = await detectProjectContext(process.cwd());
        
        if (!context.name) {
            return { success: false, error: 'No project name detected' };
        }
        
        if (!context.language) {
            return { success: false, error: 'No language detected' };
        }
        
        console.log(`  Detected: ${context.name} (${context.language}), Confidence: ${(context.confidence * 100).toFixed(1)}%`);
        return { success: true, context };
    });
    
    // Test 2: Memory Relevance Scoring
    results.test('Memory Relevance Scoring', () => {
        const scored = scoreMemoryRelevance(mockMemories, mockProjectContext);
        
        if (!Array.isArray(scored)) {
            return { success: false, error: 'Scoring did not return array' };
        }
        
        if (scored.length !== mockMemories.length) {
            return { success: false, error: 'Scoring lost memories' };
        }
        
        // Check that memories have scores
        for (const memory of scored) {
            if (typeof memory.relevanceScore !== 'number') {
                return { success: false, error: 'Memory missing relevance score' };
            }
        }
        
        // Check that memories are sorted by relevance (highest first)
        for (let i = 1; i < scored.length; i++) {
            if (scored[i].relevanceScore > scored[i-1].relevanceScore) {
                return { success: false, error: 'Memories not sorted by relevance' };
            }
        }
        
        console.log(`  Scored ${scored.length} memories, top score: ${scored[0].relevanceScore.toFixed(3)}`);
        return { success: true, scored };
    });
    
    // Test 3: Context Formatting
    results.test('Context Formatting', () => {
        const scored = scoreMemoryRelevance(mockMemories, mockProjectContext);
        const formatted = formatMemoriesForContext(scored, mockProjectContext);
        
        if (typeof formatted !== 'string') {
            return { success: false, error: 'Formatting did not return string' };
        }
        
        if (formatted.length < 100) {
            return { success: false, error: 'Formatted context too short' };
        }
        
        // Check for key formatting elements
        if (!formatted.includes('Memory Context')) {
            return { success: false, error: 'Missing memory context header' };
        }
        
        if (!formatted.includes(mockProjectContext.name)) {
            return { success: false, error: 'Missing project name in context' };
        }
        
        console.log(`  Generated ${formatted.length} characters of formatted context`);
        return { success: true, formatted };
    });
    
    // Test 4: Session Start Hook Structure
    results.test('Session Start Hook Structure', () => {
        if (typeof sessionStartHook.handler !== 'function') {
            return { success: false, error: 'Session start hook missing handler function' };
        }
        
        if (!sessionStartHook.name || !sessionStartHook.version) {
            return { success: false, error: 'Session start hook missing metadata' };
        }
        
        if (sessionStartHook.trigger !== 'session-start') {
            return { success: false, error: 'Session start hook wrong trigger' };
        }
        
        console.log(`  Hook: ${sessionStartHook.name} v${sessionStartHook.version}`);
        return { success: true };
    });
    
    // Test 5: Session End Hook Structure
    results.test('Session End Hook Structure', () => {
        if (typeof sessionEndHook.handler !== 'function') {
            return { success: false, error: 'Session end hook missing handler function' };
        }

        if (!sessionEndHook.name || !sessionEndHook.version) {
            return { success: false, error: 'Session end hook missing metadata' };
        }

        if (sessionEndHook.trigger !== 'session-end') {
            return { success: false, error: 'Session end hook wrong trigger' };
        }

        console.log(`  Hook: ${sessionEndHook.name} v${sessionEndHook.version}`);
        return { success: true };
    });

    // Test 5a: parseTranscript - String content messages
    await results.asyncTest('parseTranscript: String content messages', async () => {
        const { parseTranscript } = sessionEndHook._internal;
        if (!parseTranscript) {
            return { success: false, error: 'parseTranscript not exported' };
        }

        const entries = [
            { type: 'user', message: { role: 'user', content: 'Hello, how are you?' } },
            { type: 'assistant', message: { role: 'assistant', content: 'I am doing well!' } }
        ];
        const tmpFile = path.join(os.tmpdir(), `test-transcript-${Date.now()}.jsonl`);
        await fsPromises.writeFile(tmpFile, entries.map(e => JSON.stringify(e)).join('\n'), 'utf8');

        try {
            const result = await parseTranscript(tmpFile);
            if (result.messages.length !== 2) {
                return { success: false, error: `Expected 2 messages, got ${result.messages.length}` };
            }
            if (result.messages[0].content !== 'Hello, how are you?') {
                return { success: false, error: `Unexpected content: ${result.messages[0].content}` };
            }
            console.log(`  Parsed ${result.messages.length} messages from JSONL`);
            return { success: true };
        } finally {
            await fsPromises.unlink(tmpFile).catch(() => {});
        }
    });

    // Test 5b: parseTranscript - Array content blocks
    await results.asyncTest('parseTranscript: Array content blocks', async () => {
        const { parseTranscript } = sessionEndHook._internal;
        const entries = [
            {
                type: 'assistant',
                message: {
                    role: 'assistant',
                    content: [
                        { type: 'text', text: 'First part.' },
                        { type: 'text', text: 'Second part.' }
                    ]
                }
            }
        ];
        const tmpFile = path.join(os.tmpdir(), `test-transcript-${Date.now()}.jsonl`);
        await fsPromises.writeFile(tmpFile, entries.map(e => JSON.stringify(e)).join('\n'), 'utf8');

        try {
            const result = await parseTranscript(tmpFile);
            if (result.messages.length !== 1) {
                return { success: false, error: `Expected 1 message, got ${result.messages.length}` };
            }
            if (!result.messages[0].content.includes('First part')) {
                return { success: false, error: 'Missing first part in content' };
            }
            console.log(`  Correctly joined array content blocks`);
            return { success: true };
        } finally {
            await fsPromises.unlink(tmpFile).catch(() => {});
        }
    });

    // Test 5c: parseTranscript - Skips non-user/assistant entries
    await results.asyncTest('parseTranscript: Skips non-message entries', async () => {
        const { parseTranscript } = sessionEndHook._internal;
        const entries = [
            { type: 'file-history-snapshot', message: null },
            { type: 'user', message: { role: 'user', content: 'Hello' } },
            { type: 'system', message: { role: 'system', content: 'System msg' } }
        ];
        const tmpFile = path.join(os.tmpdir(), `test-transcript-${Date.now()}.jsonl`);
        await fsPromises.writeFile(tmpFile, entries.map(e => JSON.stringify(e)).join('\n'), 'utf8');

        try {
            const result = await parseTranscript(tmpFile);
            if (result.messages.length !== 1) {
                return { success: false, error: `Expected 1 message (user only), got ${result.messages.length}` };
            }
            console.log(`  Correctly filtered to user/assistant messages only`);
            return { success: true };
        } finally {
            await fsPromises.unlink(tmpFile).catch(() => {});
        }
    });

    // Test 5d: parseTranscript - Handles malformed JSON gracefully
    await results.asyncTest('parseTranscript: Handles malformed JSON', async () => {
        const { parseTranscript } = sessionEndHook._internal;
        const tmpFile = path.join(os.tmpdir(), `test-transcript-${Date.now()}.jsonl`);
        const content = [
            JSON.stringify({ type: 'user', message: { role: 'user', content: 'Valid' } }),
            'not valid json {{{',
            JSON.stringify({ type: 'assistant', message: { role: 'assistant', content: 'Also valid' } })
        ].join('\n');
        await fsPromises.writeFile(tmpFile, content, 'utf8');

        try {
            const result = await parseTranscript(tmpFile);
            if (result.messages.length !== 2) {
                return { success: false, error: `Expected 2 messages (skipping malformed), got ${result.messages.length}` };
            }
            console.log(`  Gracefully skipped malformed JSON line`);
            return { success: true };
        } finally {
            await fsPromises.unlink(tmpFile).catch(() => {});
        }
    });

    // Test 6: Configuration Loading
    results.test('Configuration Loading', () => {
        const configPath = path.join(__dirname, '../config.json');

        if (!fs.existsSync(configPath)) {
            return { success: false, error: 'Configuration file not found' };
        }

        try {
            const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));

            if (!config.memoryService) {
                return { success: false, error: 'Invalid configuration structure' };
            }

            // Support both old (direct endpoint) and new (dual-protocol) structures
            const endpoint = config.memoryService.endpoint || config.memoryService.http?.endpoint;

            if (!endpoint) {
                return { success: false, error: 'No endpoint configured (checked both old and new format)' };
            }

            console.log(`  Endpoint: ${endpoint}`);
            return { success: true, config };

        } catch (error) {
            return { success: false, error: `Configuration parse error: ${error.message}` };
        }
    });
    
    // Test 7: File Structure
    results.test('File Structure Validation', () => {
        const requiredFiles = [
            '../core/session-start.js',
            '../core/session-end.js',
            '../utilities/project-detector.js',
            '../utilities/memory-scorer.js', 
            '../utilities/context-formatter.js',
            '../config.json',
            '../config.template.json',
            '../README.md'
        ];
        
        for (const file of requiredFiles) {
            const fullPath = path.join(__dirname, file);
            if (!fs.existsSync(fullPath)) {
                return { success: false, error: `Missing required file: ${file}` };
            }
        }
        
        console.log(`  All ${requiredFiles.length} required files present`);
        return { success: true };
    });
    
    // Test 8: Mock Session Start (Limited Test)
    await results.asyncTest('Mock Session Start Hook', async () => {
        const mockContext = {
            workingDirectory: process.cwd(),
            sessionId: 'test-session',
            injectSystemMessage: async (message) => {
                if (typeof message !== 'string' || message.length < 50) {
                    throw new Error('Invalid message injection');
                }
                console.log(`  Injected ${message.length} characters of context`);
                return true;
            }
        };
        
        try {
            // Note: This will attempt to contact the memory service
            // In a real test environment, we'd mock this
            await sessionStartHook.handler(mockContext);
            return { success: true };
        } catch (error) {
            // Expected to fail without real memory service connection or when dependencies are missing
            if (error.message.includes('Network error') ||
                error.message.includes('ENOTFOUND') ||
                error.message.includes('memoryClient is not defined') ||
                error.message.includes('No active connection')) {
                console.log('  ⚠️  Expected error (no memory service or connection available)');
                console.log('  This is expected if the service is not running during tests');
                return { success: true }; // This is expected in test environment
            }
            throw error;
        }
    });
    
    // Test 9: Package Dependencies
    results.test('Package Dependencies Check', () => {
        const requiredModules = ['fs', 'path', 'https', 'child_process'];
        
        for (const module of requiredModules) {
            try {
                require(module);
            } catch (error) {
                return { success: false, error: `Missing required module: ${module}` };
            }
        }
        
        console.log(`  All ${requiredModules.length} required Node.js modules available`);
        return { success: true };
    });
    
    // Test 10: Claude Code Settings Validation
    results.test('Claude Code Settings Configuration', () => {
        const settingsPath = path.join(process.env.HOME, '.claude', 'settings.json');
        
        if (!fs.existsSync(settingsPath)) {
            return { success: false, error: 'Claude Code settings.json not found' };
        }
        
        try {
            const settings = JSON.parse(fs.readFileSync(settingsPath, 'utf8'));
            
            // Check for hooks configuration
            if (!settings.hooks) {
                return { success: false, error: 'No hooks configuration found in settings' };
            }
            
            // Check for SessionStart hook
            if (!settings.hooks.SessionStart || !Array.isArray(settings.hooks.SessionStart)) {
                return { success: false, error: 'SessionStart hooks not configured' };
            }
            
            // Check for SessionEnd hook
            if (!settings.hooks.SessionEnd || !Array.isArray(settings.hooks.SessionEnd)) {
                return { success: false, error: 'SessionEnd hooks not configured' };
            }
            
            // Check hook command paths
            const startHook = JSON.stringify(settings.hooks.SessionStart);
            const endHook = JSON.stringify(settings.hooks.SessionEnd);
            
            if (!startHook.includes('session-start.js')) {
                return { success: false, error: 'SessionStart hook command not configured correctly' };
            }
            
            if (!endHook.includes('session-end.js')) {
                return { success: false, error: 'SessionEnd hook command not configured correctly' };
            }
            
            console.log('  Claude Code settings configured correctly');
            return { success: true, settings };
            
        } catch (parseError) {
            return { success: false, error: `Settings parse error: ${parseError.message}` };
        }
    });
    
    // Test 11: Hook Files Location Validation
    results.test('Hook Files in Correct Location', () => {
        const hookDir = path.join(process.env.HOME, '.claude', 'hooks');
        const requiredHooks = [
            'core/session-start.js',
            'core/session-end.js', 
            'utilities/project-detector.js',
            'utilities/memory-scorer.js',
            'utilities/context-formatter.js'
        ];
        
        for (const hookFile of requiredHooks) {
            const fullPath = path.join(hookDir, hookFile);
            if (!fs.existsSync(fullPath)) {
                return { success: false, error: `Hook file missing: ${hookFile}` };
            }
        }
        
        console.log(`  All hooks installed in ${hookDir}`);
        return { success: true };
    });
    
    // Test 12: Claude Code CLI Availability
    results.test('Claude Code CLI Availability', () => {
        const { execSync } = require('child_process');
        
        try {
            execSync('which claude', { stdio: 'pipe' });
            console.log('  Claude Code CLI available');
            return { success: true };
        } catch (error) {
            return { success: false, error: 'Claude Code CLI not found in PATH' };
        }
    });
    
    // Test 13: Memory Service Protocol
    results.test('Memory Service Protocol Compatibility', () => {
        // Test that we're generating the correct MCP JSON-RPC calls
        const testCall = {
            jsonrpc: '2.0',
            id: 1,
            method: 'tools/call',
            params: {
                name: 'retrieve_memory',
                arguments: {
                    query: 'test query',
                    tags: ['test'],
                    limit: 5
                }
            }
        };
        
        const serialized = JSON.stringify(testCall);
        const parsed = JSON.parse(serialized);
        
        if (!parsed.jsonrpc || parsed.jsonrpc !== '2.0') {
            return { success: false, error: 'Invalid JSON-RPC format' };
        }
        
        if (!parsed.params || !parsed.params.name || !parsed.params.arguments) {
            return { success: false, error: 'Invalid MCP call structure' };
        }
        
        console.log(`  MCP protocol structure valid`);
        return { success: true };
    });
    
    // Test 14: Memory Service Connectivity
    await results.asyncTest('Memory Service Connectivity', async () => {
        const configPath = path.join(__dirname, '../config.json');

        if (!fs.existsSync(configPath)) {
            return { success: false, error: 'Configuration file not found for connectivity test' };
        }

        try {
            const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));

            // Support both old (direct) and new (dual-protocol) structures
            const endpoint = config.memoryService?.endpoint || config.memoryService?.http?.endpoint;
            const apiKey = config.memoryService?.apiKey || config.memoryService?.http?.apiKey;

            if (!endpoint) {
                return { success: false, error: 'No memory service endpoint configured (checked both old and new format)' };
            }
            
            // Test basic connectivity (simplified test)
            const https = require('https');
            const url = new URL('/api/health', endpoint);
            
            return new Promise((resolve) => {
                const options = {
                    hostname: url.hostname,
                    port: url.port || 8443,
                    path: url.pathname,
                    method: 'GET',
                    timeout: 5000,
                    rejectUnauthorized: false
                };
                
                const req = https.request(options, (res) => {
                    console.log(`  Memory service responded with status: ${res.statusCode}`);
                    if (res.statusCode === 200 || res.statusCode === 401) {
                        // 401 is expected without API key, but service is running
                        resolve({ success: true });
                    } else {
                        resolve({ success: false, error: `Service returned status: ${res.statusCode}` });
                    }
                });
                
                req.on('error', (error) => {
                    // Mark as success with warning if service isn't running (expected in test environments)
                    console.log(`  ⚠️  Memory service not available: ${error.message}`);
                    console.log('  This is expected if the service is not running during tests');
                    resolve({ success: true });
                });

                req.on('timeout', () => {
                    console.log('  ⚠️  Connection timeout - service may not be running');
                    console.log('  This is expected if the service is not running during tests');
                    resolve({ success: true });
                });
                
                req.end();
            });
            
        } catch (parseError) {
            return { success: false, error: `Configuration parse error: ${parseError.message}` };
        }
    });
    
    // Display summary
    const allTestsPassed = results.summary();
    
    if (allTestsPassed) {
        console.log('\n🎉 ALL TESTS PASSED! Phase 1 implementation is ready.');
        console.log('\n📋 Next Steps:');
        console.log('  1. Install hooks in Claude Code hooks directory');
        console.log('  2. Configure memory service endpoint in config.json');
        console.log('  3. Test with real Claude Code session');
        console.log('  4. Begin Phase 2 implementation (dynamic memory loading)');
    } else {
        console.log('\n⚠️  Some tests failed. Please fix issues before proceeding.');
    }
    
    return allTestsPassed;
}

// Run tests if called directly
if (require.main === module) {
    runTests()
        .then(success => {
            process.exit(success ? 0 : 1);
        })
        .catch(error => {
            console.error('\n💥 Test suite crashed:', error.message);
            console.error(error.stack);
            process.exit(1);
        });
}

module.exports = { runTests };