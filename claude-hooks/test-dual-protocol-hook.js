#!/usr/bin/env node

/**
 * Test Dual Protocol Memory Hook
 * Tests the updated session-start hook with both HTTP and MCP protocols
 */

const { onSessionStart } = require('./core/session-start.js');
const fs = require('fs');
const path = require('path');

// Test configurations for different protocol modes
const testConfigs = {
    'auto-mcp-preferred': {
        protocol: 'auto',
        preferredProtocol: 'mcp',
        fallbackEnabled: true,
        description: 'Auto mode with MCP preferred and HTTP fallback'
    },
    'auto-http-preferred': {
        protocol: 'auto',
        preferredProtocol: 'http',
        fallbackEnabled: true,
        description: 'Auto mode with HTTP preferred and MCP fallback'
    },
    'mcp-only': {
        protocol: 'mcp',
        fallbackEnabled: false,
        description: 'MCP only mode (no fallback)'
    },
    'http-only': {
        protocol: 'http',
        fallbackEnabled: false,
        description: 'HTTP only mode (no fallback)'
    }
};

// Base test configuration
const baseConfig = {
    http: {
        endpoint: 'https://localhost:8443',
        apiKey: 'test-key-123',
        healthCheckTimeout: 2000,
        useDetailedHealthCheck: true
    },
    mcp: {
        serverCommand: ['uv', 'run', 'memory', 'server', '-s', 'cloudflare'],
        serverWorkingDir: '/Users/hkr/Documents/GitHub/mcp-memory-service',
        connectionTimeout: 3000,
        toolCallTimeout: 5000
    },
    defaultTags: ['claude-code', 'test-generated'],
    maxMemoriesPerSession: 5,
    enableSessionConsolidation: false,
    injectAfterCompacting: false,
    recentFirstMode: true,
    recentMemoryRatio: 0.6,
    recentTimeWindow: 'last-week',
    fallbackTimeWindow: 'last-month',
    showStorageSource: true,
    sourceDisplayMode: 'brief'
};

// Test context template
const createTestContext = (configName) => ({
    workingDirectory: process.cwd(),
    sessionId: `dual-protocol-test-${configName}`,
    trigger: 'session-start',
    userMessage: `test dual protocol memory hook - ${configName} mode`,
    injectSystemMessage: async (message) => {
        console.log('\n' + '='.repeat(80));
        console.log(`🧠 MEMORY CONTEXT INJECTION - ${configName.toUpperCase()}`);
        console.log('='.repeat(80));
        console.log(message);
        console.log('='.repeat(80) + '\n');
        return true;
    }
});

/**
 * Update config file temporarily for testing
 */
function updateConfigForTest(testConfigName) {
    const configPath = path.join(__dirname, 'config.json');
    const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));

    // Merge test configuration
    const testConfig = testConfigs[testConfigName];
    config.memoryService = {
        ...baseConfig,
        ...testConfig
    };

    // Write temporary config
    const backupPath = configPath + '.backup';
    if (!fs.existsSync(backupPath)) {
        fs.copyFileSync(configPath, backupPath);
    }

    fs.writeFileSync(configPath, JSON.stringify(config, null, 2));

    return () => {
        // Restore original config
        if (fs.existsSync(backupPath)) {
            fs.copyFileSync(backupPath, configPath);
            fs.unlinkSync(backupPath);
        }
    };
}

/**
 * Test a specific protocol configuration
 */
async function testProtocolConfig(configName) {
    console.log(`\n🔧 Testing ${configName.toUpperCase()} Configuration`);
    console.log(`📋 Description: ${testConfigs[configName].description}`);
    console.log(`📂 Working Directory: ${process.cwd()}`);
    console.log('─'.repeat(80));

    const restoreConfig = updateConfigForTest(configName);

    try {
        const testContext = createTestContext(configName);

        // Get the session start handler
        const sessionStartModule = require('./core/session-start.js');
        const handler = sessionStartModule.handler || sessionStartModule.onSessionStart || sessionStartModule;

        if (!handler) {
            throw new Error('Could not find onSessionStart handler');
        }

        await handler(testContext);
        console.log(`✅ ${configName} test completed successfully`);
        return { success: true, config: configName };

    } catch (error) {
        console.log(`❌ ${configName} test failed: ${error.message}`);

        if (process.env.DEBUG) {
            console.error(error.stack);
        }

        return { success: false, config: configName, error: error.message };

    } finally {
        restoreConfig();
    }
}

/**
 * Run all protocol tests
 */
async function runAllTests() {
    console.log('🚀 Starting Dual Protocol Memory Hook Tests');
    console.log(`📅 Test Date: ${new Date().toISOString()}`);
    console.log(`💻 Node Version: ${process.version}`);
    console.log('='.repeat(80));

    const results = [];

    for (const [configName, testConfig] of Object.entries(testConfigs)) {
        const result = await testProtocolConfig(configName);
        results.push(result);

        // Add delay between tests to avoid resource conflicts
        await new Promise(resolve => setTimeout(resolve, 1000));
    }

    // Summary
    console.log('\n📊 TEST RESULTS SUMMARY');
    console.log('='.repeat(80));

    const successful = results.filter(r => r.success);
    const failed = results.filter(r => !r.success);

    console.log(`✅ Successful: ${successful.length}/${results.length}`);
    if (successful.length > 0) {
        successful.forEach(r => console.log(`   • ${r.config}: OK`));
    }

    console.log(`❌ Failed: ${failed.length}/${results.length}`);
    if (failed.length > 0) {
        failed.forEach(r => console.log(`   • ${r.config}: ${r.error}`));
    }

    console.log('\n🎯 Key Observations:');
    console.log('   • Hooks should gracefully handle connection failures');
    console.log('   • Git context analysis should work regardless of protocol');
    console.log('   • Storage backend detection should fall back to environment');
    console.log('   • Both HTTP and MCP protocols should be supported');

    return results;
}

// Run tests if this script is executed directly
if (require.main === module) {
    runAllTests()
        .then(results => {
            const failedCount = results.filter(r => !r.success).length;
            process.exit(failedCount > 0 ? 1 : 0);
        })
        .catch(error => {
            console.error('❌ Test suite failed:', error.message);
            if (process.env.DEBUG) {
                console.error(error.stack);
            }
            process.exit(1);
        });
}

module.exports = { runAllTests, testProtocolConfig, testConfigs };
