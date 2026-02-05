#!/usr/bin/env node

/**
 * Test script for the enhanced session-start hook
 */

const path = require('path');

// Import the enhanced hook
const sessionStartHook = require('../../claude-hooks/core/session-start.js');

async function testEnhancedHook() {
    console.log('🧪 Testing Enhanced Session Start Hook\n');

    // Mock context for testing
    const mockContext = {
        workingDirectory: process.cwd(),
        sessionId: 'test-session-' + Date.now(),
        trigger: 'session-start',
        userMessage: 'Help me understand the memory service improvements',
        injectSystemMessage: async (message) => {
            console.log('\n🎯 INJECTED CONTEXT:');
            console.log('═'.repeat(60));
            console.log(message);
            console.log('═'.repeat(60));
            return true;
        }
    };

    console.log(`📂 Testing in directory: ${mockContext.workingDirectory}`);
    console.log(`🔍 Test query: "${mockContext.userMessage}"`);
    console.log(`⚙️  Trigger: ${mockContext.trigger}\n`);

    try {
        // Execute the enhanced hook
        await sessionStartHook.handler(mockContext);

        console.log('\n✅ Hook execution completed successfully!');
        console.log('\n📊 Expected improvements:');
        console.log('   • Multi-phase memory retrieval (recent + important + fallback)');
        console.log('   • Enhanced recency indicators (🕒 today, 📅 this week)');
        console.log('   • Better semantic queries with git context');
        console.log('   • Improved categorization with "Recent Work" section');
        console.log('   • Configurable memory ratios and time windows');

    } catch (error) {
        console.error('❌ Hook execution failed:', error.message);
        console.error('Stack trace:', error.stack);
    }
}

// Run the test
if (require.main === module) {
    testEnhancedHook()
        .then(() => {
            console.log('\n🎉 Test completed');
            process.exit(0);
        })
        .catch(error => {
            console.error('\n💥 Test failed:', error.message);
            process.exit(1);
        });
}

module.exports = { testEnhancedHook };
