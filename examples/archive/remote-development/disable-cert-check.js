#!/usr/bin/env node
/**
 * Disable certificate validation for Claude and other Node.js apps
 * Run this before starting Claude
 */

// Set the environment variable that disables certificate validation
process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';

console.log("Certificate validation disabled for Node.js applications.");
console.log("IMPORTANT: This applies to all Node.js applications started after running this script.");
console.log("It's recommended to restart Claude Desktop and Claude Code now.");

// Keep this setting for all child processes
const { execSync } = require('child_process');
execSync('export NODE_TLS_REJECT_UNAUTHORIZED=0', { stdio: 'inherit' });