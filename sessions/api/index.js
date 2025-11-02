#!/usr/bin/env node

// ==== IMPORTS ===== //

// ===== LOCAL ===== //
const { routeCommand } = require('./router.js');
//-##

//-#

// ==== GLOBALS ===== //
//-#

// ==== FUNCTIONS ===== //

function main() {
    /**Main entry point for sessions.api commands.*/
    const args = process.argv.slice(2);

    // Parse flags
    const jsonOutput = args.includes('--json');
    const fromSlash = args.includes('--from-slash');

    // Remove flags from args to get command and subargs
    const cleanArgs = args.filter(arg => !arg.startsWith('--'));

    if (cleanArgs.length === 0) {
        const error = "No command specified. Usage: node api <command> [<subcommand>] [args] [--json] [--from-slash]";
        if (jsonOutput) {
            console.log(JSON.stringify({ error }, null, 2));
        } else {
            console.error(`Error: ${error}`);
        }
        process.exit(1);
    }

    // Extract command and remaining args
    const command = cleanArgs[0];
    const commandArgs = cleanArgs.slice(1);

    try {
        const result = routeCommand(command, commandArgs, jsonOutput, fromSlash);

        if (result !== undefined && result !== null) {
            if (jsonOutput && typeof result !== 'string') {
                console.log(JSON.stringify(result, null, 2));
            } else {
                console.log(result);
            }
        }
    } catch (error) {
        if (jsonOutput) {
            console.log(JSON.stringify({ error: error.message || String(error) }, null, 2));
        } else {
            console.error(`Error: ${error.message || error}`);
        }
        process.exit(1);
    }
}

//-#

// ==== EXECUTIONS ===== //

if (require.main === module) {
    main();
}

//-#

// ==== EXPORTS ===== //
// No exports needed for main entry point
//-#