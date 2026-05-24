#!/usr/bin/env node
/**
 * Compile memory-status-tui.tsx → memory-status-tui.js using Solid's babel preset.
 * opencode's plugin loader uses Bun's default JSX transform which is incompatible
 * with solid-js (which needs dom-expressions via babel-preset-solid).
 */
import { transformAsync } from "@babel/core"
import { readFile, writeFile } from "node:fs/promises"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"

const REPO_OPENCODE = "/Users/hkr/Repositories/mcp-memory-service/opencode"
const SRC = resolve(REPO_OPENCODE, "memory-status-tui.tsx")
const OUT = resolve(REPO_OPENCODE, "memory-status-tui.js")
const CONFIG_DIR = resolve(process.env.HOME, ".config", "opencode")
const LIVE_DIR = resolve(CONFIG_DIR, "plugins", "memory-status-tui")

const source = await readFile(SRC, "utf8")

const result = await transformAsync(source, {
  filename: SRC,
  babelrc: false,
  configFile: false,
  cwd: CONFIG_DIR,
  presets: [
    [resolve(CONFIG_DIR, "node_modules/@babel/preset-typescript"), { isTSX: true, allExtensions: true }],
    [resolve(CONFIG_DIR, "node_modules/babel-preset-solid"), { generate: "universal", moduleName: "@opentui/solid" }],
  ],
})

if (!result?.code) {
  throw new Error("Babel returned no code")
}

await writeFile(OUT, result.code)
console.log(`Built ${OUT}`)

// Mirror to live plugin dir
try {
  await writeFile(resolve(LIVE_DIR, "index.js"), result.code)
  console.log(`Deployed to ${LIVE_DIR}/index.js`)
} catch (err) {
  console.warn(`Could not deploy: ${err.message}`)
}
