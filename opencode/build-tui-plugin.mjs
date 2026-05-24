#!/usr/bin/env node
/**
 * Compile memory-status-tui.tsx → memory-status-tui.js using Solid's babel preset.
 * opencode's plugin loader uses Bun's default JSX transform which is incompatible
 * with solid-js (which needs dom-expressions via babel-preset-solid).
 *
 * Babel itself isn't a project dep — we reuse the copies pulled into
 * ~/.config/opencode/node_modules by `bun add @opentui/solid` (which transitively
 * installs @babel/core, @babel/preset-typescript and babel-preset-solid).
 * Override with OPENCODE_CONFIG_DIR if your opencode config lives elsewhere.
 */
import { readFile, writeFile } from "node:fs/promises"
import { resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { createRequire } from "node:module"
import { homedir } from "node:os"

const SCRIPT_DIR = fileURLToPath(new URL(".", import.meta.url))
const SRC = resolve(SCRIPT_DIR, "memory-status-tui.tsx")
const OUT = resolve(SCRIPT_DIR, "memory-status-tui.js")
const CONFIG_DIR = process.env.OPENCODE_CONFIG_DIR
  ? resolve(process.env.OPENCODE_CONFIG_DIR)
  : resolve(homedir(), ".config", "opencode")
const LIVE_DIR = resolve(CONFIG_DIR, "plugins", "memory-status-tui")

// Resolve babel from CONFIG_DIR/node_modules regardless of where this script lives.
const requireFromConfig = createRequire(resolve(CONFIG_DIR, "package.json"))
const { transformAsync } = requireFromConfig("@babel/core")
const presetTypescriptPath = requireFromConfig.resolve("@babel/preset-typescript")
const presetSolidPath = requireFromConfig.resolve("babel-preset-solid")

const source = await readFile(SRC, "utf8")

const result = await transformAsync(source, {
  filename: SRC,
  babelrc: false,
  configFile: false,
  cwd: CONFIG_DIR,
  presets: [
    [presetTypescriptPath, { isTSX: true, allExtensions: true }],
    [presetSolidPath, { generate: "universal", moduleName: "@opentui/solid" }],
  ],
})

if (!result?.code) {
  throw new Error("Babel returned no code")
}

await writeFile(OUT, result.code)
console.log(`Built ${OUT}`)

// Mirror to live plugin dir if it exists (opencode's plugin install layout).
try {
  await writeFile(resolve(LIVE_DIR, "index.js"), result.code)
  console.log(`Deployed to ${LIVE_DIR}/index.js`)
} catch (err) {
  console.warn(`Could not deploy: ${err.message}`)
}
