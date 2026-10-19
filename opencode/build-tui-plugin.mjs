#!/usr/bin/env node
/**
 * Compile the sidebar widgets in this directory and deploy them.
 *
 * opencode's plugin loader uses Bun's default JSX transform which is incompatible
 * with solid-js (which needs dom-expressions via babel-preset-solid), so both
 * widget generations are compiled ahead of time. Importing memory-plugin.js
 * directly into these tests would otherwise fail on the JSX.
 *
 * Two OpenCode generations are covered, because they load plugins differently:
 *
 *   V1  memory-status-tui.tsx    -> <plugins>/memory-status-tui/index.js
 *       registered through tui.json["plugin"] as a file:// URL that resolves the
 *       directory's package.json "main" (the shape the README documents).
 *   V2  memory-status-tui.v2.tsx -> <plugins>/opencode-memory/tui.js
 *       discovered as part of a plugin package, next to the server plugin at
 *       <plugins>/opencode-memory/index.js, through the package.json "./tui"
 *       export. V2 needs no config entry for either file.
 *
 * Install the explicit Babel 7 build dependencies documented in README.md.
 * They are resolved from ~/.config/opencode/node_modules, independently of the
 * repository checkout. Babel 8 removed the TypeScript options used below.
 * Override with OPENCODE_CONFIG_DIR if your opencode config lives elsewhere.
 */
import { copyFile, mkdir, readFile, writeFile } from "node:fs/promises"
import { resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { createRequire } from "node:module"
import { homedir } from "node:os"

const SCRIPT_DIR = fileURLToPath(new URL(".", import.meta.url))
const CONFIG_DIR = process.env.OPENCODE_CONFIG_DIR
  ? resolve(process.env.OPENCODE_CONFIG_DIR)
  : resolve(homedir(), ".config", "opencode")
const PLUGINS_DIR = resolve(CONFIG_DIR, "plugins")

const TARGETS = [
  {
    source: "memory-status-tui.tsx",
    output: "memory-status-tui.js",
    deployDir: "memory-status-tui",
    deployName: "index.js",
    packageJson: { name: "opencode-memory-status", type: "module", main: "./index.js" },
    // The directory itself is created by hand, as the README documents: V2
    // plugin discovery walks every directory under plugins/, and a V1 widget
    // package sitting there fails its own plugin load.
    createDir: false,
  },
  {
    source: "memory-status-tui.v2.tsx",
    output: "memory-status-tui.v2.js",
    deployDir: "opencode-memory",
    deployName: "tui.js",
    createDir: true,
    // V2 resolves a plugin package through its exports, so both entrypoints are
    // declared here: the copied HTTP plugin and the compiled widget. A plugin
    // package is how V2 discovers the widget without a config entry.
    packageJson: {
      name: "opencode-memory",
      type: "module",
      main: "./index.js",
      exports: {
        ".": "./index.js",
        "./tui": "./tui.js",
      },
      peerDependencies: {
        "@opentui/core": ">=0.5.8",
        "@opentui/solid": ">=0.5.8",
        "solid-js": ">=1.9.0",
      },
    },
    // V2 has no separate server-plugin registration: the HTTP plugin is the
    // package's main entrypoint, so the build copies it into the deployment.
    copyServer: { from: "memory-plugin.js", to: "index.js" },
  },
]

const requireFromConfig = createRequire(resolve(CONFIG_DIR, "package.json"))
const { transformAsync } = requireFromConfig("@babel/core")
const presetTypescriptPath = requireFromConfig.resolve("@babel/preset-typescript")
const presetSolidPath = requireFromConfig.resolve("babel-preset-solid")

async function compile(sourceFile) {
  const result = await transformAsync(await readFile(sourceFile, "utf8"), {
    filename: sourceFile,
    babelrc: false,
    configFile: false,
    cwd: CONFIG_DIR,
    presets: [
      [presetTypescriptPath, { isTSX: true, allExtensions: true }],
      [presetSolidPath, { generate: "universal", moduleName: "@opentui/solid" }],
    ],
  })

  if (!result?.code) throw new Error(`Babel returned no code for ${sourceFile}`)
  return result.code
}

for (const target of TARGETS) {
  const code = await compile(resolve(SCRIPT_DIR, target.source))
  await writeFile(resolve(SCRIPT_DIR, target.output), code)
  console.log(`Built ${target.output}`)

  const deployDir = resolve(PLUGINS_DIR, target.deployDir)
  const deployFile = resolve(deployDir, target.deployName)
  try {
    if (target.createDir) await mkdir(deployDir, { recursive: true })
    await writeFile(deployFile, code)
    await writeFile(resolve(deployDir, "package.json"), JSON.stringify(target.packageJson, null, 2) + "\n")
    if (target.copyServer) {
      await copyFile(resolve(SCRIPT_DIR, target.copyServer.from), resolve(deployDir, target.copyServer.to))
    }
    console.log(`Deployed to ${deployFile}`)
  } catch (err) {
    console.warn(`Could not deploy ${target.deployDir}: ${err.message}`)
  }
}
