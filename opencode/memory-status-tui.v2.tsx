import { createSignal } from "solid-js"
import { homedir } from "node:os"
import path from "node:path"
import { readFile } from "node:fs/promises"
import { Plugin } from "@opencode/plugin/tui"

// OpenCode V2 CLI plugin. It replaces the V1 shape in memory-status-tui.tsx:
// V2 takes Plugin.define({ id, setup }) with context.ui.slot, where V1 took
// { id, tui } with api.slots.register and the `sidebar_content` slot. The
// widget itself is unchanged: it polls the status file the HTTP plugin writes.
const STATUS_FILE = process.env.OPENCODE_MEMORY_STATUS_FILE
  || path.join(homedir(), ".local", "state", "opencode", ".memory-status.json")
const POLL_MS = 1500

type Status = {
  projectName?: string
  loadedCount?: number
  capturedCount?: number
  lastAction?: string
  lastSummaryAt?: string
  updatedAt?: string
}

async function readStatus(): Promise<Status> {
  try {
    return JSON.parse(await readFile(STATUS_FILE, "utf8")) as Status
  } catch {
    return {}
  }
}

export default Plugin.define({
  id: "opencode-memory-status",
  setup(context) {
    const [status, setStatus] = createSignal<Status>({})

    // Async tick keeps the TUI event loop unblocked. setInterval ignores the
    // returned Promise; overlapping ticks are harmless because each fully
    // resolves before calling setStatus.
    const tick = async () => setStatus(await readStatus())
    void tick()
    const timer = setInterval(() => void tick(), POLL_MS)

    const unregister = context.ui.slot({
      append: "sidebar.content",
      render: () => {
        // `context.theme.text.base` is the only theme token the V2 CLI plugin
        // API documents; the V1 widget's accent/muted tokens have no V2 name.
        const fg = context.theme.text.base
        const s = status()
        const project = s.projectName ?? ""
        return (
          <box flexDirection="column" paddingTop={1}>
            <text fg={fg}>Memory</text>
            <text fg={fg}>
              loaded {s.loadedCount ?? 0} · captured {s.capturedCount ?? 0}
            </text>
            {project ? <text fg={fg}>{project}</text> : null}
            <text fg={fg}>{s.lastAction ?? "waiting…"}</text>
          </box>
        )
      },
    })

    return () => {
      clearInterval(timer)
      if (typeof unregister === "function") unregister()
    }
  },
})
