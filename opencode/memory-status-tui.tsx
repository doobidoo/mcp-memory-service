import { createSignal } from "solid-js"
import { homedir } from "node:os"
import path from "node:path"
import { readFile } from "node:fs/promises"
import type { TuiPlugin } from "@opencode-ai/plugin/dist/tui.js"

const STATUS_FILE = path.join(homedir(), ".config", "opencode", ".memory-status.json")
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

export const tui: TuiPlugin = async (api) => {
  const [status, setStatus] = createSignal<Status>(await readStatus())

  // Async tick keeps the TUI event loop unblocked. setInterval ignores the
  // returned Promise; overlapping ticks are harmless because each fully
  // resolves before calling setStatus.
  const tick = async () => {
    const parsed = await readStatus()
    setStatus(parsed)
  }

  const interval = setInterval(tick, POLL_MS)
  api.lifecycle.onDispose(() => clearInterval(interval))

  api.slots.register({
    slots: {
      sidebar_content: (ctx) => {
        const muted = ctx.theme.current.textMuted
        const accent = ctx.theme.current.accent
        const text = ctx.theme.current.text
        const s = status()
        const loaded = s.loadedCount ?? 0
        const captured = s.capturedCount ?? 0
        const project = s.projectName ?? ""
        const last = s.lastAction ?? "waiting…"
        return (
          <box flexDirection="column" paddingTop={1}>
            <text fg={accent}>Memory</text>
            <text fg={text}>
              loaded {loaded} · captured {captured}
            </text>
            {project ? <text fg={muted}>{project}</text> : null}
            <text fg={muted}>{last}</text>
          </box>
        )
      },
    },
  })
}

export default { id: "opencode-memory-status", tui }
