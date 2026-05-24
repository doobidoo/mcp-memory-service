import { createSignal } from "solid-js"
import { homedir } from "node:os"
import path from "node:path"
import fs from "node:fs"
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

function readStatus(): Status {
  try {
    return JSON.parse(fs.readFileSync(STATUS_FILE, "utf8")) as Status
  } catch {
    return {}
  }
}

export const tui: TuiPlugin = async (api) => {
  const [status, setStatus] = createSignal<Status>(readStatus())

  const tick = () => {
    try {
      const raw = fs.readFileSync(STATUS_FILE, "utf8")
      const parsed = JSON.parse(raw) as Status
      setStatus(parsed)
    } catch {
      // file may not exist yet — keep previous state
    }
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
