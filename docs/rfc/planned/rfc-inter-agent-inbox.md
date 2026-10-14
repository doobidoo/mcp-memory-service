# RFC: Inter-agent inbox — reliable delivery over a server-side trigger

**Status:** planned · **Created:** 2026-10-14 · **Fork-only (maturing)**
**Relates to:** #1526 (Claude Code hooks inbox), #1527 (OpenCode plugin inbox), `docs/agents/README.md` ("Tags as inter-agent signals"), delta-sync (#1345), store partition (multi-store).
**Author's basis:** written from what we actually run — a 3-agent crew (Zero/Kiro local, T'Pol and Scotty on a VPS) plus 3-4 local Kiro CLI sessions on one machine — and from three delivery mechanisms we built and watched fail, and one that works.

## Problem

#1526 and #1527 add an inbox where an agent stores a memory with a sentinel tag (`msg:cluster`), the author shows up as `agent:<id>`, and other agents read everything with that tag since they last looked. The reading side is **pull**: a hook or plugin queries on session start and on prompts.

Pull delivery depends on the consuming agent running something on a schedule — a heartbeat. **In practice the heartbeat is unreliable or absent in most agent harnesses.** We have three first-hand data points, all the same shape:

1. **Checkpoint gate (failed).** We built a 45-minute checkpoint enforcer in the Kiro hooks: a session counter plus a `PostToolUse` timer that was supposed to force a memory write every 45 min. It only armed *after the first checkpoint* — if `LAST_CHECKPOINT` was 0 the timer never blocked. So the enforcement that was meant to replace the agent's discipline depended on the agent first exercising that discipline. It did not work.

2. **Pull inbox over a side channel (failed).** Our crew communicates via an HTTP message bus (Redis Streams on the VPS). Reading was manual/polled: the operator repeatedly had to tell the agent "check the inbox" because the agent did not poll on its own; a peer's reply sat unread. Same defect as #1526/#1527's pull: the message is there, nothing puts it in front of the agent.

3. **Session-registry heartbeat (failed, partial).** We run a session registry (register / heartbeat / list / resource claims) for cross-session awareness on one machine. It works for state *read at a natural point* (session start), but its heartbeat is exactly the periodic discipline that this problem is about — a session that stops heartbeating goes stale, and a peer only learns of a message by polling the registry. Awareness, not delivery.

Against those three, one mechanism works:

4. **Learning-loop scheduler (works).** The learning loop (belief derivation, consolidation, importance-triggered reflection) was originally meant to be driven by the agent. We moved it to the **server-side scheduler** (APScheduler inside the HTTP server, plus an importance trigger that fires server-side on accumulated signal). It works precisely because it does not wait for the agent to ask.

The pattern is not about the channel or the algorithm. It is about **where the trigger lives**: in the agent (fails) or in the server (works).

> **Thesis: any function that depends on the agent's periodic discipline (its heartbeat) to fire is unreliable. The trigger must be server-side.**

#1526 deliberately puts the routine in the hooks, "not the server", with a correct reason: *a scheduled job in the HTTP server would find the messages and have nowhere to deliver them* — the server does not reach into an agent's context. This RFC agrees with that limit and takes it as the starting point. Stage 3 does not ask the server to inject context (it cannot); it asks the server to **emit a signal on a passive channel the adapter already listens on**, keeping the injection in the adapter — exactly the boundary #1526 draws. And #1526 itself parks "a tag filter on the SSE stream `GET /api/events`" as an out-of-scope follow-up: Stage 3 is that follow-up, not a redesign.

A second gap: `agent:<id>` names only the *role*. We run 3-4 Kiro sessions on one machine — all are "zero". "Message to zero" cannot be delivered: which session? The current model has no way to address one instance among many of the same agent.

## Objective

Define a runtime-agnostic contract for inter-agent messages where:
- the **trigger** for delivery is server-side (event on message arrival), not a client poll, so delivery does not depend on the agent's heartbeat;
- messages can be addressed at **three levels of identity** (host / agent / session), so N instances of the same agent on one machine are individually reachable;
- the final **injection** into an agent's context stays in the per-runtime adapter (hooks / plugin / Kiro `agentSpawn`+`userPromptSubmit`), because only the adapter assembles context — but the *decision to deliver* is the server's.

The contract is one; the transports differ: local (N instances on one host) and cross-host (delta-sync between machines) are the same contract over different carriers.

Non-goal: this RFC does not propose folding our session-registry into the memory service. It borrows only the *concept* of session identity as part of an address.

## Requirements (EARS)

### Identity and addressing

- REQ-1.1: THE author of a message SHALL be stamped at three levels: `host` (machine), `agent` (role/id), and `session` (instance id). (`agent:<id>` alone is insufficient — it cannot distinguish N sessions of the same agent.)
- REQ-1.2: THE address of a message SHALL target any level of granularity: `agent` (any session of that role), `agent@host` (any such agent on that host), or `agent@host#session` (one specific instance).
- REQ-1.3: WHEN an agent reads, THE adapter SHALL drop messages authored by its own `session` (an instance must not read its own messages), not merely by its `agent` id.

### Stage 1 — pull baseline (what #1526/#1527 already do)

- REQ-2.1: THE contract SHALL reuse the #1526/#1527 pull design unchanged — last-seen marker per tag-set, 24h lookback on first run, throttle on the marker file, the "Messages from other agents (since ...)" output block, self-exclusion by author, and `POST /api/search/by-tag` (`/api/search` silently ignores `tags`). THIS RFC adds only: the `host:`/`session:` authorship tags and the `to:<address>` target tag. It does not re-specify what #1526 already defines.
- REQ-2.2: THE Kiro adapter SHALL use `agentSpawn` (session start) and `userPromptSubmit` (per prompt) hooks, echoing inbox hits to STDOUT so exit 0 injects them into context — the Kiro equivalent of the Claude Code / OpenCode hook points. (Adapter contract, not an upstream claim.)
- REQ-2.3: THE pull tier SHALL be labelled best-effort: it inherits the heartbeat dependency and is bounded by how often the agent's turn cycle runs. It is useful, not reliable.

### Address vocabulary

- REQ-A.1: `by-tag` matches tags as exact strings, so an address SHALL be encoded as a literal tag. Authorship stamps three tags (`agent:<id>`, `host:<name>`, `session:<id>`); a target stamps `to:<address>` where `<address>` is one of `agent`, `agent@host`, or `agent@host#session` as a literal string. Resolution at a coarser level (e.g. `to:zero`) is a tag match on that literal; it does not auto-expand to finer tags. (If level-spanning resolution is wanted, it is adapter-side filtering, not a backend feature.)

### Stage 2 — directed addressing + active injection

- REQ-3.1: WHEN a message is addressed to this instance (`agent@host#session`) and is unread since the marker, THE adapter SHALL inject a dedicated "message for you" block into context, separate from normal memory context.
- REQ-3.2: THE active injection SHALL be opt-in via an environment flag (off by default), consistent with the project's existing opt-in env pattern (`MCP_CONTEXT_INJECTION_ENABLED`, `MCP_IMPORTANCE_TRIGGER_ENABLED`, etc.).
- REQ-3.3: Stage 2 remains adapter-pull at the trigger level; it reduces perceived latency but does not yet solve the heartbeat dependency. It is the bridge, not the destination.

### Stage 3 — server-side trigger (the actual target)

- REQ-4.1: WHEN a message addressed to a live instance arrives (event, via delta-sync for cross-host or local IPC for same-host), THE server SHALL push a delivery signal toward that instance's adapter WITHOUT waiting for the adapter to poll.
- REQ-4.2: THE push carrier SHALL be a passive channel the adapter listens on. Primary candidate: an address-filtered SSE stream — today `GET /api/events` (`web/api/events.py`) accepts no tag/address filter, and the `SSEManager` (`web/sse.py`) filters only by connection; stores made through MCP tools do not broadcast at all (only the REST store/search paths call `broadcast_event`). Closing those is the gap. #1526 already lists "a tag filter on the SSE stream `GET /api/events`" as an out-of-scope follow-up — this is that follow-up.
- REQ-4.3: THE server SHALL know which instances are live from presence it already tracks (the hybrid/sync layer, or an external presence source), NOT from an agent heartbeat it cannot trust.
- REQ-4.4: THE adapter's role in Stage 3 SHALL be rendering only (place the pushed message into context); the decision to deliver and the timing are the server's.

### Invariants

- INV-1: Stage 1 and Stage 2 SHALL remain functional with no server changes (backward-compatible, opt-in), so adopters who want only pull are unaffected.
- INV-2: NO change SHALL require the memory service to control an agent's runtime; context injection stays in the adapter. The server decides *what and when*, the adapter decides *how to render*.
- INV-3: The three transports (local IPC, delta-sync cross-host, pull-by-tag) SHALL share one address and one message schema, so a message is portable across topologies.

## Design sketch (non-binding; Stage 3 is the open question)

- **Message = memory**, as in #1526: the inbox is the store, not a separate bus. This is an advantage over our Redis-Streams A3, where the channel is separate from memory and knowledge is lost unless re-ingested ("A3 without ingestion = forgetting"). Here the message already is a durable, searchable memory.
- **Address as tags**: `agent:<id>` + `host:<name>` + `session:<id>` on authorship; a `to:<address>` tag (or structured field) for the target, resolvable at any of the three levels.
- **Stage 1/2** need only adapter work + the addressing tags; no server change. Deliverable now, aligns with #1526/#1527.
- **Stage 3** needs a server-side push. The honest unknown: the memory service serves data, it does not reach into an agent's session. The missing link is between server and runtime. The direction proposed upstream, to be settled in a follow-up SPEC after discussion:
  - **Address-filtered SSE** — extend `GET /api/events` to broadcast store events filtered by `to:<address>`; the adapter holds a passive long-lived listener. Works for any runtime that can run a background listener. (This is the follow-up #1526 already parked.)
- **Local case first.** N Kiro instances on one host is the simplest, most controllable instance of the problem and the one we live daily. A local broker (presence it already has + IPC push) is more tractable than cross-host and proves the contract before delta-sync carries it between machines.

> **Fork/local note (not part of the upstream proposal):** for runtimes exposing a direct inject path, a local broker could push there instead of SSE — e.g. the Kiro ACP `inject_context` (injects into a session without triggering a turn), same-host only. This is Kiro-specific and unverified in the upstream codebase, so it stays in our fork notes, not in the discussion to the maintainer.

## Out of scope

- Folding session-registry into the memory service (we keep it separate; only the session-identity concept is borrowed).
- Guaranteed-once / ordered delivery, consumer groups (what Redis Streams gave A3). If needed, a later stage; the tag-as-message model does not provide it today, and this RFC does not claim it.
- `sync_events` compaction (tracked under delta-sync / ADR-0009).

## Acceptance criteria

- A message addressed `agent@host#session` reaches exactly that instance and no other same-agent instance (addressing, REQ-1.2/1.3).
- With Stage 1/2 only and no server change, the pull inbox behaves as #1526/#1527 describe (INV-1).
- A documented demonstration that a Stage-3 push reaches an idle instance that did NOT poll — the test that distinguishes this RFC from pull. (The bar our three failed mechanisms could not clear.)

## Evidence appendix (our run, factual)

| Mechanism | Trigger location | Outcome | Why |
|-----------|------------------|---------|-----|
| 45-min checkpoint gate | agent (`PostToolUse` timer) | failed | armed only after first checkpoint; depended on the discipline it replaced |
| A3 pull inbox (Redis Streams) | agent (manual/poll) | failed | operator had to say "check the inbox"; replies sat unread |
| session-registry heartbeat | agent (heartbeat) | partial | fine for state read at session start; stale when heartbeat lapses; no push |
| learning-loop scheduler | **server (APScheduler + importance trigger)** | **works** | fires without the agent asking |

The single variable that separates "works" from "failed" is whether the trigger is server-side.
