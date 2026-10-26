# Store-scoped access & per-store OAuth — test rubric

This directory holds the automated test suite for store-partitioned access control in
`mcp-memory-service`: store-scoped SSE delivery, per-store OAuth scope enforcement across every
transport, token issuance, and the **tiered anonymous-access policy** that lets authenticated and
unauthenticated callers coexist on one shared instance without an all-or-nothing flag day.

The rubric below is the contract: every row is a case the suite must assert, with its exact
precondition, action, and expected outcome. It doubles as the PR's coverage statement.

---

## 1. Why this exists

Before these changes, access was effectively binary:

- **Store partitions** isolated data on the query/write path (`?store=`), **but SSE `/api/events`
  was an unscoped firehose** — a live subscriber saw every store's events regardless of partition.
- **OAuth scopes were global** (`read`/`write`/`admin`) — any read token could read or subscribe to
  every store.
- **Anonymous access was all-or-nothing**: `MCP_ALLOW_ANONYMOUS_ACCESS=true` granted a hardcoded
  global `read write` (every store); `false` returned `401` for any un-tokened request, locking out
  every legacy client the instant OAuth was enabled.

This suite covers the three changes that close those gaps and the backward-compatibility guarantees
that keep existing deployments working untouched.

---

## 2. Feature → test-area map

| Feature under test | Rubric areas |
| --- | --- |
| **PR #1** — store-scoped SSE (`?store=`, scoped replay, `store` on events) | A, B, H |
| **PR #2** — per-store OAuth enforcement (HTTP / SSE / MCP) + narrow issuance | C, D, E, H, I |
| **Anonymous tier** — `MCP_ANONYMOUS_SCOPE` + 3-tier coexistence policy | F, G, H |
| **PR #3** — per-client scope ceiling (DCR allowed-scope persistence) | J, H |

---

## 3. The three-tier access policy under test

| Tier | Caller | Scope granted | May do |
| --- | --- | --- | --- |
| **0 — Anonymous** | no token (`MCP_ALLOW_ANONYMOUS_ACCESS=true`, `MCP_ANONYMOUS_SCOPE="read:default"`) | `read:default` | **read** the shared `default` store only — no writes, no other partition, no `store=all`, no firehose |
| **1 — Per-agent** | OAuth bearer token | `read:<store> write:<store> read:default` | read+write its own partition, read the commons |
| **2 — Privileged** | API key, or OAuth token with global `read write` | full | every store, `store=all` cross-partition, SSE firehose |

`MCP_ANONYMOUS_SCOPE` **defaults to `"read write"`** when unset, so existing anonymous deployments are
unchanged (Tier 0 = full) until an operator opts into tightening it.

---

## 4. Test environment & how to run

Two layers; both are part of the PR.

### 4a. Unit tests (`pytest`, no server)
Pure-Python assertions on the primitives — scope matrix, the SSE subscription parser, event filtering,
`client_credentials` scope resolution, anonymous-scope config resolution. These run anywhere.

```
pytest tests/web/test_sse_store_scope.py tests/web/test_oauth_store_scopes.py \
       tests/web/test_anonymous_scope.py -v
```

### 4b. Integration / E2E (live container)
The full HTTP/SSE/MCP enforcement matrix against a running instance, exercised by
`tests/integration/store_oauth/run.sh`, which stands up the image on an isolated port + DB and asserts
HTTP status codes and MCP JSON-RPC error codes per the rubric. Config per scenario is passed by env.

> **Local-env note:** the maintainer's local interpreter and some CI runners lack loadable SQLite
> extensions, so storage-backed app tests can only run inside the container (or the CI image that
> ships them). The unit layer (4a) runs on the real primitives without a backend; the E2E layer (4b)
> runs in-container. The rubric marks each case `U` (unit) or `E` (e2e) accordingly.

### Config matrix used by the E2E scenarios

| Scenario key | `OAUTH_ENABLED` | `ALLOW_ANONYMOUS` | `ANONYMOUS_SCOPE` | DCR key | Purpose |
| --- | --- | --- | --- | --- | --- |
| `anon-open` | false | true | (unset) | — | legacy baseline / PR #1 SSE + partitions |
| `oauth-strict` | true | false | — | set | pure per-agent OAuth, no anon |
| `commons-ro` | true | true | `read:default` | set | **the shipping policy** — tiered coexistence |
| `anon-compat` | true | true | (unset) | set | anon default = global (back-compat guarantee) |

---

## 5. Rubric — all cases

Legend: **Type** `U`=unit, `E`=e2e. **Exp** is the asserted outcome (HTTP status, event count, MCP
JSON-RPC error code, or boolean).

### A. Store partition isolation (data + HTTP path)

| ID | Precondition | Action | Exp | Type |
| --- | --- | --- | --- | --- |
| A1 | write to `projectA` | `GET /api/memories?store=projectA` | returns the item | E |
| A2 | write to `projectA` | `GET /api/memories?store=projectB` | 0 results (no leak) | E |
| A3 | writes to A + B | `GET /api/memories?store=all` | union of both | E |
| A4 | write with no `store` | read `?store=default` | item present (default partition) | E |
| A5 | — | `resolve_store("all")` | returns `None` (cross-partition) | U |
| A6 | item in `projectA` | `DELETE` scoped to `projectA` | removes only from A | E |
| A7 | items in A + B | tag / time / semantic search `store=projectA` | only A matches | E |

### B. Store-scoped SSE (PR #1)

| ID | Precondition | Action | Exp | Type |
| --- | --- | --- | --- | --- |
| B1 | subscribe `?store=projectA` | write to `projectA` | subscriber receives the `memory_stored` event | E |
| B2 | subscribe `?store=projectA` | write to `projectB` | event **not** delivered (0 leaks) | E |
| B3 | subscribe with no param | write to any store | all events delivered (firehose, back-compat) | E |
| B4 | subscribe `?store=all` | write to any store | all events delivered (firehose) | E |
| B5 | subscribe `?store=projectA&store=projectB` | writes to A, B, C | A + B delivered, C not | E |
| B6 | any subscription | a system/non-store event fires | delivered to all (bypasses store filter) | U+E |
| B7 | buffered events exist for A + B | reconnect with `Last-Event-ID` on a `projectA` sub | replay yields only A events (**no cross-store leak on resume**) | U |
| B8 | subscribe `?store=proj` | write to `project` / `projectA` | not delivered (exact name match, no prefix leak) | U |
| B9 | legacy caller builds event without `store` | broadcast | delivered to firehose subs; no crash (back-compat) | U |
| B10 | — | `parse_store_subscription` on varied query strings | `getlist`, `all`→None, empty ignored, exact names | U |

### C. OAuth per-store scope primitives (PR #2)

| ID | Scope on token | Check | Exp | Type |
| --- | --- | --- | --- | --- |
| C1 | `read` (global) | `has_store_scope("read","projectA")` | True (global grants all) | U |
| C2 | `read:projectA` | `has_store_scope("read","projectA")` | True | U |
| C3 | `read:projectA` | `has_store_scope("read","projectB")` | False | U |
| C4 | `read:projectA` | `has_store_scope("read", None)` | False (cross-partition needs global) | U |
| C5 | `write:projectA` | write-op matrix mirrors C1–C4 | as above | U |
| C6 | `read:projectA` | `require_store_scope("read","projectB")` | raises 403 `insufficient_scope` | U |
| C7 | — | `require_store_read`/`require_store_write` deps read `store` from query, map `all`→None | correct store passed | U |
| C8 | `read write` | `has_scope("read")` / `has_scope("write")` | unchanged (back-compat) | U |

### D. OAuth enforcement across transports (PR #2)

Token = **narrow** `read:projectA write:projectA` unless noted.

| ID | Transport / action | Exp | Type |
| --- | --- | --- | --- |
| D1 | `GET /api/memories?store=projectA` | 200 | E |
| D2 | `GET /api/memories?store=projectB` | 403 | E |
| D3 | `GET /api/memories?store=all` | 403 (cross-partition needs global) | E |
| D4 | `POST /api/memories` store=projectA | 200 | E |
| D5 | `POST /api/memories` store=projectB | 403 | E |
| D6 | read-only token (`read:projectA`) → `POST` store=projectA | 403 (no write scope) | E |
| D7 | `GET /api/events?store=projectA` (SSE) | 200 / connects | E |
| D8 | `GET /api/events?store=projectB` (SSE) | 403 | E |
| D9 | `GET /api/events` firehose (no param) | 403 (needs global read) | E |
| D10 | MCP `tools/call` `memory_search` `store=projectA` | ok result | E |
| D11 | MCP `tools/call` `memory_store` `store=projectB` | JSON-RPC error `-32003` | E |
| D12 | MCP `tools/call` `store=all` with narrow token | denied (`-32003`) | E |
| D13 | search routes (semantic/tag/time) `store=projectB` | 403 | E |
| D14 | analytics store-carrying route `store=projectB` | 403 | E |
| D15 | **global** token `read write` → D1–D12 equivalents | all succeed (full access) | E |
| D16 | **API key** → representative read+write across stores | all succeed (full access) | E |

### E. Token issuance — `client_credentials` (PR #2)

| ID | Requested scope | Exp granted | Type |
| --- | --- | --- | --- |
| E1 | `read:projectA write:projectA` | exactly that (narrow token issuable) | E |
| E2 | `read write` | `read write` (global) | E |
| E3 | `read write admin` | **400 `invalid_scope`** (admin never issued) | E |
| E4 | (none requested) | default `read write` | E |
| E5 | authorization_code flow | requested scopes passed through unchanged | U |

### F. Anonymous tier — `MCP_ANONYMOUS_SCOPE` (NEW)

| ID | Config | Action | Exp | Type |
| --- | --- | --- | --- | --- |
| F1 | `ANONYMOUS_SCOPE` unset, anon on | resolve anonymous scope | `"read write"` (back-compat default) | U |
| F2 | `commons-ro` | anon `GET ?store=default` | 200 | E |
| F3 | `commons-ro` | anon `GET ?store=projectA` | 403 | E |
| F4 | `commons-ro` | anon `POST` (any store) | 403 (read-only) | E |
| F5 | `commons-ro` | anon `GET ?store=all` | 403 | E |
| F6 | `commons-ro` | anon SSE `?store=default` | 200 / connects | E |
| F7 | `commons-ro` | anon SSE firehose (no param) | 403 | E |
| F8 | `commons-ro` | anon MCP `memory_store` | denied (`-32003`) | E |
| F9 | `commons-ro` | anon MCP `memory_search` `store=default` | ok | E |
| F10 | `ANONYMOUS_SCOPE=""`, anon on | any anon request | 403 (empty scope grants nothing) | U+E |
| F11 | `commons-ro` + valid Tier-1 token | tokened request | uses token scope, **not** anon (precedence) | E |
| F12 | `ALLOW_ANONYMOUS=false` | no-token request | 401 regardless of `ANONYMOUS_SCOPE` | E |

### G. Tier coexistence — end-to-end policy scenarios

| ID | Scenario | Exp | Type |
| --- | --- | --- | --- |
| G1 | anon reader + Tier-1 writer on same instance (`commons-ro`) | anon reads commons; token writes its cluster; neither sees the other's private partition | E |
| G2 | un-migrated bridge (no token) under `commons-ro` | can read commons, **cannot** write (mirrors the migration path — no reader lockout, writers must tokenize) | E |
| G3 | Tier-2 (API key) alongside confined Tier-1 | API key sees firehose + all stores; Tier-1 stays confined | E |
| G4 | homelab-cluster token vs ml-app store | homelab token `GET ?store=ml-app` | 403 (cross-cluster isolation) | E |

### J. Per-client scope ceiling — DCR allowed-scope persistence (PR #3)

A client's `scope` declared at registration becomes a permanent ceiling, persisted on the client
record (`allowed_scope`). A client that declares no scope (`allowed_scope` NULL/None) is unbounded —
preserving pre-ceiling behavior and covering clients registered before the field existed. Over-requests
are rejected `400 invalid_scope` (not downscoped). Unit coverage: `tests/web/test_oauth_client_scope_ceiling.py`.

| ID | Precondition | Action | Exp | Type |
| --- | --- | --- | --- | --- |
| J-u1 | — | `scope_within_ceiling` matrix (None=unlimited, exact/subset, global-covers-store, store≠global, outside→false) | per matrix | U |
| J-u2 | — | `_client_credentials_scope(req, ceiling)`: within→grant, exceed→400, none→defaults to ceiling, admin→400, no-ceiling→honor | per matrix | U |
| J-u3 | — | SQLite + in-memory store round-trip `allowed_scope`; NULL/legacy row → None (unlimited) | persists / None | U |
| J1 | register with `scope:"read:projectA write:projectA"` | registration response | echoes that `scope` | E |
| J2 | that client | token request `scope=read:projectA` | granted, scope `read:projectA` | E |
| J3 | that client | token request `scope=read:projectB` | **400 `invalid_scope`** | E |
| J4 | that client | token request with no scope | defaults to the ceiling | E |
| J5 | that client's token | `GET /api/memories?store=projectB` | 403 (ceiling enforced at the resource too) | E |
| J6 | client registered with NO declared scope | token request for an arbitrary store scope | granted (unlimited, back-compat) | E |

### H. Backward compatibility (must not break existing API)

| ID | Guarantee | Exp | Type |
| --- | --- | --- | --- |
| H1 | request with no `store` param | behaves exactly as before (`default` partition) | E |
| H2 | existing global `read write` token | unchanged full access | E |
| H3 | SSE with no `?store` | unchanged firehose | E |
| H4 | `MCP_ANONYMOUS_SCOPE` unset | pre-change anonymous behavior (global) | E |
| H5 | existing `test_oauth_hardening` + logging suites | still pass | U |
| H6 | API-key auth | unchanged full access | E |

### I. Negative / security hardening

| ID | Attack / edge | Exp | Type |
| --- | --- | --- | --- |
| I1 | malformed / unknown bearer token | 401 `invalid_token` | E |
| I2 | expired token | 401 | U |
| I3 | narrow token attempts firehose via omitted param | 403 (cannot widen by dropping the param) | E |
| I4 | store name with odd characters | handled safely, no injection, exact match only | U |
| I5 | `Last-Event-ID` resume used to fetch another store's buffered events | blocked (scoped replay) | U |
| I6 | `admin` requested via `client_credentials` | never issued (400) | E |

### K. SSE-ticket store-scope propagation (reconciliation: upstream `/events/ticket`, issue #1117)

A single-use SSE ticket must inherit only the minting credential's **read** scope, so a store-restricted
token cannot exchange it for firehose access.

| ID | Scenario (`commons-ro`) | Exp | Type |
| --- | --- | --- | --- |
| K0 | `read:projectA` token `POST /api/events/ticket` | ticket issued (a store-scoped token may mint) | E |
| K1 | narrow ticket SSE `?store=projectA` | 200 | E |
| K2 | narrow ticket SSE `?store=projectB` | 403 (ticket confined to its channel) | E |
| K3 | narrow ticket SSE firehose | 403 | E |
| K4 | global `read write` token's ticket, firehose | 200 (back-compat) | E |

### L. Channel-authorization semantics (cluster model: commons=`default`, private=`chanA/B/C`)

| ID | Actor | Action | Exp | Type |
| --- | --- | --- | --- | --- |
| L1/L2 | anonymous | read commons (HTTP + SSE) | 200 (sees cluster-wide) | E |
| L3 | anonymous | read private `chanA` | 403 | E |
| L4 | anonymous | write commons | 403 (read-only commons) | E |
| L5–L9 | authorized (`read:default write:default read:chanA write:chanA`) | read+write commons; read+write+SSE `chanA` | 200 | E |
| L10/L11 | authorized | read/write other `chanB` | 403 (own channel only) | E |
| L12–L14 | multi-channel (`…read:chanA write:chanA read:chanB write:chanB`) | read/write `chanA`+`chanB`, read commons | 200 (multiple channels) | E |
| L15 | multi-channel | read unauthorized `chanC` | 403 | E |

### M. Anonymous commons-write feature flag (`MCP_ANONYMOUS_SCOPE`)

The allow/disallow-anonymous-commons-write control **is** `MCP_ANONYMOUS_SCOPE`: `read:default` disallows
anon writes (area F), `read:default write:default` allows them — confined to the commons either way.

| ID | Scenario (`commons-rw`: anon scope `read:default write:default`) | Exp | Type |
| --- | --- | --- | --- |
| M1 | anon `GET ?store=default` | 200 | E |
| M2 | anon `POST ?store=default` | 200 (**write allowed** by the flag) | E |
| M3 | anon MCP `memory_store` `store=default` | ok | E |
| M4/M5 | anon read/write private `chanA` | 403 (flag opens only the commons) | E |
| M6 | anon write `store=all` | 403 | E |
| M7 | anon firehose SSE | 403 | E |

---

## 6. Coverage summary

- **Unit (`U`)**: ~25 cases — scope matrix, parser, event filter, replay scoping, issuance resolution,
  anonymous-scope resolution. Run in CI and locally; no backend required.
- **E2E (`E`)**: ~45 cases across 4 config scenarios — the full HTTP/SSE/MCP enforcement + tier
  coexistence matrix. Run in-container.
- Every PR feature (SSE scoping, per-store OAuth, anonymous tier) has both positive (allow) and
  negative (deny) assertions, plus an explicit backward-compat guarantee (area H).
- Security-critical invariants carry dedicated cases: no cross-store SSE leak live (B2) or on replay
  (B7/I5), cross-partition requires global (C4/D3), admin never issuable (E3/I6), and tokened
  precedence over anonymous (F11).

## 7. Design decisions (resolved 2026-10-17)

1. **Commons store name** — the Tier-0 commons is the existing `default` store, matching the upstream
   precedent (an omitted `store` param already resolves to `default`, so legacy clients need no change).
2. **Empty `MCP_ANONYMOUS_SCOPE` semantics** (F10) — `""` means "authenticated but granted nothing" →
   **403** on every store-scoped route. (To turn anonymous off entirely, set
   `MCP_ALLOW_ANONYMOUS_ACCESS=false`, which yields 401 — see F12.)
3. **PR packaging** — the `MCP_ANONYMOUS_SCOPE` knob ships **in PR #2** alongside per-store OAuth
   enforcement, since the tier policy is incomplete without it. `test_anonymous_scope.py` lands in PR #2.
4. **Analytics breadth** (D14) — the 10 store-carrying analytics read routes enforce `require_store_read`;
   `search_analytics` and `performance` report instance-level operational metrics (not a store's
   contents) and intentionally stay on the global `read` check.
