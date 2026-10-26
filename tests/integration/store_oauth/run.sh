#!/usr/bin/env bash
#
# End-to-end integration suite for store-scoped access + per-store OAuth +
# the tiered anonymous-access policy. Stands up the service in Docker under
# each config scenario and asserts the HTTP / SSE / MCP enforcement matrix
# documented in ./README.md (rubric areas A, B, D, E, F, G, H, I).
#
# Usage:
#   IMAGE=mcp-memory-service:storescope-test ./run.sh
#
# Env:
#   IMAGE     container image under test   (default mcp-memory-service:storescope-test)
#   PORT      host port to bind            (default 8091)
#   DCR       DCR registration key         (default test-dcr-key-123)
#   KEEP      if set, leave the last container running for inspection
#
# Requires: docker, curl, python3. Exits non-zero on any failed assertion.
set -uo pipefail

IMAGE="${IMAGE:-mcp-memory-service:storescope-test}"
PORT="${PORT:-8091}"
DCR="${DCR:-test-dcr-key-123}"
CTR="storescope-e2e"
B="http://127.0.0.1:${PORT}"
DATADIR="/tmp/storescope-e2e-data"   # per-run data dir (reset each scenario)

PASS=0; FAIL=0
pass(){ PASS=$((PASS+1)); printf '  \033[32mPASS\033[0m %s\n' "$1"; }
fail(){ FAIL=$((FAIL+1)); printf '  \033[31mFAIL\033[0m %s — expected %s, got %s\n' "$1" "$2" "$3"; }
# check <desc> <expected> <actual>
check(){ if [ "$2" = "$3" ]; then pass "$1"; else fail "$1" "$2" "$3"; fi; }

# Always print the summary and clean up, even on an early interrupt, so the
# result line is never lost to a half-finished SSE wait.
finish(){
  [ -n "${KEEP:-}" ] || docker rm -f "$CTR" >/dev/null 2>&1 || true
  echo; echo "================= RESULT: $PASS passed, $FAIL failed ================="
}
trap finish EXIT

jget(){ python3 -c "import sys,json;print(json.load(sys.stdin).get('$1',''))" 2>/dev/null; }
# HTTP status code of a request
code(){ curl -s -o /dev/null -w "%{http_code}" "$@"; }
# MCP tools/call -> prints 'ok' | 'err:<code>' (JSON-RPC/gate error) | 'errresult'
# (tool-level isError) | 'bad'. A denied scope gate returns a JSON-RPC error;
# a tool that merely failed returns result.isError — distinguishing them keeps a
# failed tool call from being mistaken for an authorized success.
mcp(){ # $1=auth header (may be empty), $2=json body
  local h=(); [ -n "$1" ] && h=(-H "$1")
  curl -s "${h[@]}" -H 'Content-Type: application/json' -X POST "$B/mcp" -d "$2" \
    | python3 -c "import sys,json
d=json.load(sys.stdin)
e=d.get('error')
if e: print('err:'+str(e.get('code')))
else:
    r=d.get('result')
    if isinstance(r,dict) and r.get('isError'): print('errresult')
    elif r is not None: print('ok')
    else: print('bad')" 2>/dev/null
}

start_instance(){ # $@ = extra -e env flags
  docker rm -f "$CTR" >/dev/null 2>&1 || true
  rm -rf "$DATADIR"; mkdir -p "$DATADIR"; chmod 777 "$DATADIR"
  docker run -d --name "$CTR" -p "${PORT}:8000" \
    -e MCP_MODE=http \
    -e MCP_MEMORY_STORAGE_BACKEND=sqlite_vec \
    -e MCP_MEMORY_SQLITE_PATH=/data/memory.db \
    "$@" \
    -v "$DATADIR":/data "$IMAGE" >/dev/null
  for i in $(seq 1 30); do curl -s "$B/api/health" 2>/dev/null | grep -q healthy && return 0; sleep 2; done
  echo "INSTANCE NOT HEALTHY"; docker logs "$CTR" 2>&1 | tail -20; return 1
}

# Register a confidential client + mint a client_credentials token for a scope.
# echoes the access_token (empty on failure)
mint(){ # $1=scope
  local reg cid csec
  reg=$(curl -s -X POST "$B/oauth/register" -H "Authorization: Bearer $DCR" -H 'Content-Type: application/json' \
    -d '{"client_name":"e2e","grant_types":["client_credentials"],"token_endpoint_auth_method":"client_secret_basic","redirect_uris":[]}')
  cid=$(echo "$reg" | jget client_id); csec=$(echo "$reg" | jget client_secret)
  curl -s -X POST "$B/oauth/token" \
    -d "grant_type=client_credentials&client_id=$cid&client_secret=$csec&scope=$1" | jget access_token
}

seed_anon(){ # seed one memory into each store while anonymous+full (helper for read tests)
  curl -s -X POST "$B/api/memories" -H 'Content-Type: application/json' -d '{"content":"seedA","store":"projectA","tags":["s"]}' >/dev/null
  curl -s -X POST "$B/api/memories" -H 'Content-Type: application/json' -d '{"content":"seedB","store":"projectB","tags":["s"]}' >/dev/null
  curl -s -X POST "$B/api/memories" -H 'Content-Type: application/json' -d '{"content":"seedD","store":"default","tags":["s"]}' >/dev/null
}

###############################################################################
echo "== IMAGE=$IMAGE PORT=$PORT =="
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "image not found: $IMAGE"; exit 2; }

# ---------------------------------------------------------------------------
echo; echo "### Scenario anon-open — PR#1 SSE + partition isolation (A, B) ###"
start_instance -e MCP_ALLOW_ANONYMOUS_ACCESS=true || exit 1
# A: partition isolation
curl -s -X POST "$B/api/memories" -H 'Content-Type: application/json' -d '{"content":"alpha","store":"projectA","tags":["t"]}' >/dev/null
curl -s -X POST "$B/api/memories" -H 'Content-Type: application/json' -d '{"content":"beta","store":"projectB","tags":["t"]}' >/dev/null
cnt(){ curl -s "$B/api/memories?store=$1" | python3 -c "import sys,json;d=json.load(sys.stdin);print(len(d.get('memories',d.get('results',[]))))"; }
check "A1/A2 store=projectA isolated (1)" "1" "$(cnt projectA)"
check "A2 store=projectB isolated (1)"    "1" "$(cnt projectB)"
check "A3 store=all cross-partition (2)"  "2" "$(cnt all)"
# B: store-scoped SSE
timeout 9 curl -sN "$B/api/events?store=projectA" >/tmp/sse_a.out 2>&1 &
sleep 2
curl -s -X POST "$B/api/memories" -H 'Content-Type: application/json' -d '{"content":"a2","store":"projectA","tags":["t"]}' >/dev/null
curl -s -X POST "$B/api/memories" -H 'Content-Type: application/json' -d '{"content":"b2","store":"projectB","tags":["t"]}' >/dev/null
sleep 3.5; wait 2>/dev/null
A_ON_A=$(grep -c '"store": "projectA"' /tmp/sse_a.out)
B_ON_A=$(grep -c '"store": "projectB"' /tmp/sse_a.out)
[ "$A_ON_A" -ge 1 ] && pass "B1 projectA event delivered to projectA sub (>=1, got $A_ON_A)" || fail "B1 projectA event delivered" ">=1" "$A_ON_A"
check "B2 projectB event does NOT leak to projectA sub (0)" "0" "$B_ON_A"

# ---------------------------------------------------------------------------
echo; echo "### Scenario commons-ro — tiered policy: anon read:default + narrow token (D, F, G) ###"
start_instance -e MCP_ALLOW_ANONYMOUS_ACCESS=true -e MCP_ANONYMOUS_SCOPE='read:default' \
               -e MCP_OAUTH_ENABLED=true -e MCP_DCR_REGISTRATION_KEY="$DCR" || exit 1

# --- Tier 1: narrow token read:projectA write:projectA ---
AT=$(mint 'read:projectA%20write:projectA')
[ -n "$AT" ] && pass "E1 narrow client_credentials token issued" || fail "E1 narrow token issued" "token" "empty"
HN="Authorization: Bearer $AT"
check "D1  GET  store=projectA (200)" "200" "$(code -H "$HN" "$B/api/memories?store=projectA")"
check "D2  GET  store=projectB (403)" "403" "$(code -H "$HN" "$B/api/memories?store=projectB")"
check "D3  GET  store=all      (403)" "403" "$(code -H "$HN" "$B/api/memories?store=all")"
check "D4  POST store=projectA (200)" "200" "$(code -H "$HN" -H 'Content-Type: application/json' -X POST "$B/api/memories" -d '{"content":"x","store":"projectA","tags":["t"]}')"
check "D5  POST store=projectB (403)" "403" "$(code -H "$HN" -H 'Content-Type: application/json' -X POST "$B/api/memories" -d '{"content":"x","store":"projectB","tags":["t"]}')"
check "D7  SSE  ?store=projectA (200)" "200" "$(code -H "$HN" --max-time 2 "$B/api/events?store=projectA")"
check "D8  SSE  ?store=projectB (403)" "403" "$(code -H "$HN" --max-time 2 "$B/api/events?store=projectB")"
check "D9  SSE  firehose        (403)" "403" "$(code -H "$HN" --max-time 2 "$B/api/events")"
check "D10 MCP  memory_search projectA (ok)"   "ok"      "$(mcp "$HN" '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"memory_search","arguments":{"query":"x","store":"projectA"}}}')"
check "D11 MCP  memory_store projectB (err)"   "err:-32003" "$(mcp "$HN" '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"memory_store","arguments":{"content":"z","store":"projectB"}}}')"
check "D12 MCP  memory_search store=all (err)" "err:-32003" "$(mcp "$HN" '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"memory_search","arguments":{"query":"x","store":"all"}}}')"

# --- read-only token: write must be denied (D6) ---
ATR=$(mint 'read:projectA')
check "D6  read-only token POST (403)" "403" "$(code -H "Authorization: Bearer $ATR" -H 'Content-Type: application/json' -X POST "$B/api/memories" -d '{"content":"x","store":"projectA","tags":["t"]}')"

# --- Tier 0: anonymous read:default (F) ---
check "F2 anon GET  ?store=default  (200)" "200" "$(code "$B/api/memories?store=default")"
check "F3 anon GET  ?store=projectA (403)" "403" "$(code "$B/api/memories?store=projectA")"
check "F4 anon POST ?store=default  (403)" "403" "$(code -H 'Content-Type: application/json' -X POST "$B/api/memories" -d '{"content":"x","store":"default","tags":["t"]}')"
check "F5 anon GET  ?store=all      (403)" "403" "$(code "$B/api/memories?store=all")"
check "F6 anon SSE  ?store=default  (200)" "200" "$(code --max-time 2 "$B/api/events?store=default")"
check "F7 anon SSE  firehose        (403)" "403" "$(code --max-time 2 "$B/api/events")"
check "F8 anon MCP  memory_store    (err)" "err:-32003" "$(mcp "" '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"memory_store","arguments":{"content":"z","store":"default"}}}')"
check "F9 anon MCP  memory_search default (ok)"  "ok"        "$(mcp "" '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"memory_search","arguments":{"query":"x","store":"default"}}}')"

# --- Tier 2 / back-compat: global token = full (D15, H2), API key skipped (no API_KEY set) ---
ATG=$(mint 'read%20write')
HG="Authorization: Bearer $ATG"
check "D15 global token GET projectB (200)" "200" "$(code -H "$HG" "$B/api/memories?store=projectB")"
check "D15 global token GET all      (200)" "200" "$(code -H "$HG" "$B/api/memories?store=all")"
check "H3  global token SSE firehose (200)" "200" "$(code -H "$HG" --max-time 2 "$B/api/events")"

# --- Issuance hardening (E3) ---
REG=$(curl -s -X POST "$B/oauth/register" -H "Authorization: Bearer $DCR" -H 'Content-Type: application/json' -d '{"client_name":"e2e2","grant_types":["client_credentials"],"token_endpoint_auth_method":"client_secret_basic","redirect_uris":[]}')
CID=$(echo "$REG" | jget client_id); CSEC=$(echo "$REG" | jget client_secret)
check "E3 client_credentials admin blocked (400)" "400" "$(code -X POST "$B/oauth/token" -d "grant_type=client_credentials&client_id=$CID&client_secret=$CSEC&scope=read%20write%20admin")"

# --- G4 cross-cluster isolation (homelab token cannot read ml-app) ---
ATH=$(mint 'read:homelab%20write:homelab')
check "G4 homelab token GET ?store=ml-app (403)" "403" "$(code -H "Authorization: Bearer $ATH" "$B/api/memories?store=ml-app")"

# --- J: per-client scope ceiling (PR#3) — client declares scope at registration ---
REGC=$(curl -s -X POST "$B/oauth/register" -H "Authorization: Bearer $DCR" -H 'Content-Type: application/json' \
  -d '{"client_name":"ceil","grant_types":["client_credentials"],"token_endpoint_auth_method":"client_secret_basic","redirect_uris":[],"scope":"read:projectA write:projectA"}')
CCID=$(echo "$REGC" | jget client_id); CCSEC=$(echo "$REGC" | jget client_secret)
check "J1 register echoes declared scope ceiling" "read:projectA write:projectA" "$(echo "$REGC" | jget scope)"
TOKW=$(curl -s -X POST "$B/oauth/token" -d "grant_type=client_credentials&client_id=$CCID&client_secret=$CCSEC&scope=read:projectA")
check "J2 token within ceiling granted" "read:projectA" "$(echo "$TOKW" | jget scope)"
check "J3 token exceeding ceiling (400)" "400" "$(code -X POST "$B/oauth/token" -d "grant_type=client_credentials&client_id=$CCID&client_secret=$CCSEC&scope=read:projectB")"
check "J4 no-scope request defaults to ceiling" "read:projectA write:projectA" "$(curl -s -X POST "$B/oauth/token" -d "grant_type=client_credentials&client_id=$CCID&client_secret=$CCSEC" | jget scope)"
ATC=$(echo "$TOKW" | jget access_token)
check "J5 ceiling token enforced at resource: GET projectB (403)" "403" "$(code -H "Authorization: Bearer $ATC" "$B/api/memories?store=projectB")"
UNL=$(mint 'read:projectZ')
check "J6 unlimited client (no declared scope) back-compat" "present" "$([ -n "$UNL" ] && echo present || echo empty)"

# --- K: SSE ticket inherits the minting credential's store scope ---
# POST /api/events/ticket exchanges a real credential for a single-use ticket
# (issue #1117). The ticket must carry only the caller's read scope, else a
# store-restricted token could mint a ticket and open the firehose. Each GET
# mints a FRESH ticket ($(tkt ...)) because tickets are single-use.
tkt(){ curl -s -X POST "$B/api/events/ticket" -H "$1" | jget ticket; }
check "K0 narrow token mints an SSE ticket"        "present" "$([ -n "$(tkt "$HN")" ] && echo present || echo empty)"
check "K1 narrow ticket SSE ?store=projectA (200)" "200" "$(code --max-time 2 "$B/api/events?store=projectA&ticket=$(tkt "$HN")")"
check "K2 narrow ticket SSE ?store=projectB (403)" "403" "$(code --max-time 2 "$B/api/events?store=projectB&ticket=$(tkt "$HN")")"
check "K3 narrow ticket SSE firehose        (403)" "403" "$(code --max-time 2 "$B/api/events?ticket=$(tkt "$HN")")"
check "K4 global ticket SSE firehose        (200)" "200" "$(code --max-time 2 "$B/api/events?ticket=$(tkt "$HG")")"

# ---------------------------------------------------------------------------
echo; echo "### Scenario oauth-strict — anon off => no-token lockout (F12, H) ###"
start_instance -e MCP_ALLOW_ANONYMOUS_ACCESS=false -e MCP_OAUTH_ENABLED=true -e MCP_DCR_REGISTRATION_KEY="$DCR" || exit 1
check "F12 no token GET (401)"  "401" "$(code "$B/api/memories?store=default")"
check "F12 no token SSE (401)"  "401" "$(code --max-time 2 "$B/api/events?store=default")"

# ---------------------------------------------------------------------------
echo; echo "### Scenario anon-compat — ANONYMOUS_SCOPE unset => full anon (H1, H4) ###"
start_instance -e MCP_ALLOW_ANONYMOUS_ACCESS=true -e MCP_OAUTH_ENABLED=true -e MCP_DCR_REGISTRATION_KEY="$DCR" || exit 1
check "H4 anon(full) GET  ?store=projectA (200)" "200" "$(code "$B/api/memories?store=projectA")"
check "H4 anon(full) POST ?store=projectA (200)" "200" "$(code -H 'Content-Type: application/json' -X POST "$B/api/memories" -d '{"content":"x","store":"projectA","tags":["t"]}')"
check "H4 anon(full) SSE  firehose        (200)" "200" "$(code --max-time 2 "$B/api/events")"

###############################################################################
# summary + cleanup handled by the EXIT trap (finish)
[ "$FAIL" -eq 0 ]
