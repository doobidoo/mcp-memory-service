"""Training and evaluation data for the fine-tuned 4-way head (#1460).

The head replaces the bidirectional 3-way run + fourway_decide with one
inference trained on the store's actual question — "does B make A out of
date?" — so its data has to teach that question, not NLI. Three groups,
strictly disjoint (the __main__ check refuses any overlap):

  train/val  generated + hand-written pairs: homelab/dev-tool notes, 24
             life domains (meetings, budgets, flights, doctors, school,
             insurance, garden…), superset updates, corrected diagnoses,
             minimal changes-vs-ends pairs, mixed registers, long notes
  testA      the shipped 40-pair sentence-style set, verbatim (kept in
             sync with tests/reasoning/test_nli_fourway.py::FOURWAY_SET)
  testB      30 note-style pairs: the 15 that reproduce the #1460 audit
             shape (different subjects sharing one word — 37 of the 44
             real-store false positives) plus 15 fresh (5/5/5)
  testC      40 pairs in ten domains that appear nowhere in training
             (chess club, wedding, choir, allotment…), mixed registers —
             the closest thing to an unseen store

Everything is seeded: build(seed) is deterministic across machines, and
train_fourway_head.py pins the seed it reports with.

Labels: changes / ends / repeats / unrelated (the harness's four).
"""

import random

# ── testA: shipped 40-pair set, verbatim ────────────────────────────────────

TEST_A = [
    ("changes", "The cron job for backups runs every night at 2am.", "The backup cron moved to 4am after the maintenance window changed."),
    ("changes", "The API service is deployed at version 2.3.1.", "We shipped API service 2.4.0 this morning; 2.3.1 is no longer live."),
    ("changes", "The embedding service version is 3.11.", "The embedding service version is 3.12."),
    ("changes", "The cache TTL is 5 minutes.", "The cache TTL is 5 hours now."),
    ("changes", "The service runs on port 8000.", "The service runs on port 3000 since the refactor."),
    ("changes", "Dark mode is enabled by default.", "Dark mode is disabled by default."),
    ("changes", "The rate limit is 100 requests per minute.", "The rate limit was raised to 1000 requests per minute."),
    ("changes", "The model is loaded lazily on first use.", "The model is loaded on startup as of last week's change."),
    ("changes", "Memory records go to the default store.", "Memory records now go to the project store by default."),
    ("changes", "The earlier diagnosis was a disk permissions issue.", "On closer inspection the failures come from a failing SATA cable, not disk permissions."),
    ("ends", "Project Atlas is active and we meet weekly.", "Project Atlas is closed; the team was reassigned last month."),
    ("ends", "The nightly sync job runs at midnight.", "The nightly sync job was cancelled when we moved to streaming."),
    ("ends", "We use the legacy import tool for migrations.", "The legacy import tool is deprecated and removed from the repo."),
    ("ends", "OAuth tokens are cached for a week.", "We stopped caching OAuth tokens; the cache was dropped entirely."),
    ("ends", "The staging cluster is running.", "The staging cluster was shut down in March."),
    ("ends", "Feature flags are read from the config service.", "The config service is retired; feature flags are gone with it."),
    ("ends", "We maintain a mirror of the wiki.", "The wiki mirror is no longer maintained."),
    ("ends", "The metrics dashboard is available at port 9090.", "The metrics dashboard was taken offline."),
    ("ends", "Backups are sent to the offsite box.", "Offsite backups ended; the box was returned."),
    ("ends", "The experimental search backend is enabled for all users.", "The experimental search backend was disabled for everyone."),
    ("repeats", "The API rate limit is 100 requests per minute.", "Requests on the API are limited to 100 per minute."),
    ("repeats", "Embeddings use all-MiniLM-L6-v2 with 384 dimensions.", "The embedding model is all-MiniLM-L6-v2, producing 384-dim vectors."),
    ("repeats", "Backups run daily.", "A backup happens every day."),
    ("repeats", "The service requires an API key for write access.", "Writes need an API key; that has been the rule all along."),
    ("repeats", "The default port is 8000.", "By default the service listens on port 8000."),
    ("repeats", "Tags help organize memories.", "Tags are useful for organizing memories."),
    ("repeats", "Consolidation runs every six hours.", "Every six hours, consolidation runs."),
    ("repeats", "OAuth tokens expire after one hour.", "OAuth tokens are valid for sixty minutes."),
    ("repeats", "The store lives in a single sqlite file.", "All memories are kept in one sqlite database file."),
    ("repeats", "Dark mode follows the system preference.", "The theme switches with the OS setting."),
    ("unrelated", "Never commit directly to main.", "The cache was never warmed after deploy."),
    ("unrelated", "The deploy pipeline runs on Fridays.", "The cache layer was tuned for read-heavy workloads."),
    ("unrelated", "Errors are retried three times.", "Error messages are sanitized before logging."),
    ("unrelated", "Tags help organize memories.", "Tag filters support wildcards."),
    ("unrelated", "The vector store uses cosine distance.", "Vector embeddings are cached in memory."),
    ("unrelated", "Repo A lint rule: never use print statements.", "Repo B lint rule: never use print statements."),
    ("unrelated", "The dashboard shows sync status.", "The dashboard language can be switched."),
    ("unrelated", "We migrated the auth service to OAuth 2.1.", "The auth service logs were moved to a separate store."),
    ("unrelated", "The search index was rebuilt last night.", "Search latency improved after the rebuild."),
    ("unrelated", "The homelab VPN uses wireguard on port 51820.", "The homelab wiki runs in a container behind traefik."),
]

# ── testB: note-style; the 15 reproduction pairs + 15 fresh ──────────────────
# reproduction pairs (margin_experiment2.py, words disjoint from the train pool)

TEST_B = [
    # 12 different-subject shared-word (the audit's 37/44 shape)
    ("unrelated", "zotero: better-bibtex auto-export to /papers", "zotero ios app can't open annotated pdfs"),
    ("unrelated", "statusline: added battery segment", "statusline lualine rewrite broke the git branch segment"),
    ("unrelated", "keys: gpg key expired, renewed to 2027", "ssh keys for the build host regenerated as ed25519"),
    ("unrelated", "memory: sqlite wal mode enabled", "memory pressure from the embedding cache"),
    ("unrelated", "lint: ruff replaces flake8", "lint errors counted per file in the CI summary"),
    ("unrelated", "upstream: nixpkgs pinned to 25.05", "upstream pr 412 still unresolved"),
    ("unrelated", "port 8123: home assistant ingress", "port knocking sequence changed for the lab"),
    ("unrelated", "backup: restic to b2, 30d retention", "backup laptop switched to arch"),
    ("unrelated", "sync: caldav to nextcloud works", "sync wrapper adds 2s latency to every request"),
    ("unrelated", "cache: warmed after deploy", "cache stampede during the friday demo"),
    ("unrelated", "token: hf token in envpass", "token count explodes on chinese docs"),
    ("unrelated", "vpn: wireguard on 51820", "vpn dns leak on android"),
    # 3 related-but-both-stand (the audit's 6/44 shape)
    ("unrelated", "traefik: wildcard cert via dns-01", "traefik dashboard behind basic auth now"),
    ("unrelated", "cron: fstrim monthly", "cron mail spam from the backup job"),
    ("unrelated", "gpu: nvidia driver 550 breaks sleep", "gpu passthrough to the win11 vm works"),
    # 5 fresh note-style changes
    ("changes", "immich on 2283", "immich moved to 2284"),
    ("changes", "postgres 15 on the vps", "postgres 16 now, migrated last weekend"),
    ("changes", "rss refresh 15m", "rss refresh 2h, feeds were hammering miniflux"),
    ("changes", "wifi channel 36", "wifi channel 149 after the neighbour ap moved"),
    ("changes", "scanner on the printer usb", "scanner moved to the nas, sane over network"),
    # 5 fresh note-style ends
    ("ends", "matrix-sms bridge up", "matrix-sms bridge shut down in may"),
    ("ends", "rss via miniflux", "miniflux retired, moved to freshrss"),
    ("ends", "pihole on the pi3", "pihole taken offline when the udm took over dns"),
    ("ends", "hugo theme anatole", "anatole removed in the redesign"),
    ("ends", "kafka for events", "kafka dropped; redis streams now"),
    # 5 fresh note-style repeats
    ("repeats", "grafana on 3000", "grafana listens on 3000"),
    ("repeats", "vimrc 200 lines", "the vimrc is 200 lines"),
    ("repeats", "tmux prefix ctrl-a", "tmux prefix is ctrl-a"),
    ("repeats", "mail via msmtp", "msmtp sends the mail"),
    ("repeats", "fonts: jetbrains mono", "jetbrains mono is the font"),
]

# ── train pool: audit-shape words, disjoint from TEST_B's words ──────────────
# (zotero, statusline, keys, memory, lint, upstream, port, backup, sync,
#  cache, token, vpn, traefik, cron, gpu + the immich/postgres/rss/wifi/
#  scanner/matrix/miniflux/pihole/anatole/kafka/grafana/vimrc/tmux/mail/
#  fonts entities appear in TEST_B changes/ends/repeats only)

AUDIT_WORDS = {
    "nginx": ["nginx: tls via letsencrypt", "nginx now behind cloudflare", "nginx worker_connections 4096", "nginx configtest fails on alpine", "nginx access logs rotated weekly"],
    "docker": ["docker: rootless mode on", "docker compose v2 replaced the python one", "docker: watchtower auto-updates", "docker hub rate limits hit daily", "docker devicemapper removed upstream"],
    "git": ["git: rebase by default", "git lfs for the design assets", "git hooks skipped in ci", "git-sign with the ssh key", "git gc aggressive monthly"],
    "ssh": ["ssh: key auth only, passwords off", "ssh multiplexing saves latency", "ssh tunnel for the db", "ssh-agent in the systemd user unit", "ssh jumphost via tailscale"],
    "mail": ["mail via nullmailer", "maildir synced with mbsync", "spamassassin learned 200 hams", "smtp relay on the vps", "mail filters in sieve"],
    "calendar": ["calendar: caldav week numbers", "calendar week starts monday", "family calendar invite links", "calendar reminders 10min before", "calendar: birthdays from contacts"],
    "obsidian": ["obsidian: vault in git", "obsidian daily notes template v3", "obsidian sync via syncthing", "obsidian plugins: dataview, templater", "obsidian mobile on the ipad"],
    "firefox": ["firefox: arkenfox user.js", "firefox containers for work", "firefox: ublock origin everywhere", "firefox syncs passwords", "firefox on android drains battery"],
    "vim": ["vim: leader is space", "vim lsp via coc-nvim", "vim: spellcheck for markdown only", "vim remaps jk to escape", "vim sessions per project"],
    "tmux2": ["tmux prefix ctrl-space", "tmux: mouse mode on", "tmux status bar shows cpu", "tmux resurrect saves sessions", "tmux on the phone via termux"],
    "font": ["fontconfig: emoji fallback noto", "font rendering with subpixel on wayland", "font ligatures off in the editor", "new font: berkeley mono", "font size 14 on the laptop"],
    "theme": ["theme: catppuccin everywhere", "dark theme follows sunset", "theme switch breaks gtk apps", "gruvbox on the terminal", "theme synced between editors"],
    "monitor": ["monitor: dp 1.4 cable needed", "second monitor on the usb-c dock", "monitor color profile calibrated", "monitor: hdr washed out on linux", "ultrawide at 100hz"],
    "printer": ["printer: brother hl-l2350", "printer driverless via airprint", "printer on the guest wifi", "printer duplex off by default", "printer toner low alert"],
    "scanner2": ["scanner: fujitsu snapsnap", "scanner to the nas inbox", "scanner ocr with tesseract", "scanner feeder jams on cardstock", "scanner powered over usb"],
    "wifi2": ["wifi: 5ghz only for the tv", "wifi roaming aggressiveness high", "guest wifi isolated vlan", "wifi 6e on the laptop", "wifi calling broken at home"],
    "dns": ["dns: unbound as resolver", "dns over https in firefox", "dns ttl 300 for the app domain", "local dns for *.home", "dns cache poisoned once on travel"],
    "logs": ["logs to journald only", "logs shipped with vector", "logs rotated at 100m", "log noise from the cron mailer", "structured logs in json"],
    "python": ["python: uv replaces pipenv", "python 3.12 on the box", "python venv per project", "python: ruff + mypy in pre-commit", "python free-threading still waiting"],
    "rust": ["rust: stable toolchain", "rust-analyzer in helix", "rust build cache on the nas", "rust cross-compile to arm", "rust: cargo deny in ci"],
    "postgres2": ["postgres: pgbouncer in front", "postgres backups via wal-g", "postgres autovacuum tuned", "postgres replication to the nas", "postgres on zfs, recordsize 16k"],
    "redis2": ["redis: cache only, no persistence", "redis memory limit 512mb", "redis: keyspace notifications on", "redis cluster overkill here", "redis next to the app on the box"],
    "tailscale": ["tailscale: subnet routes on", "tailscale exit node abroad", "tailscale acl deny inbound", "tailscale key expiry 90d", "tailscale on the tv? no"],
    "hass": ["hass on the pi 4", "hass: zigbee2mqtt network", "hass db purged to 7d", "hass automations in yaml", "hass remote via nabu casa"],
    "jellyfin": ["jellyfin on the nas", "jellyfin: hw transcode via vaapi", "jellyfin users: family of 4", "jellyfin metadata refresh nightly", "jellyfin remote via tailscale"],
    "nextcloud2": ["nextcloud: aio on docker", "nextcloud files synced 20gb", "nextcloud maintenance sundays", "nextcloud on postgres2", "nextcloud preview generator slow"],
    "proxmox": ["proxmox on the beelink", "proxmox: lxc for the resolver", "proxmox backups to the nas", "proxmox cluster of 2", "proxmox: gpu passthrough tests"],
    "matrix2": ["matrix: synapse on the vps", "matrix bridges: discord, signal", "matrix element on desktop", "matrix federation rate-limited", "matrix db vacuumed monthly"],
    "prometheus": ["prometheus scrapes 40 targets", "prometheus retention 30d", "prometheus remote write off", "prometheus on the monitor box", "prometheus alerts to ntfy"],
    "ansible": ["ansible: pull mode on hosts", "ansible vault for secrets", "ansible on the nas via cron", "ansible: roles split per host", "ansible idempotence checked weekly"],
}

REPO_RULES = [
    "never use print statements", "no default exports", "always rebase, never merge",
    "tailwind only, no css files", "semicolons required", "2-space indent",
    "no classes, functions only", "tests colocated with source",
]

SENTENCE_EXTRA = [
    # changes
    ("changes", "The API gateway listens on 9080.", "The API gateway moved to 9090 last sprint."),
    ("changes", "Postgres runs on the vps.", "Postgres moved to the nas."),
    ("changes", "Tokens live one day.", "Tokens live one hour as of the audit."),
    ("changes", "Feature X is beta.", "Feature X is GA now."),
    ("changes", "The queue is rabbitmq.", "The queue is sqs since the cloud move."),
    ("changes", "Retries wait 5 seconds.", "Retries wait 500ms now."),
    ("changes", "The wiki is dokuwiki.", "The wiki moved to obsidian publish."),
    # ends
    ("ends", "The irc bridge works.", "The irc bridge was shut down in march."),
    ("ends", "We run jenkins.", "Jenkins was deprecated and removed."),
    ("ends", "The minecraft server is up.", "The minecraft server was taken offline."),
    ("ends", "Sheets drive the roster.", "The sheet was retired for airtable."),
    ("ends", "The old search stack still serves.", "The old search stack was retired with the v2 launch."),
    ("ends", "We keep the fax number.", "The fax line was cancelled in spring."),
    ("ends", "The status page is manual.", "The manual status page was removed for uptime-kuma."),
    # repeats
    ("repeats", "The hub runs on the pi.", "The pi hosts the hub."),
    ("repeats", "Certificates renew every 90 days.", "Certs renew on a 90-day cycle."),
    ("repeats", "The team stands up at 9:30.", "Standup happens at 9:30 daily."),
    ("repeats", "Backups keep 30 days.", "Retention for backups is one month."),
    ("repeats", "The dashboard is internal only.", "Only staff can see the dashboard."),
    ("repeats", "Deploys need two approvals.", "Two approvals are required to deploy."),
    ("repeats", "The store is append-only.", "Records are only ever added, never edited."),
    # unrelated
    ("unrelated", "Deploys happen on fridays.", "Lunch is at noon."),
    ("unrelated", "The api rate limits per key.", "The office plants need water."),
    ("unrelated", "Logs go to the graylog.", "The meeting notes live in the wiki."),
    ("unrelated", "We use terraform for the cloud.", "The coffee machine is on the smart plug."),
    ("unrelated", "The nas has 4 bays.", "The on-call rotation is weekly."),
    ("unrelated", "Search relevance was tuned last month.", "The fridge cam detects the cat."),
    ("unrelated", "The vpn endpoints are split by region.", "The 3d printer needs a new nozzle."),
]

# note-style changes/ends/repeats template slots
SVC = ["vaultwarden", "uptime-kuma", "ntfy", "wallabag", "gitea", "woodpecker", "caddy", "authelia"]
PORTS = [8080, 8443, 3000, 9090, 8123, 8920, 8096, 8200, 2342, 9443]
TOOLS = ["terraform", "starship", "fzf", "bat", "zola", "pandoc", "wezterm", "k9s"]
TIMES = ["02:00", "03:30", "04:00", "23:00", "01:15"]
DURS = ["5m", "1h", "6h", "24h", "48h"]
PATHS = ["/var/log/app", "/data/logs", "/mnt/nas/logs", "~/logs/app"]
FEATS = ["read-only mode", "the maintenance banner", "telemetry", "signups", "the beta flag"]
APPS = ["the dashboard", "the api", "the sync job", "the exporter", "the ingest worker"]
MODE_PAIRS = [("sqlite", "postgres"), ("docker", "podman"), ("bash", "nushell"), ("pip", "uv"), ("s3", "minio")]
HOSTS = ["the vps", "the nas", "the pi", "the beelink", "the cloud box"]
NUM_PAIRS = [(100, 1000), (3, 1), (64, 512), (8, 2), (30, 90)]
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august"]
JOBS = ["the nightly rebuild", "the index refresh", "the thumbnailer", "the digest mail", "the prune job"]
THINGS = ["the legacy exporter", "the old auth proxy", "the static gallery", "the palm script", "the v1 ingest"]


def _pairs_changes(rng):
    out = []
    for _ in range(9):
        a, b = rng.sample(PORTS, 2)
        s = rng.choice(SVC)
        out.append(("changes", f"{s} on port {a}", f"{s} moved to port {b}"))
    for _ in range(9):
        t1, t2 = rng.sample(TIMES, 2)
        out.append(("changes", f"maintenance at {t1}", f"maintenance moved to {t2}"))
    for _ in range(9):
        d1, d2 = rng.sample(DURS, 2)
        out.append(("changes", f"session ttl {d1}", f"session ttl {d2} now"))
    for _ in range(8):
        p1, p2 = rng.sample(PATHS, 2)
        out.append(("changes", f"app logs at {p1}", f"app logs moved to {p2}"))
    for _ in range(8):
        f = rng.choice(FEATS)
        out.append(("changes", f"{f} off by default", f"{f} on by default now"))
    for _ in range(8):
        m1, m2 = rng.choice(MODE_PAIRS)
        out.append(("changes", f"packaging via {m1}", f"packaging switched to {m2}"))
    for _ in range(8):
        n1, n2 = rng.choice(NUM_PAIRS)
        out.append(("changes", f"workers {n1}", f"workers scaled to {n2}"))
    for _ in range(8):
        h1, h2 = rng.sample(HOSTS, 2)
        app = rng.choice(APPS)
        out.append(("changes", f"{app} on {h1}", f"{app} migrated to {h2}"))
    return out


def _pairs_ends(rng):
    out = []
    for _ in range(10):
        j = rng.choice(JOBS)
        out.append(("ends", f"{j} still runs", f"{j} cancelled — no longer needed"))
    for _ in range(9):
        t = rng.choice(THINGS)
        out.append(("ends", f"we keep {t}", f"{t} deprecated and removed"))
    for _ in range(9):
        s = rng.choice(SVC)
        out.append(("ends", f"{s} still live", f"{s} taken offline in {rng.choice(MONTHS)}"))
    for _ in range(9):
        app = rng.choice(APPS)
        out.append(("ends", f"{app} healthy", f"{app} shut down in {rng.choice(MONTHS)}"))
    for _ in range(8):
        t = rng.choice(TOOLS)
        out.append(("ends", f"metrics for {t}", f"metrics for {t} dropped entirely"))
    return out


def _pairs_repeats(rng):
    out = []
    for _ in range(9):
        s = rng.choice(SVC)
        out.append(("repeats", f"{s} on the nas", f"the nas runs {s}"))
    for _ in range(9):
        t = rng.choice(TOOLS)
        out.append(("repeats", f"{t} installed", f"{t} is set up"))
    for _ in range(9):
        d = rng.choice(DURS)
        out.append(("repeats", f"refresh every {d}", f"refreshes at {d} intervals"))
    for _ in range(9):
        h = rng.choice(HOSTS)
        out.append(("repeats", f"backups on {h}", f"{h} handles the backups"))
    for _ in range(9):
        n, _ = rng.choice(NUM_PAIRS)
        out.append(("repeats", f"retention {n}d", f"{n}-day retention"))
    return out


def _pairs_audit(rng):
    out = []
    for word, notes in AUDIT_WORDS.items():
        pairs = []
        for i in range(len(notes)):
            for j in range(i + 1, len(notes)):
                pairs.append((notes[i], notes[j]))
        rng.shuffle(pairs)
        for a, b in pairs[:4]:  # 4 pairs per word keeps unrelated bounded
            out.append(("unrelated", a, b))
    repos = ["alpha", "beta", "gamma", "delta", "core", "infra"]
    for _ in range(8):
        r1, r2 = rng.sample(repos, 2)
        rule = rng.choice(REPO_RULES)
        out.append(("unrelated", f"repo {r1} lint: {rule}", f"repo {r2} lint: {rule}"))
    return out


SENTENCE_CHANGES_EXTRA = [
    ("changes", "The export job runs on sundays.", "The export job moved to mondays."),
    ("changes", "Gitea runs on 3200.", "Gitea moved to 3201 after the clash."),
    ("changes", "The retry budget is 5.", "The retry budget is 2 now."),
    ("changes", "Headers are cached for an hour.", "Headers are cached for five minutes now."),
    ("changes", "The default branch is master.", "The default branch is main since the cleanup."),
    ("changes", "Notifications go to email.", "Notifications moved to the ntfy channel."),
    ("changes", "The search index lives on the vps.", "The search index moved to the nas."),
    ("changes", "The limit is 3 devices.", "The limit is 10 devices as of this week."),
    ("changes", "Tokens rotate monthly.", "Tokens rotate weekly after the audit."),
    ("changes", "The registry is docker hub.", "The registry moved to ghcr."),
    ("changes", "The app targets python 3.11.", "The app targets python 3.12 now."),
    ("changes", "The dashboards refresh hourly.", "The dashboards refresh every five minutes."),
    ("changes", "The api prefix is /v1.", "The api prefix is /v2 since the relaunch."),
    ("changes", "The bucket is in eu-west-1.", "The bucket moved to eu-central-1."),
    ("changes", "Backups keep a week.", "Backups keep a month now."),
    ("changes", "The pool has 2 nodes.", "The pool grew to 6 nodes."),
    ("changes", "The record ttl is 3600.", "The record ttl is 300 now."),
    ("changes", "Signups are closed.", "Signups are open as of monday."),
    ("changes", "The feed updates nightly.", "The feed updates hourly now."),
    ("changes", "The vpn endpoint is helsinki.", "The vpn endpoint moved to stockholm."),
    ("changes", "The mail relay is on port 25.", "The mail relay moved to 587."),
    ("changes", "The wiki runs mediawiki.", "The wiki moved to outline."),
    ("changes", "The camera saves jpegs.", "The camera saves raw now."),
    ("changes", "The archive is cold storage.", "The archive moved to the hot tier."),
    ("changes", "Escalation pages the on-call.", "Escalation opens a ticket now."),
    ("changes", "The store syncs on write.", "The store syncs hourly since the outage."),
    ("changes", "The booth shoots f/2.8.", "The booth shoots f/4 now."),
    ("changes", "The cli is pip-only.", "The cli moved to uv."),
]

# v2 pools: broaden the changes/repeats/ends families so the head does not
# learn one surface pattern per class (v1 collapsed changes recall to 0.2-0.3)
REFRESH = ["30s", "1m", "5m", "15m", "2h", "6h"]
RETN = [7, 14, 30, 90, 365]
TIMEOUTS = [5, 10, 30, 60, 120]
REPLICAS = [1, 2, 3, 6, 10]
THRESH = [0.5, 0.8, 0.9, 0.95, 0.99]
BRANCHES = ["master", "main", "develop", "release", "staging"]
DOMAINS = ["app.example.org", "status.example.org", "wiki.example.org", "api.internal", "dash.internal"]
APIS = [(1, 2), (2, 3)]
CHANNELS = ["telegram", "matrix", "discord", "email", "ntfy"]
OWNERS = ["sre", "platform", "data", "team-a", "team-b"]
QUOTAS = [(50, 200), (100, 500), (10, 100)]
SIZES = [(16, 64), (32, 256), (8, 32)]
VERSIONS2 = ["0.9", "1.2", "2.0", "3.4", "4.1"]
CIS = ["gitea actions", "github actions", "gitlab ci"]
CAS = ["letsencrypt", "buypass", "zeroalpha"]
AUTHM = ["oauth", "sso", "passkeys"]
SSHPORTS = [22, 2222, 2022]


def _pairs_changes2(rng):
    out = []
    for _ in range(8):
        a, b = rng.sample(REFRESH, 2)
        out.append(("changes", f"refresh {a}", f"refresh {b}, too chatty"))
    for _ in range(7):
        a, b = rng.sample(RETN, 2)
        out.append(("changes", f"retention {a}d", f"retention cut to {b}d"))
    for _ in range(7):
        a, b = rng.sample(TIMEOUTS, 2)
        out.append(("changes", f"timeout {a}s", f"timeout {b}s now"))
    for _ in range(7):
        a, b = rng.sample(REPLICAS, 2)
        out.append(("changes", f"replicas {a}", f"replicas {b}"))
    for _ in range(7):
        a, b = rng.sample(THRESH, 2)
        out.append(("changes", f"alert above {a}", f"alert above {b} now"))
    for _ in range(7):
        a, b = rng.sample(BRANCHES, 2)
        out.append(("changes", f"deploy from {a}", f"deploy from {b} now"))
    for _ in range(7):
        a, b = rng.sample(DOMAINS, 2)
        out.append(("changes", f"app at {a}", f"app moved to {b}"))
    for _ in range(7):
        a, b = rng.sample(VERSIONS2, 2)
        out.append(("changes", f"api v{a}", f"api v{b} now"))
    for _ in range(8):
        a, b = rng.sample(CHANNELS, 2)
        out.append(("changes", f"alerts on {a}", f"alerts moved to {b}"))
    for _ in range(6):
        o1, o2 = rng.sample(OWNERS, 2)
        thing = rng.choice(["the ingest", "the gateway", "the pipeline"])
        out.append(("changes", f"{thing} owned by {o1}", f"{thing} owned by {o2} now"))
    for _ in range(6):
        a, b = rng.choice(QUOTAS)
        out.append(("changes", f"quota {a}gb", f"quota raised to {b}gb"))
    for _ in range(6):
        a, b = rng.choice(SIZES)
        out.append(("changes", f"batch {a}", f"batch {b}"))
    return out


def _pairs_repeats2(rng):
    out = []
    for _ in range(8):
        t = rng.choice(TOOLS)
        v = rng.choice(VERSIONS2)
        out.append(("repeats", f"{t} v{v}", f"running {t} {v}"))
    for _ in range(7):
        h = rng.choice(DOMAINS)
        out.append(("repeats", f"api at {h}", f"the api lives on {h}"))
    for _ in range(7):
        n = rng.choice([3, 5, 8, 12])
        out.append(("repeats", f"team of {n}", f"{n} people on the team"))
    for _ in range(7):
        c = rng.choice(CIS)
        out.append(("repeats", f"ci on {c}", f"{c} runs the ci"))
    for _ in range(7):
        ca = rng.choice(CAS)
        out.append(("repeats", f"tls via {ca}", f"certs from {ca}"))
    for _ in range(7):
        m = rng.choice(AUTHM)
        out.append(("repeats", f"auth via {m}", f"sign in with {m}"))
    for _ in range(7):
        p = rng.choice(SSHPORTS)
        out.append(("repeats", f"ssh on {p}", f"ssh port {p}"))
    for _ in range(7):
        c = rng.choice(CHANNELS)
        out.append(("repeats", f"alerts to {c}", f"{c} gets the alerts"))
    for _ in range(6):
        d = rng.choice(RETN)
        out.append(("repeats", f"keep {d}d", f"retained {d} days"))
    for _ in range(6):
        n = rng.choice(REPLICAS)
        out.append(("repeats", f"{n} replicas", f"running {n} copies"))
    return out


def _pairs_ends2(rng):
    out = []
    for _ in range(6):
        t = rng.choice(TOOLS)
        out.append(("ends", f"{t} nightly build", f"{t} build cancelled — ci covers it"))
    for _ in range(6):
        s = rng.choice(SVC)
        out.append(("ends", f"{s} beta invite", f"{s} beta ended"))
    for _ in range(6):
        j = rng.choice(JOBS)
        out.append(("ends", f"{j} alert", f"{j} alert retired with the job"))
    for _ in range(6):
        p = rng.choice(HOSTS)
        out.append(("ends", f"staging on {p}", f"staging shut down in {rng.choice(MONTHS)}"))
    return out


# ── testC: held-out life domains, mixed registers, minimal-pair confusables ──
# none of these domains (chess, wedding, choir, volunteering, uni,
# boardgame, fantasy, neighbors, allotment, side-hustle; names tom/lena)
# appear anywhere in the training families.

TEST_C = [
    # 10 unrelated — audit shape in held-out domains, mixed registers
    ("unrelated", "chess club tuesdays at the library", "chess: lost on time again, blundered the endgame"),
    ("unrelated", "wedding venue deposit paid", "The wedding playlist stands at 40 shortlisted songs."),
    ("unrelated", "choir rehearsal thursdays 7pm", "choir tenors still short two people"),
    ("unrelated", "volunteering shifts on the roster app", "volunteering photos for the newsletter needed"),
    ("unrelated", "uni fees paid for semester 2", "The uni gym is free for students."),
    ("unrelated", "boardgame night last friday of the month", "boardgame shelf overflow again"),
    ("unrelated", "fantasy league draft on sunday", "fantasy league rival talk is 90% noise"),
    ("unrelated", "The neighbours borrowed our ladder last week.", "neighbors' cat visits daily"),
    ("unrelated", "allotment plot 12 taken", "allotment strawberries ripening early"),
    ("unrelated", "side-hustle shop made 3 sales", "side-hustle taxes go to the separate account"),
    # 10 changes — incl. superset, correction, register mix, one long pair
    ("changes", "chess club at the library", "chess club moved to the community center"),
    ("changes", "wedding set for june 12", "wedding moved to june 26 — venue conflict"),
    ("changes", "choir membership 15 a month", "choir dues raised to 20 a month"),
    ("changes", "volunteering every saturday", "volunteering switched to every other sunday"),
    ("changes", "uni: enrolled in intro stats for the spring term. Lectures are mondays.",
     "Swapped intro stats for the data science track after week two — same credits, better fit. Lectures are tuesdays now."),
    ("changes", "boardgame night at tom's", "boardgame night moved to lena's place"),
    ("changes", "fantasy draft sunday 8pm", "fantasy draft moved to saturday 3pm"),
    ("changes", "allotment watered mondays", "allotment watering now wednesdays"),
    ("changes", "prints at 12 each", "prints now 15 after paper costs rose"),
    ("changes", "neighbors keep the spare key under the mat", "The neighbours now leave the spare key with us."),
    # 10 ends — incl. one minimal-pair contrast with the changes above
    ("ends", "chess club tuesdays", "chess club disbanded — the library lost the room"),
    ("ends", "wedding photographer booked", "photographer cancelled; deposit refunded"),
    ("ends", "choir rehearsal thursdays", "dropped out of the choir in june"),
    ("ends", "volunteering at the shelter", "stopped the shelter shifts — schedule conflict"),
    ("ends", "uni german course b1", "german course dropped for this year"),
    ("ends", "boardgame: catan kickstarter backed", "kickstarter refunded, project cancelled"),
    ("ends", "fantasy league team set", "quit the fantasy league, too much time"),
    ("ends", "allotment fees for 2026", "gave up the allotment plot"),
    ("ends", "side-hustle stickers line", "discontinued the stickers line"),
    ("ends", "neighbors watering our plants", "neighbors moved away in march"),
    # 10 repeats
    ("repeats", "chess rating 1650", "rated 1650 at chess"),
    ("repeats", "wedding june 12", "the wedding is on 12 june"),
    ("repeats", "choir of 24 singers", "24 people sing in the choir"),
    ("repeats", "volunteering 4h monthly", "four hours a month of volunteering"),
    ("repeats", "uni semester starts april", "the semester begins in april"),
    ("repeats", "boardgame night monthly", "monthly boardgame night"),
    ("repeats", "draft sunday 8pm", "the draft is sunday at 8"),
    ("repeats", "allotment plot 12", "we hold plot 12 on the allotment"),
    ("repeats", "prints cost 15", "The prints go for 15."),
    ("repeats", "neighbors on holiday in july", "the neighbours are away in july"),
]

# ══ v3: life-domain scenarios, mixed registers, long notes, minimal pairs ═══
# testC domains (chess / wedding / choir / volunteering / uni / boardgame /
# fantasy / neighbors / allotment / side-hustle, names tom & lena) are held
# out — none of these strings appear in any training family.

LIFE_WORDS = {
    "meeting": ["standup notes go to the wiki", "monday meeting: sam takes notes", "meeting room b2 booked weekly", "client meeting agendas in the shared drive", "meeting-free wednesdays now"],
    "report": ["q3 report draft in docs", "report charts rebuilt in plotly", "expense report due the 5th", "annual report reviewed by legal", "report template v2"],
    "budget": ["budget for the reno: 40k", "monthly budget review with sam", "marketing budget cut 20%", "budget app switched to ynab", "team budget resets in april"],
    "flight": ["flight to lisbon booked, 6am", "flight delays: lounge access tips", "return flight changed to sunday", "flight credit expires in may", "red-eye flight sleep kit"],
    "doctor": ["dr lee on tuesdays", "doctor referral for the knee", "bloodwork every 6 months", "doctor: new clinic on 5th street", "insurance covers the doctor visits"],
    "school": ["school pickup at 3:15", "school trip payment due", "school newsletter fridays", "snow day means school closed", "school photos in october"],
    "kids": ["kids swim lessons saturdays", "kids share the ipad 30min", "kids dentist in june", "kids school lunch prefs", "kids flu shots done"],
    "car": ["car service every 10k km", "car insurance renews in march", "car snow tires in the garage", "car pool tuesdays with ana", "car charger lives in the trunk"],
    "gym2": ["gym peak hours 6-8pm", "gym buddy: dmitri", "gym closed on holidays", "home gym rack arrived", "gym towel rule actually enforced"],
    "recipe": ["recipe blog: no-knead bread", "recipe box synced to phone", "recipe swap with the neighbors once", "recipe fail: souffle collapsed", "family recipe book scanned"],
    "book": ["book budget 20 a month", "book list in the notes app", "book swap shelf in the lobby", "book: dune reread started", "book club pickup: thursday deliveries"],
    "phone": ["phone plan 10gb shared", "phone screen repaired, 80", "phone photos auto-upload on wifi", "phone storage full again", "work phone separate sim"],
    "landlord": ["landlord: rent due the 1st", "landlord fixing the balcony", "landlord contact via building mgmt", "landlord ok with the cat", "landlord inspection in april"],
    "insurance": ["insurance bundled home and car", "insurance deductible 500", "insurance renewal quotes compared", "travel insurance per trip", "insurance agent: maria"],
    "tax": ["tax filing deadline april 30", "tax folder receipts scanned", "tax software switched to the free one", "tax refund pending", "tax accountant meeting in feb"],
    "garden": ["garden tomatoes in july", "garden drip system on timer", "garden gophers defeated", "garden beds rebuilt", "garden compost starter working"],
    "dog": ["dog walk 7am and 6pm", "dog groomer every 6 weeks", "dog food subscription monthly", "dog park new hours", "dog scared of the vacuum"],
    "piano": ["piano lesson wednesdays 6", "piano exam grade 3 in spring", "piano tuned last month", "piano practice 20 min daily", "piano bench cushion added"],
    "camera": ["camera bag ready for hikes", "camera firmware updated", "camera club monthly challenge", "camera sensor cleaning kit", "camera strap replacement"],
    "birthday": ["birthday list in october", "birthday dinner reservations", "birthday budget 30 each", "birthday card box in the drawer", "mom's birthday flowers standing order"],
    "visa": ["visa appointment booked", "visa photos taken, 2 copies", "visa processing 3 weeks", "visa-free entry 90 days", "visa fee receipt saved"],
    "hotel": ["hotel in porto, refundable", "hotel breakfast included", "hotel late checkout 20", "hotel points via the card", "hotel reviews checked for noise"],
    "rental": ["rental car full-to-full", "rental deposit returned", "rental agreement pdf signed", "rental ski gear cheaper weekly", "rental reviews: avoid the local one"],
    "printer": ["printer ink xl now", "printer drivers finally work", "printer lives in the kids' room", "printer paper under the desk", "printer duplex jams fixed"],
}

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]
MONTHS_L = ["january", "february", "march", "april", "may", "june", "july", "september", "october"]
NAMES = ["sam", "ana", "priya", "noor", "marco", "dmitri"]
LESSONS = ["yoga", "pottery", "spanish", "swim", "pilates"]
DOCS = ["the essay", "the report", "tax filing", "the visa app", "the paper review"]
DEADLINES = ["mar 10", "mar 24", "apr 5", "apr 20", "jun 1"]
PRICE_ITEMS = ["netflix", "spotify", "hosting", "the phone plan", "the storage plan"]
PRICE_PAIRS = [(10, 16), (9, 14), (12, 20), (5, 9), (6, 11)]
DECISION_PAIRS = [("postgres", "mongo"), ("slack", "teams"), ("mac", "linux"), ("gcp", "aws"), ("notion", "obsidian"), ("chrome", "firefox")]
RENAME_PAIRS = [("the api team", "the services team"), ("growth", "product marketing"), ("site reliability", "platform"), ("the loft", "the studio")]
SUPERSET_PAIRS = [
    ("standup 9:30", "standup 9:30, mondays now async"),
    ("inbox zero friday", "inbox zero friday plus saturday sweep"),
    ("budget 40k reno", "budget 40k reno, 6k garden added"),
    ("renters insurance", "renters insurance plus valuation rider now"),
    ("chords c g d", "chords c g d, now also f and am"),
    ("coffee beans 1kg", "coffee beans 1kg monthly plus grinder service"),
    ("wifi 2.4ghz", "wifi 2.4 and 5ghz now"),
    ("house keys with ana", "house keys with ana and priya now"),
]
CORRECT_PAIRS = [
    ("the noise is the fan", "it was the pump bearing, not the fan"),
    ("diagnosed as tennis elbow", "corrected to golfer's elbow after the mri"),
    ("blamed the router", "it was the switch all along, not the router"),
    ("thought the leak was the roof", "the leak came from the balcony drain, not the roof"),
]
PHONES = ["07700 900123", "07700 900456", "07700 900789", "07700 900222"]
PLACES = ["the community center", "the studio on 5th", "the annex", "the main hall", "the park pavilion"]
TRIPS = ["the kyoto trip", "the roadtrip", "the lake week", "the city break"]
CANCEL_REASONS = ["flights too expensive", "the dates clash", "work freeze", "everyone dropped out"]
RACES = ["the 10k", "the half marathon", "the triathlon", "the trail race"]
RACE_REASONS = ["knee", "schedule", "shin splints", "work travel"]
SOLD_ITEMS = ["the treadmill", "the projector", "the bike trainer", "the cds", "the ski gear"]
CLUBS_LIFE = ["the photography circle", "the running club", "the supper club", "the film society"]
SUBS_LIFE = ["paramount", "the newspaper", "the cheese box", "adobe"]
WORK_SVCS2 = ["spotify", "hosting", "the storage plan"]

MINIMAL_PAIRS = [
    # same base, value moved (changes) vs thing cancelled (ends) — the
    # boundary the v2 head was soft on
    ("gym membership 30 a month", "gym raised it to 45 a month", "changes"),
    ("gym membership 30 a month", "cancelled the gym membership", "ends"),
    ("spanish on tuesdays", "spanish moved to thursdays", "changes"),
    ("spanish on tuesdays", "dropped the spanish course", "ends"),
    ("trip to kyoto in may", "kyoto trip moved to june", "changes"),
    ("trip to kyoto in may", "kyoto trip cancelled", "ends"),
    ("blog on wordpress", "blog moved to ghost", "changes"),
    ("blog on wordpress", "shut the blog down", "ends"),
    ("storage plan 100gb", "storage upgraded to 200gb", "changes"),
    ("storage plan 100gb", "cancelled the storage plan", "ends"),
    ("weekly review fridays", "weekly review moved to saturdays", "changes"),
    ("weekly review fridays", "dropped the weekly review", "ends"),
]

MIXED_REGISTER = [
    # one side note-style, the other sentence-style — real stores mix registers
    ("changes", "printer ink xl", "We switched to the high-yield cartridges; the regular ones ran out too fast."),
    ("changes", "The standing meeting is on wednesdays.", "standup moved to tuesdays"),
    ("changes", "rent 950", "Rent went up to 1,020 at renewal."),
    ("changes", "insurance deductible 500", "The deductible is 800 on the new policy."),
    ("changes", "car serviced at joes", "The car now goes to the dealer service center."),
    ("changes", "hosting on the starter plan", "Hosting moved to the pro tier after the traffic spike."),
    ("ends", "landline 555-0134", "The landline was cancelled; mobile only now."),
    ("ends", "magazine subs active", "Cancelled all the magazine subs in the declutter."),
    ("ends", "paper invoices by mail", "Switched everything to e-invoice; the paper ones are discontinued."),
    ("ends", "lunch deliveries mondays", "Dropped the monday lunch deliveries, cooking again."),
    ("ends", "The aquarium is running.", "tank taken down during the move"),
    ("ends", "family chat on slack", "deleted the family slack, back to whatsapp"),
    ("repeats", "library card 2201", "Our library card number is 2201."),
    ("repeats", "Bike serviced every spring.", "yearly bike service each spring"),
    ("repeats", "bins out tuesdays", "Tuesday is bin day"),
    ("repeats", "The course costs 300.", "course fee: 300"),
    ("repeats", "kids swim saturdays", "Saturday swimming lessons for the kids"),
    ("repeats", "passport expires 2028", "The passport is valid until 2028."),
    ("unrelated", "cat food auto-ship monthly", "The cat started sleeping on the printer."),
    ("unrelated", "The office moved to floor 4.", "office party budget 200"),
    ("unrelated", "monday bin day", "monday pilates fully booked"),
    ("unrelated", "flight seat 12a", "Flight anxiety tips from the podcast."),
    ("unrelated", "The garage door remote broke.", "garage now stores the bikes, not the car"),
    ("unrelated", "insurance for the laptop", "travel insurance excludes wintersports"),
]

LONG_PAIRS = [
    ("changes", "The team standardized on Postgres at the offsite. Old MySQL services were left for later.",
     "We finished the MySQL-to-Postgres migration last quarter. Everything runs on Postgres now, legacy included."),
    ("changes", "Dad takes 10mg of the medication each morning. The doctor said to watch for dizziness.",
     "Dad's dose went up to 20mg after the checkup. No dizziness so far, so they are keeping it."),
    ("ends", "We run the community newsletter every month. About 200 subscribers, all hand-assembled.",
     "The newsletter stopped in May. The subscribers moved to the town's official channel."),
    ("ends", "The startup pitch is scheduled for March. Both founders rehearse weekly.",
     "The pitch was cancelled — the lead investor pulled out and the round is shelved."),
    ("repeats", "The library is a ten-minute walk from the flat. It closes at eight on weekdays.",
     "From our flat, the library is ten minutes on foot. Weekday closing time is eight."),
    ("repeats", "Anna leads the design guild. The guild meets on the first Thursday of each month.",
     "The design guild, run by Anna, gathers every first Thursday."),
    ("unrelated", "The team offsite is planned for June in Lisbon. Flights are booked for nine people.",
     "Lisbon's tram 28 is the tourist trap everyone warns about. Still worth one ride."),
    ("unrelated", "We switched the office coffee to the dark roast. The old beans went to the kitchen at home.",
     "The kitchen tap drips unless tightened fully. A washer is on the shopping list."),
]


def _pairs_changes_life(rng):
    out = []
    for _ in range(8):
        item, (a, b) = rng.choice(list(zip(PRICE_ITEMS, PRICE_PAIRS)))
        out.append(("changes", f"{item} {a} a month", f"{item} raised to {b} a month"))
    for _ in range(9):
        c = rng.choice(LESSONS)
        d1, d2 = rng.sample(DAYS, 2)
        out.append(("changes", f"{c} on {d1}s", f"{c} moved to {d2}s"))
    for _ in range(8):
        doc = rng.choice(DOCS)
        d1, d2 = rng.sample(DEADLINES, 2)
        out.append(("changes", f"{doc} due {d1}", f"{doc} deadline moved to {d2}"))
    for _ in range(7):
        thing = rng.choice(["the standup rota", "the book box", "the on-call"])
        n1, n2 = rng.sample(NAMES, 2)
        out.append(("changes", f"{thing} run by {n1}", f"{thing} run by {n2} now"))
    for a, b in DECISION_PAIRS:
        out.append(("changes", f"went with {a}", f"switched to {b} after the pilot"))
    for old, new in RENAME_PAIRS:
        out.append(("changes", old, f"renamed to {new}"))
    for a, b in SUPERSET_PAIRS:
        out.append(("changes", a, b))
    for a, b in CORRECT_PAIRS:
        out.append(("changes", a, b))
    for _ in range(6):
        n = rng.choice(NAMES)
        p1, p2 = rng.sample(PHONES, 2)
        out.append(("changes", f"{n} at {p1}", f"{n}'s new number {p2}"))
    for _ in range(6):
        p1, p2 = rng.sample(PLACES, 2)
        out.append(("changes", f"yoga at {p1}", f"yoga moved to {p2}"))
    out.append(("changes", "vitamin d 1000", "vitamin d 2000 now"))
    out.append(("changes", "spotify free", "upgraded spotify to family"))
    return out


def _pairs_ends_life(rng):
    out = []
    for _ in range(9):
        t = rng.choice(TRIPS)
        out.append(("ends", f"{t} in {rng.choice(MONTHS_L)}", f"{t} cancelled — {rng.choice(CANCEL_REASONS)}"))
    for _ in range(8):
        r = rng.choice(RACES)
        out.append(("ends", f"training for {r}", f"dropped {r} — {rng.choice(RACE_REASONS)}"))
    for _ in range(7):
        i = rng.choice(SOLD_ITEMS)
        out.append(("ends", f"the {i} in the garage", f"sold the {i} last week"))
    for _ in range(7):
        c = rng.choice(CLUBS_LIFE)
        out.append(("ends", f"{c} meets monthly", f"{c} disbanded in {rng.choice(MONTHS_L)}"))
    for _ in range(7):
        s = rng.choice(SUBS_LIFE)
        out.append(("ends", f"{s} subscription", f"cancelled {s}, barely used it"))
    for _ in range(7):
        m = rng.choice(MONTHS_L)
        out.append(("ends", "the flat on oak street", f"moved out of oak street in {m}"))
    for _ in range(6):
        n = rng.choice(NAMES)
        out.append(("ends", f"{n} on the team", f"{n} left the team in {rng.choice(MONTHS_L)}"))
    out.append(("ends", "spanish b1 ongoing", "spanish dropped for this year"))
    out.append(("ends", "5am runs", "stopped the 5am runs — evenings now"))
    return out


def _pairs_repeats_life(rng):
    out = []
    for _ in range(8):
        n = rng.choice(NAMES)
        d = rng.choice(DAYS)
        out.append(("repeats", f"{n} appointment on {d}", f"{d} appointment with {n}"))
    for _ in range(8):
        k = rng.choice([3, 5, 8, 10, 21])
        out.append(("repeats", f"{k}k run", f"run of {k} kilometres"))
    for _ in range(8):
        item, (p, _) = rng.choice(list(zip(WORK_SVCS2, PRICE_PAIRS)))
        out.append(("repeats", f"{item} {p} a month", f"paying {p} monthly for {item}"))
    for _ in range(8):
        d = rng.choice(DAYS)
        out.append(("repeats", f"meeting every {d}", f"{d}s meeting"))
    for _ in range(7):
        d = rng.choice([250, 500, 800])
        out.append(("repeats", f"insurance deductible {d}", f"deductible of {d} on the insurance"))
    for _ in range(7):
        n = rng.choice([4, 6, 9, 12])
        out.append(("repeats", f"team of {n} reps", f"{n} reps on the team"))
    return out


def _pairs_audit_life(rng):
    out = []
    for word, notes in LIFE_WORDS.items():
        pairs = [(notes[i], notes[j])
                 for i in range(len(notes)) for j in range(i + 1, len(notes))]
        rng.shuffle(pairs)
        for a, b in pairs[:4]:
            out.append(("unrelated", a, b))
    return out


def build(seed=13):
    """train/val split, stratified-ish by simple shuffle; 12% val."""
    rng = random.Random(seed)
    data = (_pairs_changes(rng) + _pairs_changes2(rng) + _pairs_changes_life(rng)
            + _pairs_ends(rng) + _pairs_ends2(rng) + _pairs_ends_life(rng)
            + _pairs_repeats(rng) + _pairs_repeats2(rng) + _pairs_repeats_life(rng)
            + _pairs_audit(rng) + _pairs_audit_life(rng)
            + [(lbl, a, b) for a, b, lbl in MINIMAL_PAIRS]
            + MIXED_REGISTER + LONG_PAIRS
            + SENTENCE_EXTRA + SENTENCE_CHANGES_EXTRA)
    # dedupe exact pairs
    seen, uniq = set(), []
    for row in data:
        key = (row[1], row[2])
        if key not in seen:
            seen.add(key)
            uniq.append(row)
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * 0.12))
    return uniq[n_val:], uniq[:n_val]


if __name__ == "__main__":
    import json
    import sys
    train, val = build()
    test_keys = {(o, n) for _, o, n in TEST_A + TEST_B + TEST_C}
    overlap = [(o, n) for _, o, n in train + val if (o, n) in test_keys]
    counts = {}
    for lbl, *_ in train + val:
        counts[lbl] = counts.get(lbl, 0) + 1
    print(json.dumps({
        "train": len(train), "val": len(val),
        "label_counts_train_val": counts,
        "overlap_with_tests": overlap,
        "testA": len(TEST_A), "testB": len(TEST_B), "testC": len(TEST_C),
    }, indent=1))
    sys.exit(1 if overlap else 0)
