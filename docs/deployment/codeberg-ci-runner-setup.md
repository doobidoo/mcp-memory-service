# Codeberg CI: self-hosted Forgejo runner on Hetzner (tinyclaw)

> **Historical, as of 2026-09-05.** Development moved back to GitHub and CI runs on
> GitHub-hosted `ubuntu-latest` runners via [`.github/workflows/`](../../.github/workflows/).
> Nothing below is applicable any more. It is kept because the runner is still a working
> reference for anyone setting up Forgejo Actions elsewhere, and because the constraints
> it documents (github.com unreachable from the box, hence the action mirror and the raw
> `docker buildx` calls) explain why the ported workflows look the way they do.

Sets up a Forgejo `act_runner` on the Hetzner box so Codeberg Actions can run the
release pipeline at [`.forgejo/workflows/release.yml`](../../.forgejo/workflows/release.yml).
No GitHub auth involved — does not touch the GitHub lockout.

**Target box (verified 2026-06-01):** `tinyclaw` = `ubuntu-4gb-nbg1-1`, Ubuntu 24.04.4, x86_64,
3.7 GB RAM, 26 GB free. Docker NOT installed, no runner present.

> Image-arch decision (final): the full torch image is published **amd64-only**; the
> slim/ONNX image stays **multi-arch (amd64+arm64)**. arm64 of the full image is not
> published — emulated builds are slow/memory-heavy and ARM users who need it build it
> locally on native ARM hardware (no emulation). So this 4 GB box is sufficient as-is —
> no RAM upgrade and no second ARM runner needed.

---

## Prerequisites the USER must do in the Codeberg web UI

These cannot be automated (account/UI scope):

1. **Enable Actions for the repo:** `doobidoo/mcp-memory-service` -> Settings -> Advanced
   (or Units) -> tick **Actions** -> Update. *(Codeberg Actions is open-alpha — if the
   toggle is absent, request access per Codeberg docs. This is the one hard dependency:
   if Codeberg won't enable Actions / external runners for the repo, fall back to
   Woodpecker CI at ci.codeberg.org instead.)*
2. **Create a runner registration token:** Settings -> Actions -> Runners ->
   **Create new Runner** -> copy the token (looks like a long hex string).
3. **Add repo secrets:** Settings -> Actions -> Secrets -> add:
   - `PYPI_TOKEN` (classic PyPI API token, project-scoped, starts with `pypi-`)
   - `DOCKER_USERNAME`, `DOCKER_PASSWORD` (Docker Hub username + access token)

Hand over only the **registration token** for the steps below. Secrets stay in the UI.

---

## Step 1 — Install Docker (on tinyclaw)

```bash
ssh tinyclaw
# Docker engine + buildx + compose plugins (official convenience script)
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu        # run docker without sudo
# QEMU binfmt for multi-arch (slim image arm64)
sudo docker run --privileged --rm tonistiigi/binfmt --install all
# log out / back in so the docker group applies, then verify:
exit
ssh tinyclaw 'docker run --rm hello-world && docker buildx version'
```

## Step 2 — Install the Forgejo runner binary (on tinyclaw)

```bash
ssh tinyclaw
RUNNER_VERSION=12.10.2   # check https://code.forgejo.org/forgejo/runner/releases for latest
curl -fsSL -o /tmp/forgejo-runner \
  "https://code.forgejo.org/forgejo/runner/releases/download/v${RUNNER_VERSION}/forgejo-runner-${RUNNER_VERSION}-linux-amd64"
sudo install -m 0755 /tmp/forgejo-runner /usr/local/bin/forgejo-runner
forgejo-runner --version
```

## Step 3 — Register against Codeberg (needs the registration token)

The registration token can also be fetched via the API (no UI copy needed) if your
Codeberg token has repo scope:
`GET /api/v1/repos/doobidoo/mcp-memory-service/actions/runners/registration-token`.

Label maps to a capable runner image (has node + python + git + docker CLI + buildx):

```bash
sudo mkdir -p /etc/forgejo-runner && sudo chown ubuntu:ubuntu /etc/forgejo-runner
cd /etc/forgejo-runner
forgejo-runner register --no-interactive \
  --instance https://codeberg.org \
  --token "<REGISTRATION_TOKEN>" \
  --name tinyclaw \
  --labels "docker:docker://catthehacker/ubuntu:act-22.04"
# writes .runner (owned by ubuntu) in the cwd
```

## Step 4 — Config (mount host Docker socket into job containers)

Do NOT point actions at github.com — github.com is poisoned on this box (suspended
account + an `insteadOf` rewrite). Codeberg resolves bare `uses: actions/x` via its own
mirror (`data.forgejo.org`) server-side, so leave action resolution alone and just keep
workflow `uses:` refs BARE (never `https://github.com/...`).

The only config we need is mounting the host Docker socket so `docker buildx` works inside
job containers:

```yaml
# /etc/forgejo-runner/config.yaml
log:
  level: info
runner:
  capacity: 1
  timeout: 3h
cache:
  enabled: true
container:
  network: bridge
  privileged: false
  # Lets buildx/docker build in job containers reach the host daemon.
  # Trade-off: any job gets root-equivalent host Docker access — fine for a
  # trusted-maintainer-only repo.
  options: "-v /var/run/docker.sock:/var/run/docker.sock"
  valid_volumes:
    - /var/run/docker.sock
  force_pull: false
```

## Step 5 — systemd service

```bash
sudo tee /etc/systemd/system/forgejo-runner.service >/dev/null <<'UNIT'
[Unit]
Description=Forgejo Actions runner
After=docker.service
Requires=docker.service

[Service]
ExecStart=/usr/local/bin/forgejo-runner daemon --config /etc/forgejo-runner/config.yaml
WorkingDirectory=/etc/forgejo-runner
User=ubuntu
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now forgejo-runner
systemctl status forgejo-runner --no-pager
```

After this the runner shows up green under Settings -> Actions -> Runners.

---

## Verify

1. Push a throwaway tag or use **workflow_dispatch** on the release workflow.
2. Watch the run under the repo's **Actions** tab.
3. First run will be slow (pulls images, no cache). Subsequent runs reuse the local
   Docker layer cache on the box.

## Rollback / teardown

```bash
sudo systemctl disable --now forgejo-runner
sudo rm -f /etc/systemd/system/forgejo-runner.service /usr/local/bin/forgejo-runner
sudo rm -rf /etc/forgejo-runner
```

## Gotchas (learned 2026-06-01)

- **Never reference `https://github.com/...` in `uses:`.** The box has a global
  `url.git@github.com:.insteadOf https://github.com/` rewrite + stored creds for the
  suspended account, so any github.com git op fails with "Your account is suspended".
  Bare `uses: actions/checkout@v4` resolves via `data.forgejo.org` and works.
- **Prefer raw `docker buildx` over `docker/*` actions** in build jobs — avoids action
  mirror gaps. The runner image already ships docker CLI + buildx, host socket is mounted.
- **`requires-python >=3.10`** matches the runner image's python3.10, so no
  `actions/setup-python` (which would fetch from github.com) is needed.
- **`workflow_dispatch` no-ops** unless the workflow file also exists on the default
  branch; use a tag push (or push to a branch the workflow's `on:` matches) to test.

## Scope note

This migrates only the **release path** (test -> PyPI -> Docker Hub). The other ~20
GitHub workflows (CI on PR, quality gates, CodeQL, link-checks, etc.) are NOT migrated
yet — do those incrementally once the release path is proven on the runner.
