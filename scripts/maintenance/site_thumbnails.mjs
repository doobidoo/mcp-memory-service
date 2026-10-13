#!/usr/bin/env node
// Regenerates the link thumbnails on the landing page from the real pages in site/,
// and the Open Graph card (the picture shown when a link is shared) from og_card.html.
//
//   node scripts/maintenance/site_thumbnails.mjs [name ...]
//
// No dependencies: Node 22 or newer (global WebSocket) and a Chromium or Chrome
// (set CHROME_BIN to pick one). It serves site/ on a free port, drives the browser
// over the DevTools protocol and writes site/assets/images/thumb-<name>.webp at
// 1200x675. Run it again whenever one of the four pages changes visibly.
import { spawn, spawnSync } from 'node:child_process';
import { createServer } from 'node:http';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const SITE = resolve(dirname(fileURLToPath(import.meta.url)), '../../site');
const OUT = join(SITE, 'assets/images');
const HERE = dirname(fileURLToPath(import.meta.url));
const WIDTH = 1440, HEIGHT = 810, OUT_WIDTH = 1200;

// Each shot waits for the page to be in the state worth showing.
const SHOTS = {
  architecture: { path: '/architecture/?theme=dark&embed=1', scheme: 'dark' },
  deployment: { path: '/deployment/?theme=dark&embed=1', scheme: 'dark' },
  // Not part of site/: served from this directory, written as a 1200x630 JPEG, the size social networks crop to.
  og: { path: '/__og.html', file: join(HERE, 'og_card.html'), scheme: 'dark', width: 1200, height: 630, scale: 1, format: 'jpeg', out: 'og-home.jpg' },
  blog: { path: '/blog/remote-mcp-tutorial.html', scheme: 'light' },
  demo: {
    path: '/demo/',
    scheme: 'dark',
    // The model (about 23 MB) loads first; then one search so the card shows results.
    run: `(async () => {
      const until = async (test, ms) => { const end = Date.now() + ms; while (!test()) { if (Date.now() > end) throw new Error('timeout'); await new Promise(r => setTimeout(r, 250)); } };
      await until(() => /^Ready/.test(document.getElementById('status-text').textContent), 120000);
      document.getElementById('queryInput').value = 'security credentials expiring';
      document.getElementById('searchBtn').click();
      await until(() => document.querySelectorAll('#results .result').length > 0, 15000);
      window.scrollTo(0, document.querySelector('.demo-card').getBoundingClientRect().top + scrollY - 40);
    })()`,
  },
};

const TYPES = { '.html': 'text/html; charset=utf-8', '.css': 'text/css', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.json': 'application/json', '.wasm': 'application/wasm', '.woff2': 'font/woff2', '.png': 'image/png', '.jpg': 'image/jpeg',
  '.webp': 'image/webp', '.onnx': 'application/octet-stream', '.svg': 'image/svg+xml', '.ico': 'image/x-icon' };

function serve() {
  const server = createServer(async (req, res) => {
    let p = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname));
    if (p.endsWith('/')) p += 'index.html';
    try {
      const file = p === '/__og.html' ? SHOTS.og.file : join(SITE, p);
      if (!file.startsWith(SITE) && file !== SHOTS.og.file) throw new Error('outside site');
      res.writeHead(200, { 'content-type': TYPES[extname(file)] || 'application/octet-stream' });
      res.end(await readFile(file));
    } catch { res.writeHead(404).end('not found'); }
  });
  return new Promise((ok) => server.listen(0, '127.0.0.1', () => ok(server)));
}

function findChrome() {
  const candidates = [process.env.CHROME_BIN, 'chromium', 'chromium-browser', 'google-chrome-stable', 'google-chrome', 'chrome'].filter(Boolean);
  for (const c of candidates) if (spawnSync('which', [c]).status === 0) return c;
  throw new Error('No Chromium or Chrome found; set CHROME_BIN');
}

async function launch(userDir) {
  const proc = spawn(findChrome(), ['--headless=new', '--disable-gpu', '--hide-scrollbars', '--no-first-run', '--remote-debugging-port=0',
    `--user-data-dir=${userDir}`, 'about:blank'], { stdio: ['ignore', 'ignore', 'pipe'] });
  let timer;
  try {
    const wsUrl = await new Promise((ok, fail) => {
      let buf = '';
      proc.stderr.on('data', (d) => { buf += d; const m = buf.match(/DevTools listening on (ws:\/\/\S+)/); if (m) ok(m[1]); });
      proc.on('error', fail);
      proc.on('exit', () => fail(new Error('Chrome exited before it was ready')));
      timer = setTimeout(() => fail(new Error('Chrome did not start within 20 s')), 20000);
    });
    return { proc, wsUrl };
  } catch (err) {
    proc.kill();
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

function cdp(wsUrl) {
  const ws = new WebSocket(wsUrl);
  let id = 0, dead = null, rejectOpen;
  const pending = new Map(); const waiters = [];
  const die = (err) => {
    dead ||= err;
    for (const p of pending.values()) p.fail(err);
    for (const w of waiters) w.fail(err);
    pending.clear(); waiters.length = 0; rejectOpen(err);
  };
  ws.onmessage = (e) => {
    const m = JSON.parse(e.data);
    if (m.id && pending.has(m.id)) { const { ok, fail } = pending.get(m.id); pending.delete(m.id); m.error ? fail(new Error(m.error.message)) : ok(m.result); }
    else if (m.method) for (const w of [...waiters]) if (w.method === m.method && (!w.session || w.session === m.sessionId)) { waiters.splice(waiters.indexOf(w), 1); w.ok(m.params); }
  };
  ws.onerror = () => die(new Error('DevTools connection failed'));
  ws.onclose = () => die(new Error('DevTools connection closed (did Chrome crash?)'));
  const opened = new Promise((ok, no) => { ws.onopen = ok; rejectOpen = no; });
  return {
    opened,
    send: (method, params = {}, sessionId, ms = 30000) => dead ? Promise.reject(dead) : new Promise((ok, no) => {
      const i = ++id;
      const t = setTimeout(() => { pending.delete(i); no(new Error(`${method} timed out after ${ms / 1000} s`)); }, ms);
      pending.set(i, { ok: (r) => { clearTimeout(t); ok(r); }, fail: (e) => { clearTimeout(t); no(e); } });
      ws.send(JSON.stringify({ id: i, method, params, sessionId }));
    }),
    once: (method, session, ms = 60000) => dead ? Promise.reject(dead) : new Promise((ok, no) => {
      const w = { method, session,
        ok: (r) => { clearTimeout(t); ok(r); }, fail: (e) => { clearTimeout(t); no(e); } };
      const t = setTimeout(() => { waiters.splice(waiters.indexOf(w), 1); no(new Error(`${method} timed out after ${ms / 1000} s`)); }, ms);
      waiters.push(w);
    }),
    close: () => ws.close(),
  };
}

async function shoot(browser, base, name, shot) {
  const { targetId } = await browser.send('Target.createTarget', { url: 'about:blank' });
  const { sessionId } = await browser.send('Target.attachToTarget', { targetId, flatten: true });
  const s = (m, p, ms) => browser.send(m, p, sessionId, ms);
  await s('Page.enable');
  await s('Emulation.setDeviceMetricsOverride', { width: shot.width || WIDTH, height: shot.height || HEIGHT, deviceScaleFactor: shot.scale || OUT_WIDTH / WIDTH, mobile: false });
  await s('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-color-scheme', value: shot.scheme }, { name: 'prefers-reduced-motion', value: 'reduce' }] });
  // Await both together: if the browser dies during navigation, both reject, and neither may go unhandled.
  await Promise.all([browser.once('Page.loadEventFired', sessionId), s('Page.navigate', { url: base + shot.path })]);
  const wait = `(async () => { await document.fonts.ready; ${shot.run ? `await ${shot.run};` : ''} await new Promise(r => setTimeout(r, 600)); })()`;
  const r = await s('Runtime.evaluate', { expression: wait, awaitPromise: true }, 180000);
  if (r.exceptionDetails) throw new Error(`${name}: ${r.exceptionDetails.exception?.description || r.exceptionDetails.text}`);
  const { data } = await s('Page.captureScreenshot', { format: shot.format || 'webp', quality: 82 });
  const file = join(OUT, shot.out || `thumb-${name}.webp`);
  await writeFile(file, Buffer.from(data, 'base64'));
  await browser.send('Target.closeTarget', { targetId });
  return file;
}

const names = process.argv.slice(2).length ? process.argv.slice(2) : Object.keys(SHOTS);
for (const n of names) if (!SHOTS[n]) { console.error(`Unknown thumbnail "${n}". Known: ${Object.keys(SHOTS).join(', ')}`); process.exit(2); }

const server = await serve();
let userDir, proc, browser;
try {
  userDir = await mkdtemp(join(tmpdir(), 'site-thumbs-'));
  let wsUrl;
  ({ proc, wsUrl } = await launch(userDir));
  browser = cdp(wsUrl);
  await browser.opened;
  const base = `http://127.0.0.1:${server.address().port}`;
  for (const n of names) console.log('wrote', await shoot(browser, base, n, SHOTS[n]));
} catch (err) {
  console.error(`site_thumbnails: ${err.message}`);
  process.exitCode = 1;
} finally {
  browser?.close(); server.close();
  if (proc && proc.exitCode === null) {
    // Chrome keeps writing to its profile while it shuts down, so wait for it before deleting the directory.
    const exited = new Promise((ok) => proc.once('exit', ok));
    proc.kill();
    await Promise.race([exited, new Promise((ok) => setTimeout(ok, 5000))]);
  }
  if (userDir) await rm(userDir, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 }).catch(() => {});
}
