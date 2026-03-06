"""MCP Apps renderer for mcp-memory-service.

Generates interactive HTML UIs for MCP Apps-capable clients.

Protocol: MCP Apps uses EmbeddedResource with mimeType=text/html.
The host (VS Code Insiders) renders the HTML in a sandboxed iframe.

Activation:
  Remote server: MEMORY_MCP_APPS=true  OR  X-MCP-Insiders: true header
  Local server:  MEMORY_MCP_APPS=true environment variable

Security:
  - All HTML is self-contained (no external CDN dependencies)
  - Compatible with strict sandbox iframe policies
  - No external domain CSP declarations needed

Ref: https://github.com/github/github-mcp-server/discussions/2048
"""

import os
import html as html_module
from datetime import datetime, timezone
from typing import Any

try:
    from mcp.types import EmbeddedResource, TextResourceContents
    _MCP_TYPES_AVAILABLE = True
except ImportError:
    _MCP_TYPES_AVAILABLE = False


def mcp_apps_enabled() -> bool:
    """Check if MCP Apps UI rendering is enabled.

    Controlled by MEMORY_MCP_APPS=true environment variable.
    When using the GitHub MCP Server's Insiders Mode endpoint,
    also check for runtime header detection (requires FastMCP support).
    """
    return os.environ.get("MEMORY_MCP_APPS", "").lower() in ("true", "1", "yes")


def _escape(value: Any) -> str:
    """HTML-escape a value for safe embedding."""
    return html_module.escape(str(value) if value is not None else "")


def _format_ts(ts: Any) -> str:
    """Format a timestamp for display."""
    if not ts:
        return ""
    try:
        if isinstance(ts, (int, float)):
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        elif isinstance(ts, str):
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        else:
            return str(ts)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)


_CSS = """
:root {
  --bg: #1e1e2e;
  --surface: #2a2a3e;
  --surface2: #313145;
  --border: #3d3d5c;
  --text: #cdd6f4;
  --text-dim: #7f849c;
  --accent: #89b4fa;
  --green: #a6e3a1;
  --yellow: #f9e2af;
  --red: #f38ba8;
  --purple: #cba6f7;
  --teal: #94e2d5;
  --radius: 6px;
  --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  --font-mono: 'Cascadia Code', 'Fira Code', 'JetBrains Mono', monospace;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html { font-size: 13px; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font);
  padding: 12px;
  line-height: 1.5;
}
a { color: var(--accent); text-decoration: none; }
.header {
  display: flex; align-items: center; gap: 8px;
  margin-bottom: 12px; padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
}
.header h1 { font-size: 14px; font-weight: 600; color: var(--text); }
.header .meta { font-size: 11px; color: var(--text-dim); margin-left: auto; }
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 10px 12px;
  margin-bottom: 8px;
  transition: border-color 0.15s;
}
.card:hover { border-color: var(--accent); }
.card-header {
  display: flex; align-items: flex-start;
  gap: 8px; margin-bottom: 6px;
}
.score {
  flex-shrink: 0;
  font-size: 10px; font-weight: 700; font-family: var(--font-mono);
  padding: 2px 6px; border-radius: 3px;
  background: var(--surface2); color: var(--accent);
}
.score.high { background: #1e3a2a; color: var(--green); }
.score.mid  { background: #3a3a1e; color: var(--yellow); }
.score.low  { background: #3a1e1e; color: var(--red); }
.content {
  font-size: 12px; color: var(--text);
  line-height: 1.6;
  word-break: break-word;
  flex: 1;
}
.content.truncated {
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.tags { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; }
.tag {
  font-size: 10px; padding: 1px 6px;
  border-radius: 10px;
  background: var(--surface2); color: var(--purple);
  border: 1px solid #4a3a6a;
}
.card-footer {
  display: flex; align-items: center; gap: 8px;
  margin-top: 8px; padding-top: 6px;
  border-top: 1px solid var(--border);
}
.hash {
  font-family: var(--font-mono); font-size: 10px;
  color: var(--text-dim); flex: 1;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.btn {
  font-size: 10px; padding: 2px 8px;
  border-radius: 3px; cursor: pointer;
  background: var(--surface2); color: var(--text-dim);
  border: 1px solid var(--border);
  transition: all 0.15s;
}
.btn:hover { border-color: var(--accent); color: var(--accent); }
.btn.copied { color: var(--green); border-color: var(--green); }
.ts { font-size: 10px; color: var(--text-dim); }
.type-badge {
  font-size: 10px; padding: 1px 6px;
  border-radius: 3px;
  background: #1e2a3a; color: var(--teal);
  border: 1px solid #2a4a5a;
}
.empty {
  text-align: center; padding: 32px;
  color: var(--text-dim); font-size: 12px;
}
.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 8px; margin-bottom: 12px;
}
.stat-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 10px;
  text-align: center;
}
.stat-value {
  font-size: 22px; font-weight: 700;
  color: var(--accent); font-family: var(--font-mono);
}
.stat-label { font-size: 10px; color: var(--text-dim); margin-top: 2px; }
.status-dot {
  width: 8px; height: 8px; border-radius: 50%;
  display: inline-block; margin-right: 4px;
}
.status-dot.healthy { background: var(--green); box-shadow: 0 0 6px var(--green); }
.status-dot.error   { background: var(--red);   box-shadow: 0 0 6px var(--red); }
.info-row {
  display: flex; justify-content: space-between;
  padding: 5px 0; font-size: 11px;
  border-bottom: 1px solid var(--border);
}
.info-row:last-child { border-bottom: none; }
.info-key { color: var(--text-dim); }
.info-val { color: var(--text); font-family: var(--font-mono); font-size: 10px; }
.pagination {
  display: flex; align-items: center; gap: 8px;
  margin-top: 12px; justify-content: center;
  font-size: 11px; color: var(--text-dim);
}
.section {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 10px 12px;
  margin-bottom: 10px;
}
.section h2 {
  font-size: 11px; font-weight: 600;
  color: var(--text-dim); text-transform: uppercase;
  letter-spacing: 0.5px; margin-bottom: 8px;
}
"""


def _base_page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_escape(title)}</title>
<style>{_CSS}</style>
</head>
<body>
{body}
<script>
function copyHash(hash, btn) {{
  navigator.clipboard.writeText(hash).then(() => {{
    btn.textContent = 'Copied!';
    btn.classList.add('copied');
    setTimeout(() => {{
      btn.textContent = 'Copy hash';
      btn.classList.remove('copied');
    }}, 1500);
  }});
}}
function toggleExpand(id) {{
  const el = document.getElementById(id);
  if (el) {{
    el.classList.toggle('truncated');
  }}
}}
</script>
</body>
</html>"""


def _score_class(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.50:
        return "mid"
    return "low"


def render_memory_search_results(memories: list, query: str) -> Any:
    """Render retrieve_memory results as an interactive HTML card list."""
    count = len(memories) if memories else 0
    header = f"""<div class="header">
  <h1>🔍 Search Results</h1>
  <span class="meta">{count} result{'s' if count != 1 else ''} for &ldquo;{_escape(query)}&rdquo;</span>
</div>"""

    if not memories:
        body = header + '<div class="empty">No memories found for this query.</div>'
        return _wrap("Memory Search", body)

    cards = []
    for i, mem in enumerate(memories):
        content = mem.get("content", "")
        score = float(mem.get("similarity_score", mem.get("score", 0)))
        tags = mem.get("tags", mem.get("metadata", {}).get("tags", []))
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        mem_type = mem.get("memory_type", mem.get("type", ""))
        content_hash = mem.get("content_hash", "")
        created = mem.get("created_at", mem.get("timestamp", ""))
        card_id = f"content-{i}"

        score_html = f'<span class="score {_score_class(score)}">{score:.2f}</span>' if score else ""
        type_html = f'<span class="type-badge">{_escape(mem_type)}</span>' if mem_type else ""
        tags_html = (
            '<div class="tags">' +
            "".join(f'<span class="tag">{_escape(t)}</span>' for t in tags) +
            "</div>"
        ) if tags else ""
        ts_html = f'<span class="ts">{_escape(_format_ts(created))}</span>' if created else ""
        hash_html = f'<span class="hash">{_escape(content_hash[:48])}…</span>' if content_hash else ""
        copy_btn = (
            f'<button class="btn" onclick="copyHash(\'{_escape(content_hash)}\', this)">Copy hash</button>'
            if content_hash else ""
        )
        expand_btn = (
            f'<button class="btn" onclick="toggleExpand(\'{card_id}\')" style="margin-left:4px">Expand</button>'
            if len(content) > 200 else ""
        )

        cards.append(f"""<div class="card">
  <div class="card-header">
    {score_html}
    <div class="content truncated" id="{card_id}">{_escape(content)}</div>
  </div>
  {tags_html}
  <div class="card-footer">
    {hash_html}
    {ts_html}
    {type_html}
    {copy_btn}
    {expand_btn}
  </div>
</div>""")

    body = header + "\n".join(cards)
    return _wrap("Memory Search", body)


def render_memory_list(result: dict) -> Any:
    """Render list_memories results as a paginated browsable list."""
    memories = result.get("memories", [])
    total = result.get("total", len(memories))
    page = result.get("page", 1)
    page_size = result.get("page_size", 10)
    total_pages = result.get("total_pages", 1)

    header = f"""<div class="header">
  <h1>📋 Memory Browser</h1>
  <span class="meta">{total} total · page {page}/{total_pages}</span>
</div>"""

    if not memories:
        body = header + '<div class="empty">No memories stored yet.</div>'
        return _wrap("Memory Browser", body)

    cards = []
    for i, mem in enumerate(memories):
        content = mem.get("content", "")
        tags = mem.get("tags", mem.get("metadata", {}).get("tags", []))
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        mem_type = mem.get("memory_type", mem.get("type", ""))
        content_hash = mem.get("content_hash", "")
        created = mem.get("created_at", mem.get("timestamp", ""))
        card_id = f"list-content-{i}"

        type_html = f'<span class="type-badge">{_escape(mem_type)}</span>' if mem_type else ""
        tags_html = (
            '<div class="tags">' +
            "".join(f'<span class="tag">{_escape(t)}</span>' for t in tags) +
            "</div>"
        ) if tags else ""
        ts_html = f'<span class="ts">{_escape(_format_ts(created))}</span>' if created else ""
        hash_html = f'<span class="hash">{_escape(content_hash[:48])}…</span>' if content_hash else ""
        copy_btn = (
            f'<button class="btn" onclick="copyHash(\'{_escape(content_hash)}\', this)">Copy hash</button>'
            if content_hash else ""
        )

        cards.append(f"""<div class="card">
  <div class="content truncated" id="{card_id}">{_escape(content)}</div>
  {tags_html}
  <div class="card-footer">
    {hash_html}
    {ts_html}
    {type_html}
    {copy_btn}
  </div>
</div>""")

    pagination_html = ""
    if total_pages > 1:
        pagination_html = f"""<div class="pagination">
  Page {page} of {total_pages} &nbsp;·&nbsp; {total} total memories
</div>"""

    body = header + "\n".join(cards) + pagination_html
    return _wrap("Memory Browser", body)


def render_health_dashboard(health: dict) -> Any:
    """Render check_database_health results as a stats dashboard."""
    status = health.get("status", "unknown")
    is_healthy = status == "healthy"
    total = health.get("total_memories", health.get("count", "?"))
    backend = health.get("backend", health.get("storage_backend", "?"))
    db_info = health.get("database_info", {})
    ts = health.get("timestamp", "")

    dot_class = "healthy" if is_healthy else "error"
    status_color = "var(--green)" if is_healthy else "var(--red)"

    stats_html = f"""<div class="stats-grid">
  <div class="stat-card">
    <div class="stat-value">{_escape(str(total))}</div>
    <div class="stat-label">Memories</div>
  </div>
  <div class="stat-card">
    <div class="stat-value" style="font-size:14px;color:var(--teal)">{_escape(backend)}</div>
    <div class="stat-label">Backend</div>
  </div>
  <div class="stat-card">
    <div class="stat-value" style="font-size:18px;color:{status_color}">
      <span class="status-dot {dot_class}"></span>{_escape(status)}
    </div>
    <div class="stat-label">Status</div>
  </div>
</div>"""

    # Database details section
    detail_rows = ""
    for key, val in db_info.items():
        if val is not None:
            detail_rows += f"""<div class="info-row">
  <span class="info-key">{_escape(key)}</span>
  <span class="info-val">{_escape(str(val))}</span>
</div>"""

    # Additional health fields
    for key in ("embedding_model", "index_name", "sync_status"):
        val = health.get(key)
        if val:
            detail_rows += f"""<div class="info-row">
  <span class="info-key">{_escape(key)}</span>
  <span class="info-val">{_escape(str(val))}</span>
</div>"""

    if ts:
        detail_rows += f"""<div class="info-row">
  <span class="info-key">checked at</span>
  <span class="info-val">{_escape(_format_ts(ts))}</span>
</div>"""

    details_section = ""
    if detail_rows:
        details_section = f"""<div class="section">
  <h2>Details</h2>
  {detail_rows}
</div>"""

    error_section = ""
    if not is_healthy and health.get("error"):
        error_section = f"""<div class="section" style="border-color:var(--red)">
  <h2 style="color:var(--red)">Error</h2>
  <div style="font-size:11px;color:var(--text)">{_escape(str(health['error']))}</div>
</div>"""

    header = """<div class="header">
  <h1>💾 Memory Service Health</h1>
</div>"""

    body = header + stats_html + details_section + error_section
    return _wrap("Memory Health", body)


def _wrap(title: str, body: str) -> Any:
    """Wrap HTML body in a full page and return as MCP EmbeddedResource."""
    full_html = _base_page(title, body)

    if _MCP_TYPES_AVAILABLE:
        return EmbeddedResource(
            type="resource",
            resource=TextResourceContents(
                uri=f"mcp-memory://ui/{title.lower().replace(' ', '-')}",
                mimeType="text/html",
                text=full_html,
            ),
        )
    # Fallback: return raw HTML string if MCP types not available
    return full_html
