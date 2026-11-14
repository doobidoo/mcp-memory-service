"""Static checks on site/demo/index.html: the agent prompt keeps its safety rules, and the
hooks other code relies on stay in place. The state changes of the status island need a
browser and are not covered here."""
import html
import re
import unittest
from pathlib import Path

PAGE = (Path(__file__).resolve().parents[2] / "site" / "demo" / "index.html").read_text(encoding="utf-8")


def element(tag, element_id):
    match = re.search(rf'<{tag}\b[^>]*\bid="{element_id}"[^>]*>(.*?)</{tag}>', PAGE, re.S)
    assert match, f"#{element_id} not found"
    return match


class AgentPromptTest(unittest.TestCase):
    def setUp(self):
        body = element("pre", "agent-prompt").group(1)
        self.assertNotIn("<", body, "the prompt must be plain text: it is copied as textContent")
        self.prompt = html.unescape(body)

    def test_agent_asks_before_it_acts(self):
        self.assertIn("wait for my OK before you install or register anything", self.prompt)

    def test_trial_stays_in_its_own_folder_and_server_name(self):
        self.assertIn("new folder named mcp-memory-trial", self.prompt)
        self.assertIn('leave an existing MCP server named "memory" alone', self.prompt)
        self.assertIn("claude mcp add --scope project memory-trial ", self.prompt)

    def test_both_commands_pin_the_database_path(self):
        # MCP_MEMORY_SQLITE_PATH from the environment wins over MCP_MEMORY_BASE_DIR, so a
        # command without it would run against the visitor's existing database (#1985).
        commands = [line for line in self.prompt.splitlines() if "TRIAL/.venv/bin/memory" in line]
        self.assertEqual(len(commands), 2, commands)
        for command in commands:
            self.assertIn("MCP_MEMORY_SQLITE_PATH=TRIAL/data/sqlite_vec.db", command)
            self.assertIn("MCP_MEMORY_STORAGE_BACKEND=sqlite_vec", command)

    def test_fits_the_documented_deep_link_limit(self):
        # Claude Desktop truncates q at roughly 14,000 characters.
        self.assertLess(len(self.prompt), 10_000)


class EntryPointTest(unittest.TestCase):
    def test_links_carry_no_prompt_in_the_markup(self):
        # The Claude Code link is built on click from the visible text; ChatGPT gets none.
        for element_id, href in (("open-claude", "https://claude.com/download"), ("open-chatgpt", "https://chatgpt.com/")):
            tag = re.search(rf'<a\b[^>]*\bid="{element_id}"[^>]*>', PAGE).group(0)
            self.assertIn(f'href="{href}"', tag)

    def test_external_tab_is_announced_and_isolated(self):
        tag = re.search(r'<a\b[^>]*\bid="open-chatgpt"[^>]*>', PAGE).group(0)
        self.assertIn('target="_blank"', tag)
        self.assertIn('rel="noopener"', tag)
        self.assertIn("new tab", element("a", "open-chatgpt").group(1))

    def test_every_entry_reads_the_visible_prompt(self):
        self.assertEqual(PAGE.count("getElementById('agent-prompt').textContent"), 1)
        self.assertIn("'claude://code/new?q=' + encodeURIComponent(promptText())", PAGE)


class StatusIslandTest(unittest.TestCase):
    def test_hooks_used_by_the_thumbnail_script(self):
        # scripts/maintenance/site_thumbnails.mjs waits for #status-text to start with "Ready".
        element("span", "status-text")
        self.assertIn("setStatus('ready', 'Ready ", PAGE)

    def test_one_live_region_and_a_real_toggle(self):
        island = PAGE[PAGE.index('id="status"'):PAGE.index('id="memInput"')]
        self.assertEqual(island.count('role="status"'), 1)
        self.assertRegex(island, r'<button[^>]*id="island-toggle"[^>]*aria-expanded="false"[^>]*aria-controls="island-detail"')

    def test_ready_is_announced_once_everything_is_running(self):
        ready = PAGE.index("setStatus('ready', 'Ready ")
        for step in ("await pipeline(", "sqlite3InitModule(cfg)", "await store(s)"):
            self.assertLess(PAGE.index(step), ready, step)


if __name__ == "__main__":
    unittest.main()
