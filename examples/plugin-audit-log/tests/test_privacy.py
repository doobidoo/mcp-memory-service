import asyncio
import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class AuditLogPrivacyTest(unittest.TestCase):
    def reload_plugin(self, tmp_path: Path, *, mode="safe", key=None):
        env = {
            "MCP_PLUGIN_AUDIT_LOG_PATH": str(tmp_path / "audit.jsonl"),
            "MCP_PLUGIN_AUDIT_LOG_PRIVACY_MODE": mode,
        }
        if key is not None:
            env["MCP_PLUGIN_AUDIT_LOG_HMAC_KEY"] = key

        patcher = mock.patch.dict(os.environ, env, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        if key is None:
            os.environ.pop("MCP_PLUGIN_AUDIT_LOG_HMAC_KEY", None)

        import mcp_memory_plugin_audit_log as plugin

        return importlib.reload(plugin)

    @staticmethod
    def read_events(path: Path):
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_safe_mode_omits_raw_query_tags_and_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            plugin = self.reload_plugin(tmp_path, key="test-key")

            asyncio.run(plugin.on_store({
                "content_hash": "raw-content-hash",
                "memory_type": "decision",
                "tags": ["secret-project", "customer-x"],
                "content": "the raw secret memory body",
            }))
            asyncio.run(plugin.on_retrieve("deploy secret-project with token sk_test", [{"id": 1}]))

            log_path = tmp_path / "audit.jsonl"
            serialized = log_path.read_text()
            events = self.read_events(log_path)

            self.assertNotIn("secret-project", serialized)
            self.assertNotIn("customer-x", serialized)
            self.assertNotIn("the raw secret memory body", serialized)
            self.assertNotIn("deploy secret-project", serialized)
            self.assertNotIn("sk_test", serialized)
            self.assertTrue(all(event["privacy_mode"] == "safe" for event in events))
            self.assertTrue(events[0]["memory_hash_hmac"].startswith("hmac-sha256:"))
            self.assertTrue(events[1]["query_hash_hmac"].startswith("hmac-sha256:"))
            self.assertEqual(events[1]["query_length"], len("deploy secret-project with token sk_test"))
            self.assertEqual(events[1]["result_count"], 1)

    def test_safe_mode_without_hmac_omits_stable_identifier_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            plugin = self.reload_plugin(tmp_path)

            asyncio.run(plugin.on_delete("predictable-content-hash"))
            asyncio.run(plugin.on_retrieve("find private customer note", []))

            log_path = tmp_path / "audit.jsonl"
            serialized = log_path.read_text()
            events = self.read_events(log_path)

            self.assertNotIn("predictable-content-hash", serialized)
            self.assertNotIn("private customer", serialized)
            self.assertNotIn("memory_hash_hmac", events[0])
            self.assertNotIn("query_hash_hmac", events[1])
            self.assertEqual(events[0]["hash_algorithm"], "none")
            self.assertEqual(
                events[1]["identifier_hashes_omitted_reason"],
                "MCP_PLUGIN_AUDIT_LOG_HMAC_KEY not set",
            )

    def test_raw_mode_preserves_original_debug_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            plugin = self.reload_plugin(tmp_path, mode="raw")

            asyncio.run(plugin.on_retrieve("debug query", []))

            events = self.read_events(tmp_path / "audit.jsonl")
            self.assertEqual(events[0]["privacy_mode"], "raw")
            self.assertEqual(events[0]["query"], "debug query")


if __name__ == "__main__":
    unittest.main()
