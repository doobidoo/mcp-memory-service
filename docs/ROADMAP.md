# Development Roadmap

**The official roadmap has moved to the Wiki for easier maintenance and community collaboration.**

📖 **[View Development Roadmap on Wiki](https://github.com/doobidoo/mcp-memory-service/wiki/13-Development-Roadmap)**

The Wiki version includes:
- ✅ Completed milestones (v8.0–v10.47; v11.x is summarized below)
- 🎯 Current focus areas
- 🚀 Future enhancements
- 🌟 Long-term aspirations (2027+)
- 📊 Success metrics and KPIs
- 🤝 Community contribution opportunities

## Recent Milestone Summary

Since the wiki roadmap was last reviewed (v10.47.2, May 2026), the project has
shipped **v11.0–v11.14** with significant changes:

| Version | Date | Highlights |
|---------|------|------------|
| **v11.0.0** | 2026-06-13 | Major version bump |
| **v11.1.0** | 2026-06-18 | `memory_explore` and `memory_detail` entity tools |
| **v11.4.0** | 2026-07-04 | Memory merge action; pluggable domain NER extractors |
| **v11.5.x** | 2026-07-10–24 | Temporal decay, belief derivation, consolidation fixes, Docker tokenizer fix |
| **v11.6.x** | 2026-08-02–03 | Locale-aware NER/NLI (YAML plugins), migration graph-safety fix |
| **v11.7.0** | 2026-08-05 | Security release — TLS verification bypass gates (claude-hooks, opencode plugin, HTTP bridge) |
| **v11.8.x** | 2026-08-09+ | Knowledge-graph entity extraction fix; token-efficient retrieval docs; transport/HTTPS/API-key-logging security fixes (v11.8.3) |
| **v11.9.0** | 2026-08-27 | Security — transformers 5.x closes two high-severity advisories; ONNX quality-ranker fixes; `ml`/`nli` extras tested in CI |
| **v11.10.0** | 2026-08-28 | Consolidation correctness — every time horizon honors its window; configured clustering algorithm actually runs (fails loudly without scikit-learn) |
| **v11.11.0** | 2026-09-05 | Security — three critical advisories (local-only tools on all transports, SSE auth, DCR token scope); development moves back to GitHub |
| **v11.12.0** | 2026-09-14 | Retrieval filters applied before KNN; `match_all` is true AND; consolidation dedup/forgetting/health fixes; ONNX honors `MCP_EMBEDDING_MODEL`; scheduled session harvest |
| **v11.13.0** | 2026-09-19 | Harvest provenance tags and `verify_session_coverage`; Milvus / Cloudflare / hybrid-sync retrieval fixes |
| **v11.14.0** | 2026-09-25 | `agent_id` author identity and search filter; `/mcp/health` no longer discloses storage stats (GHSA-7w86-2vmv-fqwm); decay uses access timestamps |

For full details, see [CHANGELOG.md](../CHANGELOG.md).

## Why the Wiki?

The Wiki provides several advantages for roadmap documentation:
- ✅ **Easier Updates**: No PR required for roadmap changes
- ✅ **Better Navigation**: Integrated with other wiki guides
- ✅ **Community Collaboration**: Lower barrier for community input
- ✅ **Rich Formatting**: Enhanced markdown features
- ✅ **Cleaner Repository**: Reduces noise in commit history

## For Active Development Tracking

The roadmap on the Wiki tracks strategic direction. For day-to-day development:

- **[Open Issues](https://github.com/doobidoo/mcp-memory-service/issues)** — Bug reports and feature requests
- **[Pull Requests](https://github.com/doobidoo/mcp-memory-service/pulls)** — Active code changes
- **[CHANGELOG.md](../CHANGELOG.md)** — Release history and completed features

---

**Maintainer**: @doobidoo
**Last Updated**: September 2026
