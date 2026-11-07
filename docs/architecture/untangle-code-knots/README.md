# Untangle code dependencies

These diagrams document the original Antigravity analysis and the intended refactoring, not a live inventory or test report.

- [Interactive dependency map](wiring_diagram.html): original analysis before the five refactorings.
- [Before/after comparison](comparison_view.html): historical baseline versus the intended public interfaces and managed task lifecycle. Open the HTML locally in a browser.

The historical baseline predates current `main`. Upstream has since implemented guarded SQLite untagged deletion, including confirmation checks, rollback, citation retraction and sync events. This branch preserves that implementation and exposes it through the storage interface with Hybrid delegation. Namespace registration and unknown memory-type hydration from current `main` are also preserved.

The standalone ontology import retains a sanitizer fallback for tests that load the module without the package. Counts in the original diagrams describe the original analysis, not repository-wide guarantees. Test results are recorded in the pull request.
