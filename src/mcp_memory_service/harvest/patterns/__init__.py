"""Locale-based pattern loader for harvest extraction.

Loads regex patterns from YAML files in the patterns/ directory.
Supports multiple locales loaded simultaneously for bilingual sessions.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Type alias: dict of memory_type -> list of (compiled_regex, confidence)
PatternDict = Dict[str, List[Tuple[re.Pattern, float]]]

PATTERNS_DIR = Path(__file__).parent


def load_patterns(locales: str = "en") -> PatternDict:
    """Load and merge patterns from one or more locale files.

    Args:
        locales: Comma-separated locale codes (e.g., "en", "en,pt_BR").

    Returns:
        Merged pattern dict ready for use by PatternExtractor.
    """
    merged: PatternDict = {}
    locale_list = [loc.strip() for loc in locales.split(",") if loc.strip()]

    if not locale_list:
        locale_list = ["en"]

    for locale in locale_list:
        filepath = PATTERNS_DIR / f"{locale}.yaml"
        if not filepath.exists():
            logger.warning(f"Locale file not found: {filepath}")
            continue

        patterns = _parse_yaml_patterns(filepath)
        for memory_type, pat_list in patterns.items():
            merged.setdefault(memory_type, []).extend(pat_list)

    if not merged:
        logger.warning("No patterns loaded — falling back to en")
        fallback = PATTERNS_DIR / "en.yaml"
        if fallback.exists():
            merged = _parse_yaml_patterns(fallback)

    return merged


def _parse_yaml_patterns(filepath: Path) -> PatternDict:
    """Parse a YAML pattern file into compiled regex patterns."""
    try:
        import yaml
    except ImportError:
        # Fallback: simple line-based parser for environments without PyYAML
        return _parse_yaml_simple(filepath)

    with open(filepath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if not data or "patterns" not in data:
        return {}

    result: PatternDict = {}
    for memory_type, entries in data["patterns"].items():
        compiled = []
        for entry in entries:
            try:
                pattern = re.compile(entry["pattern"], re.IGNORECASE)
                confidence = float(entry.get("confidence", 0.6))
                compiled.append((pattern, confidence))
            except re.error as e:
                logger.warning(f"Invalid regex in {filepath.name}/{memory_type}: {e}")
        if compiled:
            result[memory_type] = compiled

    return result


def _parse_yaml_simple(filepath: Path) -> PatternDict:
    """Minimal YAML parser for pattern files (no PyYAML dependency)."""
    result: PatternDict = {}
    current_type = None

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.rstrip()
            # Skip comments and metadata
            if not stripped or stripped.startswith("#") or stripped.startswith("language:") or stripped.startswith("version:"):
                continue
            if stripped == "patterns:":
                continue

            # Memory type header (e.g., "  decision:")
            if stripped.endswith(":") and not stripped.strip().startswith("- ") and "pattern" not in stripped:
                current_type = stripped.strip().rstrip(":")
                result.setdefault(current_type, [])
                continue

            # Pattern line
            if "pattern:" in stripped and current_type:
                # Extract pattern value between quotes
                match = re.search(r"pattern:\s*['\"](.+?)['\"]", stripped)
                if match:
                    pat_str = match.group(1)
                    try:
                        compiled = re.compile(pat_str, re.IGNORECASE)
                        result[current_type].append((compiled, 0.6))  # default confidence
                    except re.error:
                        pass  # Skip invalid regex patterns from locale YAML files

            # Confidence line (update last pattern if available)
            if "confidence:" in stripped and current_type:
                match = re.search(r"confidence:\s*([\d.]+)", stripped)
                if match and result.get(current_type):
                    conf = float(match.group(1))
                    pat, _ = result[current_type][-1]
                    result[current_type][-1] = (pat, conf)

    return result
