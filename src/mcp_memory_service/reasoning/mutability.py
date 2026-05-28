"""Fact mutability classification (RFC #1008 §5).

Classifies memories as stable/volatile/ephemeral to inform contradiction handling.
"""

import re

_VOLATILE_PATTERNS = [
    re.compile(r'v(?:ersion)?\s*(?:is\s+)?v?\d+[\d.]+', re.I),
    re.compile(r'\b(?:currently|at the moment|as of)\b', re.I),
    re.compile(r'\b(?:running|listening|deployed|active)\s+(?:on|at)\b', re.I),
    re.compile(r'\bport\s+\d+\b', re.I),
    re.compile(r'\d{4}-\d{2}-\d{2}'),
]

_EPHEMERAL_PATTERNS = [
    re.compile(r'\b(?:this session|current session|working on)\b', re.I),
    re.compile(r'\bbranch\s+(?:feat|fix|hotfix)/', re.I),
    re.compile(r'\b(?:right now|just now|at this moment)\b', re.I),
]

_VALID_MUTABILITIES = {"stable", "volatile", "ephemeral"}


def classify_mutability(content: str) -> str:
    """Classify content mutability: stable, volatile, or ephemeral."""
    for pat in _EPHEMERAL_PATTERNS:
        if pat.search(content):
            return "ephemeral"
    for pat in _VOLATILE_PATTERNS:
        if pat.search(content):
            return "volatile"
    return "stable"


def contradiction_action(mutability_a: str, mutability_b: str) -> str:
    """Decide action when two memories contradict."""
    if mutability_a not in _VALID_MUTABILITIES:
        raise ValueError(f"Invalid mutability: {mutability_a!r}. Must be one of {_VALID_MUTABILITIES}")
    if mutability_b not in _VALID_MUTABILITIES:
        raise ValueError(f"Invalid mutability: {mutability_b!r}. Must be one of {_VALID_MUTABILITIES}")
    if "ephemeral" in (mutability_a, mutability_b):
        return "ignore"
    if "volatile" in (mutability_a, mutability_b):
        return "supersede"
    return "flag"
