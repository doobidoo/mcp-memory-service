"""The four updated facts from the #1938 reproduction.

Cosines are the ones measured with all-MiniLM-L6-v2 against v12.7.0. They are
fixtures for the supersede band, not a tuning set: every one sits between the
default dedup threshold (0.85) and the default near-identical cutoff (0.97).
"""

# (stored fact, later update, cosine similarity of the update to the stored fact)
XERJ_UPDATE_PAIRS = (
    (
        "The Lark API listens on port 8443 in production.",
        "Change: the Lark API now listens on port 9443 in production.",
        0.928,
    ),
    (
        "The auth service issues JWTs that expire after 15 minutes.",
        "Auth JWTs now expire after 60 minutes.",
        0.927,
    ),
    (
        "Dana's desk is on the 7th floor.",
        "Dana's desk moved to the 3rd floor.",
        0.857,
    ),
    (
        "The Lark API rate limit is 600 requests per minute per key.",
        "The Lark API rate limit was raised to 1200 requests per minute per key.",
        0.933,
    ),
)
