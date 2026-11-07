"""Numeric workflow bounds shared by config and the config-independent network protocol."""

TUNING_BOUNDS: dict[str, tuple[int, int]] = {
    "max_subtasks": (1, 12),
    "max_parts": (2, 6),
    "context_tokens": (1000, 16000),
    "max_attempts": (2, 8),
    "max_transient": (2, 10),
}
"""The least and the most each number of `Tuning` may be."""

PAGE_TUNING_BOUNDS: dict[str, tuple[int, int]] = {"workers": (1, 8)}
"""Page concurrency, set only by the person; never lent to network jobs."""
