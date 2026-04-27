"""Headroom runtime stats tool for DeepClaw.

Exposes live prompt-compression savings observed by the in-process Headroom
wrapper. Use this when the agent wants to inspect whether Headroom is enabled,
how many requests have been optimized since startup, and recent savings details.
"""

from __future__ import annotations

from typing import Any

from deepclaw.config import load_config
from deepclaw.headroom import get_headroom_runtime_stats


def available() -> bool:
    """Always available — reads local config and in-process runtime state."""
    return True


def headroom_stats(recent_limit: int = 5) -> dict[str, Any]:
    """Return live Headroom compression stats for the running DeepClaw process.

    Args:
        recent_limit: Number of recent optimization events to include (default 5).

    Returns:
        Dictionary containing:
        - config_enabled: whether Headroom is enabled in config
        - runtime active flag and wrapper metadata
        - cumulative savings summary since process start
        - recent optimization events with token savings details
    """
    config = load_config()
    runtime = get_headroom_runtime_stats(recent_limit=recent_limit)
    return {
        "config_enabled": bool(config.headroom.enabled),
        **runtime,
    }


def get_tools() -> list:
    """Return the tool callables for this plugin."""
    return [headroom_stats]
