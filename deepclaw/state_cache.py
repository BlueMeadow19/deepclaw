"""Helpers for invalidating cached thread-local state without clearing conversation history."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.checkpoint.base import copy_checkpoint, create_checkpoint

logger = logging.getLogger(__name__)

CACHED_STATE_KEYS = frozenset({"memory_contents", "skills_metadata"})


async def invalidate_thread_state(
    checkpointer: Any,
    thread_id: str,
    *,
    keys: set[str] | frozenset[str] | tuple[str, ...] = CACHED_STATE_KEYS,
) -> bool:
    """Drop selected cached state keys for a thread while preserving message history.

    Returns True when a new checkpoint was written, False when there was nothing to do.
    """
    if checkpointer is None or not thread_id:
        return False

    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    checkpoint_tuple = await checkpointer.aget_tuple(config)
    if checkpoint_tuple is None:
        return False

    checkpoint = copy_checkpoint(checkpoint_tuple.checkpoint)
    channel_values = checkpoint.get("channel_values", {})
    removed = False
    for key in keys:
        if key in channel_values:
            channel_values.pop(key, None)
            removed = True

    if not removed:
        return False

    metadata = dict(checkpoint_tuple.metadata or {})
    step = int(metadata.get("step", -1)) + 1
    metadata["source"] = "invalidate_state"
    new_checkpoint = create_checkpoint(checkpoint, None, step)
    await checkpointer.aput(checkpoint_tuple.config, new_checkpoint, metadata, {})
    logger.info(
        "Invalidated cached state for thread %s: %s",
        thread_id,
        ", ".join(sorted(keys)),
    )
    return True
