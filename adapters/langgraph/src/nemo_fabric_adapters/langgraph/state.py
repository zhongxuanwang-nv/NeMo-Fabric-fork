# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime-to-thread mapping and adapter-owned checkpoint persistence.

Resume is keyed by the Fabric ``runtime_id``, which is stable across ``invoke``
calls on a started runtime and fresh for each one-shot ``run``. The LangGraph
``thread_id`` is derived from it deterministically, so no adapter-owned
correlation file has to be written or repaired. LangGraph owns the transcript.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import nemo_fabric_adapters.common.utils as common_utils

from nemo_fabric_adapters.langgraph.config import AdapterConfigError

HARNESS = "langgraph"


def thread_id_for(runtime_id: str) -> str:
    """Derive a stable LangGraph thread id from a Fabric runtime id."""

    return hashlib.sha256(runtime_id.encode("utf-8")).hexdigest()[:32]


def state_dir(payload: dict[str, Any]) -> Path:
    base = Path(common_utils.base_dir(payload)).resolve()
    configured = common_utils.settings_payload(payload).get("state_dir")
    if configured:
        path = Path(str(configured))
        return path if path.is_absolute() else base / path
    artifacts = common_utils.runtime_context(payload).get("artifacts") or {}
    root = artifacts.get("root") or os.environ.get("FABRIC_ARTIFACTS")
    if root:
        return Path(str(root)).resolve() / ".fabric" / HARNESS
    return base / "artifacts" / HARNESS / ".fabric"


def checkpoint_path(payload: dict[str, Any], runtime_id: str) -> Path:
    key = hashlib.sha256(runtime_id.encode("utf-8")).hexdigest()
    return state_dir(payload) / "runtimes" / f"{key}.sqlite"


async def open_checkpointer(path: Path) -> Any:
    """Open a persistent async LangGraph checkpointer so resume works across processes.

    The graph is driven with async entry points, so the checkpointer must be async:
    the synchronous ``SqliteSaver`` raises ``NotImplementedError`` from its async
    methods.
    """

    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    except ImportError as exc:
        raise AdapterConfigError(
            "state 'adapter' requires a checkpoint backend; install the adapter's 'state' extra "
            "(pip install 'nemo-fabric-adapters-langgraph[state]')."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    saver_cm = AsyncSqliteSaver.from_conn_string(str(path))
    saver = await saver_cm.__aenter__()
    # Keep the context manager alive for the duration of the invocation.
    saver._fabric_cm = saver_cm  # type: ignore[attr-defined]
    return saver


async def close_checkpointer(checkpointer: Any) -> None:
    saver_cm = getattr(checkpointer, "_fabric_cm", None)
    if saver_cm is not None:
        await saver_cm.__aexit__(None, None, None)


async def has_checkpoint(checkpointer: Any, thread_id: str) -> bool:
    """Return whether this thread already has committed state.

    Asking the checkpointer is the source of truth for resume, so the adapter does
    not need to keep a parallel record that could disagree with it.
    """

    try:
        existing = await checkpointer.aget_tuple({"configurable": {"thread_id": thread_id}})
    except Exception:  # a missing or unreadable checkpoint is simply not a resume
        return False
    return existing is not None
