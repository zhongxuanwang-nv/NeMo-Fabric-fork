# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The public graph-authoring contract for Fabric-managed LangGraph agents.

A ``factory`` entry point receives a :class:`LangGraphAdapterContext` and returns a
compiled graph. The context is the only surface a custom agent needs; it must not
import Fabric internals or this package's other modules.

The context intentionally exposes normalized resources and JSON-shaped settings
rather than the raw Fabric invocation, so an agent keeps working when Fabric's
internal payload shape changes.
"""

from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import nemo_fabric_adapters.common.utils as common_utils

from nemo_fabric_adapters.langgraph import models as model_support
from nemo_fabric_adapters.langgraph import tools as tool_support
from nemo_fabric_adapters.langgraph.config import LangGraphSettings

_CURRENT: ContextVar[LangGraphAdapterContext | None] = ContextVar(
    "nemo_fabric_langgraph_context", default=None
)


class LangGraphAdapterContext:
    """Normalized Fabric resources handed to a LangGraph graph factory."""

    def __init__(
        self,
        *,
        payload: dict[str, Any],
        settings: LangGraphSettings,
        runnable_config: dict[str, Any],
        checkpointer: Any | None,
        callbacks: list[Any],
        mcp_tools: list[Any],
    ) -> None:
        self._payload = payload
        self._settings = settings
        self._runnable_config = runnable_config
        self._checkpointer = checkpointer
        self._callbacks = callbacks
        self._mcp_tools = mcp_tools
        self._blocked = set(common_utils.blocked_tools(payload))
        self._models: dict[str, Any] = {}
        self._bound_models: list[dict[str, Any]] = []

    # --- configuration ----------------------------------------------------

    @property
    def agent_settings(self) -> Mapping[str, Any]:
        """Read-only, JSON-shaped ``harness.settings.langgraph.agent`` values.

        Fabric does not interpret these fields beyond checking that they are JSON
        compatible; their meaning belongs to the agent.
        """

        return dict(self._settings.agent)

    @property
    def agent_name(self) -> str:
        return common_utils.agent_name(self._payload)

    @property
    def blocked_tool_names(self) -> frozenset[str]:
        """Tool names Fabric ``tools.blocked`` denies for this run."""

        return frozenset(self._blocked)

    # --- models -----------------------------------------------------------

    def chat_model(self, alias: str = "default") -> Any:
        """Return a LangChain chat model built from the Fabric ``models`` alias.

        The same alias returns the same instance for the lifetime of one invocation.
        """

        if alias not in self._models:
            model, descriptor = model_support.build_chat_model(self._payload, alias)
            self._models[alias] = model
            self._bound_models.append(descriptor)
        return self._models[alias]

    # --- tools ------------------------------------------------------------

    def mcp_tools(self) -> list[Any]:
        """Return LangChain tools from Fabric MCP servers, with tool policy applied."""

        return self.guard_tools(self._mcp_tools)

    def guard_tools(self, tools: list[Any]) -> list[Any]:
        """Apply Fabric ``tools.blocked`` to ``tools``.

        Fabric can only enforce policy on tools passed through this helper. A graph
        that wires unwrapped tools owns that permission boundary.
        """

        return tool_support.guard_tools(list(tools), self._blocked)

    # --- execution --------------------------------------------------------

    @property
    def checkpointer(self) -> Any | None:
        """The adapter-owned checkpointer for ``state: adapter``, otherwise ``None``.

        A factory that wants durable multi-turn state must compile its graph with
        this checkpointer.
        """

        return self._checkpointer

    @property
    def runnable_config(self) -> dict[str, Any]:
        """The LangGraph run configuration the adapter will use.

        Contains the thread id and any supported callbacks. A factory that builds its
        own configuration must merge this one rather than replace it.
        """

        return dict(self._runnable_config)

    @property
    def callbacks(self) -> list[Any]:
        """Telemetry callbacks to merge with the graph's own callbacks.

        Empty until a LangGraph telemetry provider is validated end to end and
        declared in the adapter descriptor.
        """

        return list(self._callbacks)

    @property
    def thread_id(self) -> str | None:
        return (self._runnable_config.get("configurable") or {}).get("thread_id")

    # --- environment ------------------------------------------------------

    @property
    def workspace(self) -> Path | None:
        """The Fabric workspace directory, when the environment provides one.

        The adapter does not sandbox filesystem access; a graph that needs
        confinement must enforce it.
        """

        return self._resolved_path(common_utils.environment_payload(self._payload).get("workspace"))

    @property
    def artifact_dir(self) -> Path | None:
        """The Fabric artifact root, when the environment provides one."""

        artifacts = common_utils.runtime_context(self._payload).get("artifacts") or {}
        return self._resolved_path(artifacts.get("root"))

    def _resolved_path(self, value: Any) -> Path | None:
        if not value:
            return None
        path = Path(str(value))
        return path if path.is_absolute() else Path(common_utils.base_dir(self._payload)) / path

    # --- adapter-internal -------------------------------------------------

    def bound_models(self) -> list[dict[str, Any]]:
        """JSON-safe descriptors of models the factory requested, for result metadata."""

        return list(self._bound_models)


def current_context() -> LangGraphAdapterContext:
    """Return the context for the graph currently being constructed.

    This exists for agents that build their graph at module import time and cannot
    accept a factory argument, which is the shape the NeMo Agent Toolkit's builder
    accessor supports. It is only valid while the adapter is loading the entry
    point; prefer a ``factory`` entry point, which makes the dependency explicit.
    """

    context = _CURRENT.get()
    if context is None:
        raise RuntimeError(
            "no LangGraph adapter context is active. current_context() is only available while the "
            "Fabric LangGraph adapter is constructing the graph; use a 'factory' entry point that "
            "receives the context as an argument."
        )
    return context


@contextmanager
def active_context(context: LangGraphAdapterContext) -> Iterator[LangGraphAdapterContext]:
    """Publish ``context`` to :func:`current_context` for the duration of graph construction."""

    token = _CURRENT.set(context)
    try:
        yield context
    finally:
        _CURRENT.reset(token)
