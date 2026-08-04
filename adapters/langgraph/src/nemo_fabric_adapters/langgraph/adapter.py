#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic LangGraph adapter for Fabric.

Loads a customer-owned LangGraph graph by an explicit Python reference, runs it
through the Fabric adapter lifecycle, and returns a normalized Fabric result. One
adapter id serves every custom LangGraph agent: a graph that already compiles
itself runs unchanged, and a graph that wants Fabric-managed models, MCP tools,
tool policy, or durable state opts in through a factory entry point that receives
:class:`~nemo_fabric_adapters.langgraph.context.LangGraphAdapterContext`.
"""

from __future__ import annotations

import json
from typing import Any

import nemo_fabric_adapters.common.utils as common_utils

from nemo_fabric_adapters.langgraph import loading
from nemo_fabric_adapters.langgraph import normalize
from nemo_fabric_adapters.langgraph import state as state_support
from nemo_fabric_adapters.langgraph import tools as tool_support
from nemo_fabric_adapters.langgraph.config import AdapterConfigError
from nemo_fabric_adapters.langgraph.config import LangGraphSettings
from nemo_fabric_adapters.langgraph.config import parse_settings
from nemo_fabric_adapters.langgraph.context import LangGraphAdapterContext
from nemo_fabric_adapters.langgraph.context import active_context

HARNESS = "langgraph"
# Binding kinds that receive the adapter context and can therefore consume
# Fabric-managed models, MCP tools, tool policy, and adapter-owned state.
CONTEXT_AWARE_KINDS = frozenset({"factory"})


def main() -> None:
    """Subprocess entrypoint used by the ``python -m`` process path."""

    output = run(common_utils.load_payload())
    print(json.dumps(output, sort_keys=True))
    if output.get("failed"):
        raise SystemExit(2)


def run(payload: dict[str, Any]) -> dict[str, Any]:
    """Fabric adapter entrypoint used by the ``python -m`` and native SDK paths."""

    import asyncio

    return asyncio.run(run_langgraph(payload))


def preflight_check() -> None:
    """Validate invocation-time prerequisites and fail fast with clear errors.

    Fabric core has no adapter-doctor hook, so this runs at invocation time.
    """

    import importlib.util

    if importlib.util.find_spec("langgraph") is None:
        raise AdapterConfigError(
            "the 'langgraph' package is required for the LangGraph adapter; install it with "
            "pip install nemo-fabric-adapters-langgraph."
        )


def validate_capabilities(payload: dict[str, Any], settings: LangGraphSettings) -> None:
    """Reject configurations the selected binding kind cannot honor.

    Fabric must not report a capability it cannot enforce. A precompiled graph owns
    its own tools and checkpointer, so Fabric can neither deny its tool calls nor
    retrofit persistence; asking for either is a configuration error rather than a
    silently ignored policy.
    """

    kind = settings.entrypoint.kind
    if kind in CONTEXT_AWARE_KINDS:
        return

    if common_utils.blocked_tools(payload):
        raise AdapterConfigError(
            f"tools.blocked cannot be enforced for a '{kind}' entry point, because the graph builds its "
            "own tool surface. Use a 'factory' entry point and build tools from "
            "LangGraphAdapterContext.mcp_tools() or guard_tools(), or remove tools.blocked."
        )
    if (common_utils.capability_plan(payload).get("native") or {}).get("mcp_servers"):
        raise AdapterConfigError(
            f"Fabric MCP servers cannot be supplied to a '{kind}' entry point, because the graph builds "
            "its own tool surface. Use a 'factory' entry point and call "
            "LangGraphAdapterContext.mcp_tools(), or remove the mcp configuration."
        )
    if settings.state == "adapter":
        raise AdapterConfigError(
            f"state 'adapter' cannot be applied to a '{kind}' entry point, because a compiled graph "
            "cannot be given a new checkpointer. Use a 'factory' entry point and compile with "
            "LangGraphAdapterContext.checkpointer, or use state 'graph' to let the graph own persistence."
        )


async def run_langgraph(payload: dict[str, Any]) -> dict[str, Any]:
    request = common_utils.request_payload(payload)
    runtime_id = common_utils.runtime_context(payload).get("runtime_id")

    settings: LangGraphSettings | None = None
    thread_id: str | None = None
    resumed = False
    final_state: Any = None
    response: Any = None
    events: list[dict[str, Any]] = []
    turn_messages: list[dict[str, Any]] = []
    bound_models: list[dict[str, Any]] = []
    error: str | None = None
    checkpointer: Any = None

    try:
        # Configuration, graph loading, and graph construction all run inside the
        # guarded scope so any misconfiguration returns a normalized failure result
        # rather than a raw traceback.
        preflight_check()
        settings = parse_settings(common_utils.settings_payload(payload))
        validate_capabilities(payload, settings)

        if settings.state != "none":
            if not runtime_id:
                raise AdapterConfigError(
                    f"state '{settings.state}' requires runtime_context.runtime_id, which this "
                    "invocation did not provide."
                )
            thread_id = state_support.thread_id_for(str(runtime_id))

        if settings.state == "adapter":
            checkpointer = await state_support.open_checkpointer(
                state_support.checkpoint_path(payload, str(runtime_id))
            )
            resumed = await state_support.has_checkpoint(checkpointer, str(thread_id))

        runnable_config = build_runnable_config(settings, thread_id)
        context = LangGraphAdapterContext(
            payload=payload,
            settings=settings,
            runnable_config=runnable_config,
            checkpointer=checkpointer,
            callbacks=[],
            mcp_tools=await tool_support.load_mcp_tools(payload)
            if settings.entrypoint.kind in CONTEXT_AWARE_KINDS
            else [],
        )

        with active_context(context):
            target = loading.load_entrypoint(settings.entrypoint, common_utils.base_dir(payload))
            factory_argument = context if settings.entrypoint.kind == "factory" else runnable_config
            graph = await loading.build_graph(settings.entrypoint, target, factory_argument)

        bound_models = context.bound_models()
        graph_input = normalize.graph_input(settings.input, request.get("input"))
        final_state, events, turn_messages = await invoke_graph(
            graph, graph_input, runnable_config, settings
        )
        response = normalize.project_output(settings.output, final_state)
    except Exception as exc:  # normalized adapter failure
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if checkpointer is not None:
            await state_support.close_checkpointer(checkpointer)

    return normalize_output(
        settings=settings,
        response=response,
        final_state=final_state,
        events=events,
        turn_messages=turn_messages,
        bound_models=bound_models,
        runtime_id=runtime_id,
        thread_id=thread_id,
        resumed=resumed,
        error=error,
    )


def build_runnable_config(settings: LangGraphSettings, thread_id: str | None) -> dict[str, Any]:
    """Build the LangGraph run configuration for this invocation."""

    config: dict[str, Any] = {}
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}
    if settings.recursion_limit is not None:
        config["recursion_limit"] = settings.recursion_limit
    return config


async def invoke_graph(
    graph: Any,
    graph_input: Any,
    runnable_config: dict[str, Any],
    settings: LangGraphSettings,
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the graph and return its final state, bounded events, and this turn's messages.

    The graph is streamed when it supports ``astream`` so token usage can be
    attributed to this turn: on a resumed thread the final state replays earlier
    turns, which must not be counted again. Events are always bounded, and are only
    reported when ``events.mode`` is ``summary``.
    """

    config = runnable_config or None
    if not hasattr(graph, "astream"):
        final = await graph.ainvoke(graph_input, config) if config else await graph.ainvoke(graph_input)
        return final, [], _state_messages(settings, final)

    events: list[dict[str, Any]] = []
    turn_messages: list[Any] = []
    final: Any = None
    stream = (
        graph.astream(graph_input, config, stream_mode=["updates", "values"])
        if config
        else graph.astream(graph_input, stream_mode=["updates", "values"])
    )
    async for mode, chunk in stream:
        if mode == "updates" and isinstance(chunk, dict):
            if len(events) < settings.events.limit:
                events.append({"nodes": [str(node) for node in chunk]})
            for node_output in chunk.values():
                if isinstance(node_output, dict):
                    new = node_output.get(settings.output.messages_key)
                    if isinstance(new, list):
                        turn_messages.extend(new)
                    elif new is not None:
                        turn_messages.append(new)
        elif mode == "values":
            final = chunk
    return final, events, _dedup_messages(normalize.messages_to_dicts(turn_messages))


def _state_messages(settings: LangGraphSettings, final_state: Any) -> list[dict[str, Any]]:
    if not isinstance(final_state, dict):
        return []
    return normalize.messages_to_dicts(final_state.get(settings.output.messages_key))


def _dedup_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop duplicate messages by id, preserving first-seen order."""

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for message in messages:
        message_id = message.get("id")
        if isinstance(message_id, str):
            if message_id in seen:
                continue
            seen.add(message_id)
        unique.append(message)
    return unique


def normalize_output(
    *,
    settings: LangGraphSettings | None,
    response: Any,
    final_state: Any,
    events: list[dict[str, Any]],
    turn_messages: list[dict[str, Any]],
    bound_models: list[dict[str, Any]],
    runtime_id: str | None,
    thread_id: str | None,
    resumed: bool,
    error: str | None,
) -> dict[str, Any]:
    """Build the normalized Fabric result for this invocation."""

    messages = _state_messages(settings, final_state) if settings is not None else []
    # Report one event list and a count derived from it, so the count never
    # describes events that were suppressed or truncated.
    reported_events = events if settings and settings.events.mode == "summary" else []
    output: dict[str, Any] = {
        "harness": HARNESS,
        "adapter": "python",
        "mode": settings.entrypoint.kind if settings else None,
        "entrypoint": settings.entrypoint.target if settings else None,
        "response": response,
        "output_mode": settings.output.mode if settings else None,
        "messages": messages,
        "message_count": len(messages),
        "events": reported_events,
        "event_count": len(reported_events),
        "usage": normalize.aggregate_usage(turn_messages),
        "models": bound_models,
        "runtime_id": runtime_id,
        "thread_id": thread_id,
        "state_mode": settings.state if settings else None,
        "resumed": resumed,
        "completed": error is None,
        "failed": error is not None,
        "error": error,
    }
    return output


if __name__ == "__main__":
    main()
