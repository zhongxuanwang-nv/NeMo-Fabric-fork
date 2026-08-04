# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declarative request-to-state input mapping and state-to-result projection."""

from __future__ import annotations

import json
from typing import Any

from nemo_fabric_adapters.langgraph.config import AdapterConfigError
from nemo_fabric_adapters.langgraph.config import InputSettings
from nemo_fabric_adapters.langgraph.config import OutputSettings


def graph_input(settings: InputSettings, request_input: Any) -> Any:
    """Convert a Fabric request into graph input using the configured mapping."""

    if settings.mode == "passthrough":
        return request_input
    text = _require_text(settings.mode, request_input)
    if settings.mode == "messages":
        return {settings.messages_key: [{"role": "user", "content": text}]}
    return {settings.field: text}


def _require_text(mode: str, request_input: Any) -> str:
    # Coercing structured input to text would silently change what the graph sees,
    # so a mismatch is a configuration error instead.
    if not isinstance(request_input, str):
        raise AdapterConfigError(
            f"harness.settings.langgraph.input.mode '{mode}' requires a text request, but the request "
            f"input is {type(request_input).__name__}. Use input.mode 'passthrough' to forward "
            "structured input unchanged."
        )
    return request_input


def project_output(settings: OutputSettings, final_state: Any) -> Any:
    """Select the primary normalized response from the graph's final state."""

    if settings.mode == "state":
        return json_safe("the graph's final state", final_state)
    if settings.mode == "field":
        state = _require_mapping(settings.mode, final_state)
        if settings.field not in state:
            raise AdapterConfigError(
                f"harness.settings.langgraph.output.field '{settings.field}' is not present in the "
                f"graph's final state; available fields are {sorted(str(key) for key in state)}."
            )
        return json_safe(f"output field '{settings.field}'", state[settings.field])

    state = _require_mapping(settings.mode, final_state)
    raw = state.get(settings.messages_key)
    if raw is None:
        raise AdapterConfigError(
            f"harness.settings.langgraph.output.mode 'last_message' expects a "
            f"'{settings.messages_key}' key in the graph's final state; available fields are "
            f"{sorted(str(key) for key in state)}. Use output.mode 'field' or 'state' for a graph that "
            "does not keep a message list."
        )
    messages = messages_to_dicts(raw)
    return final_response(messages)


def _require_mapping(mode: str, final_state: Any) -> dict[str, Any]:
    if not isinstance(final_state, dict):
        raise AdapterConfigError(
            f"harness.settings.langgraph.output.mode '{mode}' expects the graph to return a state "
            f"mapping, but it returned {type(final_state).__name__}."
        )
    return final_state


def json_safe(label: str, value: Any) -> Any:
    """Return ``value`` if it is JSON-serializable, otherwise fail with a clear error.

    The adapter cannot invent a serialization for arbitrary LangGraph state, so an
    unserializable projection becomes a normalized failure rather than a corrupted
    result.
    """

    coerced = _coerce(value)
    try:
        json.dumps(coerced)
    except (TypeError, ValueError) as exc:
        raise AdapterConfigError(
            f"{label} is not JSON-serializable ({exc}). Project a JSON-safe field with "
            "harness.settings.langgraph.output, or convert the value inside the graph."
        ) from exc
    return coerced


def _coerce(value: Any) -> Any:
    """Convert LangChain messages to dicts wherever they appear in a state value."""

    if isinstance(value, dict):
        return {str(key): _coerce(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce(item) for item in value]
    if _is_message(value):
        return message_to_dict(value)
    return value


def _is_message(value: Any) -> bool:
    return hasattr(value, "content") and (hasattr(value, "type") or hasattr(value, "role"))


def messages_to_dicts(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, (list, tuple)):
        return []
    return [item if isinstance(item, dict) else message_to_dict(item) for item in raw]


def message_to_dict(message: Any) -> dict[str, Any]:
    role = getattr(message, "type", None) or getattr(message, "role", None) or "assistant"
    entry: dict[str, Any] = {"role": str(role), "content": getattr(message, "content", "")}
    message_id = getattr(message, "id", None)
    if message_id is not None:
        entry["id"] = message_id
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        entry["tool_calls"] = tool_calls
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict):
        entry["usage"] = usage
    metadata = getattr(message, "response_metadata", None)
    if isinstance(metadata, dict) and metadata:
        entry["response_metadata"] = metadata
    return entry


def final_response(messages: list[dict[str, Any]]) -> Any:
    for message in reversed(messages):
        if str(message.get("role", "")) in ("ai", "assistant"):
            return message.get("content")
    return messages[-1].get("content") if messages else None


def aggregate_usage(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Sum token usage reported by the messages produced during this turn."""

    totals: dict[str, Any] = {}
    for message in messages:
        usage = message.get("usage") if isinstance(message.get("usage"), dict) else {}
        for source, target in (
            ("input_tokens", "prompt_tokens"),
            ("output_tokens", "completion_tokens"),
            ("total_tokens", "total_tokens"),
        ):
            value = usage.get(source)
            if isinstance(value, int) and not isinstance(value, bool):
                totals[target] = int(totals.get(target, 0)) + value
    return totals or None
