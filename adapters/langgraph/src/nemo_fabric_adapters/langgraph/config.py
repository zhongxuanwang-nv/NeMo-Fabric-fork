# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated ``harness.settings.langgraph`` configuration.

Only documented, JSON-compatible values are accepted. Unknown keys fail closed so
a typo becomes a clear configuration error instead of silently ignored intent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field
from typing import Any

DEFAULT_MESSAGES_KEY = "messages"
DEFAULT_EVENT_LIMIT = 50

BINDING_KINDS = frozenset({"compiled", "factory", "runnable_factory"})
INPUT_MODES = frozenset({"passthrough", "messages", "field"})
OUTPUT_MODES = frozenset({"last_message", "field", "state"})
EVENT_MODES = frozenset({"none", "summary"})
STATE_MODES = frozenset({"none", "graph", "adapter"})

_SETTINGS_KEYS = frozenset({"entrypoint", "input", "output", "events", "state", "agent", "recursion_limit"})
_ENTRYPOINT_KEYS = frozenset({"target", "kind", "search_paths"})
_INPUT_KEYS = frozenset({"mode", "field", "messages_key"})
_OUTPUT_KEYS = frozenset({"mode", "field", "messages_key"})
_EVENT_KEYS = frozenset({"mode", "limit"})


class AdapterConfigError(RuntimeError):
    """Raised for invalid LangGraph adapter configuration (normalized to a failure)."""


@dataclass(frozen=True)
class EntrypointSettings:
    """Explicit selection of the graph object to run."""

    target: str
    kind: str
    search_paths: tuple[str, ...] = ()

    @property
    def module(self) -> str:
        return self.target.split(":", 1)[0]

    @property
    def attribute(self) -> str:
        return self.target.split(":", 1)[1]


@dataclass(frozen=True)
class InputSettings:
    mode: str = "passthrough"
    field: str | None = None
    messages_key: str = DEFAULT_MESSAGES_KEY


@dataclass(frozen=True)
class OutputSettings:
    mode: str = "last_message"
    field: str | None = None
    messages_key: str = DEFAULT_MESSAGES_KEY


@dataclass(frozen=True)
class EventSettings:
    mode: str = "none"
    limit: int = DEFAULT_EVENT_LIMIT


@dataclass(frozen=True)
class LangGraphSettings:
    entrypoint: EntrypointSettings
    input: InputSettings = field(default_factory=InputSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    events: EventSettings = field(default_factory=EventSettings)
    state: str = "none"
    agent: dict[str, Any] = field(default_factory=dict)
    recursion_limit: int | None = None


def parse_settings(settings: dict[str, Any]) -> LangGraphSettings:
    """Validate ``harness.settings.langgraph`` and return typed settings."""

    raw = settings.get("langgraph")
    if raw is None:
        raise AdapterConfigError(
            "harness.settings.langgraph is required; it must at least set "
            "entrypoint.target (for example {'entrypoint': {'target': 'my_agent.graph:graph'}})."
        )
    section = _mapping("harness.settings.langgraph", raw)
    _reject_unknown("harness.settings.langgraph", section, _SETTINGS_KEYS)

    return LangGraphSettings(
        entrypoint=_parse_entrypoint(section.get("entrypoint")),
        input=_parse_input(section.get("input")),
        output=_parse_output(section.get("output")),
        events=_parse_events(section.get("events")),
        state=_parse_choice("harness.settings.langgraph.state", section.get("state"), STATE_MODES, "none"),
        agent=_parse_agent(section.get("agent")),
        recursion_limit=_parse_positive_int(
            "harness.settings.langgraph.recursion_limit", section.get("recursion_limit")
        ),
    )


def _parse_entrypoint(raw: Any) -> EntrypointSettings:
    if raw is None:
        raise AdapterConfigError("harness.settings.langgraph.entrypoint is required.")
    section = _mapping("harness.settings.langgraph.entrypoint", raw)
    _reject_unknown("harness.settings.langgraph.entrypoint", section, _ENTRYPOINT_KEYS)

    target = str(section.get("target") or "").strip()
    if not target:
        raise AdapterConfigError("harness.settings.langgraph.entrypoint.target is required.")
    module, separator, attribute = target.partition(":")
    if not separator or not module.strip() or not attribute.strip():
        raise AdapterConfigError(
            f"harness.settings.langgraph.entrypoint.target '{target}' must use the form "
            "'module.path:attribute'."
        )

    search_paths = section.get("search_paths")
    if search_paths is None:
        paths: tuple[str, ...] = ()
    elif isinstance(search_paths, list) and all(isinstance(item, str) for item in search_paths):
        paths = tuple(item for item in search_paths if item.strip())
    else:
        raise AdapterConfigError(
            "harness.settings.langgraph.entrypoint.search_paths must be a list of relative path strings."
        )

    return EntrypointSettings(
        target=f"{module.strip()}:{attribute.strip()}",
        kind=_parse_choice(
            "harness.settings.langgraph.entrypoint.kind", section.get("kind"), BINDING_KINDS, "compiled"
        ),
        search_paths=paths,
    )


def _parse_input(raw: Any) -> InputSettings:
    if raw is None:
        return InputSettings()
    section = _mapping("harness.settings.langgraph.input", raw)
    _reject_unknown("harness.settings.langgraph.input", section, _INPUT_KEYS)
    mode = _parse_choice("harness.settings.langgraph.input.mode", section.get("mode"), INPUT_MODES, "passthrough")
    state_field = _optional_str("harness.settings.langgraph.input.field", section.get("field"))
    if mode == "field" and not state_field:
        raise AdapterConfigError("harness.settings.langgraph.input.field is required when input.mode is 'field'.")
    return InputSettings(
        mode=mode,
        field=state_field,
        messages_key=_optional_str(
            "harness.settings.langgraph.input.messages_key", section.get("messages_key")
        )
        or DEFAULT_MESSAGES_KEY,
    )


def _parse_output(raw: Any) -> OutputSettings:
    if raw is None:
        return OutputSettings()
    section = _mapping("harness.settings.langgraph.output", raw)
    _reject_unknown("harness.settings.langgraph.output", section, _OUTPUT_KEYS)
    mode = _parse_choice("harness.settings.langgraph.output.mode", section.get("mode"), OUTPUT_MODES, "last_message")
    state_field = _optional_str("harness.settings.langgraph.output.field", section.get("field"))
    if mode == "field" and not state_field:
        raise AdapterConfigError("harness.settings.langgraph.output.field is required when output.mode is 'field'.")
    return OutputSettings(
        mode=mode,
        field=state_field,
        messages_key=_optional_str(
            "harness.settings.langgraph.output.messages_key", section.get("messages_key")
        )
        or DEFAULT_MESSAGES_KEY,
    )


def _parse_events(raw: Any) -> EventSettings:
    if raw is None:
        return EventSettings()
    section = _mapping("harness.settings.langgraph.events", raw)
    _reject_unknown("harness.settings.langgraph.events", section, _EVENT_KEYS)
    limit = _parse_positive_int("harness.settings.langgraph.events.limit", section.get("limit"))
    return EventSettings(
        mode=_parse_choice("harness.settings.langgraph.events.mode", section.get("mode"), EVENT_MODES, "none"),
        limit=limit if limit is not None else DEFAULT_EVENT_LIMIT,
    )


def _parse_agent(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    section = _mapping("harness.settings.langgraph.agent", raw)
    _require_json_safe("harness.settings.langgraph.agent", section)
    return dict(section)


def _mapping(label: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AdapterConfigError(f"{label} must be a mapping, not {type(value).__name__}.")
    return value


def _reject_unknown(label: str, section: dict[str, Any], allowed: frozenset[str]) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise AdapterConfigError(f"{label} has unsupported key(s) {unknown}; supported keys are {sorted(allowed)}.")


def _parse_choice(label: str, value: Any, allowed: frozenset[str], default: str) -> str:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text not in allowed:
        raise AdapterConfigError(f"{label} '{value}' is not supported; supported values are {sorted(allowed)}.")
    return text


def _optional_str(label: str, value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise AdapterConfigError(f"{label} must be a non-empty string.")
    return value.strip()


def _parse_positive_int(label: str, value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AdapterConfigError(f"{label} must be a positive integer.")
    return value


def _require_json_safe(label: str, value: Any) -> None:
    """Reject non-JSON values early so agent settings stay a stable, portable contract."""

    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise AdapterConfigError(f"{label} must contain only JSON-compatible values: {exc}") from exc
