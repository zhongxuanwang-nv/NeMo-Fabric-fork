# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit graph entry-point loading and shape validation.

The adapter never guesses what an entry point is. The configured
``entrypoint.kind`` decides how the imported object is used, and a mismatch fails
before the graph runs so the error names the configuration, not a deep LangGraph
traceback.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path
from typing import Any

from nemo_fabric_adapters.langgraph.config import AdapterConfigError
from nemo_fabric_adapters.langgraph.config import EntrypointSettings


def resolve_search_paths(entrypoint: EntrypointSettings, base_dir: str) -> list[Path]:
    """Resolve ``entrypoint.search_paths`` to directories confined to ``base_dir``.

    Search paths make an agent that lives in a project directory importable without
    packaging it, which is how most existing LangGraph agents are laid out. They
    come from the agent configuration, which Fabric already treats as trusted, but
    they are still confined to ``base_dir`` so a configuration cannot reach
    arbitrary locations on the host.
    """

    root = Path(base_dir).resolve()
    resolved: list[Path] = []
    for raw in entrypoint.search_paths:
        candidate = Path(raw)
        if candidate.is_absolute():
            raise AdapterConfigError(
                f"harness.settings.langgraph.entrypoint.search_paths entry '{raw}' must be relative to base_dir."
            )
        path = (root / candidate).resolve()
        if not path.is_relative_to(root):
            raise AdapterConfigError(
                f"harness.settings.langgraph.entrypoint.search_paths entry '{raw}' escapes base_dir."
            )
        if not path.is_dir():
            raise AdapterConfigError(
                f"harness.settings.langgraph.entrypoint.search_paths entry '{raw}' is not a directory."
            )
        resolved.append(path)
    return resolved


def load_entrypoint(entrypoint: EntrypointSettings, base_dir: str) -> Any:
    """Import the configured target and validate it against the declared kind."""

    for path in resolve_search_paths(entrypoint, base_dir):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)

    try:
        module = importlib.import_module(entrypoint.module)
    except ImportError as exc:
        raise AdapterConfigError(
            f"could not import '{entrypoint.module}' from the adapter environment: {exc}. Install the "
            "agent package into the adapter's Python environment, or add its directory to "
            "harness.settings.langgraph.entrypoint.search_paths."
        ) from exc

    target: Any = module
    for part in entrypoint.attribute.split("."):
        try:
            target = getattr(target, part)
        except AttributeError as exc:
            raise AdapterConfigError(
                f"'{entrypoint.module}' has no attribute '{entrypoint.attribute}'."
            ) from exc

    _validate_kind(entrypoint, target)
    return target


def _validate_kind(entrypoint: EntrypointSettings, target: Any) -> None:
    if entrypoint.kind == "compiled":
        if not is_graph(target):
            raise AdapterConfigError(
                f"entrypoint '{entrypoint.target}' is declared kind 'compiled' but "
                f"{type(target).__name__} has no invoke/ainvoke method. Compile the graph, or declare "
                "kind 'factory' (a callable taking LangGraphAdapterContext) or 'runnable_factory' (a "
                "callable taking a RunnableConfig)."
            )
        return
    # Check for a graph before checking callability: a compiled graph is not
    # callable, and "declare kind 'compiled'" is far more useful than "not callable".
    if is_graph(target):
        raise AdapterConfigError(
            f"entrypoint '{entrypoint.target}' is declared kind '{entrypoint.kind}' but looks like a "
            "compiled graph; declare kind 'compiled' instead."
        )
    if not callable(target):
        raise AdapterConfigError(
            f"entrypoint '{entrypoint.target}' is declared kind '{entrypoint.kind}' but "
            f"{type(target).__name__} is not callable."
        )


def is_graph(target: Any) -> bool:
    """Return whether ``target`` exposes the LangGraph runnable surface the adapter needs."""

    return hasattr(target, "ainvoke") or hasattr(target, "invoke")


async def build_graph(entrypoint: EntrypointSettings, target: Any, factory_argument: Any) -> Any:
    """Return a compiled graph, calling a factory entry point when configured."""

    if entrypoint.kind == "compiled":
        return target

    produced = target(factory_argument)
    if inspect.isawaitable(produced):
        produced = await produced
    if not is_graph(produced):
        raise AdapterConfigError(
            f"entrypoint '{entrypoint.target}' returned {type(produced).__name__}; a "
            f"'{entrypoint.kind}' entry point must return a compiled LangGraph graph."
        )
    return produced
