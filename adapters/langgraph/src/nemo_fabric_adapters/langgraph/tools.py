# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fabric MCP servers and tool policy expressed as LangChain tools."""

from __future__ import annotations

import os
import shlex
from typing import Any

import nemo_fabric_adapters.common.utils as common_utils
from langchain_core.tools import BaseTool

from nemo_fabric_adapters.langgraph.config import AdapterConfigError

# MCP transports langchain-mcp-adapters accepts (after normalization).
VALID_MCP_TRANSPORTS = {"stdio", "sse", "streamable_http", "websocket"}


def mcp_connection(name: str, spec: Any) -> dict[str, Any]:
    """Normalize one routed Fabric MCP server into a langchain-mcp-adapters connection."""

    # A misconfigured server must fail loudly, not be silently dropped.
    if not isinstance(spec, dict):
        raise AdapterConfigError(f"MCP server '{name}' must be a mapping.")
    transport = str(spec.get("transport") or "").strip().lower().replace("-", "_")
    # McpServerPlan carries the URL or command in ``url``; there is no ``command``.
    target = os.path.expandvars(str(spec.get("url") or "")).strip()
    if not target:
        raise AdapterConfigError(f"MCP server '{name}' requires a url (or command in url).")
    if transport in ("stdio", "command", "process"):
        parts = shlex.split(target)
        if not parts:
            raise AdapterConfigError(f"MCP server '{name}' has an empty stdio command.")
        return {"transport": "stdio", "command": parts[0], "args": parts[1:]}
    if transport in ("", "http", "streamable_http", "streamablehttp"):
        transport = "streamable_http"
    if transport not in VALID_MCP_TRANSPORTS:
        raise AdapterConfigError(f"MCP server '{name}' has unsupported transport '{transport}'.")
    return {"transport": transport, "url": target}


async def load_mcp_tools(payload: dict[str, Any]) -> list[tuple[str, Any]]:
    """Load LangChain tools from the MCP servers Fabric routed as ``harness_native``.

    Tools are loaded per server and returned paired with the server that supplied
    them, because ``MultiServerMCPClient`` names a tool by its bare MCP tool name.
    Keeping the origin lets tool policy resolve a NAT-style ``server__tool``
    selector and distinguish two servers that expose the same tool name.
    """

    native = common_utils.capability_plan(payload).get("native") or {}
    servers = native.get("mcp_servers") or {}
    connections = {name: mcp_connection(name, spec) for name, spec in servers.items()}
    if not connections:
        return []

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as exc:
        raise AdapterConfigError(
            "MCP servers are configured but langchain-mcp-adapters is not installed; install the "
            "adapter's 'mcp' extra (pip install 'nemo-fabric-adapters-langgraph[mcp]')."
        ) from exc

    client = MultiServerMCPClient(connections)
    loaded: list[tuple[str, Any]] = []
    for name in connections:
        for tool in await client.get_tools(server_name=name):
            loaded.append((name, tool))
    return loaded


def qualified_name(server: str, tool: Any) -> str:
    """Return the ``server__tool`` selector for an MCP tool.

    ``__`` matches the separator the NAT adapter uses for function-group members,
    so one Fabric ``tools.blocked`` entry means the same thing in both adapters.
    The adapter resolves the selector rather than renaming the tool, because
    renaming would change what the model and the graph's own call sites see.
    """

    return f"{server}__{getattr(tool, 'name', '')}"


def guard_mcp_tools(server_tools: list[tuple[str, Any]], blocked: set[str]) -> list[Any]:
    """Apply tool policy to MCP tools, honoring qualified and bare selectors.

    A qualified ``server__tool`` selector denies one server's tool; a bare tool
    name denies that tool on every server that exposes it.
    """

    guarded: list[Any] = []
    for server, tool in server_tools:
        name = str(getattr(tool, "name", ""))
        guarded.append(
            _blocked_tool(tool) if name in blocked or qualified_name(server, tool) in blocked else tool
        )
    return guarded


def guard_tools(tools: list[Any], blocked: set[str]) -> list[Any]:
    """Replace blocked tools with deny stubs that keep the original name and schema.

    Denying rather than removing keeps the graph's tool surface stable: a model or a
    hardcoded node that references the tool receives an explicit policy message
    instead of an unknown-tool failure, and the original implementation never runs.
    """

    if not blocked:
        return list(tools)
    return [_blocked_tool(tool) if str(getattr(tool, "name", "")) in blocked else tool for tool in tools]


class BlockedTool(BaseTool):  # type: ignore[misc]
    """Deny stub for a tool blocked by Fabric ``tools.blocked``."""

    denial: str

    def _run(self, *args: Any, **kwargs: Any) -> str:
        return self.denial

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        return self.denial


def _blocked_tool(tool: Any) -> BlockedTool:
    name = str(getattr(tool, "name", ""))
    kwargs: dict[str, Any] = {
        "name": name,
        "description": str(getattr(tool, "description", "") or f"Tool '{name}' is blocked."),
        "denial": f"Tool '{name}' is blocked by the configured tools policy.",
    }
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is not None:
        kwargs["args_schema"] = args_schema
    return BlockedTool(**kwargs)
