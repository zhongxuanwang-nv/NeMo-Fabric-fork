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


async def load_mcp_tools(payload: dict[str, Any]) -> list[Any]:
    """Load LangChain tools from the MCP servers Fabric routed as ``harness_native``."""

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
    return list(await client.get_tools())


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
