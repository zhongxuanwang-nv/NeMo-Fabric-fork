# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A custom LangGraph case-review agent that opts into Fabric-managed resources.

This agent needs more than execution: a configurable model, Fabric MCP tools, an
enforced tool policy, durable multi-turn state, and a domain-specific result field.
It gets all of that from one adapter-owned argument, so it stays an ordinary
LangGraph agent and no agent-specific Fabric adapter is required.

The only Fabric-facing surface is ``LangGraphAdapterContext``. The graph topology,
state schema, prompts, application tools, and output field belong to the agent.
"""

from __future__ import annotations

from typing import Annotated
from typing import Any
from typing import TypedDict

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

DEFAULT_SYSTEM_PROMPT = "You are a case reviewer. Answer using only the findings provided."

REVIEW_PROMPT = (
    "{system_prompt}\n\nQuestion: {question}\n\nFindings:\n{findings}\n\n"
    "Write the reviewer's answer."
)


class CaseReviewState(TypedDict, total=False):
    """This agent's own state schema, including its domain-specific ``answer`` field."""

    messages: Annotated[list, add_messages]
    findings: list[str]
    answer: str
    turns: int


@tool
def lookup_precedent(query: str) -> str:
    """Look up an internal precedent for a case question."""

    return f"precedent:{query}"


def build_graph(context: Any) -> Any:
    """Build the graph from Fabric-managed resources.

    ``context`` is a ``LangGraphAdapterContext``. It is annotated loosely so this
    module stays importable without the adapter installed, which keeps the agent
    testable on its own.
    """

    settings = context.agent_settings
    system_prompt = settings.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    max_findings = int(settings.get("max_findings", 3))
    model = context.chat_model("default")

    # Fabric MCP tools already have tool policy applied; application tools must be
    # passed through guard_tools() for the same policy to cover them.
    tools = [*context.mcp_tools(), *context.guard_tools([lookup_precedent])]

    async def research(state: CaseReviewState) -> dict[str, Any]:
        question = _latest_question(state)
        findings: list[str] = []
        for candidate in tools[:max_findings]:
            findings.append(f"{candidate.name}: {await candidate.ainvoke({'query': question})}")
        return {"findings": findings, "turns": int(state.get("turns", 0)) + 1}

    async def review(state: CaseReviewState) -> dict[str, Any]:
        prompt = REVIEW_PROMPT.format(
            system_prompt=system_prompt,
            question=_latest_question(state),
            findings="\n".join(state.get("findings") or ["(none)"]),
        )
        completion = await model.ainvoke([{"role": "user", "content": prompt}])
        answer = completion.content if isinstance(completion, AIMessage) else str(completion)
        return {"answer": answer, "messages": [completion]}

    workflow = StateGraph(CaseReviewState)
    workflow.add_node("research", research)
    workflow.add_node("review", review)
    workflow.add_edge(START, "research")
    workflow.add_edge("research", "review")
    workflow.add_edge("review", END)

    # Compiling with the adapter-owned checkpointer is what makes multi-turn state
    # durable across separate Fabric invocations on the same runtime.
    return workflow.compile(checkpointer=context.checkpointer)


def _latest_question(state: CaseReviewState) -> str:
    for message in reversed(state.get("messages") or []):
        role = getattr(message, "type", None) or getattr(message, "role", None)
        if role in ("human", "user"):
            content = getattr(message, "content", None)
            return content if isinstance(content, str) else str(content)
    return ""
