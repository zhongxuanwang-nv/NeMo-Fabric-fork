# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A custom LangGraph support-triage agent that predates any Fabric integration.

This agent is deliberately written the way an existing LangGraph agent normally is:
it owns its state schema, its routing, its prompts, and its own chat model, and it
compiles itself at import time. It does not import Fabric or the Fabric LangGraph
adapter, and nothing in this module was changed to make it runnable under Fabric.

It validates that the generic adapter can run an unmodified compiled graph through
the ``compiled`` binding kind.
"""

from __future__ import annotations

import os
from typing import Annotated
from typing import Any
from typing import TypedDict

from langchain_core.messages import AIMessage
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

URGENT_TERMS = ("outage", "data loss", "security", "cannot log in", "production down")
BILLING_TERMS = ("invoice", "refund", "billing", "charge", "payment")

DRAFT_PROMPT = (
    "You are a support agent. Write a short, concrete reply to the customer ticket below. "
    "The ticket has been classified as category '{category}' with priority '{priority}'.\n\n"
    "Ticket: {ticket}"
)


class TriageState(TypedDict, total=False):
    """This agent's own state schema; Fabric does not define or inspect it."""

    ticket: str
    category: str
    priority: str
    messages: Annotated[list, add_messages]


def build_model() -> Any:
    """Build this agent's chat model from its own environment configuration."""

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.environ.get("TRIAGE_AGENT_MODEL", "meta/llama-3.3-70b-instruct"),
        base_url=os.environ.get("TRIAGE_AGENT_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        api_key=os.environ.get("TRIAGE_AGENT_API_KEY") or os.environ.get("NVIDIA_API_KEY", ""),
        temperature=0,
    )


def classify(state: TriageState) -> dict[str, Any]:
    ticket = (state.get("ticket") or "").lower()
    if any(term in ticket for term in URGENT_TERMS):
        return {"category": "incident", "priority": "p1"}
    if any(term in ticket for term in BILLING_TERMS):
        return {"category": "billing", "priority": "p3"}
    return {"category": "general", "priority": "p2"}


def escalate(state: TriageState) -> dict[str, Any]:
    """Escalate without spending a model call; p1 tickets go straight to on-call."""

    return {
        "messages": [
            AIMessage(
                content=(
                    "Escalated to the on-call engineer as a p1 incident. "
                    "A responder will follow up on this ticket directly."
                )
            )
        ]
    }


async def draft_reply(state: TriageState) -> dict[str, Any]:
    prompt = DRAFT_PROMPT.format(
        category=state.get("category", "general"),
        priority=state.get("priority", "p2"),
        ticket=state.get("ticket", ""),
    )
    reply = await build_model().ainvoke([{"role": "user", "content": prompt}])
    return {"messages": [reply]}


def route_after_classify(state: TriageState) -> str:
    return "escalate" if state.get("priority") == "p1" else "draft_reply"


def build() -> Any:
    workflow = StateGraph(TriageState)
    workflow.add_node("classify", classify)
    workflow.add_node("escalate", escalate)
    workflow.add_node("draft_reply", draft_reply)
    workflow.add_edge(START, "classify")
    workflow.add_conditional_edges(
        "classify", route_after_classify, {"escalate": "escalate", "draft_reply": "draft_reply"}
    )
    workflow.add_edge("escalate", END)
    workflow.add_edge("draft_reply", END)
    return workflow.compile()


graph = build()
