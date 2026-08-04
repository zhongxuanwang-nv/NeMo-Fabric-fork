# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation suite for the generic LangGraph adapter.

The two representative agents under ``examples/langgraph/`` are exercised as real
compiled LangGraph graphs, not stubs: the only substituted component is the chat
model, so the graph topology, state reducers, routing, tool calls, and checkpoint
persistence under test are the real ones.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("langgraph.graph")

from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from nemo_fabric_adapters.langgraph import adapter  # noqa: E402
from nemo_fabric_adapters.langgraph import context as context_module  # noqa: E402
from nemo_fabric_adapters.langgraph import models as model_support  # noqa: E402
from nemo_fabric_adapters.langgraph import tools as tool_support  # noqa: E402
from nemo_fabric_adapters.langgraph.config import AdapterConfigError  # noqa: E402
from nemo_fabric_adapters.langgraph.config import parse_settings  # noqa: E402

TRIAGE_TARGET = "examples.langgraph.triage_agent.graph:graph"
CASE_REVIEW_TARGET = "examples.langgraph.case_review_agent.graph:build_graph"

DESCRIPTOR_PATH = Path(__file__).resolve().parents[2] / "adapters" / "langgraph" / "fabric-adapter.json"


class FakeChatModel:
    """Minimal stand-in for a LangChain chat model, with usage metadata."""

    def __init__(self, reply: str = "drafted reply") -> None:
        self.reply = reply
        self.calls: list[Any] = []

    async def ainvoke(self, messages: Any, *args: Any, **kwargs: Any) -> AIMessage:
        self.calls.append(messages)
        return AIMessage(
            content=self.reply,
            usage_metadata={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
        )


@tool
def search_docs(query: str) -> str:
    """Search internal documentation."""

    return f"docs:{query}"


def build_payload(
    tmp_path: Path,
    langgraph_settings: dict[str, Any],
    *,
    request_input: Any = "the invoice charge looks wrong",
    runtime_id: str | None = "runtime-1",
    models: dict[str, Any] | None = None,
    blocked: list[str] | None = None,
    mcp_servers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an AdapterInvocation-shaped payload for the adapter entrypoint."""

    config: dict[str, Any] = {
        "harness": {
            "adapter_id": "nvidia.fabric.langgraph",
            "resolution": "preinstalled",
            "settings": {"langgraph": langgraph_settings},
        },
        "models": models or {},
    }
    if blocked:
        config["tools"] = {"blocked": blocked}

    return {
        "agent_name": "langgraph-agent",
        "base_dir": str(tmp_path),
        "config": config,
        "runtime_context": {
            "runtime_id": runtime_id,
            "invocation_id": "invocation-1",
            "request_id": "request-1",
            "environment": {"workspace": str(tmp_path / "workspace")},
            "artifacts": {"root": str(tmp_path / "artifacts")},
        },
        "request": {"input": request_input},
        "capability_plan": {"native": {"mcp_servers": mcp_servers or {}}},
    }


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> FakeChatModel:
    """Bind a fake chat model to both the Fabric alias path and the agent-owned path."""

    model = FakeChatModel()
    monkeypatch.setattr(
        model_support,
        "build_chat_model",
        lambda payload, alias: (
            model,
            {"alias": alias, "model": "fake-model", "provider": "nvidia", "base_url": None},
        ),
    )
    import examples.langgraph.triage_agent.graph as triage_graph

    monkeypatch.setattr(triage_graph, "build_model", lambda: model)
    return model


@pytest.fixture
def fake_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve one MCP tool without contacting a server."""

    class FakeClient:
        def __init__(self, connections: dict[str, Any]) -> None:
            self.connections = connections

        async def get_tools(self) -> list[Any]:
            return [search_docs]

    import langchain_mcp_adapters.client as mcp_client

    monkeypatch.setattr(mcp_client, "MultiServerMCPClient", FakeClient)


# --- descriptor -----------------------------------------------------------


def test_descriptor_declares_only_supported_surface() -> None:
    descriptor = json.loads(DESCRIPTOR_PATH.read_text(encoding="utf-8"))

    assert descriptor["contract_version"] == "fabric.adapter/v1alpha1"
    assert descriptor["adapter_id"] == "nvidia.fabric.langgraph"
    assert descriptor["harness"] == "langgraph"
    assert descriptor["adapter_kind"] == "python"
    assert descriptor["runner"] == {
        "module": "nemo_fabric_adapters.langgraph.adapter",
        "callable": "run",
    }
    assert descriptor["config"]["accepts"] == ["models", "tools", "tools.blocked", "mcp"]
    # No telemetry provider is validated end to end yet, so none is advertised.
    assert descriptor["telemetry"]["providers"] == {}


# --- agent 1: unmodified compiled graph ----------------------------------


def test_compiled_agent_runs_unchanged(tmp_path: Path, fake_model: FakeChatModel) -> None:
    payload = build_payload(
        tmp_path,
        {
            "entrypoint": {"target": TRIAGE_TARGET, "kind": "compiled"},
            "input": {"mode": "field", "field": "ticket"},
            "output": {"mode": "last_message"},
        },
    )

    result = adapter.run(payload)

    assert result["failed"] is False, result["error"]
    assert result["harness"] == "langgraph"
    assert result["mode"] == "compiled"
    assert result["entrypoint"] == TRIAGE_TARGET
    assert result["response"] == "drafted reply"
    assert result["output_mode"] == "last_message"
    assert result["message_count"] == 1
    assert result["usage"] == {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
    # No Fabric-managed resources were requested by a compiled graph.
    assert result["models"] == []
    assert result["state_mode"] == "none"
    assert result["thread_id"] is None
    # The graph's own routing and prompt were used.
    assert "billing" in str(fake_model.calls[0])


def test_compiled_agent_field_projection_reports_graph_state(
    tmp_path: Path, fake_model: FakeChatModel
) -> None:
    payload = build_payload(
        tmp_path,
        {
            "entrypoint": {"target": TRIAGE_TARGET, "kind": "compiled"},
            "input": {"mode": "field", "field": "ticket"},
            "output": {"mode": "field", "field": "priority"},
        },
        request_input="production down, total outage",
    )

    result = adapter.run(payload)

    assert result["failed"] is False, result["error"]
    assert result["response"] == "p1"
    # The p1 branch escalates without a model call, so no usage is reported.
    assert result["usage"] is None
    assert fake_model.calls == []


def test_compiled_agent_state_projection_is_json_safe(
    tmp_path: Path, fake_model: FakeChatModel
) -> None:
    payload = build_payload(
        tmp_path,
        {
            "entrypoint": {"target": TRIAGE_TARGET, "kind": "compiled"},
            "input": {"mode": "field", "field": "ticket"},
            "output": {"mode": "state"},
        },
    )

    result = adapter.run(payload)

    assert result["failed"] is False, result["error"]
    assert result["response"]["category"] == "billing"
    assert result["response"]["messages"][-1]["content"] == "drafted reply"
    json.dumps(result)


def test_passthrough_input_forwards_native_state(tmp_path: Path, fake_model: FakeChatModel) -> None:
    payload = build_payload(
        tmp_path,
        {
            "entrypoint": {"target": TRIAGE_TARGET, "kind": "compiled"},
            "input": {"mode": "passthrough"},
            "output": {"mode": "field", "field": "category"},
        },
        request_input={"ticket": "please refund my invoice"},
    )

    result = adapter.run(payload)

    assert result["failed"] is False, result["error"]
    assert result["response"] == "billing"


def test_events_summary_is_reported_and_bounded(tmp_path: Path, fake_model: FakeChatModel) -> None:
    settings = {
        "entrypoint": {"target": TRIAGE_TARGET, "kind": "compiled"},
        "input": {"mode": "field", "field": "ticket"},
        "events": {"mode": "summary", "limit": 1},
    }

    result = adapter.run(build_payload(tmp_path, settings))

    assert result["failed"] is False, result["error"]
    assert result["event_count"] == 1
    assert result["events"] == [{"nodes": ["classify"]}]


def test_events_none_suppresses_event_reporting(tmp_path: Path, fake_model: FakeChatModel) -> None:
    result = adapter.run(
        build_payload(
            tmp_path,
            {
                "entrypoint": {"target": TRIAGE_TARGET, "kind": "compiled"},
                "input": {"mode": "field", "field": "ticket"},
            },
        )
    )

    assert result["failed"] is False, result["error"]
    assert result["events"] == []


# --- agent 1: capabilities a compiled graph cannot honor -----------------


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ({"blocked": ["search_docs"]}, "tools.blocked cannot be enforced"),
        ({"mcp_servers": {"docs": {"transport": "streamable_http", "url": "http://localhost:1"}}},
         "MCP servers cannot be supplied"),
    ],
)
def test_compiled_mode_rejects_unenforceable_capabilities(
    tmp_path: Path, extra: dict[str, Any], expected: str
) -> None:
    payload = build_payload(
        tmp_path,
        {
            "entrypoint": {"target": TRIAGE_TARGET, "kind": "compiled"},
            "input": {"mode": "field", "field": "ticket"},
        },
        **extra,
    )

    result = adapter.run(payload)

    assert result["failed"] is True
    assert expected in result["error"]


def test_compiled_mode_rejects_adapter_owned_state(tmp_path: Path) -> None:
    payload = build_payload(
        tmp_path,
        {
            "entrypoint": {"target": TRIAGE_TARGET, "kind": "compiled"},
            "input": {"mode": "field", "field": "ticket"},
            "state": "adapter",
        },
    )

    result = adapter.run(payload)

    assert result["failed"] is True
    assert "state 'adapter' cannot be applied" in result["error"]


def test_compiled_mode_accepts_graph_owned_state(tmp_path: Path, fake_model: FakeChatModel) -> None:
    payload = build_payload(
        tmp_path,
        {
            "entrypoint": {"target": TRIAGE_TARGET, "kind": "compiled"},
            "input": {"mode": "field", "field": "ticket"},
            "state": "graph",
        },
    )

    result = adapter.run(payload)

    assert result["failed"] is False, result["error"]
    assert result["state_mode"] == "graph"
    assert result["thread_id"]


# --- agent 2: factory with Fabric-managed resources ----------------------


def case_review_settings(**overrides: Any) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "entrypoint": {"target": CASE_REVIEW_TARGET, "kind": "factory"},
        "input": {"mode": "messages"},
        "output": {"mode": "field", "field": "answer"},
        "state": "adapter",
        "agent": {"system_prompt": "Review the case.", "max_findings": 3},
    }
    settings.update(overrides)
    return settings


def test_factory_agent_binds_model_tools_policy_and_state(
    tmp_path: Path, fake_model: FakeChatModel, fake_mcp: None
) -> None:
    payload = build_payload(
        tmp_path,
        case_review_settings(output={"mode": "state"}),
        request_input="is the termination clause enforceable?",
        models={"default": {"provider": "nvidia", "model": "fake-model"}},
        blocked=["lookup_precedent"],
        mcp_servers={"docs": {"transport": "streamable_http", "url": "http://localhost:1"}},
    )

    result = adapter.run(payload)

    assert result["failed"] is False, result["error"]
    assert result["mode"] == "factory"
    assert result["response"]["answer"] == "drafted reply"
    assert result["models"] == [
        {"alias": "default", "model": "fake-model", "provider": "nvidia", "base_url": None}
    ]
    assert result["state_mode"] == "adapter"
    assert result["thread_id"]
    assert result["resumed"] is False

    findings = result["response"]["findings"]
    # The Fabric MCP tool ran.
    assert any("docs:is the termination clause enforceable?" in finding for finding in findings)
    # The blocked application tool was denied, and its real output never appeared.
    assert any("blocked by the configured tools policy" in finding for finding in findings)
    assert not any("precedent:is the termination clause enforceable?" in finding for finding in findings)
    # The agent's own settings reached the graph.
    assert "Review the case." in str(fake_model.calls[0])


def test_factory_agent_projects_domain_field(
    tmp_path: Path, fake_model: FakeChatModel, fake_mcp: None
) -> None:
    payload = build_payload(
        tmp_path,
        case_review_settings(),
        models={"default": {"provider": "nvidia", "model": "fake-model"}},
    )

    result = adapter.run(payload)

    assert result["failed"] is False, result["error"]
    assert result["response"] == "drafted reply"
    assert result["output_mode"] == "field"


def test_factory_agent_resumes_state_across_invocations(
    tmp_path: Path, fake_model: FakeChatModel, fake_mcp: None
) -> None:
    settings = case_review_settings(output={"mode": "state"})

    first = adapter.run(
        build_payload(
            tmp_path,
            settings,
            request_input="first question",
            models={"default": {"provider": "nvidia", "model": "fake-model"}},
        )
    )
    second = adapter.run(
        build_payload(
            tmp_path,
            settings,
            request_input="second question",
            models={"default": {"provider": "nvidia", "model": "fake-model"}},
        )
    )

    assert first["failed"] is False, first["error"]
    assert second["failed"] is False, second["error"]
    # The same Fabric runtime maps to the same LangGraph thread.
    assert first["thread_id"] == second["thread_id"]
    assert first["resumed"] is False
    assert second["resumed"] is True
    # The checkpointed state carried the first turn into the second.
    assert first["response"]["turns"] == 1
    assert second["response"]["turns"] == 2
    assert any("first question" in str(message) for message in second["response"]["messages"])


def test_distinct_runtimes_do_not_share_state(
    tmp_path: Path, fake_model: FakeChatModel, fake_mcp: None
) -> None:
    settings = case_review_settings(output={"mode": "state"})
    models = {"default": {"provider": "nvidia", "model": "fake-model"}}

    first = adapter.run(build_payload(tmp_path, settings, models=models, runtime_id="runtime-a"))
    second = adapter.run(build_payload(tmp_path, settings, models=models, runtime_id="runtime-b"))

    assert first["thread_id"] != second["thread_id"]
    assert second["resumed"] is False
    assert second["response"]["turns"] == 1


def test_adapter_state_requires_a_runtime_id(tmp_path: Path) -> None:
    payload = build_payload(tmp_path, case_review_settings(), runtime_id=None)

    result = adapter.run(payload)

    assert result["failed"] is True
    assert "requires runtime_context.runtime_id" in result["error"]


def test_missing_model_alias_is_a_normalized_failure(
    tmp_path: Path, fake_mcp: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    payload = build_payload(tmp_path, case_review_settings())

    result = adapter.run(payload)

    assert result["failed"] is True
    assert "models.default is not configured" in result["error"]


def test_missing_credential_env_var_is_a_normalized_failure(
    tmp_path: Path, fake_mcp: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    payload = build_payload(
        tmp_path,
        case_review_settings(),
        models={"default": {"provider": "nvidia", "model": "fake-model"}},
    )

    result = adapter.run(payload)

    assert result["failed"] is True
    assert "NVIDIA_API_KEY' is not set" in result["error"]


# --- NeMo Agent Toolkit-compatible factory shape -------------------------


RUNNABLE_FACTORY_AGENT = '''
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict, total=False):
    question: str
    answer: str
    thread: str


def build(config: Any) -> Any:
    thread = ((config or {}).get("configurable") or {}).get("thread_id", "")

    def answer(state: State) -> dict[str, Any]:
        return {"answer": f"answered:{state.get('question', '')}", "thread": thread}

    workflow = StateGraph(State)
    workflow.add_node("answer", answer)
    workflow.add_edge(START, "answer")
    workflow.add_edge("answer", END)
    return workflow.compile()
'''


def write_external_agent(tmp_path: Path, body: str, name: str = "external_agent") -> str:
    directory = tmp_path / "agents"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.py").write_text(body, encoding="utf-8")
    return "agents"


def test_runnable_factory_receives_the_run_config(tmp_path: Path) -> None:
    search_path = write_external_agent(tmp_path, RUNNABLE_FACTORY_AGENT)
    payload = build_payload(
        tmp_path,
        {
            "entrypoint": {
                "target": "external_agent:build",
                "kind": "runnable_factory",
                "search_paths": [search_path],
            },
            "input": {"mode": "field", "field": "question"},
            "output": {"mode": "state"},
            "state": "graph",
        },
        request_input="what is the status?",
    )

    result = adapter.run(payload)

    assert result["failed"] is False, result["error"]
    assert result["mode"] == "runnable_factory"
    assert result["response"]["answer"] == "answered:what is the status?"
    # The NAT-shaped factory received Fabric's thread id through the run config.
    assert result["response"]["thread"] == result["thread_id"]


def test_runnable_factory_cannot_receive_adapter_owned_state(tmp_path: Path) -> None:
    search_path = write_external_agent(tmp_path, RUNNABLE_FACTORY_AGENT)
    payload = build_payload(
        tmp_path,
        {
            "entrypoint": {
                "target": "external_agent:build",
                "kind": "runnable_factory",
                "search_paths": [search_path],
            },
            "state": "adapter",
        },
    )

    result = adapter.run(payload)

    assert result["failed"] is True
    assert "state 'adapter' cannot be applied to a 'runnable_factory'" in result["error"]


AMBIENT_CONTEXT_AGENT = '''
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from nemo_fabric_adapters.langgraph import current_context


class State(TypedDict, total=False):
    question: str
    answer: str


def build(_context: Any) -> Any:
    # A module that builds its graph from ambient configuration rather than an
    # explicit argument, as the NeMo Agent Toolkit builder accessor allows.
    prompt = current_context().agent_settings["system_prompt"]

    def answer(state: State) -> dict[str, Any]:
        return {"answer": f"{prompt}|{state.get('question', '')}"}

    workflow = StateGraph(State)
    workflow.add_node("answer", answer)
    workflow.add_edge(START, "answer")
    workflow.add_edge("answer", END)
    return workflow.compile()
'''


def test_ambient_context_is_available_during_graph_construction(tmp_path: Path) -> None:
    search_path = write_external_agent(tmp_path, AMBIENT_CONTEXT_AGENT, name="ambient_agent")
    payload = build_payload(
        tmp_path,
        {
            "entrypoint": {
                "target": "ambient_agent:build",
                "kind": "factory",
                "search_paths": [search_path],
            },
            "input": {"mode": "field", "field": "question"},
            "output": {"mode": "field", "field": "answer"},
            "agent": {"system_prompt": "ambient"},
        },
        request_input="hello",
    )

    result = adapter.run(payload)

    assert result["failed"] is False, result["error"]
    assert result["response"] == "ambient|hello"


def test_ambient_context_outside_graph_construction_raises() -> None:
    with pytest.raises(RuntimeError, match="no LangGraph adapter context is active"):
        context_module.current_context()


# --- entry-point loading -------------------------------------------------


def test_search_paths_must_stay_within_base_dir(tmp_path: Path) -> None:
    payload = build_payload(
        tmp_path,
        {
            "entrypoint": {
                "target": "external_agent:build",
                "kind": "factory",
                "search_paths": ["../escape"],
            }
        },
    )

    result = adapter.run(payload)

    assert result["failed"] is True
    assert "escapes base_dir" in result["error"]


def test_absolute_search_paths_are_rejected(tmp_path: Path) -> None:
    payload = build_payload(
        tmp_path,
        {
            "entrypoint": {
                "target": "external_agent:build",
                "kind": "factory",
                "search_paths": [str(tmp_path)],
            }
        },
    )

    result = adapter.run(payload)

    assert result["failed"] is True
    assert "must be relative to base_dir" in result["error"]


def test_unimportable_entrypoint_names_the_remedy(tmp_path: Path) -> None:
    payload = build_payload(tmp_path, {"entrypoint": {"target": "no_such_module:graph"}})

    result = adapter.run(payload)

    assert result["failed"] is True
    assert "could not import 'no_such_module'" in result["error"]
    assert "search_paths" in result["error"]


def test_missing_attribute_is_reported_clearly(tmp_path: Path) -> None:
    payload = build_payload(
        tmp_path, {"entrypoint": {"target": "examples.langgraph.triage_agent.graph:missing"}}
    )

    result = adapter.run(payload)

    assert result["failed"] is True
    assert "has no attribute 'missing'" in result["error"]


def test_compiled_kind_pointing_at_a_factory_fails_before_running(tmp_path: Path) -> None:
    payload = build_payload(
        tmp_path, {"entrypoint": {"target": CASE_REVIEW_TARGET, "kind": "compiled"}}
    )

    result = adapter.run(payload)

    assert result["failed"] is True
    assert "declared kind 'compiled'" in result["error"]
    assert "has no invoke/ainvoke method" in result["error"]


def test_factory_kind_pointing_at_a_compiled_graph_fails(tmp_path: Path) -> None:
    payload = build_payload(tmp_path, {"entrypoint": {"target": TRIAGE_TARGET, "kind": "factory"}})

    result = adapter.run(payload)

    assert result["failed"] is True
    assert "declare kind 'compiled' instead" in result["error"]


NON_GRAPH_FACTORY = '''
def build(_context):
    return {"not": "a graph"}
'''


def test_factory_returning_a_non_graph_fails(tmp_path: Path) -> None:
    search_path = write_external_agent(tmp_path, NON_GRAPH_FACTORY, name="bad_factory")
    payload = build_payload(
        tmp_path,
        {
            "entrypoint": {
                "target": "bad_factory:build",
                "kind": "factory",
                "search_paths": [search_path],
            }
        },
    )

    result = adapter.run(payload)

    assert result["failed"] is True
    assert "must return a compiled LangGraph graph" in result["error"]


# --- input and output projection failures --------------------------------


def test_messages_input_rejects_structured_requests(tmp_path: Path) -> None:
    payload = build_payload(
        tmp_path,
        {
            "entrypoint": {"target": TRIAGE_TARGET, "kind": "compiled"},
            "input": {"mode": "messages"},
        },
        request_input={"ticket": "structured"},
    )

    result = adapter.run(payload)

    assert result["failed"] is True
    assert "requires a text request" in result["error"]
    assert "passthrough" in result["error"]


def test_last_message_output_requires_a_messages_key(tmp_path: Path) -> None:
    search_path = write_external_agent(tmp_path, RUNNABLE_FACTORY_AGENT)
    payload = build_payload(
        tmp_path,
        {
            "entrypoint": {
                "target": "external_agent:build",
                "kind": "runnable_factory",
                "search_paths": [search_path],
            },
            "input": {"mode": "field", "field": "question"},
            "output": {"mode": "last_message"},
        },
    )

    result = adapter.run(payload)

    assert result["failed"] is True
    assert "expects a 'messages' key" in result["error"]


def test_unknown_output_field_lists_available_fields(tmp_path: Path, fake_model: FakeChatModel) -> None:
    payload = build_payload(
        tmp_path,
        {
            "entrypoint": {"target": TRIAGE_TARGET, "kind": "compiled"},
            "input": {"mode": "field", "field": "ticket"},
            "output": {"mode": "field", "field": "nope"},
        },
    )

    result = adapter.run(payload)

    assert result["failed"] is True
    assert "'nope' is not present" in result["error"]
    assert "category" in result["error"]


# --- configuration validation --------------------------------------------


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        ({}, "harness.settings.langgraph is required"),
        ({"langgraph": {"entrypoint": {"target": "m:g"}, "nope": 1}}, "unsupported key(s) ['nope']"),
        ({"langgraph": {}}, "entrypoint is required"),
        ({"langgraph": {"entrypoint": {"target": "no-colon"}}}, "must use the form 'module.path:attribute'"),
        ({"langgraph": {"entrypoint": {"target": "m:g", "kind": "weird"}}}, "'weird' is not supported"),
        ({"langgraph": {"entrypoint": {"target": "m:g"}, "input": {"mode": "field"}}},
         "input.field is required"),
        ({"langgraph": {"entrypoint": {"target": "m:g"}, "output": {"mode": "field"}}},
         "output.field is required"),
        ({"langgraph": {"entrypoint": {"target": "m:g"}, "state": "elsewhere"}},
         "'elsewhere' is not supported"),
        ({"langgraph": {"entrypoint": {"target": "m:g"}, "events": {"limit": 0}}},
         "must be a positive integer"),
        ({"langgraph": {"entrypoint": {"target": "m:g"}, "entrypoint_typo": 1}},
         "unsupported key(s) ['entrypoint_typo']"),
        ({"langgraph": {"entrypoint": {"target": "m:g", "search_paths": "agents"}}},
         "must be a list of relative path strings"),
        ({"langgraph": {"entrypoint": {"target": "m:g"}, "agent": {"value": {1, 2}}}},
         "must contain only JSON-compatible values"),
    ],
)
def test_settings_validation_fails_closed(settings: dict[str, Any], expected: str) -> None:
    with pytest.raises(AdapterConfigError) as failure:
        parse_settings(settings)

    assert expected in str(failure.value)


def test_settings_defaults_are_conservative() -> None:
    settings = parse_settings({"langgraph": {"entrypoint": {"target": "my_agent.graph:graph"}}})

    assert settings.entrypoint.kind == "compiled"
    assert settings.entrypoint.module == "my_agent.graph"
    assert settings.entrypoint.attribute == "graph"
    assert settings.input.mode == "passthrough"
    assert settings.output.mode == "last_message"
    assert settings.events.mode == "none"
    assert settings.state == "none"
    assert settings.agent == {}
    assert settings.recursion_limit is None


# --- MCP normalization ----------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ({"transport": "streamable_http", "url": "http://localhost:9"},
         {"transport": "streamable_http", "url": "http://localhost:9"}),
        ({"url": "http://localhost:9"}, {"transport": "streamable_http", "url": "http://localhost:9"}),
        ({"transport": "stdio", "url": "uvx server --flag"},
         {"transport": "stdio", "command": "uvx", "args": ["server", "--flag"]}),
        ({"transport": "sse", "url": "http://localhost:9/sse"},
         {"transport": "sse", "url": "http://localhost:9/sse"}),
    ],
)
def test_mcp_connections_are_normalized(spec: dict[str, Any], expected: dict[str, Any]) -> None:
    assert tool_support.mcp_connection("docs", spec) == expected


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ({"transport": "carrier-pigeon", "url": "http://localhost:9"}, "unsupported transport"),
        ({"transport": "streamable_http"}, "requires a url"),
        ("not-a-mapping", "must be a mapping"),
    ],
)
def test_invalid_mcp_configuration_fails_closed(spec: Any, expected: str) -> None:
    with pytest.raises(AdapterConfigError, match=expected):
        tool_support.mcp_connection("docs", spec)


# --- tool policy ----------------------------------------------------------


async def test_guard_tools_denies_without_executing() -> None:
    guarded = tool_support.guard_tools([search_docs], {"search_docs"})

    assert len(guarded) == 1
    assert guarded[0].name == "search_docs"
    # The schema is preserved so the graph's tool surface does not change shape.
    assert guarded[0].description == search_docs.description
    result = await guarded[0].ainvoke({"query": "anything"})
    assert "blocked by the configured tools policy" in result
    assert "docs:" not in result


async def test_guard_tools_leaves_allowed_tools_intact() -> None:
    guarded = tool_support.guard_tools([search_docs], {"other_tool"})

    assert guarded[0] is search_docs
    assert await guarded[0].ainvoke({"query": "x"}) == "docs:x"
