# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real LangGraph smoke for both representative custom agents.

The compiled-graph escalation path needs no model call, so it runs without
credentials. The factory-graph tests bind a Fabric model and need a key.

    RUN_FABRIC_LANGGRAPH_INTEGRATION=1 NVIDIA_API_KEY=... \
        pytest tests/e2e/test_langgraph.py
"""

from __future__ import annotations

import importlib.util
import os

import pytest


@pytest.fixture(name="_require_integration")
def _require_integration_fixture() -> None:
    if os.environ.get("RUN_FABRIC_LANGGRAPH_INTEGRATION") != "1":
        pytest.skip("set RUN_FABRIC_LANGGRAPH_INTEGRATION=1 to run")
    if importlib.util.find_spec("langgraph") is None:
        pytest.fail("the langgraph package is required (pip install -e '.[langgraph]')")
    if importlib.util.find_spec("nemo_fabric._native") is None:
        pytest.fail("the nemo_fabric native extension is required (pip install -e .)")


@pytest.fixture(name="_require_credentials")
def _require_credentials_fixture(_require_integration: None) -> None:
    if not os.environ.get("NVIDIA_API_KEY"):
        pytest.fail("NVIDIA_API_KEY is required for the factory-graph smoke")


@pytest.mark.usefixtures("_require_integration")
async def test_compiled_agent_runs_unchanged_through_fabric():
    # The p1 branch escalates without a model call, so this validates the full
    # descriptor-discovery and subprocess path with no credentials.
    from examples.langgraph.config import BASE_DIR, triage_config
    from nemo_fabric import Fabric

    client = Fabric()
    result = await client.run(
        triage_config(), base_dir=BASE_DIR, input="production down, total outage"
    )

    assert result["status"] == "succeeded", result.to_mapping()
    assert result["adapter_id"] == "nvidia.fabric.langgraph"
    assert result["output"]["mode"] == "compiled"
    assert "Escalated" in result["output"]["response"], result.to_mapping()
    # events: summary reports the nodes the graph actually traversed
    assert result["output"]["events"] == [{"nodes": ["classify"]}, {"nodes": ["escalate"]}]


@pytest.mark.usefixtures("_require_credentials")
async def test_factory_agent_binds_fabric_resources():
    from examples.langgraph.config import BASE_DIR, case_review_config
    from nemo_fabric import Fabric

    client = Fabric()
    result = await client.run(
        case_review_config(),
        base_dir=BASE_DIR,
        input="Is a five-year non-compete enforceable in California?",
    )

    mapping = result.to_mapping()
    assert result["status"] == "succeeded", mapping
    output = result["output"]
    assert output["mode"] == "factory"
    # the graph received the model from the Fabric alias, not its own hardcoded client
    assert output["models"] == [
        {
            "alias": "default",
            "model": "meta/llama-3.3-70b-instruct",
            "provider": "nvidia",
            "base_url": "https://integrate.api.nvidia.com/v1",
        }
    ], mapping
    # the domain result field is projected as the normalized response
    assert output["output_mode"] == "field"
    assert output["response"], mapping


@pytest.mark.usefixtures("_require_credentials")
async def test_factory_agent_resumes_across_turns():
    from examples.langgraph.config import BASE_DIR, case_review_config
    from nemo_fabric import Fabric

    client = Fabric()
    async with await client.start_runtime(case_review_config(), base_dir=BASE_DIR) as runtime:
        first = await runtime.invoke(input="Summarize the risk in a fixed-fee contract.")
        second = await runtime.invoke(input="What did I ask about first? One short sentence.")

    results = (first.to_mapping(), second.to_mapping())
    assert first["status"] == second["status"] == "succeeded", results
    # one started runtime maps to one adapter-owned LangGraph thread
    assert first["output"]["thread_id"] == second["output"]["thread_id"], results
    assert first["output"]["resumed"] is False, results
    assert second["output"]["resumed"] is True, results
