# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fabric configs for the two representative custom LangGraph agents.

Both configs use the same adapter id, ``nvidia.fabric.langgraph``. The only
differences are the entry point and how much of Fabric each agent opts into,
which is the point the design is validating.
"""

from __future__ import annotations

from pathlib import Path

from nemo_fabric import EnvironmentConfig
from nemo_fabric import FabricConfig
from nemo_fabric import HarnessConfig
from nemo_fabric import MetadataConfig
from nemo_fabric import ModelConfig
from nemo_fabric import RuntimeConfig

ADAPTER_ID = "nvidia.fabric.langgraph"
# The repository root, so the example agent packages are importable by the adapter.
BASE_DIR = Path(__file__).resolve().parents[2]
# Artifact paths are relative to base_dir, so keep them beside the example rather
# than at the repository root.
ARTIFACTS = "./examples/langgraph/artifacts"


def triage_config() -> FabricConfig:
    """Config for the unmodified compiled graph.

    The agent owns its model and tools, so Fabric supplies only execution and
    result normalization. Nothing in the agent had to change.
    """

    return FabricConfig(
        metadata=MetadataConfig(
            name="langgraph-triage-agent",
            description="Classifies a support ticket and drafts a reply.",
        ),
        harness=HarnessConfig(
            adapter_id=ADAPTER_ID,
            resolution="preinstalled",
            settings={
                "langgraph": {
                    "entrypoint": {
                        "target": "examples.langgraph.triage_agent.graph:graph",
                        "kind": "compiled",
                        "search_paths": ["."],
                    },
                    "input": {"mode": "field", "field": "ticket"},
                    "output": {"mode": "last_message"},
                    "events": {"mode": "summary"},
                }
            },
        ),
        runtime=RuntimeConfig(
            input_schema="text", output_schema="message", artifacts=f"{ARTIFACTS}/triage"
        ),
        environment=EnvironmentConfig(provider="local", artifacts=f"{ARTIFACTS}/triage"),
    )


def case_review_config() -> FabricConfig:
    """Config for the factory graph that opts into Fabric-managed resources.

    Fabric supplies the model, the MCP tool surface, the tool policy, and durable
    multi-turn state; the agent still owns its topology, prompts, and result field.
    """

    config = FabricConfig(
        metadata=MetadataConfig(
            name="langgraph-case-review-agent",
            description="Researches a case question and drafts a reviewed answer.",
        ),
        harness=HarnessConfig(
            adapter_id=ADAPTER_ID,
            resolution="preinstalled",
            settings={
                "langgraph": {
                    "entrypoint": {
                        "target": "examples.langgraph.case_review_agent.graph:build_graph",
                        "kind": "factory",
                        "search_paths": ["."],
                    },
                    "input": {"mode": "messages"},
                    "output": {"mode": "field", "field": "answer"},
                    "state": "adapter",
                    "agent": {
                        "system_prompt": "You are a case reviewer. Be concise and cite findings.",
                        "max_findings": 3,
                    },
                }
            },
        ),
        models={
            "default": ModelConfig(
                provider="nvidia",
                model="meta/llama-3.3-70b-instruct",
                temperature=0.0,
                api_key_env="NVIDIA_API_KEY",
            )
        },
        runtime=RuntimeConfig(
            input_schema="text", output_schema="message", artifacts=f"{ARTIFACTS}/case-review"
        ),
        environment=EnvironmentConfig(provider="local", artifacts=f"{ARTIFACTS}/case-review"),
    )
    # Fabric cannot enforce this policy for a compiled graph; it can for a factory
    # graph that builds its tools from the adapter context.
    config.block_tools("lookup_precedent")
    return config
