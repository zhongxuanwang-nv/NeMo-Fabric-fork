# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed consumer configuration for the first-party Pi adapter example."""

from __future__ import annotations

from pathlib import Path

from nemo_fabric import DiscoveryConfig
from nemo_fabric import EnvironmentConfig
from nemo_fabric import FabricConfig
from nemo_fabric import HarnessConfig
from nemo_fabric import MetadataConfig
from nemo_fabric import ModelConfig
from nemo_fabric import RuntimeConfig
from nemo_fabric import SkillConfig

ROOT = Path(__file__).resolve().parents[2]
DESCRIPTOR = (ROOT / "adapters/pi/pi.fabric-adapter.json").resolve()
WORKSPACE = (ROOT / "examples/code_review_agent/repos/my-service").resolve()
CODE_REVIEW_SKILL = (ROOT / "examples/code_review_agent/skills/code-review").resolve()
ARTIFACTS = (Path(__file__).parent / "artifacts").resolve()


def code_review_config(pi_executable: Path) -> FabricConfig:
    """Return the complete Pi code-review configuration."""

    if not pi_executable.is_absolute():
        raise ValueError("pi_executable must be an absolute path")

    return FabricConfig(
        metadata=MetadataConfig(
            name="pi-jsonl-rpc-code-review",
            description="Reviews the sample Python service through Pi JSONL RPC.",
        ),
        discovery=DiscoveryConfig(local_paths=[DESCRIPTOR]),
        harness=HarnessConfig(
            adapter_id="nvidia.fabric.pi",
            resolution="preinstalled",
            settings={"pi_executable": str(pi_executable)},
        ),
        models={
            "default": ModelConfig(
                provider="nvidia",
                model="nvidia/nemotron-3-super-120b-a12b",
                api_key_env="NVIDIA_API_KEY",
            )
        },
        skills=SkillConfig(paths=[CODE_REVIEW_SKILL]),
        runtime=RuntimeConfig(
            input_schema="text",
            output_schema="message",
            artifacts=ARTIFACTS,
        ),
        environment=EnvironmentConfig(
            provider="local",
            workspace=WORKSPACE,
            artifacts=ARTIFACTS,
        ),
    )
