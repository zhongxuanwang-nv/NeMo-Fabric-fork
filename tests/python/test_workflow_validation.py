# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Target-driven workflow discovery, validation, and projection tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from nemo_fabric import DiscoveryConfig
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import FabricConfigError
from nemo_fabric import HarnessConfig
from nemo_fabric import MetadataConfig
from nemo_fabric import WorkflowConfig


TARGET_ID = "test.fabric.workflow.email"
ADAPTER_ID = "test.fabric.workflow"


def _settings_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"llm_name": {"type": "string"}},
        "required": ["llm_name"],
        "additionalProperties": False,
    }


def _write_descriptors(
    base_dir: Path,
    *,
    target_adapter_id: str = ADAPTER_ID,
    target_types: list[str] | None = None,
) -> tuple[Path, Path]:
    root = base_dir / "descriptors"
    root.mkdir(parents=True, exist_ok=True)
    adapter_path = root / "workflow.fabric-adapter.json"
    adapter_path.write_text(
        json.dumps(
            {
                "contract_version": "fabric.adapter/v1alpha2",
                "adapter_id": ADAPTER_ID,
                "adapter_kind": "python",
                "runner": {"module": "test.fabric.workflow"},
                "target_types": ["workflow"] if target_types is None else target_types,
            }
        ),
        encoding="utf-8",
    )
    target_path = root / "email.fabric-target.json"
    target_path.write_text(
        json.dumps(
            {
                "contract_version": "fabric.adapter/v1alpha2",
                "id": TARGET_ID,
                "adapter_id": target_adapter_id,
                "type": "workflow",
                "spec": {
                    "entrypoint": {"kind": "factory", "ref": "email_analyzer"},
                    "settings_schema": _settings_schema(),
                },
            }
        ),
        encoding="utf-8",
    )
    return adapter_path, target_path


def _config(
    base_dir: Path,
    *,
    adapter_id: str | None = None,
    target_id: str = TARGET_ID,
    **settings: str | int | float | bool | None,
) -> FabricConfig:
    return FabricConfig(
        metadata=MetadataConfig(name="workflow-test"),
        harness=(
            HarnessConfig(adapter_id=adapter_id, resolution="preinstalled")
            if adapter_id is not None
            else None
        ),
        workflow=WorkflowConfig(target_id=target_id, settings=settings),
        discovery=DiscoveryConfig(local_paths=[base_dir / "descriptors"]),
    )


def test_target_selects_adapter_validates_settings_and_projects_entrypoint(tmp_path: Path):
    adapter_path, target_path = _write_descriptors(tmp_path)
    config = _config(tmp_path, llm_name="default")

    plan = Fabric().plan(config, base_dir=tmp_path)

    assert plan.config.workflow == config.to_mapping()["workflow"]
    assert plan["adapter_descriptor"]["descriptor"]["adapter_id"] == ADAPTER_ID
    assert Path(plan["adapter_descriptor"]["provenance"][0]["path"]).samefile(adapter_path)
    assert plan["adapter_target_descriptor"]["descriptor"]["id"] == TARGET_ID
    assert Path(plan["adapter_target_descriptor"]["provenance"][0]["path"]).samefile(target_path)
    assert plan["agent_config"]["workflow"] == {
        "entrypoint": {"kind": "factory", "ref": "email_analyzer"},
        "settings": {"llm_name": "default"},
    }


def test_target_schema_reports_exact_invalid_setting_path(tmp_path: Path):
    _write_descriptors(tmp_path)

    with pytest.raises(FabricConfigError, match=r"workflow\.settings\.llm_name"):
        Fabric().plan(_config(tmp_path, llm_name=7), base_dir=tmp_path)


def test_matching_harness_selector_is_allowed(tmp_path: Path):
    _write_descriptors(tmp_path)

    plan = Fabric().plan(
        _config(tmp_path, adapter_id=ADAPTER_ID, llm_name="default"),
        base_dir=tmp_path,
    )

    assert plan["adapter_descriptor"]["descriptor"]["adapter_id"] == ADAPTER_ID


def test_mismatched_harness_selector_is_rejected(tmp_path: Path):
    _write_descriptors(tmp_path)

    with pytest.raises(FabricConfigError, match=r"harness\.adapter_id"):
        Fabric().plan(
            _config(tmp_path, adapter_id="nvidia.fabric.claude", llm_name="default"),
            base_dir=tmp_path,
        )


def test_unknown_target_is_rejected(tmp_path: Path):
    _write_descriptors(tmp_path)

    with pytest.raises(FabricConfigError, match="unknown adapter target"):
        Fabric().plan(
            _config(tmp_path, target_id="test.fabric.missing", llm_name="default"),
            base_dir=tmp_path,
        )


def test_target_with_unknown_adapter_is_rejected(tmp_path: Path):
    _write_descriptors(tmp_path, target_adapter_id="test.fabric.missing")

    with pytest.raises(FabricConfigError, match="unknown adapter"):
        Fabric().plan(_config(tmp_path, llm_name="default"), base_dir=tmp_path)


def test_adapter_must_advertise_workflow_targets(tmp_path: Path):
    _write_descriptors(tmp_path, target_types=[])

    with pytest.raises(FabricConfigError, match="target_types"):
        Fabric().plan(_config(tmp_path, llm_name="default"), base_dir=tmp_path)
