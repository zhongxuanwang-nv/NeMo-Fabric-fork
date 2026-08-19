# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in live E2E coverage for the first-party Pi adapter.

Refer to ``examples/pi_jsonl_rpc_code_review/README.md`` for setup and the
credential-safe command.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from examples.pi_jsonl_rpc_code_review.config import ROOT
from examples.pi_jsonl_rpc_code_review.config import code_review_config
from nemo_fabric import Fabric


def _live_pi_executable() -> Path:
    if os.environ.get("RUN_FABRIC_PI_INTEGRATION") != "1":
        pytest.skip("set RUN_FABRIC_PI_INTEGRATION=1 to run Pi integration")
    if not os.environ.get("NVIDIA_API_KEY"):
        pytest.skip("NVIDIA_API_KEY is required for Pi integration")
    value = os.environ.get("FABRIC_TEST_PI_EXECUTABLE")
    if value is None:
        pytest.skip("FABRIC_TEST_PI_EXECUTABLE must name an absolute Pi executable")
    executable = Path(value)
    if not executable.is_absolute() or not executable.is_file():
        pytest.skip("FABRIC_TEST_PI_EXECUTABLE must name an absolute Pi executable")
    return executable


async def test_live_pi_code_review_reuses_one_process_and_stops(tmp_path: Path):
    executable = _live_pi_executable()
    config = code_review_config(executable)
    artifacts = tmp_path / "artifacts"
    config.runtime.artifacts = artifacts
    assert config.environment is not None
    config.environment.artifacts = artifacts
    prompt = (
        "/skill:code-review Review calculator.py. For this exercise, answer() is "
        "required to return 42. Identify the correctness defect, cite the file and "
        "line, and propose the narrowest regression test."
    )

    async with await Fabric().start_runtime(config, base_dir=ROOT) as runtime:
        first = await runtime.invoke(input=prompt)
        second = await runtime.invoke(
            input="State the earlier defect and its regression assertion in one sentence."
        )
        first_output = first.to_mapping()["output"]
        second_output = second.to_mapping()["output"]
        process_id = first_output["pi_process_id"]

        assert second_output["pi_process_id"] == process_id
        assert "calculator.py" in first_output["response"]
        assert "41" in first_output["response"]
        assert "42" in first_output["response"]
        assert "assert" in second_output["response"]

    with pytest.raises(ProcessLookupError):
        os.kill(process_id, 0)
