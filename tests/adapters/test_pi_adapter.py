# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Credential-free contract and lifecycle tests for the Pi adapter."""

from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from examples.pi_jsonl_rpc_code_review.config import ROOT
from examples.pi_jsonl_rpc_code_review.config import code_review_config
from nemo_fabric import Fabric
from nemo_fabric import FabricConfigError
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.common import lifecycle
from nemo_fabric_adapters.pi import adapter as pi_adapter

DESCRIPTOR = (ROOT / "adapters/pi/pi.fabric-adapter.json").resolve()


@pytest.fixture(name="mock_pi")
def mock_pi_fixture(tmp_path: Path) -> Path:
    executable = tmp_path / "mock-pi"
    executable.write_text(
        f"""#!{sys.executable}
import json
import os
import signal
import sys
import time
from pathlib import Path


def option(name):
    return sys.argv[sys.argv.index(name) + 1]


def send(value):
    print(json.dumps(value), flush=True)


provider = option("--provider")
model = option("--model")
mode = model.rsplit("/", 1)[-1]
skills = [sys.argv[index + 1] for index, value in enumerate(sys.argv) if value == "--skill"]
Path(".mock-pi-pid").write_text(str(os.getpid()), encoding="utf-8")
if mode == "stubborn":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if os.environ.get("NVIDIA_API_KEY") != "runtime-key" or os.environ.get("UNRELATED_SECRET"):
    raise SystemExit(91)
if os.environ.get("PI_OFFLINE") != "1" or os.environ.get("PI_SKIP_VERSION_CHECK") != "1":
    raise SystemExit(92)
if not Path(os.environ["PI_CODING_AGENT_DIR"]).is_dir():
    raise SystemExit(93)

last_text = None
for line in sys.stdin:
    request = json.loads(line)
    request_id = request["id"]
    command = request["type"]
    if command == "get_state" and mode == "noresponse":
        continue
    if command == "get_state" and mode == "eof":
        raise SystemExit(0)
    if command == "get_state" and mode == "partial":
        sys.stdout.write('{{"type":"response"')
        sys.stdout.flush()
        raise SystemExit(0)
    if command == "get_state" and mode == "malformed":
        sys.stdout.write("{{bad\\n")
        sys.stdout.flush()
        continue
    if command == "get_state" and mode == "oversized":
        sys.stdout.write('{{"value":"' + ("x" * 1024) + '"}}\\n')
        sys.stdout.flush()
        continue
    response_id = "wrong-id" if mode == "id-mismatch" and command == "get_state" else request_id
    response_command = "wrong" if mode == "command-mismatch" and command == "get_state" else command
    if command == "get_state":
        send({{
            "id": response_id,
            "type": "response",
            "command": response_command,
            "success": True,
            "data": {{"model": {{"provider": provider, "id": model}}}},
        }})
    elif command == "get_commands":
        send({{
            "id": request_id,
            "type": "response",
            "command": command,
            "success": True,
            "data": {{"commands": [
                {{"name": "skill:code-review", "source": "skill", "sourceInfo": {{"path": f"{{path}}/SKILL.md"}}}}
                for path in skills
            ]}},
        }})
    elif command == "prompt":
        send({{
            "id": request_id,
            "type": "response",
            "command": command,
            "success": True,
        }})
        if mode == "hang":
            continue
        stop_reason = "error" if mode == "error" else "aborted" if mode == "aborted" else "stop"
        if stop_reason == "stop":
            last_text = f"reviewed: {{request['message']}}"
        send({{
            "type": "agent_end",
            "messages": [{{"role": "assistant", "stopReason": stop_reason}}],
            "willRetry": False,
        }})
        send({{"type": "agent_settled"}})
    elif command == "get_last_assistant_text":
        send({{
            "id": request_id,
            "type": "response",
            "command": command,
            "success": True,
            "data": {{"text": last_text}},
        }})
if mode == "stubborn":
    while True:
        time.sleep(1)
""",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    return executable


@pytest.fixture(name="skill_path")
def skill_path_fixture(tmp_path: Path) -> Path:
    skill = tmp_path / "skills" / "code-review"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: code-review\ndescription: Reviews code.\n---\n",
        encoding="utf-8",
    )
    return skill


def _config(
    mock_pi: Path,
    skill_path: Path,
    model: str = "nvidia/runtime-key",
) -> AgentConfig:
    return AgentConfig.from_mapping(
        {
            "harness": {"settings": {"pi_executable": str(mock_pi)}},
            "models": {
                "default": {
                    "provider": "nvidia",
                    "model": model,
                    "api_key_env": "NVIDIA_API_KEY",
                }
            },
            "skills": {"paths": [str(skill_path)]},
        }
    )


def _context(
    tmp_path: Path,
    *,
    runtime_id: str = "runtime-one",
    invocation_id: str = "invocation-one",
    environment: dict[str, str] | None = None,
    artifact_root: Path | None = None,
) -> RuntimeContext:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    artifacts = artifact_root or tmp_path / "artifacts"
    return RuntimeContext.from_mapping(
        {
            "runtime_id": runtime_id,
            "invocation_id": invocation_id,
            "request_id": f"request-{invocation_id}",
            "environment": {
                "environment_id": "environment-one",
                "provider": "local",
                "control_location": "in_env_control",
                "workspace": str(workspace),
                "artifacts": str(artifacts),
                "env": environment or {},
                "ownership": "caller_owned",
            },
            "artifacts": {"root": str(artifacts)},
        }
    )


def _start(config: AgentConfig, context: RuntimeContext) -> dict[str, object]:
    return {"config": config, "runtime_context": context.to_mapping()}


def _invocation(
    context: RuntimeContext,
    text: str,
) -> tuple[AgentRunRequest, RuntimeContext]:
    return AgentRunRequest(input=text), context


def _mock_process_id(tmp_path: Path) -> int:
    return int((tmp_path / "workspace/.mock-pi-pid").read_text(encoding="utf-8"))


def _assert_process_stopped(process_id: int) -> None:
    with pytest.raises(ProcessLookupError):
        os.kill(process_id, 0)


def test_pi_descriptor_claims_only_the_portable_surface():
    descriptor = json.loads(DESCRIPTOR.read_text(encoding="utf-8"))

    assert descriptor["contract_version"] == "fabric.adapter/v1alpha2"
    assert descriptor["adapter_id"] == "nvidia.fabric.pi"
    assert descriptor["runner"] == {"module": "nemo_fabric_adapters.pi.adapter"}
    assert descriptor["config"] == {
        "input": "agent_config",
        "accepts": ["models", "skills"],
    }
    assert descriptor["model_schema"]["properties"].keys() == {
        "provider",
        "model",
        "api_key_env",
    }
    assert descriptor["model_schema"]["additionalProperties"] is False
    assert descriptor["settings_schema"]["required"] == ["pi_executable"]
    assert descriptor["settings_schema"]["additionalProperties"] is False
    assert descriptor["capabilities"] == {
        "service": False,
        "streaming": False,
        "updates": False,
        "cancellation": False,
    }
    assert "telemetry" not in descriptor


def test_pi_module_entrypoint_exits_cleanly():
    result = subprocess.run(
        [sys.executable, "-m", "nemo_fabric_adapters.pi.adapter"],
        input="",
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


async def test_runtime_reuses_one_process_and_scrubs_credentials(
    tmp_path: Path,
    mock_pi: Path,
    skill_path: Path,
):
    os.environ["NVIDIA_API_KEY"] = "host-key"
    os.environ["UNRELATED_SECRET"] = "must-not-reach-pi"
    config = _config(mock_pi, skill_path)
    first_context = _context(
        tmp_path,
        environment={"NVIDIA_API_KEY": "runtime-key"},
    )
    runtime = pi_adapter.PiRuntime()

    await runtime.start(_start(config, first_context))
    first = await runtime.invoke(*_invocation(first_context, "first review"))
    second_context = _context(
        tmp_path,
        invocation_id="invocation-two",
        environment={"NVIDIA_API_KEY": "runtime-key"},
    )
    second = await runtime.invoke(*_invocation(second_context, "second review"))
    first_output = first.output
    second_output = second.output
    process_id = first_output["pi_process_id"]

    assert first.status == "succeeded"
    assert first_output["response"] == "reviewed: first review"
    assert second_output["response"] == "reviewed: second review"
    assert second_output["pi_process_id"] == process_id
    assert first_output["model"] == {
        "provider": "nvidia",
        "model": "nvidia/runtime-key",
    }
    assert first_output["skills"] == [str(skill_path.resolve())]
    auth_paths = list((tmp_path / "artifacts").rglob("auth.json"))
    assert len(auth_paths) == 1
    assert auth_paths[0].read_text(encoding="utf-8") == (
        '{"nvidia": {"type": "api_key", "key": "${NVIDIA_API_KEY}"}}\n'
    )
    assert stat.S_IMODE(auth_paths[0].stat().st_mode) == 0o600
    assert "runtime-key" not in auth_paths[0].read_text(encoding="utf-8")

    await runtime.stop()
    _assert_process_stopped(process_id)


async def test_two_runtimes_use_isolated_processes_and_state(
    tmp_path: Path,
    mock_pi: Path,
    skill_path: Path,
):
    config = _config(mock_pi, skill_path)
    artifacts = tmp_path / "artifacts"
    first_context = _context(
        tmp_path,
        runtime_id="runtime-one",
        environment={"NVIDIA_API_KEY": "runtime-key"},
        artifact_root=artifacts,
    )
    second_context = _context(
        tmp_path,
        runtime_id="runtime-two",
        invocation_id="invocation-two",
        environment={"NVIDIA_API_KEY": "runtime-key"},
        artifact_root=artifacts,
    )
    first_runtime = pi_adapter.PiRuntime()
    second_runtime = pi_adapter.PiRuntime()

    await first_runtime.start(_start(config, first_context))
    await second_runtime.start(_start(config, second_context))
    first = await first_runtime.invoke(*_invocation(first_context, "first"))
    second = await second_runtime.invoke(*_invocation(second_context, "second"))

    assert first.output["pi_process_id"] != second.output["pi_process_id"]
    state_dirs = sorted(path.parent for path in artifacts.rglob("auth.json"))
    assert len(state_dirs) == 2
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o700 for path in state_dirs)

    await first_runtime.stop()
    await second_runtime.stop()
    _assert_process_stopped(first.output["pi_process_id"])
    _assert_process_stopped(second.output["pi_process_id"])


@pytest.mark.parametrize(
    ("model", "message"),
    [
        ("nvidia/command-mismatch", "wrong JSONL RPC command"),
        ("nvidia/id-mismatch", "unexpected JSONL RPC response id"),
    ],
)
async def test_runtime_rejects_mismatched_responses_and_cleans_up(
    tmp_path: Path,
    mock_pi: Path,
    skill_path: Path,
    model: str,
    message: str,
):
    context = _context(tmp_path, environment={"NVIDIA_API_KEY": "runtime-key"})

    with pytest.raises(lifecycle.LifecycleError, match=message):
        await pi_adapter.PiRuntime().start(
            _start(_config(mock_pi, skill_path, model=model), context)
        )

    _assert_process_stopped(_mock_process_id(tmp_path))


@pytest.mark.parametrize(
    ("model", "message", "record_limit"),
    [
        ("nvidia/malformed", "invalid JSONL", None),
        ("nvidia/oversized", "too large", 128),
        ("nvidia/eof", "closed its JSONL RPC output", None),
        ("nvidia/partial", "before an LF delimiter", None),
    ],
)
async def test_runtime_rejects_invalid_jsonl_and_cleans_up(
    tmp_path: Path,
    mock_pi: Path,
    skill_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    message: str,
    record_limit: int | None,
):
    if record_limit is not None:
        monkeypatch.setattr(pi_adapter, "_MAX_RECORD_BYTES", record_limit)
    context = _context(tmp_path, environment={"NVIDIA_API_KEY": "runtime-key"})

    with pytest.raises(lifecycle.LifecycleError, match=message):
        await pi_adapter.PiRuntime().start(
            _start(_config(mock_pi, skill_path, model=model), context)
        )

    _assert_process_stopped(_mock_process_id(tmp_path))


async def test_runtime_start_timeout_closes_the_pi_process(
    tmp_path: Path,
    mock_pi: Path,
    skill_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(pi_adapter, "_RPC_COMMAND_TIMEOUT_SECONDS", 0.01)
    context = _context(tmp_path, environment={"NVIDIA_API_KEY": "runtime-key"})

    with pytest.raises(lifecycle.LifecycleError, match="did not respond"):
        await pi_adapter.PiRuntime().start(
            _start(_config(mock_pi, skill_path, "nvidia/noresponse"), context)
        )

    _assert_process_stopped(_mock_process_id(tmp_path))


async def test_runtime_turn_timeout_stops_the_pi_process(
    tmp_path: Path,
    mock_pi: Path,
    skill_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(pi_adapter, "_TURN_SETTLE_TIMEOUT_SECONDS", 0.01)
    config = _config(mock_pi, skill_path, model="nvidia/hang")
    context = _context(tmp_path, environment={"NVIDIA_API_KEY": "runtime-key"})
    runtime = pi_adapter.PiRuntime()

    await runtime.start(_start(config, context))
    process_id = runtime._client.process.pid
    with pytest.raises(lifecycle.LifecycleError, match="did not settle"):
        await runtime.invoke(*_invocation(context, "wait forever"))

    _assert_process_stopped(process_id)
    await runtime.stop()


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group signal ordering")
async def test_stop_uses_kill_only_after_graceful_and_terminate_deadlines(
    tmp_path: Path,
    mock_pi: Path,
    skill_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(pi_adapter, "_PROCESS_EXIT_TIMEOUT_SECONDS", 0.01)
    sent_signals: list[int] = []
    signal_process_group = pi_adapter._signal_process_group

    def record_signal(process, signal_value: int) -> None:
        sent_signals.append(signal_value)
        signal_process_group(process, signal_value)

    monkeypatch.setattr(pi_adapter, "_signal_process_group", record_signal)
    context = _context(tmp_path, environment={"NVIDIA_API_KEY": "runtime-key"})
    runtime = pi_adapter.PiRuntime()
    await runtime.start(
        _start(_config(mock_pi, skill_path, "nvidia/stubborn"), context)
    )
    process_id = runtime._client.process.pid

    await runtime.stop()

    assert sent_signals == [signal.SIGTERM, signal.SIGKILL]
    _assert_process_stopped(process_id)


@pytest.mark.parametrize("model", ["nvidia/error", "nvidia/aborted"])
async def test_runtime_returns_failed_result_for_failed_terminal_turn(
    tmp_path: Path,
    mock_pi: Path,
    skill_path: Path,
    model: str,
):
    config = _config(mock_pi, skill_path, model=model)
    context = _context(tmp_path, environment={"NVIDIA_API_KEY": "runtime-key"})
    runtime = pi_adapter.PiRuntime()

    await runtime.start(_start(config, context))
    result = await runtime.invoke(*_invocation(context, "failed review"))

    assert result.status == "failed"
    assert result.output is None
    assert result.error.code == "pi_invocation_failed"
    await runtime.stop()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("pi_executable", "relative/pi", "pi_executable must be an absolute path"),
        (
            "workspace",
            "relative/workspace",
            "environment.workspace must be an absolute path",
        ),
        (
            "artifacts",
            "relative/artifacts",
            "runtime.artifacts must be an absolute path",
        ),
        ("skill", "relative/skill", "skills.paths must be an absolute path"),
    ],
)
def test_launch_spec_requires_explicit_absolute_paths(
    tmp_path: Path,
    mock_pi: Path,
    skill_path: Path,
    field: str,
    value: str,
    message: str,
):
    config = _config(mock_pi, skill_path)
    context = _context(tmp_path, environment={"NVIDIA_API_KEY": "runtime-key"})
    if field == "pi_executable":
        config.harness.settings["pi_executable"] = value
    elif field == "skill":
        config.skills.paths = [Path(value)]
    else:
        mapping = context.to_mapping()
        if field == "workspace":
            mapping["environment"]["workspace"] = value
        else:
            mapping["artifacts"]["root"] = value
        context = RuntimeContext.from_mapping(mapping)

    with pytest.raises(lifecycle.LifecycleError, match=message):
        pi_adapter.PiLaunchSpec.from_fabric(config, context)


def test_state_path_hashes_runtime_id_and_rejects_symlink_escape(
    tmp_path: Path,
    mock_pi: Path,
    skill_path: Path,
):
    config = _config(mock_pi, skill_path)
    context = _context(
        tmp_path,
        runtime_id="../../outside",
        environment={"NVIDIA_API_KEY": "runtime-key"},
    )
    spec = pi_adapter.PiLaunchSpec.from_fabric(config, context)

    assert spec.state_dir.parent == (tmp_path / "artifacts/.fabric/pi").resolve()
    assert spec.state_dir.is_dir()
    assert not (tmp_path / "outside").exists()

    escaped_root = tmp_path / "escaped-artifacts"
    escaped_root.mkdir()
    outside = tmp_path / "outside-state"
    outside.mkdir()
    (escaped_root / ".fabric").symlink_to(outside, target_is_directory=True)
    escaped_context = _context(
        tmp_path,
        runtime_id="escape",
        environment={"NVIDIA_API_KEY": "runtime-key"},
        artifact_root=escaped_root,
    )

    with pytest.raises(lifecycle.LifecycleError, match="below runtime.artifacts"):
        pi_adapter.PiLaunchSpec.from_fabric(config, escaped_context)
    assert not (outside / "pi").exists()


def test_auth_file_creation_does_not_follow_symlinks(
    tmp_path: Path,
    mock_pi: Path,
    skill_path: Path,
):
    context = _context(tmp_path, environment={"NVIDIA_API_KEY": "runtime-key"})
    spec = pi_adapter.PiLaunchSpec.from_fabric(_config(mock_pi, skill_path), context)
    target = tmp_path / "target-auth"
    target.write_text("unchanged", encoding="utf-8")
    (spec.state_dir / "auth.json").symlink_to(target)

    with pytest.raises(lifecycle.LifecycleError, match="authentication state"):
        pi_adapter.PiRpcProcess(spec)._write_auth_reference()

    assert target.read_text(encoding="utf-8") == "unchanged"


def test_plan_rejects_relative_executable_and_extra_model_fields(mock_pi: Path):
    pytest.importorskip("nemo_fabric._native")
    config = code_review_config(mock_pi)
    config.harness.settings["pi_executable"] = "relative/pi"

    with pytest.raises(FabricConfigError, match="pi_executable"):
        Fabric().plan(config, base_dir=ROOT)

    config = code_review_config(mock_pi)
    config.models["default"].temperature = 0.0
    with pytest.raises(FabricConfigError, match="temperature"):
        Fabric().plan(config, base_dir=ROOT)


async def test_fabric_plans_diagnoses_and_runs_with_explicit_paths(
    tmp_path: Path,
    mock_pi: Path,
):
    pytest.importorskip("nemo_fabric._native")
    os.environ["ADAPTER_PYTHON"] = sys.executable
    os.environ["PYTHONPATH"] = str(ROOT)
    config = code_review_config(mock_pi)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    artifacts = tmp_path / "artifacts"
    config.discovery.local_paths = [DESCRIPTOR]
    config.runtime.artifacts = artifacts.resolve()
    config.environment.workspace = workspace.resolve()
    config.environment.artifacts = artifacts.resolve()
    config.environment.env = {"NVIDIA_API_KEY": "runtime-key"}
    fabric = Fabric()

    plan = fabric.plan(config, base_dir=ROOT)
    report = await fabric.doctor(config, base_dir=ROOT)
    async with await fabric.start_runtime(config, base_dir=ROOT) as runtime:
        first = await runtime.invoke(input="first review")
        second = await runtime.invoke(input="second review")
        process_id = first.output["pi_process_id"]

        assert second.output["pi_process_id"] == process_id

    assert plan["adapter_descriptor"]["descriptor"]["adapter_id"] == (
        "nvidia.fabric.pi"
    )
    assert Path(plan["adapter_descriptor"]["provenance"][0]["path"]).is_absolute()
    assert report.status == "pass", report
    assert first.status == "succeeded"
    _assert_process_stopped(process_id)
