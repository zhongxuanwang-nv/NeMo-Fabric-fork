# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed NVIDIA NeMo Fabric adapter for Pi's JSONL RPC mode."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import signal
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentModelConfig
from nemo_fabric_adapter_contract.models import AgentRunError
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import AgentRunResult
from nemo_fabric_adapter_contract.models import AgentRunStatus
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.common import lifecycle

_MAX_RECORD_BYTES = 16 * 1024 * 1024
_RPC_COMMAND_TIMEOUT_SECONDS = 30.0
_TURN_SETTLE_TIMEOUT_SECONDS = 600.0
_PROCESS_EXIT_TIMEOUT_SECONDS = 2.0
_PLATFORM_ENV_NAMES = (
    "COMSPEC",
    "LANG",
    "LC_ALL",
    "PATH",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "SYSTEMROOT",
    "SystemRoot",
    "TEMP",
    "TMP",
    "TMPDIR",
    "WINDIR",
)


def _error(code: str, message: str) -> lifecycle.LifecycleError:
    return lifecycle.LifecycleError(code, message)


@dataclass(frozen=True, slots=True)
class PiModelSelection:
    """The supported typed subset of one Fabric model role."""

    provider: str
    model_id: str
    api_key_env: str
    api_key: str = field(repr=False, compare=False)

    @classmethod
    def from_fabric(
        cls,
        config: AgentConfig,
        environment: Mapping[str, str],
    ) -> PiModelSelection:
        if set(config.models) != {"default"}:
            raise _error(
                "pi_invalid_models",
                "Pi requires exactly one model role named 'default'",
            )
        model: AgentModelConfig = config.models["default"]
        if (
            model.api_key_env is None
            or model.base_url is not None
            or model.temperature is not None
            or model.settings
            or model.extensions
        ):
            raise _error(
                "pi_unsupported_model_configuration",
                "Pi supports provider, model, and api_key_env only",
            )
        api_key = environment.get(model.api_key_env)
        if api_key is None:
            api_key = os.environ.get(model.api_key_env)
        if not api_key:
            raise _error(
                "pi_missing_model_credential",
                f"{model.api_key_env} is required for the selected Pi model",
            )
        return cls(
            provider=model.provider,
            model_id=model.model,
            api_key_env=model.api_key_env,
            api_key=api_key,
        )


@dataclass(frozen=True, slots=True)
class PiLaunchSpec:
    """Fully resolved local paths and typed model mapping for one Pi process."""

    executable: Path
    workspace: Path
    state_dir: Path
    model: PiModelSelection
    skill_paths: tuple[Path, ...]

    @classmethod
    def from_fabric(
        cls,
        config: AgentConfig,
        context: RuntimeContext,
    ) -> PiLaunchSpec:
        settings = config.harness.settings if config.harness is not None else {}
        executable = _absolute_path(settings.get("pi_executable"), "pi_executable")
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise _error(
                "pi_executable_invalid",
                "pi_executable must be an executable local file",
            )

        workspace = _absolute_path(
            context.environment.workspace,
            "environment.workspace",
        )
        if not workspace.is_dir():
            raise _error(
                "pi_workspace_invalid",
                "environment.workspace must be an existing local directory",
            )

        artifact_root = _absolute_path(context.artifacts.root, "runtime.artifacts")
        if artifact_root.exists() and not artifact_root.is_dir():
            raise _error(
                "pi_artifact_root_invalid",
                "runtime.artifacts must be a local directory path",
            )
        model = PiModelSelection.from_fabric(config, context.environment.env)

        skill_paths: list[Path] = []
        for raw_path in config.skills.paths if config.skills is not None else []:
            skill_path = _absolute_path(raw_path, "skills.paths")
            if not skill_path.is_dir() or not (skill_path / "SKILL.md").is_file():
                raise _error(
                    "pi_skill_path_invalid",
                    "every Pi skill path must be a local directory containing SKILL.md",
                )
            skill_paths.append(skill_path)

        state_dir = _runtime_state_dir(artifact_root, context.runtime_id)

        return cls(
            executable=executable,
            workspace=workspace,
            state_dir=state_dir,
            model=model,
            skill_paths=tuple(skill_paths),
        )

    def argv(self) -> list[str]:
        argv = [
            str(self.executable),
            "--mode",
            "rpc",
            "--no-session",
            "--provider",
            self.model.provider,
            "--model",
            self.model.model_id,
            "--no-skills",
        ]
        for path in self.skill_paths:
            argv.extend(("--skill", str(path)))
        return argv


def _absolute_path(value: Any, name: str) -> Path:
    if value is None:
        raise _error("pi_path_required", f"{name} is required")
    path = Path(str(value))
    if not path.is_absolute():
        raise _error("pi_path_not_absolute", f"{name} must be an absolute path")
    try:
        return path.resolve()
    except OSError as error:
        raise _error("pi_path_invalid", f"{name} cannot be resolved") from error


def _runtime_state_dir(artifact_root: Path, runtime_id: str) -> Path:
    """Create one contained, private state directory for an opaque runtime ID."""

    try:
        artifact_root.mkdir(parents=True, exist_ok=True)
        artifact_root = artifact_root.resolve(strict=True)
        if not artifact_root.is_dir():
            raise _error(
                "pi_artifact_root_invalid",
                "runtime.artifacts must be a local directory path",
            )
        fabric_root = artifact_root / ".fabric"
        if fabric_root.is_symlink():
            raise _error(
                "pi_state_root_invalid",
                "Pi state root must remain below runtime.artifacts",
            )
        fabric_root.mkdir(exist_ok=True)
        fabric_root = fabric_root.resolve(strict=True)
        if fabric_root.parent != artifact_root or not fabric_root.is_dir():
            raise _error(
                "pi_state_root_invalid",
                "Pi state root must remain below runtime.artifacts",
            )
        state_root = fabric_root / "pi"
        if state_root.is_symlink():
            raise _error(
                "pi_state_root_invalid",
                "Pi state root must remain below runtime.artifacts",
            )
        state_root.mkdir(parents=True, exist_ok=True)
        state_root = state_root.resolve(strict=True)
        if not state_root.is_dir() or not state_root.is_relative_to(artifact_root):
            raise _error(
                "pi_state_root_invalid",
                "Pi state root must remain below runtime.artifacts",
            )
        component = hashlib.sha256(runtime_id.encode("utf-8")).hexdigest()
        state_dir = state_root / component
        state_dir.mkdir(mode=0o700)
        state_dir = state_dir.resolve(strict=True)
    except FileExistsError as error:
        raise _error(
            "pi_state_dir_exists",
            "Pi state directory already exists for this runtime",
        ) from error
    except OSError as error:
        raise _error(
            "pi_state_dir_invalid",
            "Pi state directory cannot be created",
        ) from error

    if state_dir.parent != state_root:
        raise _error(
            "pi_state_dir_invalid",
            "Pi state directory must remain under runtime.artifacts",
        )
    try:
        os.chmod(state_dir, 0o700)
    except OSError as error:
        raise _error(
            "pi_state_dir_invalid",
            "Pi state directory permissions cannot be set",
        ) from error
    return state_dir


class PiRpcProcess:
    """One Pi JSONL-RPC child process owned by one Fabric runtime."""

    def __init__(self, spec: PiLaunchSpec) -> None:
        self.spec = spec
        self.process: asyncio.subprocess.Process | None = None
        self._next_id = 0

    async def start(self) -> None:
        self._write_auth_reference()
        environment = {
            name: value
            for name in _PLATFORM_ENV_NAMES
            if (value := os.environ.get(name)) is not None
        }
        environment[self.spec.model.api_key_env] = self.spec.model.api_key
        environment["PI_CODING_AGENT_DIR"] = str(self.spec.state_dir)
        environment["PI_OFFLINE"] = "1"
        environment["PI_SKIP_VERSION_CHECK"] = "1"
        self.process = await asyncio.create_subprocess_exec(
            *self.spec.argv(),
            cwd=self.spec.workspace,
            env=environment,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            limit=_MAX_RECORD_BYTES + 2,
            start_new_session=os.name != "nt",
        )

        state, _ = await self.request("get_state")
        selected = state.get("data", {}).get("model") or {}
        if (
            selected.get("provider") != self.spec.model.provider
            or selected.get("id") != self.spec.model.model_id
        ):
            raise _error(
                "pi_model_mapping_failed",
                "Pi did not select the configured Fabric model",
            )

        commands, _ = await self.request("get_commands")
        loaded_skills = {
            Path(command["sourceInfo"]["path"]).resolve()
            for command in commands.get("data", {}).get("commands", [])
            if command.get("source") == "skill"
            and isinstance(command.get("sourceInfo", {}).get("path"), str)
        }
        expected_skills = {
            (path / "SKILL.md").resolve() for path in self.spec.skill_paths
        }
        if not expected_skills.issubset(loaded_skills):
            raise _error(
                "pi_skill_mapping_failed",
                "Pi did not load every configured Fabric skill path",
            )

    def _write_auth_reference(self) -> None:
        auth_path = self.spec.state_dir / "auth.json"
        value = {
            self.spec.model.provider: {
                "type": "api_key",
                "key": "${" + self.spec.model.api_key_env + "}",
            }
        }
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(auth_path, flags, 0o600)
        except FileExistsError as error:
            raise _error(
                "pi_auth_file_exists",
                "Pi authentication state already exists for this runtime",
            ) from error
        except OSError as error:
            raise _error(
                "pi_auth_file_invalid",
                "Pi authentication state cannot be created",
            ) from error
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                descriptor = -1
                json.dump(value, stream)
                stream.write("\n")
        except OSError as error:
            raise _error(
                "pi_auth_file_invalid",
                "Pi authentication state cannot be written",
            ) from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    async def invoke(self, prompt: str) -> str | None:
        try:
            _, events = await self.request("prompt", message=prompt)
            await self._wait_for_settlement(events)
            if not _agent_succeeded(events):
                return None

            response, _ = await self.request("get_last_assistant_text")
            text = response.get("data", {}).get("text")
            if not isinstance(text, str):
                raise _error(
                    "pi_missing_response",
                    "Pi completed without a final assistant text response",
                )
            return text
        except lifecycle.LifecycleError:
            await self.close()
            raise

    async def request(
        self,
        command: str,
        timeout_seconds: float | None = None,
        **fields: Any,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if timeout_seconds is None:
            timeout_seconds = _RPC_COMMAND_TIMEOUT_SECONDS
        self._next_id += 1
        request_id = f"fabric-{self._next_id}"
        events: list[dict[str, Any]] = []
        try:
            await self._write_record(
                {"id": request_id, "type": command, **fields},
                timeout_seconds,
            )
            async with asyncio.timeout(timeout_seconds):
                while True:
                    record = await self._read_record()
                    if record.get("type") != "response":
                        events.append(record)
                        continue
                    if record.get("id") != request_id:
                        raise _error(
                            "pi_rpc_correlation_failed",
                            "Pi returned an unexpected JSONL RPC response id",
                        )
                    if record.get("command") != command:
                        raise _error(
                            "pi_rpc_command_mismatch",
                            "Pi returned a response for the wrong JSONL RPC command",
                        )
                    if record.get("success") is not True:
                        raise _error(
                            "pi_rpc_command_failed",
                            f"Pi rejected the {command!r} RPC command",
                        )
                    return record, events
        except TimeoutError as error:
            await self.close()
            raise _error(
                "pi_rpc_timeout",
                f"Pi did not respond to the {command!r} JSONL RPC command before the deadline",
            ) from error
        except lifecycle.LifecycleError:
            await self.close()
            raise

    async def _wait_for_settlement(self, events: list[dict[str, Any]]) -> None:
        if any(event.get("type") == "agent_settled" for event in events):
            return
        try:
            async with asyncio.timeout(_TURN_SETTLE_TIMEOUT_SECONDS):
                while True:
                    record = await self._read_record()
                    events.append(record)
                    if record.get("type") == "agent_settled":
                        return
        except TimeoutError as error:
            await self.close()
            raise _error(
                "pi_turn_timeout",
                "Pi did not settle the invocation before the deadline",
            ) from error

    async def _write_record(
        self,
        record: dict[str, Any],
        timeout_seconds: float,
    ) -> None:
        process = self.process
        if process is None or process.stdin is None or process.returncode is not None:
            raise _error("pi_process_unavailable", "Pi process is not running")
        process.stdin.write(json.dumps(record, separators=(",", ":")).encode() + b"\n")
        try:
            await asyncio.wait_for(process.stdin.drain(), timeout_seconds)
        except TimeoutError as error:
            await self.close()
            raise _error(
                "pi_rpc_timeout",
                "Pi did not accept a JSONL RPC command before the POC deadline",
            ) from error
        except (BrokenPipeError, ConnectionResetError) as error:
            raise _error(
                "pi_rpc_write_failed", "Pi closed its JSONL RPC input"
            ) from error

    async def _read_record(self) -> dict[str, Any]:
        process = self.process
        if process is None or process.stdout is None:
            raise _error("pi_process_unavailable", "Pi process is not running")
        try:
            raw = await process.stdout.readuntil(b"\n")
        except asyncio.LimitOverrunError as error:
            raise _error(
                "pi_rpc_record_too_large", "Pi JSONL record is too large"
            ) from error
        except asyncio.IncompleteReadError as error:
            if error.partial:
                raise _error(
                    "pi_rpc_incomplete_record",
                    "Pi closed its JSONL RPC output before an LF delimiter",
                ) from error
            raise _error("pi_rpc_eof", "Pi closed its JSONL RPC output") from error
        body = raw[:-1]
        if body.endswith(b"\r"):
            body = body[:-1]
        if len(body) > _MAX_RECORD_BYTES:
            raise _error("pi_rpc_record_too_large", "Pi JSONL record is too large")
        try:
            value = json.loads(body.decode("utf-8"), parse_constant=_invalid_constant)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise _error("pi_rpc_invalid_json", "Pi returned invalid JSONL") from error
        if not isinstance(value, dict):
            raise _error("pi_rpc_invalid_record", "Pi JSONL record must be an object")
        return value

    async def close(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
            try:
                await process.stdin.wait_closed()
            except (BrokenPipeError, ConnectionResetError):
                pass
        if await _wait_for_exit(process):
            return
        _signal_process_group(process, signal.SIGTERM)
        if await _wait_for_exit(process):
            return
        _signal_process_group(process, signal.SIGKILL)
        if not await _wait_for_exit(process):
            raise _error("pi_process_stop_failed", "Pi process did not stop")


async def _wait_for_exit(process: asyncio.subprocess.Process) -> bool:
    try:
        await asyncio.wait_for(process.wait(), _PROCESS_EXIT_TIMEOUT_SECONDS)
    except TimeoutError:
        return False
    if process.stdout is not None:
        try:
            await asyncio.wait_for(
                process.stdout.read(),
                _PROCESS_EXIT_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            return False
    return True


def _signal_process_group(
    process: asyncio.subprocess.Process, signal_value: int
) -> None:
    if process.returncode is not None:
        return
    if os.name == "nt":
        if signal_value == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
        return
    try:
        os.killpg(process.pid, signal_value)
    except ProcessLookupError:
        if signal_value == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()


def _agent_succeeded(events: list[dict[str, Any]]) -> bool:
    """Return whether the final Pi agent run ended with an assistant response."""

    for event in reversed(events):
        if event.get("type") != "agent_end":
            continue
        messages = event.get("messages")
        if not isinstance(messages, list):
            continue
        assistants = [
            message
            for message in messages
            if isinstance(message, dict) and message.get("role") == "assistant"
        ]
        if not assistants:
            return False
        return assistants[-1].get("stopReason") not in {"aborted", "error"}
    return False


def _invalid_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


class PiRuntime:
    """Fabric lifecycle binding that owns exactly one Pi process."""

    def __init__(self) -> None:
        self._runtime_id: str | None = None
        self._client: PiRpcProcess | None = None

    async def start(self, payload: dict[str, Any]) -> None:
        if self._client is not None:
            raise _error("pi_runtime_already_started", "Pi runtime is already started")
        config = payload.get("config")
        if not isinstance(config, AgentConfig):
            raise _error("pi_invalid_config", "Pi requires a validated AgentConfig")
        context = RuntimeContext.from_mapping(payload.get("runtime_context"))
        client = PiRpcProcess(PiLaunchSpec.from_fabric(config, context))
        self._client = client
        self._runtime_id = context.runtime_id
        try:
            await client.start()
        except BaseException:
            await self.stop()
            raise

    async def invoke(
        self,
        request: AgentRunRequest,
        context: RuntimeContext,
    ) -> AgentRunResult:
        client = self._client
        if client is None or client.process is None or self._runtime_id is None:
            raise _error("pi_runtime_not_started", "Pi runtime is not started")
        if context.runtime_id != self._runtime_id:
            raise _error(
                "pi_runtime_mismatch",
                "Pi invocation does not match the active Fabric runtime",
            )
        prompt = request.input or ""
        if not isinstance(prompt, str):
            prompt = json.dumps(prompt, sort_keys=True)
        process_id = client.process.pid
        response = await client.invoke(prompt)
        if response is None:
            return AgentRunResult(
                status=AgentRunStatus.FAILED,
                output=None,
                error=AgentRunError(
                    code="pi_invocation_failed",
                    message="Pi did not complete the invocation",
                    retryable=False,
                ),
            )
        return AgentRunResult(
            status=AgentRunStatus.SUCCEEDED,
            output={
                "response": response,
                "pi_process_id": process_id,
                "model": {
                    "provider": client.spec.model.provider,
                    "model": client.spec.model.model_id,
                },
                "skills": [str(path) for path in client.spec.skill_paths],
            },
        )

    async def stop(self) -> None:
        client = self._client
        self._client = None
        self._runtime_id = None
        if client is not None:
            await client.close()


def main() -> None:
    lifecycle.serve(PiRuntime, config_loader=AgentConfig.from_mapping)


if __name__ == "__main__":
    main()
