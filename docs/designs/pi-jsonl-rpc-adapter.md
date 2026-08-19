<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Pi JSONL-RPC Adapter Plan for NVIDIA NeMo Fabric

**Status:** Implemented as `nvidia.fabric.pi` in
`nemo-fabric-adapters-pi`; this document records the source POC and ship
decision.

## Decision

Use the existing Python local-host lifecycle. One host owns exactly one local Pi
child for each NeMo Fabric runtime and communicates through
[`pi --mode rpc`](https://pi.dev/docs/latest/rpc). Reuse that process for ordered
invocations and close it during runtime stop. This follows the persistent runtime
pattern of every bundled adapter; the Codex adapter is the closest JSON-RPC
precedent.

The source POC was checked with Pi `0.84.2`, which is also the pinned production
version. Pi RPC uses LF-delimited
JSON records. A prompt is complete at `agent_settled`; `agent_end` can precede a
retry or continuation. The POC applies fixed startup, RPC, and turn deadlines so
a stalled process is closed instead of blocking the Fabric runtime indefinitely.

## Adapter Contract

The descriptor accepts only `models` and `skills`.

| Fabric value | Pi mapping | Rejection |
| --- | --- | --- |
| `models.default.provider` | `--provider` | Missing or additional model role |
| `models.default.model` | `--model` | Missing value |
| `models.default.api_key_env` | Runtime-local `auth.json` environment reference | Missing variable |
| `skills.paths[]` | [Pi skill CLI](https://pi.dev/docs/latest/skills): `--no-skills` plus repeated `--skill <absolute-path>` | Relative path or missing `SKILL.md` |

`base_dir`, the descriptor, Pi executable, workspace, artifact root, and every
skill directory are explicit absolute local paths. The adapter resolves the
paths canonically, hashes opaque runtime IDs before creating state directories,
and creates state files privately below the artifact root. The model and loaded
skill paths are verified with `get_state` and `get_commands` before start succeeds.

Lifecycle:

1. `start`: validate typed `AgentConfig`, create runtime-local Pi state, spawn one
   `pi --mode rpc --no-session` process, and verify model and skill mappings.
2. `invoke`: send one correlated `prompt`, wait for `agent_settled`, then read
   `get_last_assistant_text`.
3. `stop`: close stdin, wait, terminate on timeout, and kill only as a final
   fallback.

The maintained example is the code-review agent in
[`examples/pi_jsonl_rpc_code_review`](../../examples/pi_jsonl_rpc_code_review/README.md).
It reuses the existing sample repository and `code-review` skill. The adapter passes
only platform execution variables and the selected model credential to Pi.

```bash
uv venv .venv-pi --python 3.12
pi_python="$PWD/.venv-pi/bin/python"
uv pip install --python "$pi_python" \
  -e sdk/python/nemo-fabric-runtime \
  -e adapter-contract/python \
  -e adapters/common \
  -e adapters/pi

pi_executable="/absolute/path/to/pi"
PYTHONPATH="$PWD" ADAPTER_PYTHON="$pi_python" \
  "$pi_python" -m examples.pi_jsonl_rpc_code_review \
  --pi-executable "$pi_executable" --plan

export NVIDIA_API_KEY="..."
PYTHONPATH="$PWD" ADAPTER_PYTHON="$pi_python" \
  "$pi_python" -m examples.pi_jsonl_rpc_code_review \
  --pi-executable "$pi_executable"
```

## Shipped Packaging

The first-party `nemo-fabric-pi-cli-bin` wheel vendors the official
shrinkwrap-locked `@earendil-works/pi-coding-agent` 0.84.2 production tree
without running npm lifecycle scripts and installs the pinned Node.js 22.19.0
wheel. The adapter's `harness` and `full` extras install that distribution and
expose `nemo-fabric-pi`. The root `nemo-fabric[pi]` extra delegates to the leaf
`harness` extra.

Runtime configuration still requires an explicit absolute Pi executable path.
The adapter never falls back to `PATH`, `npx`, or runtime network installation.
Every other normalized capability remains unsupported until separately designed
and tested.
