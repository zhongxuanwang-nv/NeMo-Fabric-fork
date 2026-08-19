<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Pi Adapter

Use the `nvidia.fabric.pi` adapter to run [Pi](https://pi.dev) 0.84.2 through
its persistent JSONL RPC mode with NVIDIA NeMo Fabric.

## Install

The following table shows which components each installation provides:

| Installation | Runtime | Adapter | Pi and Node.js |
| --- | --- | --- | --- |
| `pip install "nemo-fabric[pi]"` | Yes | Yes | Yes |
| `pip install "nemo-fabric-adapters-pi[harness]"` | No | Yes | Yes |
| `pip install "nemo-fabric-adapters-pi[full]"` | No | Yes | Yes |
| `pip install nemo-fabric-adapters-pi` | No | Yes | No |

The `harness` and `full` extras install the first-party
`nemo-fabric-pi-cli-bin` distribution. That package vendors the
shrinkwrap-locked official
`@earendil-works/pi-coding-agent` 0.84.2 npm distribution. Installation does
not run npm lifecycle scripts. A Fabric runtime never invokes `npx`, searches
`PATH` for Pi, or installs packages over the network.

The extras support Linux x86_64 and arm64 with glibc 2.28 or newer, Linux
musl x86_64 and arm64, macOS 13.5 or newer on x86_64 and arm64, and Windows
x86_64 and arm64. Install the bare adapter when an execution environment
provides another reviewed Pi 0.84.2 executable.

## Resolve Explicit Paths

Resolve the installed executable and descriptor without searching `PATH`:

```python
import os
import sysconfig
from pathlib import Path

scripts = Path(sysconfig.get_path("scripts")).resolve()
pi_executable = scripts / ("nemo-fabric-pi.exe" if os.name == "nt" else "nemo-fabric-pi")
descriptor = (
    Path(sysconfig.get_path("data")).resolve()
    / "share/nemo-fabric/adapters/pi/pi.fabric-adapter.json"
)
```

Pass absolute local paths for the Pi executable, descriptor, workspace,
artifact root, and every skill directory. A skill directory must contain a
`SKILL.md` file.

## Configure the Adapter

The adapter accepts exactly one model role named `default`. It maps only these
portable fields:

| NeMo Fabric Field | Pi Mapping |
| --- | --- |
| `models.default.provider` | `--provider` |
| `models.default.model` | `--model` |
| `models.default.api_key_env` | Runtime-local environment reference in `auth.json` |
| `skills.paths[]` | `--no-skills` and one `--skill <absolute-path>` per directory |

The closed model schema rejects `base_url`, `temperature`, model settings, and
extensions. The adapter does not map instructions, tools, MCP, telemetry,
streaming, updates, cancellation, or service behavior. Pi's ambient extension
discovery remains Pi-native and is outside this adapter's normalized surface.

The following example uses only explicit absolute paths:

```python
import asyncio
import os
import sysconfig
from pathlib import Path

from nemo_fabric import DiscoveryConfig, EnvironmentConfig, Fabric, FabricConfig
from nemo_fabric import HarnessConfig, MetadataConfig, ModelConfig, RuntimeConfig
from nemo_fabric import SkillConfig

scripts = Path(sysconfig.get_path("scripts")).resolve()
pi_executable = scripts / ("nemo-fabric-pi.exe" if os.name == "nt" else "nemo-fabric-pi")
descriptor = (
    Path(sysconfig.get_path("data")).resolve()
    / "share/nemo-fabric/adapters/pi/pi.fabric-adapter.json"
)
workspace = Path("/absolute/path/to/workspace").resolve()
artifacts = Path("/absolute/path/to/artifacts").resolve()
skill = Path("/absolute/path/to/code-review-skill").resolve()

config = FabricConfig(
    metadata=MetadataConfig(name="pi-agent"),
    discovery=DiscoveryConfig(local_paths=[descriptor]),
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
    skills=SkillConfig(paths=[skill]),
    runtime=RuntimeConfig(artifacts=artifacts, timeout_seconds=900),
    environment=EnvironmentConfig(
        provider="local",
        workspace=workspace,
        artifacts=artifacts,
    ),
)


async def main():
    fabric = Fabric()
    fabric.plan(config, base_dir=workspace)
    await fabric.doctor(config, base_dir=workspace)
    async with await fabric.start_runtime(config, base_dir=workspace) as runtime:
        first = await runtime.invoke(input="Review the workspace.")
        second = await runtime.invoke(input="Summarize the earlier finding.")
        return first, second


first, second = asyncio.run(main())
```

Set the environment variable named by `api_key_env` before starting the
runtime. The adapter first checks `RuntimeContext.environment.env`, then the
host environment. It passes only platform execution variables, Pi's isolated
state settings, and the selected credential to the child. The credential is
never written to `auth.json`; that file contains only an environment-variable
reference.

## Lifecycle and Failure Behavior

One adapter runtime owns one Pi process. It reuses that process for ordered
invocations and isolates independent runtimes. Each runtime uses a private,
hashed state directory below its artifact root.

The adapter enforces bounded command and turn deadlines, strict LF-delimited
JSONL, response ID and command matching, and `agent_settled` as the completion
boundary. A failed or aborted agent turn returns a failed normalized result.
Malformed records, oversized records, EOF, correlation failures, and timeouts
fail the lifecycle and close the process. Stop closes stdin, waits for exit,
terminates the process group when necessary, and kills it only as a final
fallback.

The adapter does not claim automated NVIDIA NeMo Fabric conformance. The public
conformance suite is not yet available.

## Run the Maintained Example

The [Pi JSONL-RPC code-review example](../../examples/pi_jsonl_rpc_code_review/README.md)
plans and runs the adapter against the shared calculator fixture. The live test
is opt-in and reads its API key only from the environment.
