<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Pi JSONL-RPC Code-Review Example

This example uses the first-party `nvidia.fabric.pi` adapter to map one typed
model and explicit local skill directories into one Pi process per NVIDIA NeMo
Fabric runtime. It does not define any other normalized capability mapping.

Run the following commands from the repository root. They create a Pi-specific
environment without the unrelated Hermes Agent checkout prerequisite:

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
```

For a live code review, export `NVIDIA_API_KEY` before running the final command:

```bash
PYTHONPATH="$PWD" ADAPTER_PYTHON="$pi_python" \
  "$pi_python" -m examples.pi_jsonl_rpc_code_review \
  --pi-executable "$pi_executable"
```

Use the opt-in E2E test after setting the same credential and absolute Pi path:

```bash
uv pip install --python "$pi_python" pytest pytest-asyncio requests
RUN_FABRIC_PI_INTEGRATION=1 \
FABRIC_TEST_PI_EXECUTABLE="$pi_executable" \
NVIDIA_API_KEY="$NVIDIA_API_KEY" \
PYTHONPATH="$PWD" ADAPTER_PYTHON="$pi_python" "$pi_python" -m pytest \
  tests/e2e/test_pi_jsonl_rpc_code_review.py -q
```

The executable, descriptor, workspace, artifact root, and skill paths in this
example are explicit absolute local paths. The live test reads the API key only
from the environment, reuses one Pi process for two invocations, and verifies
that runtime stop removes the process.
