<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Pi Adapter

[![License](https://img.shields.io/github/license/NVIDIA/NeMo-Fabric)](https://github.com/NVIDIA/NeMo-Fabric/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/NVIDIA/NeMo-Fabric/)
[![Release](https://img.shields.io/github/v/release/NVIDIA/NeMo-Fabric?color=green)](https://github.com/NVIDIA/NeMo-Fabric/releases)

`nemo-fabric-adapters-pi` connects NVIDIA NeMo Fabric to Pi 0.84.2 through
Pi's persistent JSONL RPC mode.

## Install

The following table shows which components each installation provides:

| Installation | Runtime | Adapter | Pi and Node.js |
| --- | --- | --- | --- |
| `pip install "nemo-fabric[pi]"` | Yes | Yes | Yes |
| `pip install "nemo-fabric-adapters-pi[harness]"` | No | Yes | Yes |
| `pip install "nemo-fabric-adapters-pi[full]"` | No | Yes | Yes |
| `pip install nemo-fabric-adapters-pi` | No | Yes | No |

The harness extras install the first-party `nemo-fabric-pi-cli-bin`
distribution, which packages pinned Node.js and the official shrinkwrap-locked
Pi npm distribution. Runtime startup
still requires the explicit absolute path to the installed `nemo-fabric-pi`
executable.

The adapter maps only `models.default` and explicit `skills.paths`. It does not
map tools, instructions, MCP, telemetry, streaming, updates, cancellation, or
other optional capabilities.

For configuration, lifecycle, security behavior, and supported platforms, see
the [Pi adapter README](https://github.com/NVIDIA/NeMo-Fabric/tree/main/adapters/pi/README.md).
