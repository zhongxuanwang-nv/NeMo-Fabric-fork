<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Pi CLI Distribution

`nemo-fabric-pi-cli-bin` packages the shrinkwrap-locked official
`@earendil-works/pi-coding-agent` 0.84.2 npm production tree and Node.js
22.19.0 for the NVIDIA NeMo Fabric Pi adapter. It exposes the
`nemo-fabric-pi` executable without a runtime `PATH` search, `npx`, or network
installation.

Most users should install `nemo-fabric[pi]` or
`nemo-fabric-adapters-pi[harness]` instead of installing this package directly.
