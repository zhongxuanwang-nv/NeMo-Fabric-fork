<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Design Proposals

Design proposals and specifications for NeMo Fabric changes that need alignment
before or alongside implementation.

These documents are **not** part of the published documentation site. They are
working documents for maintainers and reviewers, so they are intentionally absent
from the Fern navigation in [`docs/index.yml`](../index.yml). User-facing behavior
belongs in the published sections, and in the relevant adapter, integration, or
example `README.md`.

| Proposal | Status | Scope |
| --- | --- | --- |
| [LangGraph adapter](langgraph-adapter-proposal.md) | Draft for alignment, with a working proof of concept | A single generic adapter (`nvidia.fabric.langgraph`) for custom LangGraph agents, validated against two representative agents |

## Conventions

A proposal should state its status in the first lines, record the decision it is
asking for, and separate what has been validated from what is still outstanding.
When a proposal is implemented, keep it as the design record and move user-facing
instructions into the published docs and the relevant package `README.md`.

When a proposal cites sources that could not be verified, say so in the document
rather than presenting unverified material as settled.
