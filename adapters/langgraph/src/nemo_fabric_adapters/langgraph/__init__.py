# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic LangGraph adapter for NeMo Fabric.

The graph-authoring contract is :class:`~nemo_fabric_adapters.langgraph.context.LangGraphAdapterContext`.
Everything else in this package is adapter implementation detail and may change
between releases.
"""

from nemo_fabric_adapters.langgraph.context import LangGraphAdapterContext
from nemo_fabric_adapters.langgraph.context import current_context

__all__ = ["LangGraphAdapterContext", "current_context"]
