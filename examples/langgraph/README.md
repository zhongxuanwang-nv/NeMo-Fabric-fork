<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Representative Custom LangGraph Agents

Two custom LangGraph agents that validate the generic
[LangGraph adapter](../../adapters/langgraph/README.md). They sit at opposite ends
of the intended support range, and both run through the same adapter ID
(`nvidia.fabric.langgraph`) with no agent-specific adapter.

| Agent | What makes it representative | Adapter configuration |
| --- | --- | --- |
| [`triage_agent`](triage_agent/graph.py) | An existing agent: own state schema, conditional routing, own prompts, own chat model, compiled at import time. Does not import Fabric. | `compiled`, `input: field(ticket)`, `output: last_message`, `state: none` |
| [`case_review_agent`](case_review_agent/graph.py) | Needs Fabric-managed resources: a configurable model, MCP tools, an enforced tool policy, durable multi-turn state, and a domain result field. | `factory`, `input: messages`, `output: field(answer)`, `state: adapter`, `tools.blocked` |

`triage_agent` proves a custom graph runs **unchanged**. `case_review_agent` proves
a graph can consume the full Fabric surface through one adapter-owned argument
while remaining an ordinary LangGraph agent.

## Run them

```bash
pip install -e '.[langgraph,runtime]'
```

The triage agent's p1 branch escalates without a model call, so it runs with no
credentials:

```python
import asyncio

from examples.langgraph.config import BASE_DIR, triage_config
from nemo_fabric import Fabric


async def main() -> None:
    result = await Fabric().run(
        triage_config(), base_dir=BASE_DIR, input="production down, total outage"
    )
    print(result["output"]["response"])


asyncio.run(main())
```

The case-review agent binds a Fabric model, so it needs `NVIDIA_API_KEY`. Using a
started runtime keeps one LangGraph thread across turns:

```python
import asyncio

from examples.langgraph.config import BASE_DIR, case_review_config
from nemo_fabric import Fabric


async def main() -> None:
    async with await Fabric().start_runtime(case_review_config(), base_dir=BASE_DIR) as runtime:
        first = await runtime.invoke(input="Is a five-year non-compete enforceable in California?")
        second = await runtime.invoke(input="What did I ask about first?")
    print(first["output"]["resumed"], second["output"]["resumed"])  # False True


asyncio.run(main())
```

## Verify

```bash
# design-validation suite; substitutes only the chat model
pytest tests/adapters/test_langgraph_adapter.py

# opt-in real smoke
RUN_FABRIC_LANGGRAPH_INTEGRATION=1 NVIDIA_API_KEY=... pytest tests/e2e/test_langgraph.py
```
