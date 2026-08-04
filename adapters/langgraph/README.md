<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Fabric LangGraph Adapter

Run a custom [LangGraph](https://langchain-ai.github.io/langgraph/) agent as a NeMo
Fabric agent. One generic adapter serves every LangGraph agent: a graph that
already compiles itself runs unchanged, and a graph that wants Fabric-managed
models, MCP tools, tool policy, or durable state opts in through a factory entry
point.

- **Adapter ID:** `nvidia.fabric.langgraph`
- **Distribution:** `nemo-fabric-adapters-langgraph`
- **Import namespace:** `nemo_fabric_adapters.langgraph`

> **Status:** proof of concept. The package is hosted in the Fabric repository
> during incubation but is designed as a third-party adapter. See
> [the design proposal](../../docs/designs/langgraph-adapter-proposal.md) for the
> ownership model and promotion criteria.

## Installation

```bash
pip install nemo-fabric-adapters-langgraph
```

Optional extras add only what a given agent needs:

| Extra | Install when the graph uses |
| --- | --- |
| `models` | `context.chat_model(...)` for a Fabric model alias |
| `mcp` | `context.mcp_tools()` for Fabric MCP servers |
| `state` | `state: adapter` for adapter-owned checkpoint persistence |

```bash
pip install 'nemo-fabric-adapters-langgraph[models,mcp,state]'
```

Install the agent's own package into the same Python environment, or point
`entrypoint.search_paths` at its directory.

## Quick start: an existing compiled graph

A graph that already builds its own model and tools needs no code changes.

```python
from nemo_fabric import FabricConfig, HarnessConfig, MetadataConfig

config = FabricConfig(
    metadata=MetadataConfig(name="my-langgraph-agent"),
    harness=HarnessConfig(
        adapter_id="nvidia.fabric.langgraph",
        resolution="preinstalled",
        settings={
            "langgraph": {
                "entrypoint": {"target": "my_agent.graph:graph", "kind": "compiled"},
                "input": {"mode": "messages"},
                "output": {"mode": "last_message"},
            }
        },
    ),
)
```

## Opting into Fabric resources

A `factory` entry point receives a `LangGraphAdapterContext` and returns a compiled
graph. This is the only Fabric-facing surface an agent needs.

```python
from nemo_fabric_adapters.langgraph import LangGraphAdapterContext


def build_graph(context: LangGraphAdapterContext):
    model = context.chat_model("default")
    tools = [*context.mcp_tools(), *context.guard_tools([my_local_tool])]
    prompt = context.agent_settings["system_prompt"]

    workflow = build_my_workflow(model=model, tools=tools, system_prompt=prompt)
    return workflow.compile(checkpointer=context.checkpointer)
```

```python
settings = {
    "langgraph": {
        "entrypoint": {"target": "my_agent.graph:build_graph", "kind": "factory"},
        "input": {"mode": "messages"},
        "output": {"mode": "field", "field": "answer"},
        "state": "adapter",
        "agent": {"system_prompt": "You are a careful reviewer."},
    }
}
```

## Configuration reference

All settings live under `harness.settings.langgraph`. Unknown keys are rejected. The
same surface is published as a JSON Schema in the `settings_schema` field of
[`fabric-adapter.json`](fabric-adapter.json), so an installed adapter advertises its
own settings.

### `entrypoint`

| Key | Default | Purpose |
| --- | --- | --- |
| `target` | required | `module.path:attribute` reference to the graph or factory |
| `kind` | `compiled` | `compiled`, `factory`, or `runnable_factory` |
| `search_paths` | `[]` | Directories relative to `base_dir`, appended to the import path |

Search paths must stay inside `base_dir`, and are appended rather than prepended so
an agent directory cannot shadow the standard library or an installed package.

| `kind` | Target shape | Fabric integration |
| --- | --- | --- |
| `compiled` | A compiled graph exposing `invoke`/`ainvoke` | Execution and result normalization only |
| `factory` | `callable(LangGraphAdapterContext) -> compiled graph` | Full opt-in: models, MCP tools, tool policy, state, callbacks |
| `runnable_factory` | `callable(RunnableConfig) -> compiled graph` | Execution plus the Fabric thread id. Matches the NVIDIA NeMo Agent Toolkit `langgraph_wrapper` entry-point shape, so the same graph runs in both |

### `input`

| `mode` | Behavior |
| --- | --- |
| `passthrough` (default) | Forwards the request input to the graph unchanged |
| `messages` | Wraps a text request as `{messages: [{role: user, content: ...}]}` |
| `field` | Places a text request in the state field named by `input.field` |

`messages` and `field` reject a non-text request rather than coercing it. Use
`messages_key` to rename the message list.

### `output`

| `mode` | Behavior |
| --- | --- |
| `last_message` (default) | Returns the last assistant message under `messages_key` |
| `field` | Returns the top-level state field named by `output.field` |
| `state` | Returns the full final state, converted to JSON-safe values |

### `state`

| `mode` | Behavior |
| --- | --- |
| `none` (default) | No thread id and no adapter-owned persistence |
| `graph` | Supplies a stable thread id; the graph owns checkpoint storage |
| `adapter` | Supplies a thread id and a checkpointer through the context (`factory` only) |

The thread id is derived from the Fabric `runtime_id`, so every `invoke` on one
started runtime continues the same LangGraph thread and separate runtimes stay
isolated.

### `events`, `agent`, and `recursion_limit`

| Key | Default | Purpose |
| --- | --- | --- |
| `events.mode` | `none` | `summary` reports a bounded, post-completion node summary |
| `events.limit` | `50` | Maximum buffered events |
| `agent` | `{}` | JSON-shaped domain settings passed to a factory unchanged |
| `recursion_limit` | unset | LangGraph recursion limit for the run |

## What the adapter cannot do

- **Compiled graphs are outside Fabric policy.** A compiled graph builds its own
  models and tools, so `tools.blocked`, Fabric MCP servers, and `state: adapter`
  are rejected for `compiled` and `runnable_factory` entry points rather than
  reported as enforced. Use a `factory` entry point for those.
- **Policy covers only wrapped tools.** `tools.blocked` applies to tools passed
  through `context.mcp_tools()` or `context.guard_tools(...)`. A graph that wires
  tools directly owns that boundary. For MCP tools a selector may be the bare tool
  name, which denies it on every server, or `server__tool`, which denies one
  server's copy; the tool itself is not renamed.
- **Output must be JSON-safe.** An unserializable projection fails with a
  normalized error instead of a corrupted result.
- **No streaming or cancellation.** Fabric's Python adapter runtime invokes an
  adapter to completion and returns one result.
- **Interrupts cannot be answered.** A graph that calls `interrupt()` returns a
  normalized incomplete result with `interrupted: true` and the interrupt payload,
  rather than a partial success. Under `state: graph` or `state: adapter` the
  thread stays checkpointed, so LangGraph can resume it directly.
- **Subgraph usage depends on state merging.** A composed subgraph whose messages
  merge into the parent state is counted normally; one that keeps its messages in
  a separate channel contributes no token usage.
- **Adapter files are metadata, not artifacts.** The checkpoint database path is
  reported as `checkpoint_path`, because Fabric promotes only relay artifacts into
  the artifact manifest.
- **No graph-level telemetry yet.** The descriptor advertises no telemetry
  provider until one is validated end to end; `context.callbacks` is the merge
  point for when it is.
- **Importing a graph is not a sandbox.** The adapter runs trusted Python.

## Testing

```bash
# unit and design-validation suite, no credentials required
pytest tests/adapters/test_langgraph_adapter.py

# opt-in real smoke against the example agents
RUN_FABRIC_LANGGRAPH_INTEGRATION=1 NVIDIA_API_KEY=... pytest tests/e2e/test_langgraph.py
```

## Bug routing

File adapter behavior issues against this package. File issues in the Fabric
adapter descriptor, invocation, or result contracts against
[NeMo Fabric](https://github.com/NVIDIA/nemo-fabric/).

## License

Apache-2.0. See [LICENSE](LICENSE).
