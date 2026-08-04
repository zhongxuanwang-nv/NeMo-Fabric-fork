<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Third-Party Adapters for Custom Agents (LangGraph)

Continues **Third Party Adapters for Fabric** (Ajay Thorve), which establishes the
third-party adapter contract with NAT as the first reference implementation and
leaves this section as two options to decide between.

The purpose is not LangGraph support on its own. It is to confirm that one contract
can serve independently developed adapters, and to feed the changes that contract
still needs back into the parent document. Those are collected in
[Contract Requirements](#contract-requirements-from-two-implementations).

| | |
| --- | --- |
| **Status** | In Progress |
| **Category** | Architecture & Design |
| **Draft date** | Aug 4, 2026 |
| **Implementation** | [`adapters/langgraph/`](../../adapters/langgraph/README.md), examples at [`examples/langgraph/`](../../examples/langgraph/README.md), branch `feat/langgraph-adapter` |
| **NSPECT-ID** | TBD |

## Problem

LangGraph provides graph-building primitives but defines no Fabric-facing
application registry, configuration schema, input/output model, or cleanup
lifecycle. Each custom agent owns its own graph parameters, state schema, model and
tool injection points, checkpointer semantics, and cleanup.

NAT proved the contract against a framework that *does* supply those things: a
component registry, a typed schema per component, and a builder lifecycle.
LangGraph supplies none, which is why it is the right second implementation — if
the contract holds here, it is framework-independent rather than shaped around NAT.

## Goals

- Decide between an application-specific and a generic LangGraph adapter.
- Run representative custom agents through Fabric's public contract only.
- Keep the normalized-versus-`harness.settings` boundary identical in meaning to
  the NAT adapter.
- Reject capabilities the adapter cannot enforce instead of ignoring them.
- Specify the contract changes a standardized third-party adapter contract needs,
  each evidenced by two independent implementations rather than one framework's
  convenience.

## Success Criteria

| Criterion | Status |
| --- | --- |
| An existing compiled graph runs with no code changes | Met |
| A custom graph consumes Fabric models, MCP tools, tool policy, and durable state | Met |
| Capability planning rejects what the selected binding cannot enforce | Met |
| Runtime reuse retains graph state across `invoke` calls and isolates runtimes | Met |
| Both agents run end to end through the real Fabric runtime | Met |
| One adapter serves every custom agent, with no per-agent adapter | Met |
| Clean installation discovers an external adapter wheel | **Blocked** — requirement 1 |
| Normalized system instructions map to a graph | **Blocked** — requirement 8 |
| The contract's gaps are specified, not just encountered | Met — 10 requirements below |

## Option 1: Application-Specific Adapter

One Fabric adapter per custom agent, importing the agent directly and hardcoding its
state schema, tool wiring, and result shape. It fits one agent exactly, but costs a
descriptor, package, release, and compatibility matrix per agent; duplicates
lifecycle, model binding, MCP conversion, tool policy, and normalization in every
copy; turns an agent's state-schema change into an adapter release; and contradicts
the parent document's requirement that an adapter remain generic with no catalog of
known components.

## Option 2: Generic LangGraph Adapter — Recommended

One adapter, `nvidia.fabric.langgraph`
(`nemo-fabric-adapters-langgraph`), that loads a customer-owned graph by an
explicit Python reference. The agent selects a binding kind:

| Kind | Entry point | For | Fabric integration |
| --- | --- | --- | --- |
| `compiled` (default) | A compiled graph exposing `invoke`/`ainvoke` | An existing graph that owns its model and tools | Execution and normalization only |
| `factory` | `callable(LangGraphAdapterContext) -> compiled graph` | A graph that wants Fabric-managed resources | Models, MCP tools, tool policy, durable state |
| `runnable_factory` | `callable(RunnableConfig) -> compiled graph` | A graph already written for the NAT `langgraph_wrapper` | Execution plus the Fabric thread ID |

`compiled` is the equivalent of NAT's built-in workflow path: normalized config in,
no agent changes. `factory` is the equivalent of a custom NAT workflow that
resolves components from the builder, except the resources arrive as an explicit
argument instead of an ambient accessor.

**Use Option 1 only when** the agent needs a remote LangGraph service protocol, a
custom process or container lifecycle, server-side streaming or cancellation, or a
non-JSON request/result protocol. A new prompt, state schema, node, subgraph,
application tool, retriever, routing policy, or output field never justifies it.

## Contract Requirements From Two Implementations

The main output for the parent document. Each row is a gap that both NAT and this
adapter hit independently, so it belongs in the shared contract rather than in
either adapter. Detail for the LangGraph column is in the sections below.

| # | Gap | NAT evidence | LangGraph evidence | Contract change |
| --- | --- | --- | --- | --- |
| 1 | **Installed adapters are not discoverable.** `AdapterRegistry` scans only `<repo>/adapters/` and `<base_dir>/adapters/`. | Success criterion "discover an external adapter wheel without modifying Fabric core" is unmet. | The wheel installs its descriptor to `share/nemo-fabric/adapters/langgraph/`; nothing reads it. | Define **one** mechanism — an entry-point group or a `share/nemo-fabric/adapters` scan under `sys.prefix` — and have the registry use it. Blocking for any third party. |
| 2 | **Settings are not declared or centrally validated.** `settings_schema` is absent from `adapter-descriptor.schema.json`, so it survives only as an untyped `extensions` passthrough. | Publishes `settings_schema`, then defers authoritative validation to its own registry. | Publishes `settings_schema` and hand-validates in the adapter, rejecting unknown keys itself. | Promote `settings_schema` to a typed descriptor field and validate `harness.settings` against it at plan time, so settings errors are uniform and pre-execution. |
| 3 | **Capability acceptance cannot be conditional.** `config.accepts` is a static list, but what an adapter can enforce depends on its settings. | A custom workflow cannot take normalized instructions or auto-attached MCP; ReAct can. Same `accepts` list. | `tools.blocked`, Fabric MCP, and `state: adapter` are enforceable for `factory` but not `compiled`. Same `accepts` list. | Add a plan-time adapter validation hook, or declarative conditional accepts, so rejection is uniform and happens before execution instead of each adapter hand-rolling it at invoke. |
| 4 | **Where runtime state lives is unspecified.** `adapter_kind: python` spawns a subprocess per invoke; `start` only preflights and `stop` emits an event. | Design retains builder, workflow, sessions, and MCP clients in memory across `invoke` — not possible under this kind. | Keys a durable checkpoint to `runtime_id` instead, which the current kind supports. | State that `runtime_id` is the state key for `adapter_kind: python`, and add a distinct persistent kind if in-memory retention is required rather than overloading `python`. |
| 5 | **The adapter result envelope has no schema.** Thirteen schemas exist; none covers the adapter's stdout. `RunResult.output` is opaque. | Converts NAT results to "the Fabric adapter response contract" by convention. | Reproduces `response`/`failed`/`messages` by copying existing adapters. | Schema the stdout envelope, and add a general adapter-artifact key: only `relay_artifacts` of kind `atof`/`atif` currently reach the manifest. |
| 6 | **No status for a run that stopped needing input.** | Human-in-the-loop will need one. | `interrupt()` returns with `__interrupt__`; the adapter invents `interrupted`/`interrupts`. | Add a normalized incomplete/needs-input outcome so adapters do not each invent a different shape for the same situation. |
| 7 | **Tool selector semantics are not specified.** | `<group>__<member>` is canonical, and members are renamed. | MCP tools carry bare names; the optional prefix joins with a single ambiguous underscore. The adapter resolves both forms without renaming. | Specify `__` as the separator, whether adapters rename or resolve, and the `enabled` then `blocked` precedence once `tools.enabled` exists. |
| 8 | **Two declared capabilities do not exist.** `ToolsConfig` has only `blocked`; `FabricConfig` has no `instructions`. | Descriptor accepts `instructions.system` and `tools.enabled`. | Cannot consume either; carries prompt intent through uninterpreted `agent` settings. | Add both to `FabricConfig`, and deliver `instructions.system` only where an adapter declares a mapping, since LangGraph has no equivalent of ReAct `additional_instructions`. |
| 9 | **Normalization is copied, not shared.** | Maps `models.*` to NAT `llms` with provider, key env, base URL, temperature. | Provider defaults, key-env resolution, and MCP transport normalization were copied near-verbatim from the Deep Agents adapter. | Move model-alias and MCP-transport normalization into `nemo-fabric-adapters-common` so third parties do not re-derive provider defaults and drift. |
| 10 | **Naming is inconsistent.** | `nvidia.nemo.platform.nat`, `nemo_platform_fabric_adapter_nat`. | `nvidia.fabric.langgraph`, `nemo_fabric_adapters.langgraph`. | Adopt one provider token across repository, distribution, import package, `adapter_id`, and `harness`, as the toolkit's plugin naming table does. |

Requirements 1 through 3 gate third-party adapters at all: without discovery an
adapter cannot be installed, and without declared settings and conditional
capability acceptance every adapter invents its own validation and failure
behavior. Requirements 4 through 8 are correctness and uniformity of the shared
contract. Requirements 9 and 10 reduce duplication and drift.

## LangGraph Behaviors That Shape The Design

The graph-object, interrupt, and subgraph-streaming rows were verified empirically
against langgraph 1.2.9; the rest come from LangGraph's documented contract and
constraints the Deep Agents adapter already records.

| LangGraph behavior | Why it matters | Adapter response |
| --- | --- | --- |
| A compiled graph is a `CompiledStateGraph`: it exposes `invoke`/`ainvoke` but is **not callable**. A factory is callable but has no `invoke`. | The two cannot be distinguished by "is it callable", and a wrong guess fails deep inside LangGraph. | `entrypoint.kind` is explicit; the loader validates shape up front and names the correct kind in the error. |
| `interrupt()` does **not** raise. The graph returns normally with a private `__interrupt__` key holding `Interrupt` objects. | Projection would report a false success, or an unrelated error about the field the graph never reached. | Detect interrupts before projection; return a normalized incomplete result with `interrupted: true` and each JSON-safe payload. |
| State is an arbitrary typed dict holding LangChain messages, `Interrupt` objects, and any Python value. | Fabric results must be JSON. | Coerce messages to dicts, then fail with a normalized error naming the field if still unserializable. |
| Reducers such as `add_messages` accumulate, so a resumed thread's final state replays every prior turn. | Usage taken from final state re-bills earlier turns. | Aggregate usage from this turn's `updates` deltas only. |
| A node may return its whole message list rather than only new messages. | Usage would count earlier messages again each step. | De-duplicate turn messages by id. |
| `astream(..., subgraphs=True)` emits a composed subgraph's messages **twice** and emits subgraph-namespace `values` chunks. | Double-counted usage, and a subgraph's state overwriting the root graph's. | Do not stream subgraph namespaces; a composed subgraph merges into parent state, so the parent's update already carries its messages. |
| A checkpointer is bound at `compile()` and cannot be attached afterward. | A precompiled graph cannot be given Fabric-managed persistence. | `state: adapter` rejected for `compiled` and `runnable_factory`; a factory compiles with `context.checkpointer`. |
| Async execution needs an async saver; the sync `SqliteSaver` raises `NotImplementedError` from async methods. | Confusing resume failures. | `AsyncSqliteSaver` from `langgraph-checkpoint-sqlite>=3,<4`, opened and closed per invocation. |
| Thread identity lives in `config.configurable.thread_id`; LangGraph owns the transcript. | Fabric needs continuity without inventing a transcript. | Derive `thread_id` deterministically from `runtime_id`; ask the checkpointer whether the thread exists to report `resumed` truthfully. |
| Tools are LangChain `BaseTool` instances with an `args_schema`, and a node may call one by name. | Removing a blocked tool breaks a hardcoded call site with an unknown-tool error. | Replace a blocked tool with a deny stub preserving name, description, and `args_schema`. |
| Graphs commonly live in an unpackaged directory, which is what NAT's `dependencies` list addresses. | Otherwise "no code changes" is unachievable in practice. | `entrypoint.search_paths`, relative to `base_dir`, confined to it, and appended to `sys.path` so an agent directory cannot shadow the standard library. |
| `RemoteGraph` and remote subgraphs execute outside the process. | Fabric cannot govern their tools, state, or telemetry. | Runnable only if the client satisfies the local graph contract; out of scope for policy and telemetry claims. |

## Runtime Lifecycle

The NAT design retains the builder, workflow, sessions, and MCP clients in memory
across `invoke` calls. **That is not available to `adapter_kind: python` as
implemented today**: `run_python_adapter` spawns `python -m <module>` per
invocation, `start` performs only interpreter preflight, and `stop` emits an event.
Nothing survives in memory between invocations.

LangGraph satisfies the same runtime-reuse criterion a different way:

| Stage | This adapter |
| --- | --- |
| `start_runtime` | Fabric allocates a stable `runtime_id`. No graph is built. |
| `invoke` | Derive the thread ID, open the checkpointer, build the graph, run it, project the result, close the checkpointer. |
| `stop_runtime` | Nothing to tear down: durable state is committed and per-invocation resources close in a `finally` block. |

Continuity comes from a **durable checkpoint keyed by `runtime_id`**, not an
in-memory object graph. That costs graph construction per invocation and gains crash
and process independence. If Fabric later adds a persistent Python adapter runtime,
this adapter can cache the compiled graph without changing the agent contract.
Whether NAT's in-memory model requires that runtime, or a different adapter kind, is
an open contract question this comparison raises.

## Configuration Model

Normalized Fabric config supplies models, MCP servers, tool policy, and
environment. Adapter-owned `harness.settings.langgraph` supplies only graph
selection and the request/result shape.

| Area | Values |
| --- | --- |
| `entrypoint` | `target` (`module:attr`, required), `kind`, `search_paths` |
| `input.mode` | `passthrough` (default), `messages`, `field`; the latter two require a text request and reject structured input rather than coercing it |
| `output.mode` | `last_message` (default), `field`, `state` |
| `state` | `none` (default), `graph`, `adapter` |
| `events` | `mode` (`none`, `summary`), `limit`; bounded post-completion node summary, not progressive streaming |
| `agent` | JSON object passed to a factory unchanged; Fabric assigns it no meaning |
| `recursion_limit` | LangGraph recursion limit |

Unknown keys are rejected. Callable import strings for input transforms, output
transforms, or tool factories are deliberately not configurable; those belong in the
factory, versioned and tested with the agent.

Following NAT, the descriptor publishes a `settings_schema` so an installed adapter
advertises its own settings instead of requiring readers to find this document.
Unlike NAT, the schema is closed and authoritative, because there is no downstream
component registry to validate against.

### Factory Context

```python
def build_graph(context: LangGraphAdapterContext):
    model = context.chat_model("default")
    tools = [*context.mcp_tools(), *context.guard_tools([my_local_tool])]
    workflow = compile_my_graph(model=model, tools=tools, **context.agent_settings)
    return workflow.compile(checkpointer=context.checkpointer)
```

| Surface | Provides | Boundary |
| --- | --- | --- |
| `chat_model(alias)` | LangChain chat model from a Fabric alias, memoized per invocation | NVIDIA and OpenAI-compatible, then an `init_chat_model` fallback |
| `mcp_tools()` | Tools from Fabric MCP servers, policy applied | Transport validated; fails closed |
| `guard_tools(tools)` | `tools.blocked` applied to the agent's own tools | Only tools passed through this helper are governed |
| `checkpointer`, `runnable_config`, `thread_id` | Adapter-owned persistence and thread identity | The factory must compile with the checkpointer and merge, not replace, the config |
| `workspace`, `artifact_dir` | Resolved Fabric paths | The adapter does not sandbox the filesystem |
| `callbacks` | Telemetry callbacks to merge | Empty until a provider is validated and declared |
| `agent_settings`, `agent_name`, `blocked_tool_names` | Read-only JSON configuration | Fabric assigns domain fields no meaning |

`current_context()` is a compatibility shim for a module that builds its graph at
import time, mirroring NAT's `SyncBuilder.current()` accessor. It is valid only
during graph construction; a `factory` entry point is preferred because the
dependency is explicit.

## Supported Configuration

| `FabricConfig` field | Target | Behavior |
| --- | --- | --- |
| `models.<name>` | `context.chat_model("<name>")` | Built on request by a factory; never injected into a compiled graph |
| `models.<name>.provider` | LangChain client | `nvidia`/`openai`/`openai-compatible` via `ChatOpenAI`, else `init_chat_model` |
| `models.<name>.api_key_env` | Client credential | Resolved from the environment without logging the value |
| `models.<name>.base_url`, `.temperature`, `.settings.base_url` | Client fields | Direct mapping; NVIDIA and unspecified providers default to the NVIDIA endpoint |
| `mcp.servers.<name>` (`harness_native`) | LangChain tools via `MultiServerMCPClient` | Only for `factory`; rejected otherwise |
| `tools.blocked` | Deny stubs over context-supplied tools | Only for `factory`; rejected otherwise |
| `environment.workspace`, `artifacts` | `context.workspace`, `context.artifact_dir` | Paths only; the adapter does not confine the filesystem |
| `runtime_context.runtime_id` | LangGraph `thread_id` | Deterministic derivation; the basis of runtime reuse |

## Tool Policy

Fabric governs only tools it wraps. Effective tools are
`graph-accessible ∩ context-supplied − Fabric-blocked`, and blocking wins. A graph
that deliberately wires unwrapped tools owns that boundary.

**Selector naming did not transfer from NAT, and was reconciled.** NAT names
members `<group>__<member>` (`calculator__add`) and treats that as the canonical
Fabric selector. LangGraph has no such convention:
`MultiServerMCPClient.get_tools()` returns tools carrying the bare MCP tool name,
and its `tool_name_prefix` option defaults to `False` and joins with a **single**
underscore, which is ambiguous — server `web` with tool `search_docs` is
indistinguishable from server `web_search` with tool `docs`.

The adapter therefore loads MCP tools per server, keeps the origin, and resolves
both selector forms: a qualified `server__tool` denies one server's tool, a bare
name denies that tool on every server exposing it. It does **not** rename the
tool, because renaming changes what the model and the graph's own call sites see.
One Fabric policy now means the same thing in both adapters, without either
adapter dictating the other's tool names.

## Pending Contract Additions

The NAT descriptor declares `instructions.system` and `tools.enabled` under
`config.accepts`, but neither exists in Fabric today: `ToolsConfig` has only
`blocked`, and `FabricConfig` has no `instructions` field. When they land:

| Addition | LangGraph behavior |
| --- | --- |
| `instructions.system` | LangGraph has no equivalent of NAT's ReAct `additional_instructions`; a prompt lives wherever the graph puts it. Deliver it to a `factory` through the context, and **reject** it for `compiled` and `runnable_factory`, mirroring NAT's rule that custom workflows receive no automatic instruction mapping. Today the same intent is carried by `agent.system_prompt`, which Fabric does not interpret. |
| `tools.enabled` | Enforceable only over context-supplied tools, since a graph builds its own inventory. `enabled: []` exposes none; `None` preserves the graph's default. Unresolvable selectors must be rejected. |

That LangGraph has nowhere to put a normalized system instruction is the clearest
evidence that instruction mapping is framework-specific and belongs behind an
explicit per-adapter declaration rather than in the portable core.

## Portability Boundary

Mirrors the parent document's calculator and phishing-analyzer split. Portable:
models, MCP server declarations, tool policy, runtime and environment config. Not
portable: the graph, its state schema, and its in-process tools, which stay
LangGraph-specific until exposed through MCP. The same MCP server routed to NAT and
to this adapter is the cross-harness case; a tool compiled into a graph is not.

## Initially Unsupported

| Capability | Reason |
| --- | --- |
| Interrupt and resume | Fabric has no interaction contract; the run returns a normalized incomplete result and the thread stays checkpointed for LangGraph to resume directly |
| Streaming, cancellation, config updates, service lifecycle | Not implemented end to end; the descriptor declares all four `false` |
| Graph-level telemetry | No provider validated end to end, so the descriptor declares none |
| Fabric skills | No defined LangGraph mapping |
| `tools.blocked`, Fabric MCP, `state: adapter` for `compiled`/`runnable_factory` | The graph owns its tools and checkpointer; rejected rather than silently ignored |
| Subgraph usage without state merging | A subgraph keeping messages in a separate channel contributes no usage |
| Adapter files as artifacts | Fabric promotes only relay `atof`/`atif` from adapter output; the checkpoint path is reported as metadata |
| Installed-wheel descriptor discovery | Descriptors are found in repository or project `adapters/` directories only |

## Validation

Two custom agents, one adapter, no agent-specific code. The suite runs them as real
compiled graphs with a real SQLite checkpointer; only the chat model is substituted.

| Agent | Characteristics | Configuration |
| --- | --- | --- |
| `triage_agent` | Custom state schema, conditional routing, own prompts, own chat model, compiled at import, never imports Fabric | `compiled`, `input: field(ticket)`, `output: last_message`, `state: none` |
| `case_review_agent` | Routed research and review nodes, domain `answer` field, multi-turn accumulation, application tool alongside MCP tools | `factory`, `input: messages`, `output: field(answer)`, `state: adapter`, model alias, MCP, `tools.blocked` |

Proven by [`tests/adapters/test_langgraph_adapter.py`](../../tests/adapters/test_langgraph_adapter.py)
and [`tests/e2e/test_langgraph.py`](../../tests/e2e/test_langgraph.py): a compiled
graph running unchanged across all three output projections; compiled mode failing
clearly for unenforceable capabilities and succeeding with `state: graph`; factory
mode binding a model alias and MCP tools; a blocked application tool **and** a
blocked MCP tool denied without executing while unblocked tools run; one thread
reused across invocations with accurate `resumed` and runtime isolation; an
interrupting graph reporting an incomplete run with a preserved checkpoint; a
multi-agent graph counting a composed subgraph's tokens exactly once; specific
loading failures; and both agents running through the real Fabric runtime, with the
factory agent additionally run against a live NVIDIA endpoint across two turns.

## Open Decisions

The ten contract requirements above each need a decision in the parent document.
Decisions specific to this adapter:

1. Which requirements gate publishing this adapter? Requirement 1 is the only
   blocked success criterion, and it also removes the packaging workaround in
   `TODO.md`.
2. Should `runnable_factory` and `current_context()` remain supported for NAT
   portability, or should Fabric support only the explicit `factory` contract?
3. Which production or partner agents replace the two example fixtures as long-term
   compatibility cases?
4. Which NeMo Relay LangGraph integration version has an approved validation path,
   so telemetry can be declared?

## References

- Third Party Adapters for Fabric (Ajay Thorve) — the parent contract document.
- [NAT third-party plugin packages](https://docs.nvidia.com/nemo/agent-toolkit/latest/extend/third-party-plugins.html)
- [Running existing LangGraph agents in NAT](https://docs.nvidia.com/nemo/agent-toolkit/latest/run-workflows/existing-agents/langgraph.html)
- [`adapters/deepagents/`](../../adapters/deepagents/README.md) — in-repo precedent
  for a LangGraph-based harness.
