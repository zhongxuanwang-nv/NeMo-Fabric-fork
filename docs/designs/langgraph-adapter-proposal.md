<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Third-Party Adapters for Custom Agents (LangGraph)

Fills in the LangGraph section of **Third Party Adapters for Fabric**
(Ajay Thorve). That document owns the problem statement, goals, success criteria,
ownership model, tool-policy evaluation order, and portability boundary, and states
**NAT's** view of the unified contract. None of that is restated here.

This document states **LangGraph's** view of the same contract: what a framework
with no component registry, no per-component configuration schema, and no builder
lifecycle demands of it, and where those demands diverge from NAT's.

|                    |                                                                                                                                                                       |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**         | In Progress                                                                                                                                                           |
| **Category**       | Architecture & Design                                                                                                                                                 |
| **Draft date**     | Aug 4, 2026                                                                                                                                                           |
| **Implementation** | `[adapters/langgraph/](../../adapters/langgraph/README.md)`, examples at `[examples/langgraph/](../../examples/langgraph/README.md)`, branch `feat/langgraph-adapter` |
| **NSPECT-ID**      | TBD                                                                                                                                                                   |

## Outcome

Option 2, the generic adapter, is recommended and implemented. LangGraph meets every
parent success criterion except two, and both are contract gaps rather than
LangGraph limitations:

- **Clean installation of an external adapter wheel** — requirement 1.
- **Normalized system instructions** — requirement 8. LangGraph has nowhere to put
one, which is itself a finding about the contract.

## Option 1: Application-Specific Adapter

One adapter per custom agent, importing the agent and hardcoding its state schema,
tool wiring, and result shape. It fits one agent exactly, but costs a descriptor,
package, release, and compatibility matrix per agent; duplicates lifecycle, model
binding, MCP conversion, tool policy, and normalization in every copy; turns an
agent's state-schema change into an adapter release; and contradicts the parent
document's rule that an adapter remain generic.

## Option 2: Generic LangGraph Adapter — Recommended

One adapter, `nvidia.fabric.langgraph`
(`nemo-fabric-adapters-langgraph`), that loads a customer-owned graph by an
explicit Python reference. The agent selects a binding kind:

| Kind                 | Entry point                                           | For                                                     | Fabric integration                            |
| --- | --- | --- | --- |
| `compiled` (default) | A compiled graph exposing `invoke`/`ainvoke`          | An existing graph that owns its model and tools         | Execution and normalization only              |
| `factory`            | `callable(LangGraphAdapterContext) -> compiled graph` | A graph that wants Fabric-managed resources             | Models, MCP tools, tool policy, durable state |
| `runnable_factory`   | `callable(RunnableConfig) -> compiled graph`          | A graph already written for the NAT `langgraph_wrapper` | Execution plus the Fabric thread ID           |

`compiled` is the equivalent of NAT's built-in workflow path: normalized config in,
no agent changes. `factory` is the equivalent of a custom NAT workflow that
resolves components from the builder, except the resources arrive as an explicit
argument instead of an ambient accessor.

**Use Option 1 only when** the agent needs a remote LangGraph service protocol, a
custom process or container lifecycle, server-side streaming or cancellation, or a
non-JSON request/result protocol. A new prompt, state schema, node, subgraph,
application tool, retriever, routing policy, or output field never justifies it.

## Contract Requirements From Two Implementations

LangGraph's contribution to the unified contract. Each row leads with what building
this adapter required, then notes where NAT hits the same gap — a gap both
implementations reach independently belongs in the contract, not in either adapter.

| #   | Gap                                                                                                                                                                                  | What LangGraph required                                                                                                                                                    | NAT reaches it too                                                                                                    | Contract change                                                                                                                                                              |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Installed adapters are not discoverable.** `AdapterRegistry` scans only `<repo>/adapters/` and `<base_dir>/adapters/`.                                                             | The wheel installs its descriptor to `share/nemo-fabric/adapters/langgraph/` and nothing reads it, so the adapter cannot be consumed as a package.                         | Its "discover an external adapter wheel" criterion is unmet.                                                          | Define **one** mechanism — an entry-point group or a `share/nemo-fabric/adapters` scan under `sys.prefix` — and have the registry use it. Blocking for any third party.      |
| 2   | **Settings are not declared or centrally validated.** `settings_schema` is absent from `adapter-descriptor.schema.json`, so it survives only as an untyped `extensions` passthrough. | Hand-validates its settings and rejects unknown keys itself, because nothing downstream will.                                                                              | Publishes a schema, then defers authoritative validation to its own registry.                                         | Promote `settings_schema` to a typed descriptor field and validate `harness.settings` against it at plan time, so settings errors are uniform and pre-execution.             |
| 3   | **`config.accepts` is static, and its granularity is unspecified.** What an adapter can enforce depends on its settings, and nothing says whether to declare a section or its sub-fields. | `tools.blocked`, Fabric MCP, and `state: adapter` are enforceable for a `factory` graph but impossible for a `compiled` one, under one `accepts` list.                     | A custom workflow cannot take normalized instructions or auto-attached MCP; ReAct can. It also declares `models.base_url` and `models.temperature` sub-paths where this adapter first declared only `models`. | Add a plan-time adapter validation hook, or declarative conditional accepts, and specify the granularity. This adapter now follows NAT's sub-path precedent. |
| 4   | **Where runtime state lives is unspecified.** `adapter_kind: python` spawns a subprocess per invoke; `start` only preflights and `stop` emits an event.                              | Nothing survives in memory, so continuity had to come from a durable checkpoint keyed by `runtime_id`.                                                                     | Its design retains builder, workflow, sessions, and MCP clients in memory across `invoke`, which this kind cannot do. | State that `runtime_id` is the state key for `adapter_kind: python`, and add a distinct persistent kind if in-memory retention is required rather than overloading `python`. |
| 5   | **The adapter result envelope has no schema.** Thirteen schemas exist; none covers the adapter's stdout. `RunResult.output` is opaque.                                               | `response`/`failed`/`messages` were reproduced by copying existing adapters, and the checkpoint path has nowhere to go but ad hoc metadata.                                | Converts NAT results to "the Fabric adapter response contract" by convention.                                         | Schema the stdout envelope, and add a general adapter-artifact key: only `relay_artifacts` of kind `atof`/`atif` reach the manifest.                                         |
| 6   | **No status for a run that stopped needing input.**                                                                                                                                  | `interrupt()` returns rather than raises, so the adapter had to invent `interrupted`/`interrupts` to avoid reporting a false success.                                      | Human-in-the-loop will need the same outcome.                                                                         | Add a normalized incomplete/needs-input outcome so adapters do not each invent a different shape for the same situation.                                                     |
| 7   | **Tool selector semantics are not specified.**                                                                                                                                       | MCP tools carry bare names, and the optional prefix joins with a single ambiguous underscore, so NAT-style selectors had to be resolved per server without renaming tools. | `<group>__<member>` is canonical there, and members are renamed.                                                      | Specify `__` as the separator, whether adapters rename or resolve, and the `enabled` then `blocked` precedence once `tools.enabled` exists.                                  |
| 8   | **Two declared capabilities do not exist.** `ToolsConfig` has only `blocked`; `FabricConfig` has no `instructions`.                                                                  | Cannot consume either. There is no equivalent of ReAct `additional_instructions`, so prompt intent travels as uninterpreted `agent` settings; when the fields land, an instruction can only reach a `factory`, and `tools.enabled` reduces to all-or-nothing for a graph that builds its own inventory. | Its descriptor accepts `instructions.system` and `tools.enabled`.                                                     | Add both to `FabricConfig`, and deliver `instructions.system` only where an adapter declares a mapping.                                                                      |
| 9   | **Normalization is copied, not shared.**                                                                                                                                             | Provider defaults, key-env resolution, and MCP transport normalization were copied near-verbatim from the Deep Agents adapter.                                             | Maps `models.`* to NAT `llms` with the same provider, key env, base URL, and temperature fields.                      | Move model-alias and MCP-transport normalization into `nemo-fabric-adapters-common` so third parties do not re-derive provider defaults and drift.                           |
| 10  | **Naming is inconsistent.**                                                                                                                                                          | `nvidia.fabric.langgraph`, `nemo_fabric_adapters.langgraph`.                                                                                                               | `nvidia.nemo.platform.nat`, `nemo_platform_fabric_adapter_nat`.                                                       | Adopt one provider token across repository, distribution, import package, `adapter_id`, and `harness`.                                                                       |
| 11  | **Runtime capability flags are unverified.** `resolve_runtime_capabilities` only ANDs a descriptor claim with the adapter kind, and no runtime path implements streaming, cancellation, updates, or service lifecycle. | Declares all four `false` by convention; a `true` would be propagated as fact and silently not work. | Declares all four `false` by the same convention. | Either verify the flags against what the runtime actually implements, or state that they are advisory. The parent goal of "truthful capability declarations" currently rests on adapter honesty alone. |

Requirements 1 through 3 gate third-party adapters at all: without discovery an
adapter cannot be installed, and without declared settings and conditional
capability acceptance every adapter invents its own validation and failure
behavior. Requirements 4 through 8 and 11 are correctness and uniformity of the
shared contract. Requirements 9 and 10 reduce duplication and drift.

Not everything failed to transfer. The `capabilities` block worked: both adapters
independently declared the same four flags `false` with the same reasoning, and
neither needs any of them to ship. That contrast is useful — the descriptor's runtime
capability section is sound in intent, while `config.accepts` needed requirement 3
because it cannot express a capability that depends on settings.

## LangGraph Behaviors That Shape The Design

The graph-object, interrupt, and subgraph-streaming rows were verified empirically
against langgraph 1.2.9; the rest come from LangGraph's documented contract and
constraints the Deep Agents adapter already records.

| LangGraph behavior                                                                                                                                      | Why it matters                                                                                          | Adapter response                                                                                                                                        |
| ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A compiled graph is a `CompiledStateGraph`: it exposes `invoke`/`ainvoke` but it itself is **not callable**. A factory is callable but has no `invoke`. | The two cannot be distinguished by "is it callable", and a wrong guess fails deep inside LangGraph.     | `entrypoint.kind` is explicit; the loader validates shape up front and names the correct kind in the error.                                             |
| `interrupt()` does **not** raise. The graph returns normally with a private `__interrupt__` key holding `Interrupt` objects.                            | Projection would report a false success, or an unrelated error about the field the graph never reached. | Detect interrupts before projection; return a normalized incomplete result with `interrupted: true` and each JSON-safe payload.                         |
| State is an arbitrary typed dict holding LangChain messages, `Interrupt` objects, and any Python value.                                                 | Fabric results must be JSON.                                                                            | Coerce messages to dicts, then fail with a normalized error naming the field if still unserializable.                                                   |
| Reducers such as `add_messages` accumulate, so a resumed thread's final state replays every prior turn.                                                 | Usage taken from final state re-bills earlier turns.                                                    | Aggregate usage from this turn's `updates` deltas only.                                                                                                 |
| A node may return its whole message list rather than only new messages.                                                                                 | Usage would count earlier messages again each step.                                                     | De-duplicate turn messages by id.                                                                                                                       |
| `astream(..., subgraphs=True)` emits a composed subgraph's messages **twice** and emits subgraph-namespace `values` chunks.                             | Double-counted usage, and a subgraph's state overwriting the root graph's.                              | Do not stream subgraph namespaces; a composed subgraph merges into parent state, so the parent's update already carries its messages.                   |
| A checkpointer is bound at `compile()` and cannot be attached afterward.                                                                                | A precompiled graph cannot be given Fabric-managed persistence.                                         | `state: adapter` rejected for `compiled` and `runnable_factory`; a factory compiles with `context.checkpointer`.                                        |
| Async execution needs an async saver; the sync `SqliteSaver` raises `NotImplementedError` from async methods.                                           | Confusing resume failures.                                                                              | `AsyncSqliteSaver` from `langgraph-checkpoint-sqlite>=3,<4`, opened and closed per invocation.                                                          |
| Thread identity lives in `config.configurable.thread_id`; LangGraph owns the transcript.                                                                | Fabric needs continuity without inventing a transcript.                                                 | Derive `thread_id` deterministically from `runtime_id`; ask the checkpointer whether the thread exists to report `resumed` truthfully.                  |
| Tools are LangChain `BaseTool` instances with an `args_schema`, and a node may call one by name.                                                        | Removing a blocked tool breaks a hardcoded call site with an unknown-tool error.                        | Replace a blocked tool with a deny stub preserving name, description, and `args_schema`.                                                                |
| Graphs commonly live in an unpackaged directory, which is what NAT's `dependencies` list addresses.                                                     | Otherwise "no code changes" is unachievable in practice.                                                | `entrypoint.search_paths`, relative to `base_dir`, confined to it, and appended to `sys.path` so an agent directory cannot shadow the standard library. |
| `RemoteGraph` and remote subgraphs execute outside the process.                                                                                         | Fabric cannot govern their tools, state, or telemetry.                                                  | Runnable only if the client satisfies the local graph contract; out of scope for policy and telemetry claims.                                           |

## Runtime Lifecycle

The NAT design retains the builder, workflow, sessions, and MCP clients in memory
across `invoke` calls. **That is not available to** `adapter_kind: python` **as
implemented today**: `run_python_adapter` spawns `python -m <module>` per
invocation, `start` performs only interpreter preflight, and `stop` emits an event.
Nothing survives in memory between invocations.

LangGraph satisfies the same runtime-reuse criterion a different way:

| Stage           | This adapter                                                                                                      |
| --------------- | ----------------------------------------------------------------------------------------------------------------- |
| `start_runtime` | Fabric allocates a stable `runtime_id`. No graph is built.                                                        |
| `invoke`        | Derive the thread ID, open the checkpointer, build the graph, run it, project the result, close the checkpointer. |
| `stop_runtime`  | Nothing to tear down: durable state is committed and per-invocation resources close in a `finally` block.         |

Continuity comes from a **durable checkpoint keyed by** `runtime_id`, not an
in-memory object graph. That costs graph construction per invocation and gains crash
and process independence. If Fabric later adds a persistent Python adapter runtime,
this adapter can cache the compiled graph without changing the agent contract
(requirement 4).

## Configuration Model

Where NAT's `harness.settings` mirrors a framework application structure
(`workflow`, `functions`, `function_groups`), LangGraph has no such structure to
mirror. Adapter-owned settings therefore carry only graph selection and the
request/result shape — everything else the agent decides in code.

| Area              | Values                                                                                                                                  |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `entrypoint`      | `target` (`module:attr`, required), `kind`, `search_paths`                                                                              |
| `input.mode`      | `passthrough` (default), `messages`, `field`; the latter two require a text request and reject structured input rather than coercing it |
| `output.mode`     | `last_message` (default), `field`, `state`                                                                                              |
| `state`           | `none` (default), `graph`, `adapter`                                                                                                    |
| `events`          | `mode` (`none`, `summary`), `limit`; bounded post-completion node summary, not progressive streaming                                    |
| `agent`           | JSON object passed to a factory unchanged; Fabric assigns it no meaning                                                                 |
| `recursion_limit` | LangGraph recursion limit                                                                                                               |

Unknown keys are rejected. Callable import strings for input transforms, output
transforms, or tool factories are deliberately not configurable; those belong in the
factory, versioned and tested with the agent.

The descriptor publishes a `settings_schema` as NAT's does, with one difference:
here it is closed and authoritative. NAT validates loosely and defers to its
component registry; LangGraph has no registry to defer to, so the schema and the
adapter are the only validation there will ever be.

### Factory Context

```python
def build_graph(context: LangGraphAdapterContext):
    model = context.chat_model("default")
    tools = [*context.mcp_tools(), *context.guard_tools([my_local_tool])]
    workflow = compile_my_graph(model=model, tools=tools, **context.agent_settings)
    return workflow.compile(checkpointer=context.checkpointer)
```

| Surface                                              | Provides                                                          | Boundary                                                                          |
| ---------------------------------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `chat_model(alias)`                                  | LangChain chat model from a Fabric alias, memoized per invocation | NVIDIA and OpenAI-compatible, then an `init_chat_model` fallback                  |
| `mcp_tools()`                                        | Tools from Fabric MCP servers, policy applied                     | Transport validated; fails closed                                                 |
| `guard_tools(tools)`                                 | `tools.blocked` applied to the agent's own tools                  | Only tools passed through this helper are governed                                |
| `checkpointer`, `runnable_config`, `thread_id`       | Adapter-owned persistence and thread identity                     | The factory must compile with the checkpointer and merge, not replace, the config |
| `workspace`, `artifact_dir`                          | Resolved Fabric paths                                             | The adapter does not sandbox the filesystem                                       |
| `callbacks`                                          | Telemetry callbacks to merge                                      | Empty until a provider is validated and declared                                  |
| `agent_settings`, `agent_name`, `blocked_tool_names` | Read-only JSON configuration                                      | Fabric assigns domain fields no meaning                                           |

`current_context()` is a compatibility shim for a module that builds its graph at
import time, mirroring NAT's `SyncBuilder.current()` accessor. It is valid only
during graph construction; a `factory` entry point is preferred because the
dependency is explicit.

## Supported Configuration

Only where LangGraph differs from the parent document's mappings. Credential
resolution, `runtime.*` enforcement, and provider field passthrough behave as
specified there.

| `FabricConfig` field                                           | Target                                      | Behavior                                                                        |
| -------------------------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------- |
| `models.<name>`                         | `context.chat_model("<name>")`              | Built only when a factory asks for it. There is no `llms` block to generate, and nothing can be injected into a compiled graph. |
| `models.<name>.provider`                | LangChain client                            | `nvidia`/`openai`/`openai-compatible` via `ChatOpenAI`, else `init_chat_model`. NVIDIA and unspecified default to the NVIDIA endpoint. |
| `mcp.servers.<name>` (`harness_native`) | LangChain tools via `MultiServerMCPClient`  | Only for `factory`; rejected otherwise. Loaded per server so selectors can resolve origin.                                      |
| `tools.blocked`                         | Deny stubs over context-supplied tools      | Only for `factory`; rejected otherwise. No declared inventory exists to filter.                                                |
| `environment.workspace`, `artifacts`    | `context.workspace`, `context.artifact_dir` | Paths only; the adapter does not confine the filesystem                                                                        |
| `runtime_context.runtime_id`            | LangGraph `thread_id`                       | Deterministic derivation; the basis of runtime reuse, in place of in-memory retention                                           |

## Tool Policy

The parent document's evaluation order and precedence apply unchanged. Two
LangGraph-specific differences:

**Only context-supplied tools are governable.** A graph builds its own tool surface,
so there is no declared inventory to filter. Policy reaches tools passed through
`context.mcp_tools()` or `context.guard_tools(...)`; a graph that wires tools
directly owns that boundary. This is why `tools.blocked` is rejected outright for
`compiled` and `runnable_factory`.

**Selector naming did not transfer, and was reconciled.** NAT's canonical
`<group>__<member>` has no LangGraph equivalent:
`MultiServerMCPClient.get_tools()` returns tools carrying the bare MCP tool name,
and its `tool_name_prefix` option defaults to `False` and joins with a **single**
underscore, which cannot be parsed unambiguously — server `web` with tool
`search_docs` is indistinguishable from server `web_search` with tool `docs`. The
adapter therefore loads tools per server, keeps the origin, and resolves both forms:
qualified `server__tool` denies one server's tool, a bare name denies it everywhere.
It does not rename tools, because renaming changes what the model and the graph's
own call sites see.

Beyond MCP, a tool compiled into a graph stays LangGraph-specific: unlike a NAT
function it has no registry entry to declare, so MCP is the only portable boundary
available to it.

## Initially Unsupported

Cancellation, config updates, service lifecycle, and Fabric skills are unsupported
for the same reasons the parent document gives for NAT, and the descriptor declares
all four runtime capabilities `false`. Requirements 1, 5, 6, and 8 cover the contract
gaps.

**Streaming needs a distinction.** `capabilities.streaming` means "whether
invocations can emit progressive output" — the result channel back to the Fabric
caller — and the Python adapter transport cannot do it, since it runs a subprocess to
completion and parses one stdout document. That is separate from NeMo Relay's ATOF
event stream, which is genuinely live: its endpoints support `http_post`,
`websocket`, and `ndjson`. So live observability is available through the telemetry
channel even while the result channel is not, and declaring `streaming: false` does
not contradict emitting ATOF.

This adapter does not emit ATOF **yet**, for a different reason: it declares no
telemetry provider until one is validated end to end. LangGraph makes the gap
particularly visible — `astream` already yields incremental node and message events,
and the adapter already consumes them to attribute usage to the turn and build the
bounded event summary. The incremental data exists; there is simply no channel to
forward it live, so it is buffered and reported after completion.

**Progressive output is not a prerequisite.** Nothing about LangGraph requires it for
a request/response invocation: both validation agents run end to end today, and
driving the graph with `astream` rather than `ainvoke` is an internal choice made for
usage attribution, not to satisfy Fabric. The parent document reaches the same
conclusion for NAT, declaring `streaming: false` and listing it as not implemented by
the initial adapter lifecycle. Two frameworks that both stream natively independently
concluded they do not need it to ship, which is worth recording as agreement rather
than as a gap.

Note that `capabilities` is a self-declared claim. `resolve_runtime_capabilities`
only ANDs the descriptor's flag with "is this a process or python adapter", and no
runtime path implements progressive delivery, so a descriptor claiming
`streaming: true` would be propagated into the plan as fact and simply not work.
Unlike `tools.blocked`, which fails closed with `UnsupportedToolsPolicy`, nothing
verifies these four flags (requirement 11).

What is unsupported specifically because of LangGraph:

| Capability | Reason |
| --- | --- |
| Answering an interrupt | The run returns a normalized incomplete result; the thread stays checkpointed for LangGraph to resume directly |
| `tools.blocked`, Fabric MCP, `state: adapter` for `compiled`/`runnable_factory` | The graph owns its tools and checkpointer; rejected rather than silently ignored |
| Subgraph usage without state merging | A subgraph keeping messages in a separate channel contributes no usage |
| Graph-level telemetry | No LangGraph provider validated end to end, so the descriptor declares none |

## Validation

Two custom agents, one adapter, no agent-specific code. The suite runs them as real
compiled graphs with a real SQLite checkpointer; only the chat model is substituted.

| Agent               | Characteristics                                                                                                        | Configuration                                                                                              |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `triage_agent`      | Custom state schema, conditional routing, own prompts, own chat model, compiled at import, never imports Fabric        | `compiled`, `input: field(ticket)`, `output: last_message`, `state: none`                                  |
| `case_review_agent` | Routed research and review nodes, domain `answer` field, multi-turn accumulation, application tool alongside MCP tools | `factory`, `input: messages`, `output: field(answer)`, `state: adapter`, model alias, MCP, `tools.blocked` |

Proven by `[tests/adapters/test_langgraph_adapter.py](../../tests/adapters/test_langgraph_adapter.py)`
and `[tests/e2e/test_langgraph.py](../../tests/e2e/test_langgraph.py)`: a compiled
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

The eleven contract requirements above each need a decision in the parent document.
Decisions specific to this adapter, with a recommendation where there is one:

1. **Which requirements gate publishing this adapter?** Requirement 1 blocks it
   outright: a consumer who installs the wheel cannot use it, because Fabric will not
   find the descriptor. Resolving it also removes the packaging workaround in
   `TODO.md`. Recommendation: treat 1 as blocking and 2 and 3 as required before a
   second third-party adapter is written, since without them each adapter invents its
   own validation and rejection behavior.
2. **Should `runnable_factory` and `current_context()` remain supported?** Both exist
   only for NAT portability, and both are extra public surface Fabric must keep
   working. Recommendation: keep `runnable_factory`, which is one branch in the loader
   and lets a graph written for `langgraph_wrapper` run unchanged; treat
   `current_context()` as provisional, since ambient state is easy to misuse and the
   explicit `factory` argument is always available.
3. **Which production or partner agents replace the two example fixtures?** The
   current agents were written for this validation, so they cannot prove the adapter
   was not designed around them. Needed: at least one real agent that predates any
   Fabric involvement, ideally exercising a feature neither fixture touches, such as
   `Store`, `Send` fan-out, or structured output.
4. **Which NeMo Relay LangGraph integration version has an approved validation
   path?** Until one is named, the descriptor declares no telemetry provider, so
   enabling telemetry fails at plan time and no ATOF or OTel flows from a LangGraph
   agent. The merge point already exists as `context.callbacks`; what is missing is
   the Relay-side integration equivalent to `nemo_relay.integrations.deepagents` and
   an end-to-end test pinning the version pair.

## References

- Third Party Adapters for Fabric (Ajay Thorve) — the parent contract document.
- [NAT third-party plugin packages](https://docs.nvidia.com/nemo/agent-toolkit/latest/extend/third-party-plugins.html)
- [Running existing LangGraph agents in NAT](https://docs.nvidia.com/nemo/agent-toolkit/latest/run-workflows/existing-agents/langgraph.html)
- [`adapters/deepagents/`](../../adapters/deepagents/README.md) — in-repo precedent
  for a LangGraph-based harness.

