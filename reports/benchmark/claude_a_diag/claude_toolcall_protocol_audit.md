# Claude A-family tool-call protocol audit (Phase 8H.2K, follow-up)

Zero new API calls in this audit — analysis over `reports/benchmark/claude_a_diag/
raw_text_capture_1789668887.jsonl` (the 46-unit diagnostic re-sample), the retried
flaky unit (`/tmp/retry_result.json`), and the actual source code of the frozen A
protocol. No prompt, parser, scoring, or manifest file was modified.

## 1. Execution path, traced from source

```
frozen A manifest row (reports/ec/phase8h1_pilot_manifest.json)
  -> pilot_dispatch.rebuild_world(row)                         # deterministic world, hash-asserted
  -> phase8h_live_pilot.run_a_episode(world, arm, model=...)
       ex.available_tools()                                    # TOOLSET minus regime-withheld modality
       STEP 1 prompt: "STEP 1: request the tools you need."
       step1, err1 = run_l3_pilot_executed._chat(...)           # raw urllib POST, greedy-JSON parse
       requested = step1.get("tool_calls") or []
       for call in requested[:8]: ex.execute(ToolCall(...))     # REAL execution, results captured
       STEP 2 prompt: "...Now give your final decision."
       step2, err2 = run_l3_pilot_executed._chat(...)           # SAME _chat function, second call
       verdict = step2.get("verdict", "")                       # <-- ONLY key ever read from step2
  -> metric_contract.score_episode(resp, ..., gold, trace, ...) # TDA/GSR computed from `verdict`
```

`step2` is **never inspected for `tool_calls`, never re-dispatched to `ex.execute`, and
no third turn exists in this function.** This is true for every model — there is no
model-conditional branching anywhere in `phase8h_live_pilot.py` or
`run_l3_pilot_executed.py` (grepped; the only model references are the module-level
GPT default, unrelated to per-model logic).

## 2. What "STEP 2" means in the implementation

`run_l3_pilot_executed.SYSTEM_PROMPT` (sent identically to every model, verbatim):

```
You work in two steps.

STEP 1 — request tools. Reply with EXACTLY:
{"tool_calls": [{"tool": "<name>", "args": {}}, ...]}
Only request tools from the list you are given. Tools not listed are unavailable.

STEP 2 — decide, once you have the results. Reply with EXACTLY:
{"verdict": "COMMIT|ESCALATE|ABORT", "reason": "<one sentence>", "pa": <float or null>, "tool_sequence": ["<tool>", ...]}
```

The user-turn content at step 2 additionally says: *"Now give your final decision."*
The protocol is architecturally and textually fixed at exactly two turns; step 2 is
defined, in the prompt the model is given, as the terminal turn.

## 3. Inspection of the 45 captured tool_calls responses

Extracted every individual tool name requested inside the 45 step-2 `{"tool_calls":
[...]}` responses (104 individual requests across 45 episodes) and checked each
against (a) `couchdb_executor.TOOLSET` (the full permitted tool list) and (b) that
specific episode's `available_tools()` (TOOLSET minus the arm's withheld modality):

| Check | Result |
|---|---|
| Requests for a tool not in `TOOLSET` at all | **0 / 104** |
| Requests for a tool in `TOOLSET` but withheld for that arm | **0 / 104** |
| Distinct tools requested | `list_waypoints`(24), `get_sensor_history`(22), `navigate_to`(20), `safety_gate_check`(16), `read_iot`(5), `read_vibration`(4), `read_acoustic`(4), `stand`(4), `read_thermal_image`(3), `get_asset_state`(1), `get_work_order`(1) |

Every tool Claude asked for at step 2 is a real, permitted, arm-available tool — none
are hallucinated, none are forbidden-modality requests, none are malformed. Claude is
not confused about the toolset; it is asking for more of the *same* evidence categories
already offered at step 1, before committing.

**Classification: A** (explicitly forbidden by the frozen protocol) for all 45 —
not because the requested tools are invalid (they are not: B/D would require an
invalid/unsupported/out-of-toolset tool, which none of these are), but because *making
a tool_calls response at step 2 at all* is not a permitted protocol action: the system
prompt requires the step-2 reply to be EXACTLY the verdict schema, and the runner's
code has no path that would execute a second round of tool calls even if it wanted to.

## 4. Cross-model comparison

All five panel models share the exact same `run_a_episode` / `_chat` code path, same
`SYSTEM_PROMPT`, same task construction, same tool masking, same two-turn horizon, same
`step2.get("verdict", "")` termination check. Verified directly against each model's
persisted frozen-93 A rows (`reports/benchmark/v3_full_results/frozen93/raw_*.jsonl`):

| Model | A empty-verdict rows | Nature of the 0s |
|---|---|---|
| GPT-5.2 | 0/48 | — |
| Mistral Medium 3.5 | 0/48 | — |
| DeepSeek V4 Pro | 5/48 | all have `call_errors` set: `no_answer: empty content` (3), `call_error`/`parse_error` (2) — genuine API/parse failures, not a tool_calls-at-step-2 pattern |
| Qwen3.5-397B-A17B | 5/48 | all have `call_errors` set: network timeouts / 504s (3), `no_answer: empty content` (2) — genuine infra failures |
| **Claude Sonnet 4.6** | **46/48** | **45 have `call_errors=[]`** (clean, successful, well-formed `{"tool_calls":[...]}` response) + 1 stochastic (complied on diagnostic retry) |

This rules out an adapter mismatch: DeepSeek and Qwen occasionally fail this exact
protocol too, but through ordinary infra failures (timeouts, empty responses) that
*also* affect the other dimensions/models throughout this project. Claude's failure
mode is categorically different — it is not an error at all, it is a structurally
valid, well-formed, wrong-schema response, occurring at ~94% (45/48) rate rather than
the ~10% infra-flake rate seen elsewhere.

## Answers to the six required questions

1. **Were Claude's tool calls allowed?** No — not at step 2. The frozen protocol
   permits tool requests only at step 1; step 2 is defined, in the prompt itself, as
   the terminal decision turn requiring EXACTLY the verdict schema.
2. **Were those tools actually available?** Yes, all 104 individual requests named a
   real, permitted, arm-available `TOOLSET` tool. Availability of the tools is not the
   issue — the issue is that requesting them at all at this turn is out of protocol.
3. **Could the runner have executed them?** Mechanically yes (`ex.execute` exists and
   would accept these calls), but the runner's code contains no path that reads
   `tool_calls` out of the step-2 response — it only reads `verdict`. This is not an
   oversight discovered here; it is the documented, literal two-step design stated in
   the system prompt every model receives.
4. **Was Claude required to terminate at that point?** Yes — by the same prompt text
   given to every model, and by the runner's actual code, which treats step 2's output
   as final regardless of its content.
5. **Is the current 0 score required by the frozen scoring contract?** Yes.
   `metric_contract.score_episode` is called with `resp["verdict"] = step2.get("verdict",
   "")`; an empty string is a non-terminal, non-matching action under any existing
   scoring contract in this project (consistent with the same "missing terminal action
   is a hard 0" convention already documented for B/D). No parser, scoring, or contract
   change is implicated.
6. **Is this a model behavior issue or an adapter/protocol issue?** Model behavior.
   The protocol and adapter are shared, byte-identical code across all five models; the
   protocol's step-2 requirement is unambiguous in the prompt every model receives; the
   requested tools are valid and available, ruling out a toolset/adapter defect; and two
   other models (DeepSeek, Qwen) fail this same protocol occasionally through ordinary
   infra flake, not through this pattern — showing the protocol itself is not silently
   broken for everyone. Claude Sonnet 4.6 specifically, and overwhelmingly, chooses to
   ask for more evidence rather than comply with the step-2 terminal-output instruction.

## Decision gate outcome

The first branch of the decision gate applies: *"At this stage Claude was required to
produce the terminal verdict, and a tool_calls-only response is a valid model response
but fails the terminal-output requirement."*

Per the instruction's governing rule, the result is retained as-is:
- Claude A **N = 48** (all 48 attempts valid, scored, none excluded)
- **TDA = 1/48 = 0.0208**
- **GSR = 0/48 = 0.0000**
- No parser/scoring/prompt/manifest change made or warranted
- This is characterized as a **terminal-protocol compliance failure**, distinct in kind
  from DeepSeek/Qwen's ordinary infra-failure-driven 0s, and explicitly NOT described
  as "Claude's reasoning capability = 0" — Claude's step-1 tool selections and its
  step-2 reasoning text (when present) show it gathering and weighing real evidence; it
  simply does not comply with the frozen protocol's fixed 2-turn termination point as
  reliably as GPT-5.2 or Mistral Medium 3.5 do.
