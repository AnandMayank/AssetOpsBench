# InspectionBench V3 — Construction Methodology

This document explains how the frozen V3 benchmark was built and what construct-validity
guarantees are structurally enforced, not merely asserted.

## 1. World-first gold derivation

Every episode's hidden physical state is sampled *before* any gold label exists
(`construction/scenario_gen.py::sample_world`), and gold is derived deterministically
from that state afterward. The sampler is structurally blocked from ever accepting a
gold-shaped parameter:

```python
FORBIDDEN_SAMPLER_PARAMS = {
    "gold", "gold_action", "label", "verdict", "expected", "answer",
    "target", "desired", "expected_verdict", "gold_label",
}
```

This is checked automatically, not only documented. `render_question()` (the function
that produces the actual agent-visible task text) has three *deliberate* omissions —
physical value, enterprise state, and IoT telemetry — each because stating them would
make gold reachable without any observation, each backed by a named regression test
(`test_question_rendering_cannot_leak_gold`).

## 2. Capability families (A–E)

### A — Evidence grounding
Regime withholding (`FULL` / `PHYSICAL_ONLY` / `DIGITAL_ONLY`) controls which real tools
are available; scored via `GSR` (grounded-support rate: did the decision depend on
delivered evidence) and `TDA` (terminal decision accuracy, secondary).

### B — Evidence acquisition
Built on a pre-existing, previously-unwired Inspection Capability Framework:
`ObservationStore` / `ObservationResolver` / `EvidenceLedger` resolve real sensor
observations from a 24,601-record substrate spanning 6 modalities
(`construction/inspection_capability/`). Four templates (B-ACQ-1..4) test modality-gap
recognition, count-sufficiency, genuine-unavailable handling, and cross-modal
reconciliation. Scored primarily on **ADA** (acquisition-decision accuracy: STOP vs
ACQUIRE), with **ASA** (which modality), **MAR**/**UAR** (missed/unnecessary
acquisition), and **UHA** (unavailable-handling accuracy, only applicable when gold
`acquisition_genuinely_unavailable=True`) as diagnostics — never combined into one
composite B score. A delivered-evidence firewall means a model's claimed
`observation_id` is checked against what the executor actually returned, never against
what merely exists in the store.

**Known, disclosed construct limitation (B-ACQ-4)**: gold's `acceptable_acquisition_set`
has cardinality ≤ 1 on every one of the 66 canonical B-Acquisition episodes — a
single-reference construct, not the originally-envisioned multi-reference (G2-style) one.
Flagged as future work, not fixed by loosening the construct.

### C — Procedural grounding
Required tool-call order is scored against the *real execution trace*
(`Stage.EXECUTED` events), never a self-reported sequence — `ordering_satisfied` is
computed from what the executor actually ran.

### D — Relational physical grounding
Split into **D-enterprise** (work-order/enterprise-state gating, 20 episodes) and
**D-physical** (70 episodes, a real MuJoCo-verified oracle over reach / joint-collision /
clearance / grasp-payload / stability / energy constraints, individually and jointly).
Scored via **CC** (terminal correctness, secondary) and, primarily, **CSA**/**CS-F1** and
**LCA** (constraint-set accuracy and limiting-constraint accuracy) — whether the agent
identified *which* constraint actually bound admissibility, which terminal correctness
alone cannot establish. This divergence is not hypothetical: full-scale pilots on this
exact 70-episode pool showed CC=0.71–0.87 while CSA=0.35–0.76 for the *same* model runs
(see the parent repository's evaluation reports for the measured numbers — not
republished here as this package is construction-only).

### E — Temporal grounding / validity
A real multi-turn sequence per asset (`construction/sequence_executor.py`) where the
hidden physical value may drift between visits; scores whether the agent re-observes
appropriately versus reuses stale state (`stale_state_reuse`,
`unnecessary_reobservation`), alongside `CC_grounded` and `PROC`.

## 3. Family-boundary discipline

A binding design constraint, enforced structurally: B's evidence-sufficiency axis is
scoped to modality coverage and evidence count — **freshness (`max_age_s`) is never used
as a B sufficiency criterion**, since that would collapse B's construct into E's. Verified
by a dedicated structural test scanning every B generator function's source for any
`max_age_s`-based sufficiency logic.

## 4. B-ACQ-1's tie-break repair (a worked example of the construct-validity discipline)

`motor_01`'s thermal substrate has 10 records tied at `quality=0.9` with identical
timestamps. The naive resolver tie-break (`max()` over `(quality, timestamp)`, Python's
first-element-wins-on-tie behavior over insertion-ordered candidates) deterministically
always returned the *same* record (`fault_class="Rotor-0"` → ESCALATE), which happened to
coincide with the "absent" arm's ESCALATE — eliminating terminal-action variance in that
one configuration. The repair: a **mechanical, label-blind rank-2 selection rule** —
resolve once, exclude that exact `observation_id`, resolve again — using the resolver's
existing `exclude_ids` field (already used elsewhere for legitimate multi-record
accumulation), reading no `fault_class` or label at selection time. This deterministically
resolves a *different* real record (`fault_class="Noload"` → COMMIT via the same,
unmodified capability rule table), restoring genuine terminal-action variance without
ever peeking at the answer to choose it. Threaded consistently through gold computation,
the scripted-to-gold runner, and the live model-pilot path so gold and live execution
never diverge.

## 5. Scale-generation discipline (66-episode B-Acquisition pool)

Real substrate audit first, not aspiration: the underlying observation store was queried
live for per-(asset, modality) record counts and quality/timestamp distributions before
any scaling decision. Findings directly shaped the design — e.g., only 3 of 8 substrate
modalities have a genuine present-for-some/absent-for-others split among the 4 canonical
assets, capping the modality-gap axis at exactly that many real configurations. **No
seed-padding**: a different seed for an otherwise-identical (asset, modality, arm)
configuration was explicitly rejected as non-informative, since the module's own
architecture note confirms B-Acquisition's decisive evidence is never
`WorldState.physical_value`. Every added axis is traceable to a specific, live-queried
substrate fact, documented per-axis in the generator source.

## 6. Frozen-93 historical pool

The oldest, most extensively validated subset (48 A + 12 B-legacy + 9 C + 6 D-enterprise
+ 18 E = 93 episodes) is pinned by SHA256
(`d2b48c0b0c9ef19f6c8f6ddd936da098ffd5d0d4023650619f01c1ee043087fe`) and never
regenerated. It is the only pool with a fully validated, real-model-facing execution
pipeline for every one of its five families in this package
(`evaluation/phase8h1_run_pilot.py`, dispatching via `construction/pilot_dispatch.py` on
`scenario_template`, never on `scenario_id`).

## 7. What is deliberately NOT in V3

- **No dedicated robot-failure-cascade family.** Physical-to-operational failure
  propagation motivates the overall design; it is not yet a separately-scored construct.
- **No LLM-judge scoring anywhere.** Every metric in every family is computed
  deterministically from real trace/tool-execution data or a real physics oracle
  (D-physical's MuJoCo-verified evaluator).
- **No merged 4,075-episode real-model evaluation.** As documented in the top-level
  README, only 229 of 4,075 episodes currently have a validated real-model execution
  path; extending this is future engineering work, tracked separately from the frozen
  benchmark definition itself.
