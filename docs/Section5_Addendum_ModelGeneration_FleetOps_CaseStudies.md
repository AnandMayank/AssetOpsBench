# Addendum: model-generation comparison, fleet-ops framing, case studies

Written for: paper authors. Everything below is either computed fresh from existing raw
data in this pass, or explicitly flagged as not-yet-verified. Nothing here restates the
Section 5 rewrite or the B failure-analysis addendum already committed.

## 1. GPT-5.2 -> GPT-6-Astra: a real same-episode generation comparison

You asked whether a newer model (GPT-6-Astra) actually improved over its predecessor
(GPT-5.2), or just looks different on a different pool. This is now checkable directly:
both models were run on the **identical** frozen-93 pool (93/93 episode ids match exactly),
so the comparison needs no cross-pool adjustment.

On A's 48 shared episodes:

| | TDA | GSR | TDA-GSR gap |
|---|---|---|---|
| GPT-5.2 | 66.7% | 43.8% | 22.9 pp |
| GPT-6-Astra | 77.1% | 47.9% | 29.2 pp |

At the episode level: 39/48 verdicts are identical between the two models. Of the 9 that
differ, GPT-6-Astra converts 7 of GPT-5.2's wrong answers to correct, and breaks 2 that
GPT-5.2 had right -- a net +5 correct episodes.

**This contradicts the premise you asked me to check** ("newer model performed better but
didn't show much improvement compared to earlier") -- on this specific, controlled,
same-episode comparison, GPT-6-Astra shows a genuine, non-trivial improvement on both TDA
and GSR. What's more interesting is what *doesn't* improve alongside it: the TDA-GSR gap
widens from 22.9pp to 29.2pp. The newer model is both more accurate and proportionally
*more* prone to reaching a correct-looking terminal action without the grounding evidence
behind it -- accuracy improved faster than grounding did. That's a sharper, more specific
claim than either "newer models are better" or "newer models haven't improved," and it's
the one the data actually supports.

**Caveat:** this is one model family, one pool (N=48), one generation step. It is a real,
checkable data point, not a trend -- I would not generalize this to "newer models always
widen the gap" without the same comparison on other model families where both generations
were run on identical pools (worth checking if any other pair qualifies).

## 2. Physical grounding (D): a concrete "right answer, no reasoning" case

Aggregate D numbers (GPT-6-Astra: CC=0.886, CSA=0.514, LCA=0.464) already show the gap
between terminal correctness and constraint recovery. Here is what that gap looks like at
the level of one episode, useful for an appendix good/bad-example pair:

> Episode `T-D-PHYS-REACH::GEN-DPHYS-joint_and_collision-inadmissible-hydraulic_pump_1-060014`.
> Gold: `joint_and_collision` is the sole violated constraint (all five others satisfied),
> limiting constraint `joint_and_collision`, correct terminal action `ESCALATE`.
> GPT-6-Astra's response: `terminal_action: ESCALATE` (correct, CC=1) with
> `predicted_violated_constraints: []` and `predicted_limiting_constraint: null` -- an
> **empty** constraint set and no named limiting constraint at all (CSA=0, LCA=0).

The model reached the right terminal action while naming *zero* constraints as the reason
-- not a wrong constraint, no constraint whatsoever. This is a stronger illustration than an
aggregate CSA number: it shows a model can hit the correct DISPATCH/ESCALATE decision
through something other than the six-constraint reasoning the task is built to test, most
plausibly a prior toward ESCALATE-when-uncertain (consistent with the same over-escalation
bias found in B). For fleet operators, this is the concrete failure a physical-admissibility
gate exists to catch: a terminal decision with an empty or absent justification should never
be treated as equivalent to one that correctly names the binding constraint, even when both
happen to reach the same action.

## 3. Fleet-management framing (Boston Dynamics Orbit)

You pointed at bostondynamics.com/products/orbit/ as a model for how this should read to an
industry audience. I checked what Orbit actually is rather than assume: it's Boston
Dynamics' real, shipping fleet-management product -- a cloud dashboard that aggregates Spot
robots across sites, showing live robot locations, active missions, **inspection alerts**,
and equipment-health status, with centralized analytics across a fleet (sources below).

This maps cleanly onto the five capabilities, and the mapping is worth stating explicitly
because it turns "we measured five metrics" into "here is what an operations dashboard
would do with each one":

- **A (evidence grounding) -> alert-verification gate.** Before an inspection alert
  is marked resolved on the dashboard, require the underlying decision to be GSR-passing,
  not just TDA-passing -- i.e., don't let a plausible-looking ESCALATE/COMMIT close an alert
  unless the evidence trail behind it is real.
- **B (acquisition) -> "evidence sought" indicator, separate from "decision reached."**
  A fleet dashboard already shows mission status; this argues for a second, independent
  status specifically for whether the robot acquired the evidence it needed before deciding
  -- visible even when the terminal decision looks fine.
- **C (procedural coverage) -> mission-completion percentage, not mission-success flag.**
  Orbit-style dashboards report whether a mission completed; this argues the completion
  metric should be the fraction of required steps actually executed, not a boolean, given
  GPT-6-Astra/Gemini's 100%-ordering/~2-19%-coverage pattern (Section 5.3-C) -- a mission
  can look "done" while performing almost none of the inspection.
- **D (physical admissibility) -> pre-dispatch gate, not post-hoc audit.** The case study
  above is exactly the failure a fleet operator cares about at dispatch time, before the
  robot moves, not after: a decision with no named physical justification should block
  dispatch even if the decision itself (ESCALATE) happens to be safe.
- **E (temporal validity) -> staleness flag on cached readings.** A fleet dashboard that
  shows "last known gauge value: X" needs a validity window attached to that value, not just
  a timestamp -- this is precisely what E measures (Table 9's sequence example: an agent
  that runs out of acquisition budget must ABORT rather than reuse a stale reading).

**Sources:** [Boston Dynamics -- Orbit Robot Fleet Management Software](https://bostondynamics.com/products/orbit/), [Boston Dynamics -- Robot Fleet Management Lifts Off with Spot](https://bostondynamics.com/blog/robot-fleet-management-lifts-off-with-spot/), [IoT World Today -- Boston Dynamics Launches Robotic Fleet Management Platform](https://www.iotworldtoday.com/iiot/boston-dynamics-launches-robotic-fleet-management-platform)

## 4. L1/L2/L3 evidence provenance: real in the codebase, not yet verified in results

You asked why L1/L2/L3 (Section 3.1's provenance taxonomy: L1 = capture from the inspected
asset, L2 = replay from another asset of the same class, L3 = diagnostic-only) hasn't
appeared in the results section, since the paper states it's load-bearing for scoring.

I checked this rather than assume it's wired up: the taxonomy is real and defined in code
(`src/orchestrator/scenario_contract.py`'s `PROVENANCE_VALUES`:
`L1_REAL_ASSET_EVIDENCE`, `L2_ASSET_CLASS_EVIDENCE_REPLAY`, `L3_EVIDENCE_DIAGNOSTIC`,
`SIMULATED_UNCLASSIFIED`), and it's asserted as a required, validated field per scenario. But
in the time available this session I could **not** verify whether the episodes actually
scored in the current panels (frozen-93, A-expanded, etc.) span more than one provenance
class, or whether GSR's scoring path anywhere differentiates a model that commits on L2
evidence when L1 was required (the paper's own stated purpose for the field). This needs a
dedicated pass: pull the `evidence_provenance_class` for every scored episode, check the
distribution, and if L2/L3 episodes exist in the scored pools, check whether any model
committed on non-L1 evidence when the scenario required L1. I would not add an L1/L2/L3
result to the paper without doing that check first -- it's a real gap between what the paper
claims the field does and what I've confirmed it does in the actually-evaluated data.

## 5. Temporal grounding (E): what's defensible to say now vs. what needs the pooled run

E's per-model frozen-93 numbers (N=18) are already in Section 5.3. The richer claim --
precision/recall on stale-evidence *re-observation*, at the pooled N=180 scale (E-expanded-
v1+v2, all 5 models' raw data now complete) -- has not been aggregated yet; that's the one
number still marked pending in the Section 5 rewrite. Once computed, this is the piece that
would let a per-model E discussion go beyond "these are the scores" into "here is how often
each model reuses a stale reading vs. correctly re-observes, and how that differs from
recognizing staleness after the fact" (mirroring Table 9's worked example). I did not fake
this number here; it needs the aggregation script run once background compute is free.
