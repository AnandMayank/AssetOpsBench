# Class-E multi-episode state semantics (specification only — P3)

Documented so the requirement is fixed before it is built. **Not implemented**;
classes A, C and D must be stable first.

Class E (R013 FM-7c historical outlier, R043/R044 FM-23 temporal drift) needs
state that survives across episodes. The current executor is explicitly the
opposite: `reset()` and `reset_from_world()` re-establish hidden state per
episode so scenarios cannot contaminate one another, and `FixtureSession`
reverts every precondition on exit. That isolation is what makes A/C/D
reproducible, so class E cannot simply reuse it.

## Required semantics

| Element | Requirement |
|---|---|
| **Episode sequence** | An ordered run of N episodes sharing one asset and one world lineage. Sequence identity, not episode identity, is the unit of evaluation. |
| **Carried state** | Committed readings, their timestamps and their provenance. A later episode must be able to observe what an earlier one wrote. |
| **World evolution** | The hidden physical value changes between episodes under a declared process (drift, step change, or stationary). The process is part of the sampled world, so it must be drawn before any label. |
| **Reset boundary** | Reset at *sequence* start, never between episodes. `FixtureSession` revert must become sequence-scoped. |
| **Gold derivation** | `derive_gold` becomes a function of the sequence prefix, not a single state: whether drift *should* have been detected by episode k depends on episodes 1..k. |
| **Contamination control** | Sequences must not leak into each other. Requires per-sequence asset instances or a namespaced profile key. |
| **Trace model** | The existing `ExecutionTrace` is per-episode. A sequence-level trace chaining episode traces is needed so provenance survives the boundary. |

## Constructs it would newly measure

- **belief update under repeated contradiction** — does a committed reading from
  episode 1 anchor the decision in episode 3?
- **drift detection latency** — how many episodes before the change is flagged.
- **stale-state reuse** — committing a value carried from an earlier episode
  rather than re-observing.

None is measurable today: with per-episode reset, every episode begins with no
history, so an agent cannot anchor to a prior reading and the failure cannot
occur.

## Why it is deferred

It changes the reset contract, which every current guarantee rests on. Doing it
before A/C/D are stable would put the isolation property at risk to gain a
family that has three scenarios.
