# Abstract Sentence Reconstruction — Candidate Factual Versions

These candidates are built ONLY from values verified in `ABSTRACT_FACT_SHEET_FINAL.md`
and the rest of this audit. None retains the "19 grounded worlds / 93 scenario
templates / six industrial asset types" framing, because none of those three numbers
was independently reproduced against current V3 artifacts this pass (see
`abstract_claim_inventory.csv` claims C2–C4). The paper is NOT edited by this document;
these are recommendations only.

## Candidate 1 — scale-first, most conservative

> InspectionBench is an execution-grounded benchmark for agentic industrial inspection
> comprising 4,075 canonical episodes spanning five evaluation capabilities (evidence
> grounding, evidence acquisition, procedural grounding, relational physical grounding,
> and temporal grounding) across 4 canonical industrial assets (a chiller, two pumps,
> and a motor). Episodes are produced by a deterministic world-to-scenario compiler with
> matched-group construction (e.g. 800 A-family worlds × 3 evidence-delivery regimes,
> holding gold decisions invariant while varying what evidence is delivered to the
> agent). Of the 4,075 canonical episodes, 229 have been evaluated against a panel of 5
> frontier models; the remaining episodes are scripted, generation-time constructs with
> no model execution against them to date.

## Candidate 2 — modality/provenance-forward, more cautious

> InspectionBench comprises 4,075 canonical, matched-generation inspection episodes over
> 4 grounded industrial assets, requiring agents to acquire visual, IoT/telemetry,
> acoustic, thermal, and maintenance-record evidence — under an explicit,
> generation-time lineage provenance scheme — before committing to a terminal
> operational decision. A frozen 2-turn protocol and a panel of 5 frontier models were
> used to evaluate 229 of these episodes; we find that the reported grounding gap for at
> least one evaluated model is substantially attributable to terminal-verdict
> protocol-compliance behavior rather than grounding accuracy alone, and that at least
> one capability's model ranking inverts under a corrected procedural-coverage metric
> versus the metric first used to report it.

## Candidate 3 — narrowest, safest under this audit's findings

> InspectionBench is a benchmark of 4,075 canonical inspection episodes, built by a
> deterministic compiler from a small, fixed set of 4 grounded industrial assets, with
> matched-group construction that holds gold decisions fixed while varying delivered
> evidence across regimes. A subset of 229 episodes (spanning all five capability
> dimensions) has been executed against 5 frontier models to date; canonical benchmark
> size and model-evaluated N are reported separately throughout, since coverage ranges
> from 100% (B-Acquisition, D-physical) down to roughly 1-2% (A, E) of each family's
> canonical pool.

## Explicit note on the sentence this task asked NOT to retain unmodified

> "agents acquire visual, operational, and telemetry evidence with explicit provenance
> and execute inspections before making downstream operational decisions"

This audit's finding (`provenance_audit.json`): the "explicit provenance" component is
only PARTIALLY supported. A generation-time lineage provenance mechanism is real and
spans nearly all 4,075 episodes; a separate evidence-SOURCE tier (L1/L2/L3) is defined
in code but was not found persisted in any current V3 manifest, so it cannot be claimed
to be attached to evidence benchmark-wide. If the sentence is retained, it should be
narrowed to name the generation-time lineage mechanism specifically, or should
explicitly scope the provenance claim to the 50/66 B-Acquisition episodes that carry a
`source_provenance` block, rather than asserting it benchmark-wide.
