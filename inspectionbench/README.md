# InspectionBench V3

**A frozen, capability-and-failure-mode benchmark for evaluating physical-to-operational
coupling in autonomous industrial inspection agents.**

This directory is a self-contained, clearly namespaced package within the broader
AssetOpsBench repository. It does not modify, replace, or depend on unrelated
AssetOpsBench functionality outside of it.

## What InspectionBench evaluates

InspectionBench decomposes the industrial inspection loop —

```
physical observation
  → evidence acquisition / interpretation
  → procedural / physical reasoning
  → operational decision
  → persistent enterprise state / downstream consequence
```

— into five complementary capability dimensions, evaluated separately rather than
collapsed into one leaderboard score:

| dimension | capability | primary metric(s) |
|---|---|---|
| **A** | Evidence grounding | GSR (grounded-support rate), TDA (secondary) |
| **B** | Evidence acquisition | ADA (acquisition-decision accuracy, primary); ASA, MAR, UAR, UHA (diagnostic) |
| **C** | Procedural grounding / execution | ordering-satisfaction rate, CC |
| **D** | Relational physical grounding | CSA / LCA (constraint-set / limiting-constraint accuracy, primary); CC (terminal, secondary) |
| **E** | Temporal grounding / validity | CC_grounded, PROC, stale-state-reuse rate |

**Terminal task success (whether the agent's final action matched gold) is measured, but
is explicitly NOT treated as evidence of the underlying capability.** A model can score
high on terminal correctness while failing the capability-specific metric that explains
*why* the correct answer was reached — and empirically does, in both directions, across
this benchmark's own validation pilots (see `docs/CONSTRUCTION.md` and the linked
evaluation reports for concrete, measured examples).

**Multiple valid execution paths are allowed.** Evidence contracts define *sufficient*
evidence requirements (e.g. "at least one thermal reading above quality threshold X"),
not one prescribed trajectory. An agent may satisfy a contract via any real tool call
that resolves to real delivered evidence; scoring is against the contract and the actual
delivered-evidence trace, never against a single canonical sequence of tool calls.

## The frozen benchmark

| | |
|---|---|
| **Canonical total** | **4,075 episodes** |
| Manifest | `manifests/final_benchmark_manifest_v3.json` |
| Frozen historical 93-episode SHA256 | `d2b48c0b0c9ef19f6c8f6ddd936da098ffd5d0d4023650619f01c1ee043087fe` |

Family composition (`final_canonical_total` = 4,075, computed from the rollup manifest,
**not** by summing the nominal `family_counts` — see the manifest's own
`family_counts_accounting_note` field: A and E overlap by a pre-existing 45-episode
cross-family dedup inherited unchanged from the prior manifest version):

| family | count | manifest |
|---|---|---|
| A (Evidence grounding) | 2,400 nominal (3,795 deduped jointly with E) | `manifests/a_e_pool_manifest.json` |
| E (Temporal grounding) | 1,440 nominal (deduped jointly with A, above) | same |
| B-legacy (historical, frozen) | 12 | part of `manifests/frozen_93_historical_manifest.json` |
| B-Acquisition | 66 | `manifests/b_acquisition_final_manifest.json` |
| C (Procedural grounding) | 112 | `manifests/cd_4000_final_manifest.json` |
| D-enterprise | 20 | same |
| D-physical | 70 | `manifests/d_physical_canonical_manifest.json` |

**A separate, smaller, historically frozen 93-episode subset** (`A`=48, `B`=12, `C`=9,
`D`=6, `E`=18) is the one pool with a fully validated, real-model-facing execution
pipeline shipped in this package (`evaluation/phase8h1_run_pilot.py`) — see
[Model-eligible subset](#model-eligible-subset-vs-canonical-total) below.

### `manifests/` contents

| file | content | note |
|---|---|---|
| `final_benchmark_manifest_v3.json` | the authoritative 4,075-episode rollup | canonical entry point |
| `final_benchmark_manifest_v2.json` | the immediate predecessor rollup (4,009 episodes, pre-B-Acquisition) | kept for provenance/diff |
| `frozen_93_historical_manifest.json` | the frozen, immutable 93-episode historical pilot pool | SHA-pinned, never regenerated |
| `b_acquisition_final_manifest.json` | the 66-episode B-Acquisition canonical set | |
| `d_physical_canonical_manifest.json` | the 70-episode D-physical canonical set, extracted verbatim from the V3 rollup's own embedded content | authoritative |
| `d_physical_generation_trace_pre_dedup_88.json` | the raw, 88-episode pre-dedup generation trace (10 duplicate fingerprints → 70 canonical) | provenance only, **not** the canonical set — use the file above |
| `cd_4000_final_manifest.json` | the C / D-enterprise pool (132 episodes) | |
| `a_e_pool_manifest.json` | the combined A/E pool (3,795 deduped canonical episodes) | large (4.3 MB) |

### Model-eligible subset vs. canonical total

**4,075 is the canonical benchmark size. It is not the same as how many episodes have an
existing, validated, real-model-facing execution pipeline shipped in this package.**

As of this release, **229 of the 4,075 episodes** (93 frozen-historical + 66
B-Acquisition + 70 D-physical) have a validated runner in `evaluation/` that dispatches
to a real model via a real tool-execution path and produces a deterministically-scored
result. The remaining 3,846 (the bulk of the A/E pool, and the C/D-enterprise pool beyond
the frozen-93 subset) currently carry only **scripted, generation-time admission
scores** (computed when the episodes were generated and validated against a scripted
gold-consistent response, to prove the construct and scorer are correct) — **these are
not model results**, and this package does not present them as such anywhere. Extending
real-model execution to the remaining pools is future engineering work, not a benchmark
redefinition.

## What this package is (and is not)

**This release is benchmark construction and reproducibility infrastructure.** It is
**not** a set of finalized model-evaluation results.

- Included: frozen manifests, generator/scorer source code, capability-contract
  definitions, the real executable tool/observation infrastructure, benchmark
  construction tests, an integrity-verification script, and the evaluation *runners*
  (the code that *can* execute a model against the benchmark).
- **Not included as final results**: any specific model's scores. Development-stage
  pilot runs exist in the working repository this package was extracted from and are
  explicitly out of scope for this release — in particular, an in-progress five-model
  evaluation run has two known, disclosed incompleteness issues (one model's D-physical
  coverage stopped at 39/70 episodes due to sustained low throughput, not failure; one
  model's evidence-grounding family showed a still-uninvestigated response-parsing
  anomaly). Neither is presented as a final result here or anywhere in this package.

## Construct: five capability dimensions, one physical-to-operational loop

Each family perturbs a different point in the loop above while holding a real,
deterministic, non-LLM-judged gold answer fixed:

- **A** withholds or exposes real sensor evidence and asks whether the agent's decision
  actually depended on evidence it was delivered (not merely whether the final label
  matched).
- **B** varies whether the initial evidence ledger satisfies a real, pre-authored
  capability contract, and scores whether the agent recognizes insufficiency, requests
  the right modality via a real `request_observation` tool call, and revises its
  decision — using a delivered-evidence firewall so a claimed observation ID must
  correspond to something the executor actually returned.
- **C** varies required tool-call order and scores compliance against the real execution
  trace, not a self-reported sequence.
- **D** poses real, MuJoCo-verified joint/collision/reach/grasp/energy/stability
  constraints (D-physical) and enterprise work-order gating (D-enterprise), scoring both
  the terminal decision and — separately — whether the agent identified the actual
  binding constraint.
- **E** revisits an asset across a real multi-turn sequence with a physical value that
  may drift, scoring whether stale evidence was inappropriately reused versus
  re-observed.

Construct-validity discipline enforced throughout the construction code (see
`docs/CONSTRUCTION.md` for the full account, including specific, named regression tests):
gold is *never* computed from a label the world sampler could see (`scenario_gen.py`'s
`FORBIDDEN_SAMPLER_PARAMS` blocks this structurally); evaluator-only fields never appear
in agent-visible prompts; every acquisition claim is checked against what the executor
actually delivered, never merely against what exists in the store.

## To our knowledge

To our knowledge, existing benchmarks do not jointly evaluate agent-initiated evidence
acquisition, evidence-delivery provenance, procedural compliance, relational physical
constraint identification, and temporal evidence validity within one shared,
industrial, tool-executing environment with fully deterministic (non-LLM-judge) scoring.
We do not claim this is the only benchmark that touches any *one* of these axes
individually — several do, and are surveyed in the construction documentation — nor do we
claim InspectionBench eliminates all possible benchmark shortcuts; it reduces specific,
named ones (see `docs/CONSTRUCTION.md`).

**InspectionBench V3 does not contain a dedicated robot-failure-cascade evaluation
family.** Physical-to-operational failure propagation is the benchmark's motivating
design principle, not a specific, separately-scored family in this version.

## Reproducibility

### 1. Install dependencies

This package's code targets the same Python environment as the parent repository
(Python 3.12; see the parent repo's `pyproject.toml`/`uv.lock` for the exact pinned
dependency set — `pytest`, standard library only for the construction/scoring modules
themselves). From the parent repository root:

```bash
uv sync   # or: pip install -e .
```

### 2. Obtain/configure required assets

The construction code in `construction/inspection_capability/` resolves real sensor
observations from a substrate of real, third-party-licensed records
(MIMII CC-BY-SA-4.0, REVA Mendeley, a Kaggle casting-defect dataset, a GitHub blade-defect
dataset, and a Zenodo transformer dataset — full per-record `source_license` and
`dataset_lineage` fields are carried on every observation). **That raw substrate file
(~35 MB, mixed licensing) is intentionally not redistributed in this package.** The
manifests here contain the derived, synthetic episode *metadata* (asset, modality,
world/seed, gold) — not the raw sensor payloads — which is safe to publish. To run the
executable evaluation path end-to-end (real tool calls resolving real observations), a
researcher needs to reconstruct or obtain that substrate separately, following the
per-record `dataset_lineage` provenance already present in `inspection_capabilities.json`
and the observation-resolution code.

Model API access (TokenRouter, or any OpenAI-compatible proxy) requires your own
credentials, configured via environment variables (`TOKENROUTER_API_KEY`,
`TOKENROUTER_BASE_URL`) — **never hardcoded**, and never included in this package.

### 3. Run the integrity check

```bash
python3 inspectionbench/scripts/verify_integrity.py
```

Expects `INTEGRITY OK`. Verifies the frozen-93 SHA256, the V3 canonical total, every
manifest referenced by the rollup, and the B-Acquisition/D-physical canonical counts —
entirely from files inside this package, no network access required.

### 4. Load the frozen manifest

```python
import json
v3 = json.load(open("inspectionbench/manifests/final_benchmark_manifest_v3.json"))
print(v3["final_canonical_total"])  # 4075
```

### 5. Run the construction/validation tests

```bash
cd inspectionbench
PYTHONHASHSEED=0 python3 -m pytest tests/ -q
```

(Some tests import sibling modules from the parent repository's `src/orchestrator/`
path for shared infrastructure not duplicated in this package, e.g. `pytest` fixtures for
real executor integration tests; run from the parent repository with its environment
active for full coverage. Construction/scoring-only tests run standalone.)

## Provenance and licensing

- **This package's own code and manifests** (generators, scorers, capability contracts,
  episode metadata): released under the parent repository's license (`LICENSE` at the
  repository root).
- **The underlying sensor-observation substrate** referenced by the manifests (not
  included, see above): each record carries its own third-party license
  (CC-BY-SA-4.0 and others) and dataset lineage — consult
  `construction/inspection_capability/data/inspection_capabilities.json` and the
  `dataset_lineage`/`source_license` convention documented in `docs/CONSTRUCTION.md`
  before redistributing any reconstructed copy.
- No private facility data, credentials, or internal-only infrastructure information is
  included anywhere in this package.

## Directory layout

```
inspectionbench/
  README.md                    -- this file
  manifests/                   -- frozen benchmark artifacts (episode metadata, gold)
  construction/                -- generators, scorers, capability contracts, executor
  evaluation/                  -- real-model evaluation runners (not results)
  tests/                       -- construction/validation test suite
  scripts/verify_integrity.py  -- standalone, offline integrity check
  docs/CONSTRUCTION.md         -- full construction methodology and construct-validity notes
```
