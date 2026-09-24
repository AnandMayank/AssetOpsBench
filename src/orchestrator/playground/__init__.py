"""playground — Phase 1a, B2: a stateful, multi-step evidence-acquisition
evaluation built on top of (never modifying) the frozen InspectionBench
substrate. See docs/InspectionBench_StatefulEnv_Design.md and the approved
plan (futuresim-we-now-want-parsed-bunny) for the scientific design this
package implements.

Nothing under `inspectionbench/`, no frozen manifest, and none of
`b_acquisition_scoring.py` / `b_acquisition_generator.py` / `canonical_identity.py`
/ `execution_trace.py` / `pilot_dispatch.py` is modified by this package --
it only imports from them.

Offline only (Phase 1a): no module here makes a model/API call. The
model-facing runner (`pg_b2_run.py`, not yet implemented) is gated behind
G1-G4 substrate validation and the acceptance-gate tests in this package.
"""
