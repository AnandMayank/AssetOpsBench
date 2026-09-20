# Audit note: DIGITAL_ONLY GSR is structurally zero, not a capability finding

**Finding.** In `matched_regime_delta_v1.{json,md}` (pushed in commit `3ed2e2b`), the
FULL-vs-DIGITAL_ONLY GSR delta (e.g. GPT-5.2 +0.625, Mistral +0.688) was reported
alongside the FULL-vs-PHYSICAL_ONLY delta as if both were comparable empirical
capability signals. They are not.

**Root cause (verified against source, not inferred).**
`src/orchestrator/l3_grounded_scoring.py`:
- `REQUIRED_MODALITY["FM-6a"] = "physical"`, `REQUIRED_MODALITY["FM-7a"] = "physical"`
  (lines 46-47) -- the only two failure modes the A family uses
  (`phase8h_live_pilot.run_a_episode`: `fm = "FM-7a" if is_contradiction else "FM-6a"`).
- `cc_grounded()` (lines 101-153): `need = REQUIRED_MODALITY.get(fm, "physical")`,
  `matching = {oid: e for oid, e in delivered.items() if e.modality == need}`,
  `observation_delivered = bool(matching)`, and
  `grounded = int(bool(cc) and observation_delivered and ...)`.
- `phase8h_live_pilot.py`: `REGIME_WITHHELD = {"FULL": [], "PHYSICAL_ONLY": ["digital"],
  "DIGITAL_ONLY": ["physical"]}` -- DIGITAL_ONLY withholds physical evidence entirely.

Since DIGITAL_ONLY never delivers a physical-modality observation, `matching` is the
empty set for every DIGITAL_ONLY episode, so `observation_delivered = False` and
`grounded = 0` **unconditionally** -- for every model, on every DIGITAL_ONLY episode,
regardless of what the agent does. This is correct-by-design (the grounding contract
for FM-6a/FM-7a requires physical evidence), not a bug, and not fixable in the sense of
"making it non-tautological" without redefining the evidence contract itself (out of
scope -- the frozen contract is not being changed).

**Consequence.** The FULL-vs-DIGITAL_ONLY GSR delta is mathematically `FULL_GSR - 0 =
FULL_GSR` for every model -- it re-reports each model's FULL-regime GSR under a
different label, and cannot show any DIGITAL_ONLY-specific model behavior. It must not
be presented as differentiating model capability, and is excluded from the main-paper
figure for this reason.

**What remains clean and is used going forward:**
- **GSR, FULL vs PHYSICAL_ONLY**: both regimes deliver physical evidence, so this
  is a genuine, non-tautological paired comparison (only whether *digital/IoT*
  evidence being additionally available changes grounded correctness).
- **TDA, all three regimes (FULL / PHYSICAL_ONLY / DIGITAL_ONLY)**: TDA is
  `scores.get("CC")`, a plain verdict-matches-gold check with no modality
  precondition baked in -- no structural floor or ceiling in any regime. This is
  the correct metric to show DIGITAL_ONLY behavior on: whether the agent
  recognizes it lacks the required grounding evidence and escalates/aborts
  correctly, rather than whether it "grounded" (which is untestable there).

No frozen benchmark manifest, scoring contract, or runner code was changed to
produce this note -- read-only source verification only.
