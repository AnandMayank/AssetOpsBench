"""test_scenario_contract.py — Pass R1: unified typed gold contract.

Requires the AssetOpsBenchScenarioGeneration checkout (same convention as
test_rc004_gold_trees.py). No network calls.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ORCH_SRC = Path(__file__).resolve().parents[1]  # src/orchestrator
sys.path.insert(0, str(_ORCH_SRC))

_SCENARIOS_DIR = Path.home() / "AssetOpsBenchScenarioGeneration" / "RobotInspection" / "Scenarios"
requires_scenario_repo = pytest.mark.skipif(
    not _SCENARIOS_DIR.exists(), reason="AssetOpsBenchScenarioGeneration checkout not found")

if _SCENARIOS_DIR.exists():
    import scenario_contract as SC


@requires_scenario_repo
def test_all_scenarios_validate():
    """89 at R1; R3 added R090/R091 (FM-29 acoustic) -- validate against the
    actual scenario_R* directory count, not a number frozen at R1."""
    reg = SC.build_registry()
    SC.validate_registry(reg)
    assert len(reg) == len(SC._all_scenario_ids())
    assert len(reg) >= 89


@requires_scenario_repo
def test_no_scenario_invented_gold():
    """build_registry raises rather than silently inventing a value; a clean
    return means every one of the 89 resolved through a real source."""
    reg = SC.build_registry()
    for sid, g in reg.items():
        assert g.gold_action, f"{sid}: empty gold_action"


@requires_scenario_repo
def test_r021_r025_now_parse():
    """The two dirs that parsed under neither original convention (no
    'Expected verdict:'/'Gold decision:' line -- gold lived only in a JSON
    blob) must now resolve via the typed contract."""
    reg = SC.build_registry()
    assert reg["R021"].gold_action == "COMMIT"
    assert reg["R025"].gold_action == "FLAG_SET"


@requires_scenario_repo
def test_thermal_scenarios_match_l3_arms_exactly():
    """R1 must not perturb the live thermal scoring path: the typed gold for
    R049/R050/R088/R089 must equal l3_arms.SCENARIOS's own values exactly
    (same source, cross-checked, not re-derived)."""
    import l3_arms
    reg = SC.build_registry()
    for sid in l3_arms.THERMAL_SCENARIOS:
        assert reg[sid].gold_action == l3_arms.SCENARIOS[sid]["gold"], sid
        assert reg[sid].verdict_convention == "l3_arms.SCENARIOS"


@requires_scenario_repo
def test_thermal_gold_source_and_provenance():
    reg = SC.build_registry()
    for sid in ("R049", "R050", "R088", "R089"):
        g = reg[sid]
        assert g.gold_source == "externally_mapped"
        assert g.evidence_provenance_class == "L2_ASSET_CLASS_EVIDENCE_REPLAY"


@requires_scenario_repo
def test_render_groundtruth_header_round_trips():
    """typed -> rendered header -> reparsed must reproduce gold_action
    exactly, for every scenario (not just thermal)."""
    reg = SC.build_registry()
    for sid, g in reg.items():
        header = SC.render_groundtruth_header(g)
        reparsed = SC._parse_groundtruth(header)
        expected = " / ".join(g.gold_action) if isinstance(g.gold_action, tuple) else g.gold_action
        assert reparsed["verdict"] == expected, sid


@requires_scenario_repo
def test_gold_source_values_are_declared_honestly():
    """R1 had no latent WORLD object (that's R4's job) -- at R1, no record
    could honestly claim derived_from_world. R4 (see test_world_gold.py)
    introduces the WORLD and demonstrates it for exactly the six
    gauge-band scenarios world.GAUGE_BAND_DERIVED_SCENARIOS names -- this
    test now pins that count rather than forbidding the value outright, so
    a future pass silently expanding "derived_from_world" to scenarios that
    were never actually verified against WORLD state would fail here."""
    reg = SC.build_registry()
    sources = {g.gold_source for g in reg.values()}
    assert sources <= {"sme_adjudicated", "externally_mapped", "derived_from_world"}
    derived = [sid for sid, g in reg.items() if g.gold_source == "derived_from_world"]
    assert len(derived) == 6, (
        f"expected exactly the R4-verified gauge-band family, got {sorted(derived)}")


@requires_scenario_repo
def test_conflicts_are_flagged_not_silently_resolved():
    """Real cross-source gold disagreements exist in this corpus (measured:
    12, e.g. R014 gold_robot_inspection.json=COMMIT vs groundtruth.txt=
    ESCALATE). build_registry must not pick a winner silently -- every
    flagged scenario carries extra['gold_conflict_with_groundtruth_txt']."""
    reg, conflicts = SC.build_registry_with_conflicts()
    assert len(conflicts) >= 10
    flagged_ids = {c["scenario_id"] for c in conflicts}
    for sid in flagged_ids:
        if sid in reg:  # R021/R025 conflicts are appended post-hoc in the CLI path
            assert reg[sid].extra.get("gold_conflict_with_groundtruth_txt") is True


@requires_scenario_repo
def test_dispatch_commit_equivalence_not_a_false_conflict():
    """R028/R030/R032/R034/R036/R038: gold_robot_inspection.json says COMMIT,
    groundtruth.txt says DISPATCH. l3_scoring.normalise_action treats these
    as the same action (-> PROCEED); the conflict detector must agree."""
    reg, conflicts = SC.build_registry_with_conflicts()
    flagged_ids = {c["scenario_id"] for c in conflicts}
    for sid in ("R028", "R030", "R032", "R034", "R036", "R038"):
        assert sid not in flagged_ids, f"{sid} should not be a false conflict"


@requires_scenario_repo
def test_saved_registry_round_trips_through_json():
    reg = SC.build_registry()
    SC.save_registry(reg, path=SC.TYPED_REGISTRY_PATH)
    reloaded = SC.load_registry(path=SC.TYPED_REGISTRY_PATH)
    assert set(reloaded) == set(reg)
    for sid in reg:
        assert reloaded[sid].gold_action == reg[sid].gold_action
        assert reloaded[sid].gold_source == reg[sid].gold_source


@requires_scenario_repo
def test_coverage_report_never_sums_provenance_classes():
    """R0-PRE rule 4: report per-class counts, never a summed total that
    could be misread as 'N assets with real evidence'."""
    reg = SC.build_registry()
    report = SC.coverage_report(reg)
    dist = report["evidence_provenance_distribution"]
    assert sum(dist.values()) == report["total_scenarios"]
    assert "L1_REAL_ASSET_EVIDENCE" not in dist  # measured this pass: still zero
