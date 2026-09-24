#!/usr/bin/env python3
"""phase8h2l_main_table_build.py -- builds the InspectionBench V3 main
results table (A/B/C/D/E, all 5 primary-panel models), per the schema
decided in the scoring/normalization audit plan: raw per-capability
scores, no cross-capability aggregate, no overall rank, bootstrap 95% CI
(2000 resamples, seed 42) per cell, N shown whenever n_valid < N_nominal.

Metrics (unchanged from their existing scoring code -- nothing here
recomputes or redefines a metric, it only aggregates what score_episode /
b_acquisition_scoring / dphys_scoring / l3_grounded_scoring already
produced per episode):
  A  GSR            frozen-93 pool, N=48   (runner_return.metric.GSR)
  B  B_AGS (branched, corrected) B-Acquisition pool, N=66:
       STOP                 -> ADA and TDA_post
       ACQUIRE + available  -> ADA and ASA and TDA_post
       ACQUIRE + unavailable-> ADA and ASA and UHA
  C  procedural_coverage    frozen-93 pool, N=9    (ordering.procedural_coverage)
  D  CSA                    D-physical pool, N=70  (score.CSA)
  E  CC_grounded            frozen-93 pool, N=18 slots / 6 sequences (runner_return.CC_grounded)

Reads only existing, already-scored raw files. Writes nothing back to
them. Additive-only output: reports/benchmark/main_table/.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))
from ordering import procedural_coverage  # noqa: E402

FROZEN93 = REPO_ROOT / "reports" / "benchmark" / "v3_full_results" / "frozen93"
BACQ = REPO_ROOT / "reports" / "benchmark" / "b_acquisition_pilot"
DPHYS = REPO_ROOT / "reports" / "benchmark" / "dphys_pilot"
OUT_DIR = REPO_ROOT / "reports" / "benchmark" / "main_table"

MODELS = [
    ("Claude Sonnet 4.6", "Claude_Sonnet_4.6", "anthropic_claude-sonnet-4.6"),
    ("GPT-5.2", "GPT-5.2", "openai_gpt-5.2"),
    ("DeepSeek V4 Pro", "DeepSeek_V4_Pro", "deepseek_deepseek-v4-pro"),
    ("Mistral Medium 3.5", "Mistral_Medium_3.5", "mistralai_mistral-medium-3-5"),
    ("Qwen3.5-397B-A17B", "Qwen3.5-397B-A17B", "qwen_qwen3.5-397b-a17b"),
]

N_BOOT = 2000
SEED = 42


def bootstrap_ci(values: List[float], n_boot: int = N_BOOT, seed: int = SEED):
    if not values:
        return None, None
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot) - 1]
    return round(lo, 4), round(hi, 4)


def load_jsonl(path: Path) -> List[dict]:
    return [json.loads(l) for l in path.read_text().splitlines()]


def cell(values: List[Optional[float]], n_nominal: int, markers: List[str]) -> dict:
    valid = [v for v in values if v is not None]
    n_valid = len(valid)
    if n_valid == 0:
        return {"value": None, "ci_low": None, "ci_high": None,
                "n_valid": 0, "N_nominal": n_nominal, "markers": markers + ["EXCLUDED"]}
    value = round(sum(valid) / n_valid, 4)
    lo, hi = bootstrap_ci(valid)
    m = list(markers)
    if n_valid < n_nominal:
        m.append(f"n={n_valid}/{n_nominal}")
    return {"value": value, "ci_low": lo, "ci_high": hi,
            "n_valid": n_valid, "N_nominal": n_nominal, "markers": m}


A_ROOT_CAUSE_NOTE = {
    "Claude Sonnet 4.6": "pooled across the frozen-93 A pool (2/48 evaluable) and A-expanded-v1 (12/240 evaluable) -- 14/288 total evaluable, all step2_protocol_noncompliance elsewhere; see claude_A_audit_summary.json",
    "DeepSeek V4 Pro": "5/48 empty verdict excluded (truncation/output-budget); see A_evaluable_verification.json for full diagnostic incl. 24 additional partial-evidence episodes retained per reviewed decision (lenient N)",
    "Qwen3.5-397B-A17B": "5/48 empty verdict excluded (2 truncation, 3 infrastructure); see A_evaluable_verification.json",
}

A_EXPANDED_240 = REPO_ROOT / "reports" / "benchmark" / "v3_full_results" / "a_expanded_240"


def _claude_pooled_evaluable_gsr() -> dict:
    """Claude's n=2 evaluable count on the frozen-93 A pool alone is too
    small to report a number at all (the original EXCLUDED decision).
    Rather than leave the main table cell blank, pool in Claude's
    A-expanded-v1 evaluable episodes (same construct, same protocol,
    non-overlapping seeds, already run and audited -- see
    AUDIT_n16_vs_n80_scaling.md) to get a real, if still small, N. This
    does NOT change the other four models' primary A cells, which stay on
    the original frozen-93 pool per the user-confirmed reference values."""
    frozen = [json.loads(l) for l in (FROZEN93 / "raw_Claude_Sonnet_4.6.jsonl").read_text().splitlines()]
    a_frozen = [r for r in frozen if r["dim"] == "A"]
    eval_frozen = [r for r in a_frozen if r["runner_return"].get("verdict", "") != ""]
    gsr_frozen = [r["runner_return"]["metric"]["GSR"] for r in eval_frozen]

    expanded = [json.loads(l) for l in (A_EXPANDED_240 / "raw_Claude_Sonnet_4.6.jsonl").read_text().splitlines()]
    eval_expanded = [r for r in expanded if r.get("verdict_present")]
    gsr_expanded = [r["metric"]["GSR"] for r in eval_expanded]

    combined = gsr_frozen + gsr_expanded
    n = len(combined)
    value = round(sum(combined) / n, 4)
    lo, hi = bootstrap_ci(combined)
    return {"value": value, "ci_low": lo, "ci_high": hi, "n_valid": n, "N_nominal": 48 + 240,
            "markers": [f"n={n}/288 (pooled, see below)", A_ROOT_CAUSE_NOTE["Claude Sonnet 4.6"]]}


def compute_a(frozen93_rows: List[dict], display_name: str) -> dict:
    """PRIMARY A metric: GSR on the evaluable set only (episodes with a
    valid, non-empty terminal verdict) -- NOT a hard-zero over all 48.
    Missing/empty-verdict episodes are excluded from the denominator, not
    scored as failures, because the diagnostic audit (Phase 8H.2N) showed
    the missing outputs are dominated by protocol/infra/truncation
    artifacts for DeepSeek and Qwen, and by a genuine but separately-
    reported model-behavior issue for Claude. Claude's frozen-93-only
    evaluable count (2/48) is too small on its own, so its cell pools in
    A-expanded-v1's evaluable episodes (14/288 combined) rather than
    showing a blank/EXCLUDED cell -- the other four models' cells stay on
    the original frozen-93 pool, matching the user-confirmed reference
    values. Output-validity rate (A3) is reported as a separate
    footnote/appendix quantity, never folded into this cell. See
    reports/benchmark/A_evaluable_verification.json for the full
    N-accounting this is built from."""
    a_rows = [r for r in frozen93_rows if r["dim"] == "A"]
    n_total = len(a_rows)
    evaluable = [r for r in a_rows if r["runner_return"].get("verdict", "") != ""]
    n_evaluable = len(evaluable)
    n_missing = n_total - n_evaluable

    if display_name == "Claude Sonnet 4.6":
        return _claude_pooled_evaluable_gsr()

    values = [r["runner_return"]["metric"]["GSR"] for r in evaluable]
    markers = []
    if n_missing:
        markers.append(f"n={n_evaluable}/{n_total} (output-validity {n_evaluable}/{n_total}={n_evaluable/n_total:.1%}, see appendix)")
        note = A_ROOT_CAUSE_NOTE.get(display_name)
        if note:
            markers.append(note)
    return cell(values, n_evaluable, markers)


def compute_b(b_rows: List[dict]) -> dict:
    values = []
    n_truncated = 0
    for r in b_rows:
        s = r["score"]; g = r["gold"]
        if r.get("error") is not None or r.get("response") is None:
            n_truncated += 1
            values.append(0.0)  # missing/invalid terminal_action = hard 0, matching scoring contract
            continue
        ada = bool(s["ADA"])
        tda_post = bool(s["TDA_post"])
        asa = (s.get("ASA") == 1.0)
        uha = (s.get("UHA") == 1.0)
        if not g["acquisition_required"]:
            b_ags = ada and tda_post
        elif not g["acquisition_genuinely_unavailable"]:
            b_ags = ada and asa and tda_post
        else:
            b_ags = ada and asa and uha
        values.append(1.0 if b_ags else 0.0)
    markers = []
    if n_truncated:
        markers.append(f"truncated={n_truncated}/{len(b_rows)}")
    return cell(values, 66, markers)


def compute_c(frozen93_rows: List[dict]) -> dict:
    c_rows = [r for r in frozen93_rows if r["dim"] == "C"]
    values = []
    for r in c_rows:
        rr = r["runner_return"]
        cov = procedural_coverage(rr.get("required_order") or [], rr.get("executed_order") or [])
        values.append(cov)
    return cell(values, 9, [])


def compute_d(d_rows: List[dict]) -> dict:
    values = [r["score"]["CSA"] for r in d_rows]
    return cell(values, 70, [])


def compute_e(frozen93_rows: List[dict]) -> dict:
    """CC_grounded is only present when required_modality applies to that
    slot (matches the existing phase8h2j_v3_full_aggregate.py convention:
    .get("CC_grounded") filtered for None, i.e. NOT_APPLICABLE slots are
    excluded from the mean, not scored as 0)."""
    e_rows = [r for r in frozen93_rows if r["dim"] == "E"]
    values = [r["runner_return"].get("CC_grounded") for r in e_rows]
    n_applicable = sum(1 for v in values if v is not None)
    n_seq = len({r["world_id"] for r in e_rows})
    return cell(values, n_applicable or 18,
                [f"n_sequences={n_seq}", f"CC_grounded_applicable={n_applicable}/{len(e_rows)}"])


def main() -> int:
    table = {}
    for display_name, frozen93_suffix, pilot_suffix in MODELS:
        frozen93_path = FROZEN93 / f"raw_{frozen93_suffix}.jsonl"
        b_path = BACQ / f"b_acq_pilot_raw_tokenrouter_{pilot_suffix}_v3_full.jsonl"
        d_candidates = [
            DPHYS / f"dphys_pilot_raw_tokenrouter_{pilot_suffix}_v3_full_merged.jsonl",
            DPHYS / f"dphys_pilot_raw_tokenrouter_{pilot_suffix}_v3_full.jsonl",
        ]
        d_path = next(p for p in d_candidates if p.exists())

        frozen93_rows = load_jsonl(frozen93_path)
        b_rows = load_jsonl(b_path)
        d_rows = load_jsonl(d_path)

        table[display_name] = {
            "A_GSR": compute_a(frozen93_rows, display_name),
            "B_B_AGS": compute_b(b_rows),
            "C_coverage": compute_c(frozen93_rows),
            "D_CSA": compute_d(d_rows),
            "E_CC_grounded": compute_e(frozen93_rows),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "main_table_v1.json"
    out_json.write_text(json.dumps(table, indent=2))

    # Markdown rendering: clean table, footnote markers only (a/b/c...),
    # full explanation moved to the footnote section below the table.
    footnotes: List[str] = []

    def fmt_cell(c: dict) -> str:
        markers = c["markers"]
        if c["value"] is None:
            idx = len(footnotes) + 1
            last_marker = markers[-1] if markers else "EXCLUDED"
            footnotes.append(f"[{idx}] {last_marker}")
            return f"EXCLUDED [{idx}]"
        value = c["value"]
        ci_lo = c["ci_low"]
        ci_hi = c["ci_high"]
        s = f"{value:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]" if ci_lo is not None else f"{value:.3f}"
        if markers:
            idx = len(footnotes) + 1
            footnotes.append(f"[{idx}] " + "; ".join(markers))
            s += f" [{idx}]"
        return s

    lines = ["| Model | A GSR (evaluable N) | B B_AGS (N=66) | C coverage\u2020 (N=9) | D CSA (N=70) | E CC_grounded\u2020 (N=18/6seq) |",
             "|---|---|---|---|---|---|"]
    for name, _, _ in MODELS:
        row = table[name]
        cells = [fmt_cell(row[key]) for key in ("A_GSR", "B_B_AGS", "C_coverage", "D_CSA", "E_CC_grounded")]
        lines.append("| " + name + " | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("Footnotes:")
    for fn in footnotes:
        lines.append(fn)
    lines.append("")
    lines.append("A is GSR (Grounded Success Rate) computed on the EVALUABLE episode set only "
                 "(episodes with a valid, non-empty terminal verdict) -- NOT a hard-zero over all "
                 "48. Output-validity diagnostics (why some episodes are excluded, per-model root "
                 "cause, TDA/A3 validity rate, and a hard-zero sensitivity appendix) are in "
                 "reports/benchmark/A_evaluable_verification.json and "
                 "reports/benchmark/main_table/A_appendix_v1.md -- not folded into this column.")

    md = "\n".join(lines)
    out_md = OUT_DIR / "main_table_v1.md"
    out_md.write_text(md + "\n")

    print(md)
    print()
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
