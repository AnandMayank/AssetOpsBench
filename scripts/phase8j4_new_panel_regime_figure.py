#!/usr/bin/env python3
"""phase8j4_new_panel_regime_figure.py -- matched within-world evidence-
access figure (FULL / PHYSICAL_ONLY / DIGITAL_ONLY) for the swapped
primary panel, same two-panel layout and style as
phase8h2w_a_expanded_figure.py (the original panel's n=80 figure).

Models shown: GPT-6-Astra, Claude Opus 5.5, Gemini 3.1 Pro Preview
(n_full_triplets 78-80, complete A-expanded pools) and DeepSeek V4 Pro
0813 (n=19 as of this run, still growing -- its A-expanded rerun is in
progress; this figure will need a rebuild once that finishes).
Qwen3.5-397B-A17B excluded: its A-expanded pool was never completed in
either panel (20/240 episodes at time of writing) -- queued separately.

Read-only over reports/benchmark/new_panel/new_panel_ablations_v1.json;
writes only the figure files.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = json.loads((REPO_ROOT / "reports" / "benchmark" / "new_panel" /
                   "new_panel_ablations_v1.json").read_text())["regime_delta"]
OUT_DIR = REPO_ROOT / "reports" / "benchmark" / "new_panel"

MODELS = ["GPT-6-Astra", "Claude Opus 5.5", "Gemini 3.1 Pro Preview", "DeepSeek V4 Pro 0813"]
COLOR_FULL, COLOR_PHYS, COLOR_DIG = "#1f4e79", "#c0724a", "#8a8a8a"

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

x = list(range(len(MODELS)))


def err(model, regime, metric):
    v = DATA[model][f"{regime}_{metric}"]
    lo, hi = DATA[model][f"{regime}_{metric}_ci"]
    return v, [v - lo, hi - v]


# --- Panel (a): GSR, FULL vs PHYSICAL_ONLY, with CI error bars -----------
full_gsr = [err(m, "FULL", "GSR") for m in MODELS]
phys_gsr = [err(m, "PHYSICAL_ONLY", "GSR") for m in MODELS]
full_gsr_y = [v for v, _ in full_gsr]
full_gsr_err = [[e[0] for _, e in full_gsr], [e[1] for _, e in full_gsr]]
phys_gsr_y = [v for v, _ in phys_gsr]
phys_gsr_err = [[e[0] for _, e in phys_gsr], [e[1] for _, e in phys_gsr]]

ax1.errorbar([i - 0.06 for i in x], full_gsr_y, yerr=full_gsr_err, fmt="o", ms=8,
            color=COLOR_FULL, capsize=4, label="FULL evidence", zorder=3)
ax1.errorbar([i + 0.06 for i in x], phys_gsr_y, yerr=phys_gsr_err, fmt="o", ms=8,
            color=COLOR_PHYS, capsize=4, label="PHYSICAL_ONLY (digital withheld)", zorder=3)
ax1.set_xticks(x)
ax1.set_xticklabels([m.replace(" ", "\n", 1) for m in MODELS], fontsize=9)
ax1.set_ylabel("Grounded Success Rate (GSR)")
ax1.set_ylim(-0.05, 1.05)
ax1.set_title("(a) GSR, FULL vs digital-withheld\nerror bars = bootstrap 95% CI", fontsize=10.5)
ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, fontsize=9)
for i, m in enumerate(MODELS):
    n = DATA[m]["n_full_triplets"]
    ax1.text(i, -0.16, f"n={n}", ha="center", fontsize=8, color="#666666", transform=ax1.get_xaxis_transform())

# --- Panel (b): TDA across regimes, with CI error bars --------------------
full_tda = [err(m, "FULL", "TDA") for m in MODELS]
phys_tda = [err(m, "PHYSICAL_ONLY", "TDA") for m in MODELS]
dig_tda = [err(m, "DIGITAL_ONLY", "TDA") for m in MODELS]
full_tda_y = [v for v, _ in full_tda]
full_tda_err = [[e[0] for _, e in full_tda], [e[1] for _, e in full_tda]]
phys_tda_y = [v for v, _ in phys_tda]
phys_tda_err = [[e[0] for _, e in phys_tda], [e[1] for _, e in phys_tda]]
dig_tda_y = [v for v, _ in dig_tda]
dig_tda_err = [[e[0] for _, e in dig_tda], [e[1] for _, e in dig_tda]]

ax2.errorbar([i - 0.12 for i in x], full_tda_y, yerr=full_tda_err, fmt="o", ms=8,
            color=COLOR_FULL, capsize=4, label="FULL", zorder=3)
ax2.errorbar(x, phys_tda_y, yerr=phys_tda_err, fmt="s", ms=8,
            color=COLOR_PHYS, capsize=4, label="PHYSICAL_ONLY", zorder=3)
ax2.errorbar([i + 0.12 for i in x], dig_tda_y, yerr=dig_tda_err, fmt="^", ms=8,
            color=COLOR_DIG, capsize=4, label="DIGITAL_ONLY", zorder=3)

for i, m in enumerate(MODELS):
    sig = not DATA[m]["TDA_FULL_vs_DIGITAL_ONLY_CIs_overlap"]
    marker = "*" if sig else "n.s."
    y = max(full_tda_y[i], dig_tda_y[i]) + max(full_tda_err[0][i], dig_tda_err[0][i]) + 0.08
    ax2.text(i, y, marker, ha="center", fontsize=13 if marker == "*" else 8.5,
             color="#c0392b" if marker == "*" else "#999999")

ax2.set_xticks(x)
ax2.set_xticklabels([m.replace(" ", "\n", 1) for m in MODELS], fontsize=9)
ax2.set_ylabel("Terminal Decision Accuracy (TDA)")
ax2.set_ylim(-0.05, 1.15)
ax2.set_title("(b) TDA across all regimes\n* = FULL vs DIGITAL_ONLY CIs do not overlap", fontsize=10.5)
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=9)
for i, m in enumerate(MODELS):
    n = DATA[m]["n_full_triplets"]
    ax2.text(i, -0.16, f"n={n}", ha="center", fontsize=8, color="#666666", transform=ax2.get_xaxis_transform())

fig.suptitle("Matched within-world evidence-access effect, swapped panel (A-expanded-v1)", fontsize=11.5, y=1.04)
fig.tight_layout()

for ext in ("pdf", "png"):
    out = OUT_DIR / f"figure_matched_regime_new_panel.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Wrote {out}")
