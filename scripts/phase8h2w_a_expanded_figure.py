#!/usr/bin/env python3
"""phase8h2w_a_expanded_figure.py -- main-paper figure built on the
n=80-world A-expanded-v1 pool (vs the original n=16-world pool), with
bootstrap 95% CI error bars so defensibility is visible on the figure
itself, not just in an appendix table. Qwen excluded (run incomplete,
20/240 at time of this figure -- see a_expanded_regime_ci_v1.json).
Claude shown in panel (b) FULL-only (its PHYSICAL_ONLY/DIGITAL_ONLY
evaluable N is 0 -- see claude_A_audit_summary.json; this generalizes at
n=80 scale: 228/240 empty verdicts here vs 46/48 in the original pool).
Read-only over existing data; writes only the figure files.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = json.loads((REPO_ROOT / "reports" / "benchmark" / "matched_regime_delta" /
                   "a_expanded_regime_ci_v1.json").read_text())
OUT_DIR = REPO_ROOT / "reports" / "benchmark" / "matched_regime_delta"

MODELS_FULL_SET = ["GPT-5.2", "DeepSeek V4 Pro", "Mistral Medium 3.5"]
COLOR_FULL, COLOR_PHYS, COLOR_DIG = "#1f4e79", "#c0724a", "#8a8a8a"

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

# --- Panel (a): GSR, FULL vs PHYSICAL_ONLY, with CI error bars -----------
x = list(range(len(MODELS_FULL_SET)))

def err(model, regime, metric):
    v = DATA[model][regime][metric]
    lo, hi = DATA[model][regime][f"{metric}_ci"]
    return v, [[v - lo], [hi - v]]

full_gsr_y, full_gsr_err = zip(*[err(m, "FULL", "GSR") for m in MODELS_FULL_SET])
phys_gsr_y, phys_gsr_err = zip(*[err(m, "PHYSICAL_ONLY", "GSR") for m in MODELS_FULL_SET])
full_gsr_err = [[e[0][0] for e in full_gsr_err], [e[1][0] for e in full_gsr_err]]
phys_gsr_err = [[e[0][0] for e in phys_gsr_err], [e[1][0] for e in phys_gsr_err]]

ax1.errorbar([i - 0.06 for i in x], full_gsr_y, yerr=full_gsr_err, fmt="o", ms=8,
            color=COLOR_FULL, capsize=4, label="FULL evidence", zorder=3)
ax1.errorbar([i + 0.06 for i in x], phys_gsr_y, yerr=phys_gsr_err, fmt="o", ms=8,
            color=COLOR_PHYS, capsize=4, label="PHYSICAL_ONLY (digital withheld)", zorder=3)
ax1.set_xticks(x)
ax1.set_xticklabels([m.replace(" ", "\n", 1) for m in MODELS_FULL_SET], fontsize=9.5)
ax1.set_ylabel("Grounded Success Rate (GSR)")
ax1.set_ylim(-0.05, 1.05)
ax1.set_title("(a) GSR, FULL vs digital-withheld (n=80 worlds)\nerror bars = bootstrap 95% CI -- all overlap", fontsize=10.5)
ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=False, fontsize=9)
for i in x:
    ax1.text(i, -0.14, "n=80", ha="center", fontsize=8, color="#666666", transform=ax1.get_xaxis_transform())

# --- Panel (b): TDA across regimes, with CI error bars --------------------
full_tda_y, full_tda_err = zip(*[err(m, "FULL", "TDA") for m in MODELS_FULL_SET])
phys_tda_y, phys_tda_err = zip(*[err(m, "PHYSICAL_ONLY", "TDA") for m in MODELS_FULL_SET])
dig_tda_y, dig_tda_err = zip(*[err(m, "DIGITAL_ONLY", "TDA") for m in MODELS_FULL_SET])
full_tda_err = [[e[0][0] for e in full_tda_err], [e[1][0] for e in full_tda_err]]
phys_tda_err = [[e[0][0] for e in phys_tda_err], [e[1][0] for e in phys_tda_err]]
dig_tda_err = [[e[0][0] for e in dig_tda_err], [e[1][0] for e in dig_tda_err]]

ax2.errorbar([i - 0.12 for i in x], full_tda_y, yerr=full_tda_err, fmt="o", ms=8,
            color=COLOR_FULL, capsize=4, label="FULL", zorder=3)
ax2.errorbar(x, phys_tda_y, yerr=phys_tda_err, fmt="s", ms=8,
            color=COLOR_PHYS, capsize=4, label="PHYSICAL_ONLY", zorder=3)
ax2.errorbar([i + 0.12 for i in x], dig_tda_y, yerr=dig_tda_err, fmt="^", ms=8,
            color=COLOR_DIG, capsize=4, label="DIGITAL_ONLY", zorder=3)

sig = {"GPT-5.2": True, "DeepSeek V4 Pro": False, "Mistral Medium 3.5": True}
for i, m in enumerate(MODELS_FULL_SET):
    marker = "*" if sig[m] else "n.s."
    y = max(full_tda_y[i], dig_tda_y[i]) + max(full_tda_err[0][i], dig_tda_err[0][i]) + 0.08
    ax2.text(i, y, marker, ha="center", fontsize=13 if marker == "*" else 8.5,
             color="#c0392b" if marker == "*" else "#999999")

ax2.set_xticks(x)
ax2.set_xticklabels([m.replace(" ", "\n", 1) for m in MODELS_FULL_SET], fontsize=9.5)
ax2.set_ylabel("Terminal Decision Accuracy (TDA)")
ax2.set_ylim(-0.05, 1.15)
ax2.set_title("(b) TDA across all regimes (n=80 worlds)\n* = FULL vs DIGITAL_ONLY CIs do not overlap", fontsize=10.5)
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, frameon=False, fontsize=9)
for i in x:
    ax2.text(i, -0.14, "n=80", ha="center", fontsize=8, color="#666666", transform=ax2.get_xaxis_transform())

fig.suptitle("Matched within-world evidence-access effect, A-expanded-v1 (80 worlds, n=16 pool superseded)", fontsize=11.5, y=1.04)
fig.tight_layout()

for ext in ("pdf", "png"):
    out = OUT_DIR / f"figure_matched_regime_v2_n80.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Wrote {out}")
