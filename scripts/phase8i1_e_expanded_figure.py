#!/usr/bin/env python3
"""phase8i1_e_expanded_figure.py -- main-paper figure for the E
(temporal grounding) reobservation-necessity result, n=90 (E-expanded-v1,
5x the E-v1 pool). Styled to match the existing figure language.

Panel (a): Precision/Recall/F1 per model, bootstrap 95% CI error bars.
Panel (b): output-validity rate per model (highlights Claude's 52.2%,
a striking contrast to its ~5% rate on the A family).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = json.loads((REPO_ROOT / "reports" / "benchmark" / "e_temporal_audit" /
                   "e_expanded_confusion_ci_v1.json").read_text())["per_model"]
OUT_DIR = REPO_ROOT / "reports" / "benchmark" / "e_temporal_audit"

MODELS = ["Claude Sonnet 4.6", "GPT-5.2", "DeepSeek V4 Pro", "Mistral Medium 3.5", "Qwen3.5-397B-A17B"]
COLOR_P, COLOR_R, COLOR_F1 = "#1f4e79", "#c0724a", "#4c8c4a"

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [2.2, 1]})

x = list(range(len(MODELS)))

def errb(model, key):
    v = DATA[model][key]
    lo, hi = DATA[model][f"{key}_ci"]
    return v, v - lo, hi - v

p_y, p_lo, p_hi = zip(*[errb(m, "precision") for m in MODELS])
r_y, r_lo, r_hi = zip(*[errb(m, "recall") for m in MODELS])

ax1.errorbar([i - 0.15 for i in x], p_y, yerr=[p_lo, p_hi], fmt="o", ms=8, color=COLOR_P,
            capsize=4, label="Precision (of reacquisitions, how many were needed)", zorder=3)
ax1.errorbar([i + 0.15 for i in x], r_y, yerr=[r_lo, r_hi], fmt="s", ms=8, color=COLOR_R,
            capsize=4, label="Recall (of needed reacquisitions, how many happened)", zorder=3)
f1_y = [DATA[m]["F1"] for m in MODELS]
ax1.scatter(x, f1_y, marker="_", s=400, color=COLOR_F1, linewidths=2.5, label="F1", zorder=4)

ax1.axhline(0.5, color="#bbbbbb", lw=1, ls="--", zorder=1)
ax1.set_xticks(x)
ax1.set_xticklabels([m.replace(" ", "\n", 1) for m in MODELS], fontsize=9)
ax1.set_ylabel("Reobservation-necessity score")
ax1.set_ylim(0.3, 1.0)
ax1.set_title("(a) Reacquire-exactly-when-needed (n=90/model, E-expanded-v1)\nerror bars = bootstrap 95% CI; no pairwise difference is significant", fontsize=10)
ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=False, fontsize=8.5)

validity = [DATA[m]["output_validity_rate"] for m in MODELS]
colors = ["#c0392b" if m == "Claude Sonnet 4.6" else "#1f4e79" for m in MODELS]
ax2.bar(x, validity, color=colors, width=0.6, zorder=3)
for i, v in enumerate(validity):
    ax2.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)
ax2.set_xticks(x)
short_labels = ["Claude", "GPT-5.2", "DeepSeek", "Mistral", "Qwen"]
ax2.set_xticklabels(short_labels, fontsize=8.5, rotation=20, ha="right")
ax2.set_ylabel("Output-validity rate (valid terminal verdict)")
ax2.set_ylim(0, 1.12)
ax2.set_title("(b) Output validity, E family\ncf. ~5% for Claude on the A family", fontsize=10)

fig.suptitle("Temporal grounding: reobservation-necessity (E family, 30 sequences x 3 episodes x 5 models)", fontsize=12, y=1.04)
fig.tight_layout()

for ext in ("pdf", "png"):
    out = OUT_DIR / f"figure_e_reobservation_v1.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Wrote {out}")
