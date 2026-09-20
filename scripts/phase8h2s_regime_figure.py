#!/usr/bin/env python3
"""phase8h2s_regime_figure.py -- publication-quality 2-panel figure for
the matched within-world evidence-regime comparison, styled to match the
existing paper figures (Figure 6/7: dot markers, connecting lines,
percentage/delta labels, serif font).

Panel (a): GSR, FULL vs PHYSICAL_ONLY only -- the one non-tautological
GSR comparison (see AUDIT_NOTE_digital_only_gsr.md: DIGITAL_ONLY GSR is
analytically 0 for every model by the FM-6a/FM-7a evidence contract, not
a capability signal, and is excluded here).

Panel (b): TDA across all three regimes (FULL/PHYSICAL_ONLY/DIGITAL_ONLY)
-- TDA has no modality precondition, so it is the metric that legitimately
shows DIGITAL_ONLY behavior (does the agent recognize it lacks required
grounding evidence and decide accordingly).

All values computed on the matched-triplet set per model (worlds where
all 3 regimes have a valid verdict) -- see regime_triplet_figure_data.json.
Claude Sonnet 4.6 is excluded (0 fully-matched triplets; see
claude_A_audit_summary.json).

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
                   "regime_triplet_figure_data.json").read_text())
OUT_DIR = REPO_ROOT / "reports" / "benchmark" / "matched_regime_delta"

MODELS = ["GPT-5.2", "DeepSeek V4 Pro", "Mistral Medium 3.5", "Qwen3.5-397B-A17B"]
COLOR_FULL = "#1f4e79"
COLOR_PHYS = "#c0724a"
COLOR_DIG = "#8a8a8a"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

# --- Panel (a): GSR, FULL vs PHYSICAL_ONLY --------------------------------
x = list(range(len(MODELS)))
full_gsr = [DATA[m]["FULL_GSR"] for m in MODELS]
phys_gsr = [DATA[m]["PHYSICAL_ONLY_GSR"] for m in MODELS]
n_pairs = [DATA[m]["n_full_triplets"] for m in MODELS]

for i, m in enumerate(MODELS):
    ax1.plot([i, i], [full_gsr[i], phys_gsr[i]], color="#999999", lw=1.3, zorder=1)
ax1.scatter(x, full_gsr, s=70, color=COLOR_FULL, label="FULL evidence", zorder=3)
ax1.scatter(x, phys_gsr, s=70, color=COLOR_PHYS, label="PHYSICAL_ONLY (digital withheld)", zorder=3)

for i in range(len(MODELS)):
    delta = full_gsr[i] - phys_gsr[i]
    y_top = max(full_gsr[i], phys_gsr[i])
    ax1.annotate(f"{delta:+.3f}", (i, y_top + 0.05), ha="center", fontsize=9, color="#444444")

ax1.set_xticks(x)
ax1.set_xticklabels([m.replace(" ", "\n", 1) for m in MODELS], fontsize=9.5)
ax1.set_ylabel("Grounded Success Rate (GSR)")
ax1.set_ylim(-0.05, 1.05)
ax1.set_title("(a) Matched-world GSR: FULL vs digital-withheld\n(non-tautological pair only)", fontsize=10.5)
ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=False, fontsize=9)

for i, n in enumerate(n_pairs):
    ax1.text(i, -0.13, f"n={n}", ha="center", fontsize=8, color="#666666", transform=ax1.get_xaxis_transform())

# --- Panel (b): TDA across all three regimes ------------------------------
full_tda = [DATA[m]["FULL_TDA"] for m in MODELS]
phys_tda = [DATA[m]["PHYSICAL_ONLY_TDA"] for m in MODELS]
dig_tda = [DATA[m]["DIGITAL_ONLY_TDA"] for m in MODELS]

for i in range(len(MODELS)):
    ax2.plot([i, i, i], [full_tda[i], phys_tda[i], dig_tda[i]], color="#cccccc", lw=0, zorder=1)
ax2.scatter(x, full_tda, s=65, color=COLOR_FULL, label="FULL", zorder=3, marker="o")
ax2.scatter(x, phys_tda, s=65, color=COLOR_PHYS, label="PHYSICAL_ONLY", zorder=3, marker="s")
ax2.scatter(x, dig_tda, s=65, color=COLOR_DIG, label="DIGITAL_ONLY", zorder=3, marker="^")

ax2.set_xticks(x)
ax2.set_xticklabels([m.replace(" ", "\n", 1) for m in MODELS], fontsize=9.5)
ax2.set_ylabel("Terminal Decision Accuracy (TDA)")
ax2.set_ylim(-0.05, 1.05)
ax2.set_title("(b) Matched-world TDA across all three regimes\n(no modality precondition -- clean everywhere)", fontsize=10.5)
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=9)

for i, n in enumerate(n_pairs):
    ax2.text(i, -0.13, f"n={n}", ha="center", fontsize=8, color="#666666", transform=ax2.get_xaxis_transform())

fig.suptitle("Matched within-world evidence-access effect (A family, same 16 worlds per model)", fontsize=12, y=1.03)
fig.tight_layout()

for ext in ("pdf", "png"):
    out = OUT_DIR / f"figure_matched_regime_v1.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Wrote {out}")
