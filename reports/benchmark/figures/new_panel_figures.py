#!/usr/bin/env python3
"""new_panel_figures.py -- regenerates Figures 5, 6, 7 (paper style) for
the swapped primary panel (GPT-6-Astra, Claude Opus 5.5, Gemini 3.1 Pro
Preview, DeepSeek V4 Pro 0813, Qwen3.5-397B-A17B unchanged). Reads
new_panel_main_table_v1.json (built alongside this script). Matches the
existing paper figures' visual language: Figure 5 heatmap (orange-red
sequential colormap, N annotated per cell, dashes/daggers for
unavailable/caveated cells), Figure 6 TDA-vs-GSR dumbbell with Delta
labels, Figure 7 grouped bars with the always-ESCALATE dashed baseline.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "new_panel_main_table_v1.json").read_text())

MODEL_ORDER = ["Claude Opus 5.5", "GPT-6-Astra", "DeepSeek V4 Pro 0813",
              "Gemini 3.1 Pro Preview", "Qwen3.5-397B-A17B"]
DIMS = ["A", "B", "C", "D", "E"]
DIM_LABEL = {"A": "A: Evidence\nGrounding", "B": "B: Acquisition",
            "C": "C: Procedural\nGrounding", "D": "D: Relational\nPhysical Grounding",
            "E": "E: Temporal\nGrounding"}

plt.rcParams.update({"font.family": "serif", "font.size": 10.5})

# =================== Figure 5: heatmap ===================
fig, ax = plt.subplots(figsize=(9.2, 4.6))
cmap = plt.cm.Oranges
vals = np.full((len(MODEL_ORDER), len(DIMS)), np.nan)
for i, m in enumerate(MODEL_ORDER):
    for j, d in enumerate(DIMS):
        cell = DATA[m][d]
        if cell["value"] is not None:
            vals[i, j] = cell["value"]

masked = np.ma.masked_invalid(vals)
im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")
for i, m in enumerate(MODEL_ORDER):
    for j, d in enumerate(DIMS):
        cell = DATA[m][d]
        if cell["value"] is None:
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#d9d9d9"))
            ax.text(j, i, "—\n(N/A)", ha="center", va="center", fontsize=8.5, color="#555555")
        else:
            marker = "†" if cell.get("caveat") else ""
            txt_color = "white" if cell["value"] > 0.6 else "black"
            ax.text(j, i, f"{cell['value']:.3f}{marker}\n(n={cell['n']})",
                    ha="center", va="center", fontsize=8.5, color=txt_color)
ax.set_xticks(range(len(DIMS)))
ax.set_xticklabels([f"{DIM_LABEL[d]}\n(N={DATA['_N'][d]})" for d in DIMS], fontsize=8.5)
ax.set_yticks(range(len(MODEL_ORDER)))
ax.set_yticklabels(MODEL_ORDER, fontsize=9.5)
ax.set_title("InspectionBench Results: Model × Capability (Raw Scores)", fontsize=11)
cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
cbar.set_label("Capability Score", fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "figure5_new_panel.png", dpi=300, bbox_inches="tight")
fig.savefig(HERE / "figure5_new_panel.pdf", bbox_inches="tight")
plt.close(fig)

# =================== Figure 6: A dumbbell ===================
a_models = [m for m in MODEL_ORDER if DATA[m]["A"]["value"] is not None]
fig, ax = plt.subplots(figsize=(5.6, 4.0))
x = list(range(len(a_models)))
tda_vals = [DATA[m]["A_TDA"] for m in a_models]
gsr_vals = [DATA[m]["A"]["value"] for m in a_models]
for i in x:
    ax.plot([i, i], [gsr_vals[i] * 100, tda_vals[i] * 100], color="#999999", lw=1.3, zorder=1)
ax.scatter(x, [v * 100 for v in tda_vals], s=60, color="#1f4e79", label="Terminal (TDA)", zorder=3)
ax.scatter(x, [v * 100 for v in gsr_vals], s=60, color="#c0724a", label="Grounded (GSR)", zorder=3)
for i in x:
    delta = (tda_vals[i] - gsr_vals[i]) * 100
    ymid = (tda_vals[i] + gsr_vals[i]) / 2 * 100
    gap = (tda_vals[i] - gsr_vals[i]) * 100
    if gap < 15:
        # narrow gap: the dots sit too close together for a between-dot
        # label, and a sideways offset collides with neighboring columns
        # (e.g. Qwen), so drop the label below the lower (GSR) dot instead
        ax.annotate(f"Δ = {delta:.1f} pp", (i, gsr_vals[i] * 100 - 13), ha="center", va="center",
                    fontsize=7.8, color="#444444")
    else:
        ax.annotate(f"Δ =\n{delta:.1f} pp", (i, ymid), ha="center", fontsize=7.8, color="#444444")
    ax.annotate(f"{tda_vals[i]*100:.1f}%", (i, tda_vals[i] * 100 + 3), ha="center", fontsize=8, color="#1f4e79")
    ax.annotate(f"{gsr_vals[i]*100:.1f}%", (i, gsr_vals[i] * 100 - 6), ha="center", fontsize=8, color="#c0724a")
ax.set_xticks(x)
ax.set_xticklabels([m.replace(" ", "\n", 1) for m in a_models], fontsize=8.5)
ax.set_ylabel("Score (%)")
ax.set_ylim(0, 95)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(HERE / "figure6_new_panel.png", dpi=300, bbox_inches="tight")
fig.savefig(HERE / "figure6_new_panel.pdf", bbox_inches="tight")
plt.close(fig)

# =================== Figure 7: D grouped bars ===================
fig, ax = plt.subplots(figsize=(5.6, 4.0))
d_models = MODEL_ORDER
x = np.arange(len(d_models))
width = 0.26
cc = [DATA[m]["D_CC"] for m in d_models]
csa = [DATA[m]["D"]["value"] for m in d_models]
lca = [DATA[m]["D_LCA"] for m in d_models]
ax.bar(x - width, cc, width, label="Terminal decision correctness", color="#1f4e79")
ax.bar(x, csa, width, label="Constraint-set accuracy", color="#e08e45")
ax.bar(x + width, lca, width, label="Limiting-constraint accuracy", color="#4c8c4a")
ax.axhline(0.80, color="black", ls="--", lw=1)
ax.set_xticks(x)
ax.set_xticklabels([m.replace(" ", "\n", 1) for m in d_models], fontsize=8, rotation=20, ha="right")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.05)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1, frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(HERE / "figure7_new_panel.png", dpi=300, bbox_inches="tight")
fig.savefig(HERE / "figure7_new_panel.pdf", bbox_inches="tight")
plt.close(fig)

print("Wrote figure5_new_panel.{png,pdf}, figure6_new_panel.{png,pdf}, figure7_new_panel.{png,pdf}")
