"""
InspectionBench: model x capability scores as grouped bars -- SWAPPED PRIMARY PANEL.

Same script/style as capability_bars.py (the original panel's main-image bar
chart), applied to the swapped panel (Claude Opus 5.5, GPT-6-Astra, DeepSeek
V4 Pro 0813, Gemini 3.1 Pro Preview, Qwen3.5-397B-A17B unchanged). Data comes
from reports/benchmark/figures/new_panel_main_table_v1.json, at the paper's
established Figure-5 N convention (frozen-93 only: A=48, B=66, C=9, D=70,
E=18) -- see docs/Section3_TemplateMotivation_and_Section5_Insights_Draft.md
and this session's confirmation that the newer E-expanded-v2/A-expanded
reruns feed separate pooled prose, not this figure.

Renders capability_bars_new_panel.pdf (for \\includegraphics) and
capability_bars_new_panel.png (preview). Figure width is 5.5in, the ICLR
text width, so include it at width=\\textwidth.

Palette note: same five colorblind/contrast-validated hues as the original
panel's capability_bars.py, in the same fixed slot order -- change only
after re-validating.

Usage:  python3 capability_bars_new_panel.py
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_JSON = json.loads((HERE / "new_panel_main_table_v1.json").read_text())

# ---------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": "DejaVu Sans",   # swap to "Times New Roman" to match the paper body
    "font.size": 7.2,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,             # TrueType, required by most camera-ready checkers
    "ps.fonttype": 42,
})

# ---------------------------------------------------------------- data
MODELS = ["Claude Opus 5.5", "GPT-6-Astra", "DeepSeek V4 Pro 0813",
          "Gemini 3.1 Pro Preview", "Qwen3.5-397B-A17B"]

COLORS = ["#0072B2", "#D55E00", "#CC79A7", "#7B5EA8", "#009E73"]

DIMS = ["A", "B", "C", "D", "E"]
DIM_LABEL = {"A": "A: Evidence\nGrounding", "B": "B: Active\nAcquisition",
             "C": "C: Procedural\nGrounding", "D": "D: Relational Phys.\nGrounding",
             "E": "E: Temporal\nGrounding"}
CAPS = [(DIM_LABEL[d], DATA_JSON["_N"][d]) for d in DIMS]

# rows = models (order of MODELS), cols = capabilities (order of CAPS)
DATA = np.array([[DATA_JSON[m][d]["value"] for d in DIMS] for m in MODELS])

# partial-validity cells: (row, col) -> "n=<evaluated>/<N>" sub-label, for any
# cell where the model's evaluated n fell short of the capability's full N
# (read directly from the table's per-cell "n"/"caveat" fields, not hand-copied).
PARTIAL = {}
for i, m in enumerate(MODELS):
    for j, d in enumerate(DIMS):
        cell = DATA_JSON[m][d]
        if cell.get("caveat") and cell["value"] is not None:
            PARTIAL[(i, j)] = f"n={cell['n']}/{DATA_JSON['_N'][d]}"

# ---------------------------------------------------------------- helpers
def shade(hex_c, f=0.72):
    """Darker version of a hex colour, used for the bar edge."""
    r, g, b = (int(hex_c[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % (int(r * f), int(g * f), int(b * f))


EDGES = [shade(c) for c in COLORS]

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(1, len(CAPS), figsize=(5.5, 1.95), sharey=True)

for j, ax in enumerate(axes):
    for i, v in enumerate(DATA[:, j]):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            ax.bar(i, 0.02, width=.78, facecolor="none",
                   edgecolor="#9a9a9a", hatch="///", linewidth=.6)
            ax.text(i, .05, "n/a", rotation=90, ha="center", va="bottom",
                    fontsize=6.3, color="#6a6a6a")
        else:
            ax.bar(i, v, width=.74, color=COLORS[i],
                   edgecolor=EDGES[i], linewidth=.55)
            marker = "†" if (i, j) in PARTIAL else ""
            ax.text(i, v + .025, f"{v * 100:.0f}%{marker}", rotation=90,
                    ha="center", va="bottom", fontsize=6.6, color="#333333")

    ax.set_xticks([])
    ax.set_xlim(-.7, len(MODELS) - .3)
    ax.set_ylim(0, 1.10)            # headroom for the rotated value labels
    ax.set_xlabel(f"{CAPS[j][0]}\n(N={CAPS[j][1]}) $\\uparrow$",
                  fontsize=7.0, labelpad=4)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#4a4a4a")
    ax.tick_params(left=False, labelleft=False)

handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor=e, linewidth=.55)
           for c, e in zip(COLORS, EDGES)]
fig.legend(handles, MODELS, loc="upper center", bbox_to_anchor=(.5, .995),
           ncol=3, frameon=False, fontsize=7.0, handlelength=1.1,
           handleheight=.8, columnspacing=1.4, labelspacing=.25)

# top controls the legend-to-bar gap; bottom controls room for the panel captions
fig.subplots_adjust(left=.015, right=.985, top=.86, bottom=.26, wspace=.14)

fig.savefig(HERE / "capability_bars_new_panel.pdf")
fig.savefig(HERE / "capability_bars_new_panel.png", dpi=220)
print("wrote capability_bars_new_panel.pdf and capability_bars_new_panel.png")
