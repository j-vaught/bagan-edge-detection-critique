"""WVF heatmap including Ns=1..8 to show the orientation cliff."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "single_image_ablation"

with open(OUT / "ablation_metrics.json") as f:
    data = json.load(f)
with open(OUT / "low_ns_results.json") as f:
    low_ns = json.load(f)

ns1 = [
    {"Np": 15, "Ns": 1, "d": 4, "ods": 0.5819},
    {"Np": 50, "Ns": 1, "d": 4, "ods": 0.6183},
    {"Np": 100, "Ns": 1, "d": 4, "ods": 0.6100},
    {"Np": 250, "Ns": 1, "d": 4, "ods": 0.5969},
]

all_d4 = [r for r in data["wvf"] if r["d"] == 4]
all_d4 += [r for r in low_ns if r["d"] == 4]
all_d4 += ns1

lookup = {}
for r in all_d4:
    lookup[(r["Np"], r["Ns"])] = r["ods"]

nps = sorted(set(r["Np"] for r in all_d4))
nss = sorted(set(r["Ns"] for r in all_d4))

hm = np.full((len(nps), len(nss)), np.nan)
for i, np_v in enumerate(nps):
    for j, ns_v in enumerate(nss):
        if (np_v, ns_v) in lookup:
            hm[i, j] = lookup[(np_v, ns_v)]

fig, ax = plt.subplots(figsize=(14, 7))
im = ax.imshow(hm, aspect="auto", origin="lower", cmap="RdYlGn", vmin=0.55, vmax=0.87)
ax.set_xticks(range(len(nss)))
ax.set_xticklabels(nss, fontsize=9)
ax.set_yticks(range(len(nps)))
ax.set_yticklabels(nps, fontsize=9)
ax.set_xlabel("$N_s$ (orientations)", fontsize=12)
ax.set_ylabel("$N_p$ (support size)", fontsize=12)
ax.set_title("WVF ODS Heatmap ($d$=4) — Including Low Orientations", fontsize=13)
fig.colorbar(im, ax=ax, label="ODS F-score", shrink=0.8)

for i in range(len(nps)):
    for j in range(len(nss)):
        if not np.isnan(hm[i, j]):
            color = "white" if hm[i, j] < 0.7 else "black"
            ax.text(j, i, f"{hm[i, j]:.3f}", ha="center", va="center",
                    fontsize=6, color=color)

# Vertical line between Ns=2 and Ns=3
cliff_x = nss.index(3) - 0.5
ax.axvline(x=cliff_x, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
ax.text(cliff_x + 0.15, len(nps) - 0.5, "cliff", color="red", fontsize=9, va="top")

fig.tight_layout()
fig.savefig(OUT / "plot_wvf_heatmap_full.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved plot_wvf_heatmap_full.png")
