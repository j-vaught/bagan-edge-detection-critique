"""Fill in missing Np x Ns cells for the WVF d=4 heatmap."""

import json
import numpy as np
from PIL import Image
import scipy.io as sio
from pathlib import Path

from edgecritic.wvf import wvf_image
from edgecritic.evaluation.metrics import compute_ods_ois

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "single_image_ablation"

# Load image + GT
img = np.mean(np.array(Image.open(
    ROOT / "datasets/BSDS500/BSDS500/data/images/test/100007.jpg")), axis=2)
gt_mat = sio.loadmat(str(
    ROOT / "datasets/BSDS500/BSDS500/data/groundTruth/test/100007.mat"))
gt_cell = gt_mat["groundTruth"]
gt_union = np.zeros(img.shape, dtype=bool)
for i in range(gt_cell.shape[1]):
    gt_union |= (np.asarray(gt_cell[0, i]["Boundaries"][0, 0]) > 0)

# Load existing results
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

existing = [r for r in data["wvf"] if r["d"] == 4]
existing += [r for r in low_ns if r["d"] == 4]
existing += ns1
existing_keys = {(r["Np"], r["Ns"]) for r in existing}

# Full grid
NP_ALL = [15, 20, 25, 30, 40, 50, 65, 75, 80, 100, 130, 150, 160, 200, 250, 300, 400, 500]
NS_ALL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 18, 24, 36, 48, 72, 90, 120, 180]

missing = [(np_v, ns_v) for np_v in NP_ALL for ns_v in NS_ALL
           if (np_v, ns_v) not in existing_keys]
print(f"Existing: {len(existing_keys)}, Missing: {len(missing)}, Total grid: {len(NP_ALL)*len(NS_ALL)}")

new_results = []
for idx, (np_val, ns_val) in enumerate(missing):
    min_c = (4 + 1) * (4 + 2) // 2  # 15 for d=4
    if np_val < min_c:
        continue
    mag = wvf_image(img, np_count=np_val, order=4,
                    n_orientations=ns_val, backend="cuda").gradient_mag
    ods, _, _, _ = compute_ods_ois(
        mag, gt_union.astype(np.float64), n_thresholds=500, match_radius=3)
    new_results.append({"Np": np_val, "Ns": ns_val, "d": 4, "ods": float(ods)})
    if (idx + 1) % 20 == 0:
        print(f"  {idx+1}/{len(missing)} done")

print(f"Computed {len(new_results)} new cells")

# Merge all and save
all_d4 = existing + new_results
with open(OUT / "wvf_d4_full_grid.json", "w") as f:
    json.dump(all_d4, f, indent=2)
print(f"Saved {len(all_d4)} total entries to wvf_d4_full_grid.json")

# Generate the filled heatmap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

lookup = {(r["Np"], r["Ns"]): r["ods"] for r in all_d4}

hm = np.full((len(NP_ALL), len(NS_ALL)), np.nan)
for i, np_v in enumerate(NP_ALL):
    for j, ns_v in enumerate(NS_ALL):
        if (np_v, ns_v) in lookup:
            hm[i, j] = lookup[(np_v, ns_v)]

filled = np.sum(~np.isnan(hm))
print(f"Heatmap cells filled: {filled}/{len(NP_ALL)*len(NS_ALL)}")

fig, ax = plt.subplots(figsize=(14, 7))
im = ax.imshow(hm, aspect="auto", origin="lower", cmap="RdYlGn", vmin=0.55, vmax=0.87)
ax.set_xticks(range(len(NS_ALL)))
ax.set_xticklabels(NS_ALL, fontsize=9)
ax.set_yticks(range(len(NP_ALL)))
ax.set_yticklabels(NP_ALL, fontsize=9)
ax.set_xlabel("$N_s$ (orientations)", fontsize=12)
ax.set_ylabel("$N_p$ (support size)", fontsize=12)
ax.set_title("WVF ODS Heatmap ($d$=4) — Full Grid", fontsize=13)
fig.colorbar(im, ax=ax, label="ODS F-score", shrink=0.8)

for i in range(len(NP_ALL)):
    for j in range(len(NS_ALL)):
        if not np.isnan(hm[i, j]):
            color = "white" if hm[i, j] < 0.7 else "black"
            ax.text(j, i, f"{hm[i, j]:.3f}", ha="center", va="center",
                    fontsize=5.5, color=color)

cliff_x = NS_ALL.index(3) - 0.5
ax.axvline(x=cliff_x, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
ax.text(cliff_x + 0.15, len(NP_ALL) - 0.5, "cliff", color="red", fontsize=9, va="top")

fig.tight_layout()
fig.savefig(OUT / "plot_wvf_heatmap_full.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved plot_wvf_heatmap_full.png")
