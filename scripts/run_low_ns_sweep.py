"""Sweep very low orientation counts (Ns=2..8) for WVF."""

import json
import numpy as np
from PIL import Image
import scipy.io as sio
from pathlib import Path

from edgecritic.wvf import wvf_image
from edgecritic.evaluation.metrics import compute_ods_ois

ROOT = Path(__file__).resolve().parent.parent
img = np.mean(np.array(Image.open(ROOT / "datasets/BSDS500/BSDS500/data/images/test/100007.jpg")), axis=2)
gt_mat = sio.loadmat(str(ROOT / "datasets/BSDS500/BSDS500/data/groundTruth/test/100007.mat"))
gt_cell = gt_mat["groundTruth"]
gt_union = np.zeros(img.shape, dtype=bool)
for i in range(gt_cell.shape[1]):
    b = gt_cell[0, i]["Boundaries"][0, 0]
    gt_union |= (np.asarray(b) > 0)

NS_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 12, 18, 36]
NP_VALUES = [15, 25, 50, 75, 100, 150, 250]
D_VALUES = [2, 3, 4]

results = []
for d in D_VALUES:
    for ns in NS_VALUES:
        for np_val in NP_VALUES:
            min_c = (d + 1) * (d + 2) // 2
            if np_val < min_c:
                continue
            mag = wvf_image(img, np_count=np_val, order=d,
                            n_orientations=ns, backend="cuda").gradient_mag
            ods, _, _, _ = compute_ods_ois(
                mag, gt_union.astype(np.float64),
                n_thresholds=500, match_radius=3)
            results.append({"Np": np_val, "Ns": ns, "d": d, "ods": float(ods)})

# Print tables
for d in D_VALUES:
    print(f"\nd={d}")
    header = "Ns   " + "  ".join(f"Np={n:>3}" for n in NP_VALUES)
    print(header)
    print("-" * len(header))
    for ns in NS_VALUES:
        vals = []
        for np_val in NP_VALUES:
            v = next((r["ods"] for r in results
                      if r["Np"] == np_val and r["Ns"] == ns and r["d"] == d), None)
            vals.append(f"{v:.4f}" if v else "  --- ")
        print(f"{ns:>2}   {'  '.join(vals)}")

out = ROOT / "outputs" / "single_image_ablation" / "low_ns_results.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {len(results)} results to {out}")
