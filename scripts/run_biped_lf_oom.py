"""Run the BIPED LF configs that OOM'd on RTX 6000 (48GB).

A100 has 80GB — should handle most of these. For any that still OOM,
we note them as infeasible at native BIPED resolution.
"""

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, maximum_filter

from edgecritic.lf import lf_image
from edgecritic.evaluation.metrics import compute_ods_ois

ROOT = Path(__file__).resolve().parent.parent
BIPED = ROOT / "datasets" / "BIPED" / "BIPED" / "BIPED" / "edges"
IMG_PATH = BIPED / "imgs" / "test" / "rgbr" / "RGB_008.jpg"
GT_PATH = BIPED / "edge_maps" / "test" / "rgbr" / "RGB_008.png"
OUT_DIR = ROOT / "outputs" / "biped_ablation"

print("Loading BIPED image and ground truth...")
img_gray = np.mean(np.array(Image.open(IMG_PATH)), axis=2)
gt_bool = np.array(Image.open(GT_PATH).convert("L")) > 128
print(f"  Image: {img_gray.shape}, GT: {np.sum(gt_bool)} edge px")

import torch
print(f"  GPU: {torch.cuda.get_device_name(0)}")
props = torch.cuda.get_device_properties(0)
mem_gb = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1e9
print(f"  VRAM: {mem_gb:.1f} GB")


def evaluate_config(mag, gt_b, n_thresholds=500):
    ods, ois, thresholds, f_scores = compute_ods_ois(
        mag, gt_b.astype(np.float64), n_thresholds=n_thresholds, match_radius=3)
    best_idx = np.argmax(f_scores)
    best_t = thresholds[best_idx]
    pred = mag > best_t
    dist_to_gt = distance_transform_edt(~gt_b)
    near_gt = dist_to_gt <= 3
    tp = int(np.sum(pred & near_gt))
    fp = int(np.sum(pred & ~near_gt))
    mag_max_local = maximum_filter(mag, size=7)
    gt_local_max = mag_max_local[gt_b]
    fn = int(np.sum(gt_local_max <= best_t))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "ods": float(ods), "best_threshold": float(best_t),
        "precision": float(precision), "recall": float(recall),
        "tp": tp, "fp": fp, "fn": fn,
        "n_pred": int(np.sum(pred)), "n_gt": int(np.sum(gt_b)),
    }


# Load existing results to find what's missing
with open(OUT_DIR / "ablation_metrics.json") as f:
    existing = json.load(f)

existing_lf_keys = {(r["Np"], r["Ns"], r["m"]) for r in existing["lf"]}

# Full LF grid
LF_NP = [15, 25, 50, 75, 100, 150, 250]
LF_NS = [18, 36]
LF_M = [1, 2, 3, 5, 7, 10, 14, 20]

missing = [(np_val, ns, m) for m in LF_M for ns in LF_NS for np_val in LF_NP
           if (np_val, ns, m) not in existing_lf_keys]

print(f"\nExisting LF configs: {len(existing_lf_keys)}")
print(f"Missing LF configs: {len(missing)}")

new_results = []
oom_configs = []
t0 = time.perf_counter()

for idx, (np_val, ns, m) in enumerate(missing):
    torch.cuda.empty_cache()
    try:
        mag = lf_image(img_gray, half_width=m, np_count=np_val,
                       order=4, n_orientations=ns, backend="cuda",
                       max_vram_gb=max(mem_gb * 0.7, 20)).gradient_mag
        metrics = evaluate_config(mag, gt_bool)
        metrics.update({"filter": "LF", "Np": np_val, "Ns": ns, "m": m, "d": 4})
        new_results.append(metrics)
        print(f"  [{idx+1}/{len(missing)}] LF Np={np_val} Ns={ns} m={m}: ODS={metrics['ods']:.4f}")
    except torch.cuda.OutOfMemoryError:
        oom_configs.append({"Np": np_val, "Ns": ns, "m": m})
        print(f"  [{idx+1}/{len(missing)}] LF Np={np_val} Ns={ns} m={m}: OOM on {mem_gb:.0f}GB")
        torch.cuda.empty_cache()

elapsed = time.perf_counter() - t0
print(f"\nCompleted {len(new_results)} configs, {len(oom_configs)} OOM'd, in {elapsed:.1f}s")

# Merge into existing results
existing["lf"].extend(new_results)
existing["lf_oom"] = oom_configs
existing["a100_lf_time_s"] = round(elapsed, 2)

with open(OUT_DIR / "ablation_metrics.json", "w") as f:
    json.dump(existing, f, indent=2)
print(f"Updated {OUT_DIR / 'ablation_metrics.json'}")

# Summary
if new_results:
    best_new = max(new_results, key=lambda x: x["ods"])
    print(f"\nBest new LF: ODS={best_new['ods']:.4f} Np={best_new['Np']} m={best_new['m']} Ns={best_new['Ns']}")

# Check Bagan's exact config
bagan = next((r for r in new_results if r["Np"] == 250 and r["m"] == 14 and r["Ns"] == 18), None)
if bagan:
    print(f"Bagan LF (Np=250 m=14 Ns=18): ODS={bagan['ods']:.4f}")
elif any(c["Np"] == 250 and c["m"] == 14 and c["Ns"] == 18 for c in oom_configs):
    print("Bagan LF (Np=250 m=14 Ns=18): OOM — cannot run at native BIPED resolution")

if oom_configs:
    print(f"\nOOM configs ({len(oom_configs)}):")
    for c in oom_configs:
        print(f"  Np={c['Np']} Ns={c['Ns']} m={c['m']}")
