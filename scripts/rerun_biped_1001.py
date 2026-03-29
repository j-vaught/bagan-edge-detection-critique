"""Rerun full BIPED ablation with 1001 thresholds to match Bagan's protocol."""

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, maximum_filter

from edgecritic.wvf import wvf_image
from edgecritic.lf import lf_image
from edgecritic.evaluation.metrics import compute_ods_ois

ROOT = Path(__file__).resolve().parent.parent
BIPED = ROOT / "datasets" / "BIPED" / "BIPED" / "BIPED" / "edges"
OUT = ROOT / "outputs" / "biped_ablation"

N_THRESH = 1001  # Match Bagan: 0 to 1 in 0.001 increments

print("Loading BIPED v1 RGB_008...")
img_color = np.array(Image.open(BIPED / "imgs" / "test" / "rgbr" / "RGB_008.jpg"))
img_gray = np.mean(img_color, axis=2)
gt_bool = np.array(Image.open(BIPED / "edge_maps" / "test" / "rgbr" / "RGB_008.png").convert("L")) > 128
print(f"  {img_gray.shape}, {np.sum(gt_bool)} GT edge px")

import torch
HAS_CUDA = torch.cuda.is_available()
backend = "cuda" if HAS_CUDA else "cpu"
if HAS_CUDA:
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    free_gb = torch.cuda.mem_get_info(0)[0] / 1e9
    print(f"  Free VRAM: {free_gb:.1f} GB")
VRAM_GB = max(free_gb * 0.7, 20) if HAS_CUDA else None


def evaluate_config(mag, gt_b):
    ods, ois, thresholds, f_scores = compute_ods_ois(
        mag, gt_b.astype(np.float64), n_thresholds=N_THRESH, match_radius=3)
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


# Full grids
WVF_NP = [5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 65, 80, 100, 130, 160, 200, 250, 300, 400, 500]
WVF_NS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 18, 24, 36, 48, 72, 90, 120, 180]
WVF_D = [2, 3, 4, 5]

LF_NP = [15, 25, 50, 75, 100, 150, 250]
LF_NS = [18, 36, 72]
LF_M = [1, 2, 3, 5, 7, 10, 14, 20]

# WVF
wvf_results = []
total_wvf = sum(1 for d in WVF_D for ns in WVF_NS for np_val in WVF_NP
                if np_val >= (d+1)*(d+2)//2)
print(f"\n=== WVF: {total_wvf} configs, {N_THRESH} thresholds ===")
t0 = time.perf_counter()
count = 0
for d in WVF_D:
    min_c = (d + 1) * (d + 2) // 2
    for ns in WVF_NS:
        for np_val in WVF_NP:
            if np_val < min_c:
                continue
            try:
                mag = wvf_image(img_gray, np_count=np_val, order=d,
                                n_orientations=ns, backend=backend).gradient_mag
                metrics = evaluate_config(mag, gt_bool)
                metrics.update({"filter": "WVF", "Np": np_val, "Ns": ns, "d": d})
                wvf_results.append(metrics)
            except Exception as e:
                print(f"  SKIP WVF Np={np_val} Ns={ns} d={d}: {e}")
            count += 1
            if count % 100 == 0:
                print(f"  {count}/{total_wvf} ({time.perf_counter()-t0:.1f}s)")
wvf_time = time.perf_counter() - t0
print(f"  WVF: {len(wvf_results)} in {wvf_time:.1f}s")

# LF
lf_results = []
oom_configs = []
total_lf = len(LF_NP) * len(LF_NS) * len(LF_M)
print(f"\n=== LF: {total_lf} configs, {N_THRESH} thresholds ===")
t0 = time.perf_counter()
count = 0
for m in LF_M:
    for ns in LF_NS:
        for np_val in LF_NP:
            if HAS_CUDA:
                torch.cuda.empty_cache()
            try:
                mag = lf_image(img_gray, half_width=m, np_count=np_val,
                               order=4, n_orientations=ns, backend=backend,
                               max_vram_gb=VRAM_GB).gradient_mag
                metrics = evaluate_config(mag, gt_bool)
                metrics.update({"filter": "LF", "Np": np_val, "Ns": ns, "m": m, "d": 4})
                lf_results.append(metrics)
            except Exception as e:
                if "out of memory" in str(e).lower():
                    oom_configs.append({"Np": np_val, "Ns": ns, "m": m})
                    print(f"  OOM LF Np={np_val} Ns={ns} m={m}")
                    if HAS_CUDA:
                        torch.cuda.empty_cache()
                else:
                    print(f"  SKIP LF Np={np_val} Ns={ns} m={m}: {e}")
            count += 1
            if count % 20 == 0:
                print(f"  {count}/{total_lf} ({time.perf_counter()-t0:.1f}s)")
lf_time = time.perf_counter() - t0
print(f"  LF: {len(lf_results)} in {lf_time:.1f}s, {len(oom_configs)} OOM")

# Save
all_results = {
    "dataset": "BIPED_v1",
    "image": "RGB_008",
    "image_shape": list(img_gray.shape),
    "n_gt_pixels": int(np.sum(gt_bool)),
    "evaluation": {
        "match_radius": 3,
        "n_thresholds": N_THRESH,
        "method": "distance_transform + maximum_filter + searchsorted",
        "note": "Matches Bagan protocol: 1001 thresholds, 3px match radius",
    },
    "wvf_time_s": round(wvf_time, 2),
    "lf_time_s": round(lf_time, 2),
    "wvf": wvf_results,
    "lf": lf_results,
    "lf_oom": oom_configs,
}

with open(OUT / "ablation_metrics.json", "w") as f:
    json.dump(all_results, f, indent=2)

best_wvf = max(wvf_results, key=lambda x: x["ods"])
best_lf = max(lf_results, key=lambda x: x["ods"])
paper_wvf = next((r for r in wvf_results if r["Np"]==250 and r["d"]==4 and r["Ns"]==18), None)
paper_lf = next((r for r in lf_results if r["Np"]==250 and r["m"]==14 and r["Ns"]==18), None)

print(f"\n{'='*60}")
print(f"BIPED ABLATION (1001 thresholds)")
print(f"  WVF: {len(wvf_results)}, LF: {len(lf_results)}, OOM: {len(oom_configs)}")
print(f"  Best WVF: ODS={best_wvf['ods']:.4f} Np={best_wvf['Np']} Ns={best_wvf['Ns']} d={best_wvf['d']}")
print(f"  Best LF:  ODS={best_lf['ods']:.4f} Np={best_lf['Np']} Ns={best_lf['Ns']} m={best_lf['m']}")
if paper_wvf:
    print(f"  Bagan WVF: ODS={paper_wvf['ods']:.4f}")
if paper_lf:
    print(f"  Bagan LF:  ODS={paper_lf['ods']:.4f}")
print(f"{'='*60}")
