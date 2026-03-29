"""Tier 2: High-fidelity single-image ablation on BSDS500 #100007.

Runs WVF and LF dense parameter sweeps, saves all raw metric data,
and generates visualization panels.
"""

import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io as sio

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
IMG_PATH = ROOT / "datasets/BSDS500/BSDS500/data/images/test/100007.jpg"
GT_PATH = ROOT / "datasets/BSDS500/BSDS500/data/groundTruth/test/100007.mat"
OUT_DIR = ROOT / "outputs" / "single_image_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load image and ground truth
# ---------------------------------------------------------------------------
print("Loading image and ground truth...")
img_color = np.array(Image.open(IMG_PATH))
img_gray = np.mean(img_color, axis=2)  # simple RGB average
print(f"  Image: {img_color.shape} -> grayscale {img_gray.shape}")

gt_mat = sio.loadmat(str(GT_PATH))
gt_cell = gt_mat["groundTruth"]
n_annotators = gt_cell.shape[1]
print(f"  Ground truth: {n_annotators} annotators")

# Union of all annotator boundaries
gt_union = np.zeros(img_gray.shape, dtype=bool)
gt_boundaries = []
for i in range(n_annotators):
    bdry_raw = gt_cell[0, i]["Boundaries"][0, 0]
    bdry = bdry_raw.toarray() if hasattr(bdry_raw, "toarray") else np.asarray(bdry_raw)
    gt_boundaries.append(bdry)
    gt_union |= (bdry > 0)
print(f"  GT union: {np.sum(gt_union)} edge pixels")

# ---------------------------------------------------------------------------
# Evaluation function (uses the fast vectorized pipeline)
# ---------------------------------------------------------------------------
from edgecritic.evaluation.metrics import compute_ods_ois


def evaluate_config(mag, gt_bool, n_thresholds=500):
    """Run ODS evaluation and return detailed metrics."""
    ods, ois, thresholds, f_scores = compute_ods_ois(
        mag, gt_bool.astype(np.float64), n_thresholds=n_thresholds, match_radius=3
    )
    best_idx = np.argmax(f_scores)
    best_t = thresholds[best_idx]

    # Compute P/R at best threshold
    from scipy.ndimage import distance_transform_edt, maximum_filter
    pred = mag > best_t
    dist_to_gt = distance_transform_edt(~gt_bool)
    near_gt = dist_to_gt <= 3
    tp = int(np.sum(pred & near_gt))
    fp = int(np.sum(pred & ~near_gt))
    mag_max_local = maximum_filter(mag, size=7)
    gt_local_max = mag_max_local[gt_bool]
    fn = int(np.sum(gt_local_max <= best_t))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "ods": float(ods),
        "best_threshold": float(best_t),
        "precision": float(precision),
        "recall": float(recall),
        "tp": tp, "fp": fp, "fn": fn,
        "n_pred": int(np.sum(pred)),
        "n_gt": int(np.sum(gt_bool)),
    }


# ---------------------------------------------------------------------------
# Detect backend
# ---------------------------------------------------------------------------
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
    if HAS_CUDA:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    HAS_CUDA = False

if not HAS_CUDA:
    print("  WARNING: No CUDA — will use CPU (slow for large Np)")

backend = "cuda" if HAS_CUDA else "cpu"

# ---------------------------------------------------------------------------
# WVF Dense Sweep
# ---------------------------------------------------------------------------
from edgecritic.wvf import wvf_image

NP_VALUES = [5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 65, 80, 100, 130, 160, 200, 250, 300, 400, 500]
NS_VALUES = [9, 12, 18, 24, 36, 48, 72, 90, 120, 180]
D_VALUES = [2, 3, 4, 5]

wvf_results = []
total_wvf = len(NP_VALUES) * len(NS_VALUES) * len(D_VALUES)
print(f"\n=== WVF Dense Sweep: {total_wvf} configurations ===")
t0 = time.perf_counter()
count = 0

for d in D_VALUES:
    for ns in NS_VALUES:
        for np_val in NP_VALUES:
            # Need enough neighbors for the polynomial order
            min_coeffs = (d + 1) * (d + 2) // 2
            if np_val < min_coeffs:
                count += 1
                continue

            try:
                result = wvf_image(img_gray, np_count=np_val, order=d,
                                   n_orientations=ns, backend=backend)
                mag = result.gradient_mag
                metrics = evaluate_config(mag, gt_union)
                metrics.update({"filter": "WVF", "Np": np_val, "Ns": ns, "d": d})
                wvf_results.append(metrics)
            except Exception as e:
                print(f"  SKIP WVF Np={np_val} Ns={ns} d={d}: {e}")

            count += 1
            if count % 50 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  {count}/{total_wvf} configs done ({elapsed:.1f}s)")

wvf_time = time.perf_counter() - t0
print(f"  WVF sweep: {len(wvf_results)} configs in {wvf_time:.1f}s")

# ---------------------------------------------------------------------------
# LF Dense Sweep
# ---------------------------------------------------------------------------
from edgecritic.lf import lf_image

LF_NP = [15, 25, 50, 75, 100, 150, 250]
LF_NS = [18, 36, 72]
LF_M = [1, 2, 3, 5, 7, 10, 14, 20]

lf_results = []
total_lf = len(LF_NP) * len(LF_NS) * len(LF_M)
print(f"\n=== LF Dense Sweep: {total_lf} configurations ===")
t0 = time.perf_counter()
count = 0

for m in LF_M:
    for ns in LF_NS:
        for np_val in LF_NP:
            try:
                result = lf_image(img_gray, half_width=m, np_count=np_val,
                                  order=4, n_orientations=ns, backend=backend)
                mag = result.gradient_mag
                metrics = evaluate_config(mag, gt_union)
                metrics.update({"filter": "LF", "Np": np_val, "Ns": ns,
                                "m": m, "d": 4})
                lf_results.append(metrics)
            except Exception as e:
                print(f"  SKIP LF Np={np_val} Ns={ns} m={m}: {e}")

            count += 1
            if count % 20 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  {count}/{total_lf} configs done ({elapsed:.1f}s)")

lf_time = time.perf_counter() - t0
print(f"  LF sweep: {len(lf_results)} configs in {lf_time:.1f}s")

# ---------------------------------------------------------------------------
# Save all raw metrics
# ---------------------------------------------------------------------------
all_results = {
    "image": "BSDS500_100007",
    "image_shape": list(img_gray.shape),
    "n_annotators": n_annotators,
    "n_gt_pixels": int(np.sum(gt_union)),
    "evaluation": {
        "match_radius": 3,
        "n_thresholds": 500,
        "method": "distance_transform + maximum_filter + searchsorted",
    },
    "wvf_time_s": round(wvf_time, 2),
    "lf_time_s": round(lf_time, 2),
    "wvf": wvf_results,
    "lf": lf_results,
}

json_path = OUT_DIR / "ablation_metrics.json"
with open(json_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved {len(wvf_results)} WVF + {len(lf_results)} LF results to {json_path}")

# ---------------------------------------------------------------------------
# Find best/worst configs
# ---------------------------------------------------------------------------
all_configs = wvf_results + lf_results
all_configs_sorted = sorted(all_configs, key=lambda x: x["ods"], reverse=True)
best = all_configs_sorted[0]
worst = [c for c in all_configs_sorted if c["ods"] > 0][-1]  # worst non-zero

# Best WVF and best LF specifically
wvf_sorted = sorted(wvf_results, key=lambda x: x["ods"], reverse=True)
lf_sorted = sorted(lf_results, key=lambda x: x["ods"], reverse=True)
best_wvf = wvf_sorted[0] if wvf_sorted else None
best_lf = lf_sorted[0] if lf_sorted else None

print(f"\nBest overall: {best['filter']} ODS={best['ods']:.4f} (Np={best['Np']}, Ns={best['Ns']}, d={best.get('d')}, m={best.get('m', '-')})")
print(f"Best WVF:     ODS={best_wvf['ods']:.4f} (Np={best_wvf['Np']}, Ns={best_wvf['Ns']}, d={best_wvf['d']})")
if best_lf:
    print(f"Best LF:      ODS={best_lf['ods']:.4f} (Np={best_lf['Np']}, Ns={best_lf['Ns']}, m={best_lf['m']})")
print(f"Worst:        {worst['filter']} ODS={worst['ods']:.4f} (Np={worst['Np']}, Ns={worst['Ns']})")

# ---------------------------------------------------------------------------
# Re-run best configs to get their magnitude maps for visualization
# ---------------------------------------------------------------------------
print("\nRe-running best configs for visualization...")


def run_config(cfg):
    if cfg["filter"] == "WVF":
        return wvf_image(img_gray, np_count=cfg["Np"], order=cfg["d"],
                         n_orientations=cfg["Ns"], backend=backend).gradient_mag
    else:
        return lf_image(img_gray, half_width=cfg["m"], np_count=cfg["Np"],
                        order=cfg["d"], n_orientations=cfg["Ns"],
                        backend=backend).gradient_mag


mag_best_wvf = run_config(best_wvf)
mag_best_lf = run_config(best_lf) if best_lf else None
mag_worst = run_config(worst)

# Also run a median-performing config
mid_idx = len(all_configs_sorted) // 2
median_cfg = all_configs_sorted[mid_idx]
mag_median = run_config(median_cfg)

# ---------------------------------------------------------------------------
# Figure: pipeline visualization
# ---------------------------------------------------------------------------
print("Generating figures...")

# Panel count depends on whether we have LF results
panels = []
panels.append(("Input (RGB)", img_color, {}))
panels.append(("Grayscale", img_gray, {"cmap": "gray"}))
panels.append((f"Ground Truth ({n_annotators} annotators)", gt_union.astype(float), {"cmap": "gray"}))
panels.append((
    f"Best WVF: ODS={best_wvf['ods']:.3f}\nNp={best_wvf['Np']} Ns={best_wvf['Ns']} d={best_wvf['d']}\nP={best_wvf['precision']:.2f} R={best_wvf['recall']:.2f}",
    mag_best_wvf, {"cmap": "gray"}
))
if mag_best_lf is not None:
    panels.append((
        f"Best LF: ODS={best_lf['ods']:.3f}\nNp={best_lf['Np']} Ns={best_lf['Ns']} m={best_lf['m']}\nP={best_lf['precision']:.2f} R={best_lf['recall']:.2f}",
        mag_best_lf, {"cmap": "gray"}
    ))
panels.append((
    f"Median: ODS={median_cfg['ods']:.3f}\n{median_cfg['filter']} Np={median_cfg['Np']} Ns={median_cfg['Ns']}\nP={median_cfg['precision']:.2f} R={median_cfg['recall']:.2f}",
    mag_median, {"cmap": "gray"}
))
panels.append((
    f"Worst: ODS={worst['ods']:.3f}\n{worst['filter']} Np={worst['Np']} Ns={worst['Ns']}\nP={worst['precision']:.2f} R={worst['recall']:.2f}",
    mag_worst, {"cmap": "gray"}
))

n_panels = len(panels)
fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
for ax, (title, data, kwargs) in zip(axes, panels):
    ax.imshow(data, **kwargs)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
fig.suptitle("BSDS500 #100007 — Single-Image Ablation", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "pipeline_overview.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved pipeline_overview.png")

# ---------------------------------------------------------------------------
# Figure: ODS vs Np cross-sections
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# WVF: ODS vs Np at different d, fixed Ns=36
ax = axes[0]
for d in D_VALUES:
    xs, ys = [], []
    for r in wvf_results:
        if r["d"] == d and r["Ns"] == 36:
            xs.append(r["Np"])
            ys.append(r["ods"])
    if xs:
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], "o-", markersize=4, label=f"d={d}")
ax.set_xlabel("$N_p$ (support size)")
ax.set_ylabel("ODS F-score")
ax.set_title("WVF: ODS vs $N_p$ at $N_s$=36, varying order $d$")
ax.legend()
ax.grid(True, alpha=0.3)

# LF: ODS vs Np at different m, fixed Ns=36
ax = axes[1]
for m in LF_M:
    xs, ys = [], []
    for r in lf_results:
        if r["m"] == m and r["Ns"] == 36:
            xs.append(r["Np"])
            ys.append(r["ods"])
    if xs:
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], "o-", markersize=4, label=f"m={m}")
if wvf_results:
    xs, ys = [], []
    for r in wvf_results:
        if r["d"] == 4 and r["Ns"] == 36:
            xs.append(r["Np"])
            ys.append(r["ods"])
    if xs:
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], "k--", markersize=4, label="WVF (d=4)")
ax.set_xlabel("$N_p$ (support size)")
ax.set_ylabel("ODS F-score")
ax.set_title("LF: ODS vs $N_p$ at $N_s$=36, varying half-width $m$")
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUT_DIR / "ods_vs_np.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved ods_vs_np.png")

# ---------------------------------------------------------------------------
# Figure: ODS heatmap (Np x Ns) for WVF at d=4
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
np_vals = sorted(set(r["Np"] for r in wvf_results if r["d"] == 4))
ns_vals = sorted(set(r["Ns"] for r in wvf_results if r["d"] == 4))
heatmap = np.full((len(np_vals), len(ns_vals)), np.nan)
lookup = {(r["Np"], r["Ns"]): r["ods"] for r in wvf_results if r["d"] == 4}
for i, np_v in enumerate(np_vals):
    for j, ns_v in enumerate(ns_vals):
        if (np_v, ns_v) in lookup:
            heatmap[i, j] = lookup[(np_v, ns_v)]

im = ax.imshow(heatmap, aspect="auto", origin="lower", cmap="viridis")
ax.set_xticks(range(len(ns_vals)))
ax.set_xticklabels(ns_vals)
ax.set_yticks(range(len(np_vals)))
ax.set_yticklabels(np_vals)
ax.set_xlabel("$N_s$ (orientations)")
ax.set_ylabel("$N_p$ (support size)")
ax.set_title("WVF ODS Heatmap ($d$=4)")
fig.colorbar(im, ax=ax, label="ODS F-score")

# Annotate cells
for i in range(len(np_vals)):
    for j in range(len(ns_vals)):
        if not np.isnan(heatmap[i, j]):
            ax.text(j, i, f"{heatmap[i, j]:.3f}", ha="center", va="center",
                    fontsize=6, color="white" if heatmap[i, j] < np.nanmean(heatmap) else "black")

fig.tight_layout()
fig.savefig(OUT_DIR / "wvf_heatmap.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved wvf_heatmap.png")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"ABLATION COMPLETE")
print(f"  WVF configs: {len(wvf_results)} in {wvf_time:.1f}s")
print(f"  LF configs:  {len(lf_results)} in {lf_time:.1f}s")
print(f"  Output dir:  {OUT_DIR}")
print(f"  Metrics:     {json_path}")
print(f"{'='*60}")
