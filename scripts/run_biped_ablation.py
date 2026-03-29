"""Single-image ablation on BIPED v1 test image RGB_008."""

import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from edgecritic.wvf import wvf_image
from edgecritic.lf import lf_image
from edgecritic.evaluation.metrics import compute_ods_ois
from scipy.ndimage import distance_transform_edt, maximum_filter

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
BIPED = ROOT / "datasets" / "BIPED" / "BIPED" / "BIPED" / "edges"
IMG_PATH = BIPED / "imgs" / "test" / "rgbr" / "RGB_008.jpg"
GT_PATH = BIPED / "edge_maps" / "test" / "rgbr" / "RGB_008.png"
OUT_DIR = ROOT / "outputs" / "biped_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
print("Loading BIPED image and ground truth...")
img_color = np.array(Image.open(IMG_PATH))
img_gray = np.mean(img_color, axis=2)
print(f"  Image: {img_color.shape} -> grayscale {img_gray.shape}")

gt_raw = np.array(Image.open(GT_PATH).convert("L"))
gt_bool = gt_raw > 128  # binary threshold
print(f"  GT: {gt_raw.shape}, {np.sum(gt_bool)} edge pixels")

# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
    if HAS_CUDA:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    HAS_CUDA = False
backend = "cuda" if HAS_CUDA else "cpu"

# ---------------------------------------------------------------------------
# WVF sweep (including low Ns)
# ---------------------------------------------------------------------------
NP_VALUES = [15, 20, 25, 30, 40, 50, 65, 80, 100, 130, 160, 200, 250, 300, 400, 500]
NS_VALUES = [1, 2, 3, 4, 6, 9, 12, 18, 36, 72]
D_VALUES = [2, 3, 4]

wvf_results = []
total = len(NP_VALUES) * len(NS_VALUES) * len(D_VALUES)
print(f"\n=== WVF Sweep: {total} configurations ===")
t0 = time.perf_counter()
count = 0
for d in D_VALUES:
    for ns in NS_VALUES:
        for np_val in NP_VALUES:
            min_c = (d + 1) * (d + 2) // 2
            if np_val < min_c:
                count += 1
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
            if count % 50 == 0:
                print(f"  {count}/{total} ({time.perf_counter()-t0:.1f}s)")
wvf_time = time.perf_counter() - t0
print(f"  WVF: {len(wvf_results)} configs in {wvf_time:.1f}s")

# ---------------------------------------------------------------------------
# LF sweep
# ---------------------------------------------------------------------------
LF_NP = [15, 25, 50, 75, 100, 150, 250]
LF_NS = [18, 36]
LF_M = [1, 2, 3, 5, 7, 10, 14, 20]

lf_results = []
total_lf = len(LF_NP) * len(LF_NS) * len(LF_M)
print(f"\n=== LF Sweep: {total_lf} configurations ===")
t0 = time.perf_counter()
count = 0
for m in LF_M:
    for ns in LF_NS:
        for np_val in LF_NP:
            try:
                mag = lf_image(img_gray, half_width=m, np_count=np_val,
                               order=4, n_orientations=ns, backend=backend,
                               max_vram_gb=30).gradient_mag
                metrics = evaluate_config(mag, gt_bool)
                metrics.update({"filter": "LF", "Np": np_val, "Ns": ns, "m": m, "d": 4})
                lf_results.append(metrics)
            except Exception as e:
                print(f"  SKIP LF Np={np_val} Ns={ns} m={m}: {e}")
            count += 1
            if count % 20 == 0:
                print(f"  {count}/{total_lf} ({time.perf_counter()-t0:.1f}s)")
lf_time = time.perf_counter() - t0
print(f"  LF: {len(lf_results)} configs in {lf_time:.1f}s")

# ---------------------------------------------------------------------------
# Save raw metrics
# ---------------------------------------------------------------------------
all_results = {
    "dataset": "BIPED_v1",
    "image": "RGB_008",
    "image_shape": list(img_gray.shape),
    "n_gt_pixels": int(np.sum(gt_bool)),
    "evaluation": {"match_radius": 3, "n_thresholds": 500,
                    "method": "distance_transform + maximum_filter + searchsorted"},
    "wvf_time_s": round(wvf_time, 2),
    "lf_time_s": round(lf_time, 2),
    "wvf": wvf_results,
    "lf": lf_results,
}
json_path = OUT_DIR / "ablation_metrics.json"
with open(json_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved {len(wvf_results)} WVF + {len(lf_results)} LF to {json_path}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
best_wvf = max(wvf_results, key=lambda x: x["ods"])
best_lf = max(lf_results, key=lambda x: x["ods"])
all_sorted = sorted(wvf_results + lf_results, key=lambda x: x["ods"], reverse=True)
worst = [c for c in all_sorted if c["ods"] > 0][-1]

# Bagan's params
paper_wvf = next((r for r in wvf_results if r["Np"]==250 and r["d"]==4 and r["Ns"]==18), None)
paper_lf = next((r for r in lf_results if r["Np"]==250 and r["m"]==14 and r["Ns"]==18), None)

print(f"\n{'='*65}")
print(f"BIPED v1 RGB_008 — ABLATION RESULTS")
print(f"{'='*65}")
print(f"Best WVF:  ODS={best_wvf['ods']:.4f}  Np={best_wvf['Np']} Ns={best_wvf['Ns']} d={best_wvf['d']}  P={best_wvf['precision']:.3f} R={best_wvf['recall']:.3f}")
print(f"Best LF:   ODS={best_lf['ods']:.4f}  Np={best_lf['Np']} Ns={best_lf['Ns']} m={best_lf['m']}  P={best_lf['precision']:.3f} R={best_lf['recall']:.3f}")
print(f"Worst:     ODS={worst['ods']:.4f}  {worst['filter']} Np={worst['Np']} Ns={worst['Ns']}")
if paper_wvf:
    print(f"\nBagan WVF (Np=250 d=4):  ODS={paper_wvf['ods']:.4f}  P={paper_wvf['precision']:.3f} R={paper_wvf['recall']:.3f}")
    rank = next(i for i, x in enumerate(all_sorted) if x is paper_wvf) + 1
    print(f"  Rank: {rank}/{len(all_sorted)}")
    print(f"  vs best: {best_wvf['ods']-paper_wvf['ods']:+.4f}")
if paper_lf:
    print(f"Bagan LF  (Np=250 m=14): ODS={paper_lf['ods']:.4f}  P={paper_lf['precision']:.3f} R={paper_lf['recall']:.3f}")
    rank = next(i for i, x in enumerate(all_sorted) if x is paper_lf) + 1
    print(f"  Rank: {rank}/{len(all_sorted)}")
    print(f"  vs best: {best_lf['ods']-paper_lf['ods']:+.4f}")

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
print("\nGenerating figures...")

def run_config(cfg):
    if cfg["filter"] == "WVF":
        return wvf_image(img_gray, np_count=cfg["Np"], order=cfg["d"],
                         n_orientations=cfg["Ns"], backend=backend).gradient_mag
    else:
        return lf_image(img_gray, half_width=cfg["m"], np_count=cfg["Np"],
                        order=cfg["d"], n_orientations=cfg["Ns"],
                        backend=backend).gradient_mag

# Pipeline overview
mag_best_wvf = run_config(best_wvf)
mag_best_lf = run_config(best_lf)
mag_paper_wvf = run_config(paper_wvf) if paper_wvf else None
mag_paper_lf = run_config(paper_lf) if paper_lf else None

panels = [
    ("Input (RGB)", img_color, {}),
    ("Grayscale", img_gray, {"cmap": "gray"}),
    (f"Ground Truth\n({np.sum(gt_bool)} edge px)", gt_bool.astype(float), {"cmap": "gray"}),
    (f"Best WVF: ODS={best_wvf['ods']:.3f}\nNp={best_wvf['Np']} Ns={best_wvf['Ns']} d={best_wvf['d']}\nP={best_wvf['precision']:.2f} R={best_wvf['recall']:.2f}",
     mag_best_wvf, {"cmap": "gray"}),
    (f"Best LF: ODS={best_lf['ods']:.3f}\nNp={best_lf['Np']} Ns={best_lf['Ns']} m={best_lf['m']}\nP={best_lf['precision']:.2f} R={best_lf['recall']:.2f}",
     mag_best_lf, {"cmap": "gray"}),
]
if mag_paper_wvf is not None:
    panels.append((f"Bagan WVF: ODS={paper_wvf['ods']:.3f}\nNp=250 d=4 Ns=18\nP={paper_wvf['precision']:.2f} R={paper_wvf['recall']:.2f}",
                    mag_paper_wvf, {"cmap": "gray"}))
if mag_paper_lf is not None:
    panels.append((f"Bagan LF: ODS={paper_lf['ods']:.3f}\nNp=250 m=14 Ns=18\nP={paper_lf['precision']:.2f} R={paper_lf['recall']:.2f}",
                    mag_paper_lf, {"cmap": "gray"}))

n = len(panels)
fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5))
for ax, (title, data, kwargs) in zip(axes, panels):
    ax.imshow(data, **kwargs)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
fig.suptitle("BIPED v1 RGB_008 — Single-Image Ablation", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "pipeline_overview.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved pipeline_overview.png")

# ODS vs Np
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
d_colors = {2: "#e41a1c", 3: "#377eb8", 4: "#4daf4a"}
for d in D_VALUES:
    xs, ys = [], []
    for r in wvf_results:
        if r["d"] == d and r["Ns"] == 18:
            xs.append(r["Np"]); ys.append(r["ods"])
    if xs:
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], "o-",
                color=d_colors[d], markersize=5, label=f"WVF d={d}")
ax.set_xlabel("$N_p$"); ax.set_ylabel("ODS F-score")
ax.set_title("WVF: ODS vs $N_p$ ($N_s$=18)"); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
for m in LF_M:
    xs, ys = [], []
    for r in lf_results:
        if r["m"] == m and r["Ns"] == 18:
            xs.append(r["Np"]); ys.append(r["ods"])
    if xs:
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], "o-", markersize=5, label=f"m={m}")
xs, ys = [], []
for r in wvf_results:
    if r["d"] == 4 and r["Ns"] == 18:
        xs.append(r["Np"]); ys.append(r["ods"])
if xs:
    order = np.argsort(xs)
    ax.plot(np.array(xs)[order], np.array(ys)[order], "k--", markersize=4, label="WVF (d=4)")
ax.set_xlabel("$N_p$"); ax.set_ylabel("ODS F-score")
ax.set_title("LF: ODS vs $N_p$ ($N_s$=18)"); ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUT_DIR / "ods_vs_np.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved ods_vs_np.png")

# Ns cliff
fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.cm.viridis(np.linspace(0.15, 0.85, 5))
for ci, np_val in enumerate([15, 50, 100, 200, 500]):
    ns_vals, ods_vals = [], []
    for r in wvf_results:
        if r["Np"] == np_val and r["d"] == 4:
            ns_vals.append(r["Ns"]); ods_vals.append(r["ods"])
    if ns_vals:
        order = np.argsort(ns_vals)
        ax.plot(np.array(ns_vals)[order], np.array(ods_vals)[order], "o-",
                color=colors[ci], markersize=6, label=f"$N_p$={np_val}")
import matplotlib.ticker
ax.axvline(x=3, color="red", linestyle=":", alpha=0.5, label="$N_s$=3")
ax.set_xlabel("$N_s$ (orientations)"); ax.set_ylabel("ODS F-score")
ax.set_title("Orientation Cliff on BIPED v1"); ax.legend()
ax.set_xscale("log", base=2)
ax.set_xticks([1, 2, 3, 4, 6, 9, 12, 18, 36, 72])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "ns_cliff.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved ns_cliff.png")

# WVF heatmap (d=4)
fig, ax = plt.subplots(figsize=(10, 6))
nps = sorted(set(r["Np"] for r in wvf_results if r["d"] == 4))
nss = sorted(set(r["Ns"] for r in wvf_results if r["d"] == 4))
hm = np.full((len(nps), len(nss)), np.nan)
lookup = {(r["Np"], r["Ns"]): r["ods"] for r in wvf_results if r["d"] == 4}
for i, np_v in enumerate(nps):
    for j, ns_v in enumerate(nss):
        if (np_v, ns_v) in lookup:
            hm[i, j] = lookup[(np_v, ns_v)]
im = ax.imshow(hm, aspect="auto", origin="lower", cmap="RdYlGn", vmin=0.4, vmax=np.nanmax(hm)+0.01)
ax.set_xticks(range(len(nss))); ax.set_xticklabels(nss)
ax.set_yticks(range(len(nps))); ax.set_yticklabels(nps)
ax.set_xlabel("$N_s$"); ax.set_ylabel("$N_p$")
ax.set_title("WVF ODS Heatmap — BIPED v1 ($d$=4)")
fig.colorbar(im, ax=ax, label="ODS")
cliff_x = nss.index(3) - 0.5 if 3 in nss else None
if cliff_x is not None:
    ax.axvline(x=cliff_x, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
for i in range(len(nps)):
    for j in range(len(nss)):
        if not np.isnan(hm[i, j]):
            c = "white" if hm[i, j] < np.nanmean(hm) else "black"
            ax.text(j, i, f"{hm[i, j]:.3f}", ha="center", va="center", fontsize=6, color=c)
fig.tight_layout()
fig.savefig(OUT_DIR / "wvf_heatmap.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved wvf_heatmap.png")

print(f"\nAll outputs in {OUT_DIR}")
