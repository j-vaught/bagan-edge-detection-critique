"""Smoke test: does LF with long lines beat WVF at low SNR?

Adds Gaussian noise to BIPED v1 RGB_008 at several SNR levels,
runs a small set of WVF/LF configs, and checks if the crossover exists.
"""

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from edgecritic.wvf import wvf_image
from edgecritic.lf import lf_image
from edgecritic.evaluation.metrics import compute_ods_ois

ROOT = Path(__file__).resolve().parent.parent
BIPED = ROOT / "datasets" / "BIPED" / "BIPED" / "BIPED" / "edges"
OUT = ROOT / "outputs" / "snr_smoke_test"
OUT.mkdir(parents=True, exist_ok=True)

N_THRESH = 1001

print("Loading BIPED v1 RGB_008...")
img_color = np.array(Image.open(BIPED / "imgs" / "test" / "rgbr" / "RGB_008.jpg"))
img_clean = np.mean(img_color, axis=2)
gt_bool = np.array(Image.open(BIPED / "edge_maps" / "test" / "rgbr" / "RGB_008.png").convert("L")) > 128
print(f"  {img_clean.shape}, {np.sum(gt_bool)} GT edge px")
print(f"  Intensity range: [{img_clean.min():.1f}, {img_clean.max():.1f}]")

import torch
backend = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# Signal amplitude (for SNR calculation)
signal_amplitude = img_clean.max() - img_clean.min()

# SNR levels to test
SNR_LEVELS = [0.5, 0.75, 1.0, 1.5, 2.0, 5.0, float('inf')]

# Representative configs to test
configs = [
    # WVF configs
    {"label": "WVF Np=25 d=2 (our best)",  "filter": "WVF", "Np": 25,  "d": 2, "Ns": 18},
    {"label": "WVF Np=50 d=4",             "filter": "WVF", "Np": 50,  "d": 4, "Ns": 18},
    {"label": "WVF Np=250 d=4 (Bagan)",    "filter": "WVF", "Np": 250, "d": 4, "Ns": 18},
    # LF short line
    {"label": "LF Np=25 m=1",              "filter": "LF",  "Np": 25,  "d": 4, "Ns": 18, "m": 1},
    {"label": "LF Np=25 m=3",              "filter": "LF",  "Np": 25,  "d": 4, "Ns": 18, "m": 3},
    {"label": "LF Np=50 m=3",              "filter": "LF",  "Np": 50,  "d": 4, "Ns": 18, "m": 3},
    # LF long line (Bagan's regime)
    {"label": "LF Np=100 m=7",             "filter": "LF",  "Np": 100, "d": 4, "Ns": 18, "m": 7},
    {"label": "LF Np=250 m=14 (Bagan)",    "filter": "LF",  "Np": 250, "d": 4, "Ns": 18, "m": 14},
]

results = []

for snr in SNR_LEVELS:
    if snr == float('inf'):
        img = img_clean.copy()
        snr_label = "inf (clean)"
    else:
        noise_std = signal_amplitude / snr
        np.random.seed(42)  # reproducible noise
        noise = np.random.normal(0, noise_std, img_clean.shape)
        img = np.clip(img_clean + noise, 0, 255)
        snr_label = f"{snr}"

    print(f"\n=== SNR = {snr_label} ===")

    for cfg in configs:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            if cfg["filter"] == "WVF":
                mag = wvf_image(img, np_count=cfg["Np"], order=cfg["d"],
                                n_orientations=cfg["Ns"], backend=backend).gradient_mag
            else:
                mag = lf_image(img, half_width=cfg["m"], np_count=cfg["Np"],
                               order=cfg["d"], n_orientations=cfg["Ns"],
                               backend=backend, max_vram_gb=30).gradient_mag

            ods, _, _, _ = compute_ods_ois(
                mag, gt_bool.astype(np.float64),
                n_thresholds=N_THRESH, match_radius=3)

            results.append({
                "snr": snr if snr != float('inf') else "inf",
                "config": cfg["label"],
                "filter": cfg["filter"],
                "Np": cfg["Np"],
                "Ns": cfg["Ns"],
                "d": cfg["d"],
                "m": cfg.get("m"),
                "ods": float(ods),
            })
            print(f"  {cfg['label']:<35s} ODS={ods:.4f}")

        except Exception as e:
            print(f"  {cfg['label']:<35s} ERROR: {e}")
            results.append({
                "snr": snr if snr != float('inf') else "inf",
                "config": cfg["label"],
                "filter": cfg["filter"],
                "Np": cfg["Np"],
                "Ns": cfg["Ns"],
                "d": cfg["d"],
                "m": cfg.get("m"),
                "ods": -1,
                "error": str(e),
            })

# Save results
with open(OUT / "snr_smoke_test.json", "w") as f:
    json.dump(results, f, indent=2)

# Summary table
print(f"\n{'='*90}")
print(f"{'Config':<35s}", end="")
for snr in SNR_LEVELS:
    label = "clean" if snr == float('inf') else f"{snr}"
    print(f" {label:>7s}", end="")
print()
print("-" * 90)

for cfg in configs:
    print(f"{cfg['label']:<35s}", end="")
    for snr in SNR_LEVELS:
        snr_key = "inf" if snr == float('inf') else snr
        r = next((x for x in results if x["config"] == cfg["label"]
                  and x["snr"] == snr_key), None)
        if r and r["ods"] >= 0:
            print(f" {r['ods']:>7.4f}", end="")
        else:
            print(f"    ERR", end="")
    print()
print(f"{'='*90}")

# Check for crossover
print("\n=== Crossover Analysis ===")
for snr in SNR_LEVELS:
    snr_key = "inf" if snr == float('inf') else snr
    best_wvf = max((r for r in results if r["snr"] == snr_key and r["filter"] == "WVF" and r["ods"] >= 0),
                    key=lambda x: x["ods"], default=None)
    bagan_lf = next((r for r in results if r["snr"] == snr_key and r["config"] == "LF Np=250 m=14 (Bagan)" and r["ods"] >= 0), None)
    best_lf = max((r for r in results if r["snr"] == snr_key and r["filter"] == "LF" and r["ods"] >= 0),
                   key=lambda x: x["ods"], default=None)

    label = "clean" if snr == float('inf') else f"SNR={snr}"
    if best_wvf and bagan_lf:
        gap = bagan_lf["ods"] - best_wvf["ods"]
        winner = "LF(m=14) WINS" if gap > 0 else "WVF wins"
        print(f"  {label:>10s}: best WVF={best_wvf['ods']:.4f} vs Bagan LF={bagan_lf['ods']:.4f}  gap={gap:+.4f}  {winner}")
    if best_wvf and best_lf:
        gap = best_lf["ods"] - best_wvf["ods"]
        winner = "LF wins" if gap > 0 else "WVF wins"
        print(f"{'':>12s} best WVF={best_wvf['ods']:.4f} vs best LF={best_lf['ods']:.4f}  gap={gap:+.4f}  {winner}")
