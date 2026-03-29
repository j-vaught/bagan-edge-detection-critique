"""Test batched LF on the configs that previously OOM'd."""

import logging
import numpy as np
from PIL import Image
from pathlib import Path
import torch

logging.basicConfig(level=logging.DEBUG)

from edgecritic.lf import lf_image
from edgecritic.evaluation.metrics import compute_ods_ois

ROOT = Path(__file__).resolve().parent.parent
BIPED = ROOT / "datasets" / "BIPED" / "BIPED" / "BIPED" / "edges"
img = np.mean(np.array(Image.open(BIPED / "imgs" / "test" / "rgbr" / "RGB_008.jpg")), axis=2)
gt = np.array(Image.open(BIPED / "edge_maps" / "test" / "rgbr" / "RGB_008.png").convert("L")) > 128

print(f"Image: {img.shape}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Free VRAM: {torch.cuda.mem_get_info(0)[0]/1e9:.1f} GB")

# Test the previously OOM configs from hardest to easiest
configs = [
    {"Np": 250, "m": 20, "Ns": 18},  # worst case
    {"Np": 250, "m": 14, "Ns": 18},  # Bagan's config
    {"Np": 250, "m": 10, "Ns": 72},
    {"Np": 150, "m": 20, "Ns": 72},
]

for cfg in configs:
    torch.cuda.empty_cache()
    free = torch.cuda.mem_get_info(0)[0] / 1e9
    print(f"\nLF Np={cfg['Np']} m={cfg['m']} Ns={cfg['Ns']} (free: {free:.1f} GB)...")
    try:
        result = lf_image(img, half_width=cfg["m"], np_count=cfg["Np"],
                          order=4, n_orientations=cfg["Ns"], backend="cuda",
                          max_vram_gb=25)
        ods, _, _, _ = compute_ods_ois(result.gradient_mag, gt.astype(np.float64),
                                        n_thresholds=500, match_radius=3)
        print(f"  SUCCESS: ODS={ods:.4f}")
    except torch.cuda.OutOfMemoryError:
        print(f"  OOM")
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ERROR: {e}")
