#!/usr/bin/env python3
"""
Generate qualitative panels on local maritime images using reproduced methods.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy import ndimage
from skimage.feature import canny
from skimage.transform import resize


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "src"))

from benchmark_all import (
    load_dexined,
    load_pidinet,
    load_teed,
    run_dexined,
    run_pidinet,
    run_teed,
    to_gray,
)
from wvf_lf import lf_image, wvf_image


IMAGE_DIR = ROOT / "datasets" / "maritime" / "images"
RESULTS_DIR = ROOT / "benchmarks" / "real_maritime_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def read_rgb(path: Path, max_side: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image)
    h, w = arr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        arr = resize(arr, (new_h, new_w), anti_aliasing=True, preserve_range=True).astype(np.uint8)
    return arr


def discover_readable_images(
    image_dir: Path, limit: int, max_side: int
) -> tuple[list[tuple[Path, np.ndarray]], list[dict[str, str]]]:
    readable = []
    skipped = []
    for image_path in sorted(image_dir.glob("*")):
        if not image_path.is_file():
            continue
        if image_path.stat().st_size == 0:
            skipped.append({"name": image_path.name, "reason": "empty file"})
            continue
        try:
            rgb = read_rgb(image_path, max_side=max_side)
        except (UnidentifiedImageError, OSError) as exc:
            skipped.append({"name": image_path.name, "reason": f"unreadable: {exc}"})
            continue
        readable.append((image_path, rgb))
        if len(readable) >= limit:
            break
    return readable, skipped


def normalize_map(edge_map: np.ndarray) -> np.ndarray:
    edge_map = np.asarray(edge_map, dtype=np.float64)
    if edge_map.size == 0:
        return edge_map
    lo = float(np.min(edge_map))
    hi = float(np.percentile(edge_map, 99.5))
    if hi <= lo:
        hi = float(np.max(edge_map))
    if hi <= lo:
        return np.zeros_like(edge_map, dtype=np.float64)
    scaled = np.clip((edge_map - lo) / (hi - lo), 0.0, 1.0)
    return scaled


def traditional_methods(gray: np.ndarray) -> dict[str, np.ndarray]:
    results: dict[str, np.ndarray] = {}

    sigma = (15 - 1) / 4.0
    gx = ndimage.gaussian_filter1d(gray, sigma, axis=1, order=1)
    gy = ndimage.gaussian_filter1d(gray, sigma, axis=0, order=1)
    sobel15 = np.sqrt(gx**2 + gy**2)
    results["Sobel-15x15"] = normalize_map(sobel15)

    log_sigma = 2.0
    log_mag = np.abs(ndimage.gaussian_laplace(gray, sigma=log_sigma))
    results["LoG-sigma2"] = normalize_map(log_mag)

    results["Canny-sigma2"] = canny(gray / 255.0, sigma=2.0).astype(np.float64)
    return results


def load_optional_ml_models() -> tuple[dict[str, object], dict[str, str]]:
    loaded = {}
    status = {}
    for name, loader in (
        ("TEED", load_teed),
        ("DexiNed", load_dexined),
        ("PiDiNet", load_pidinet),
    ):
        try:
            loaded[name] = loader()
            status[name] = "loaded"
        except Exception as exc:
            status[name] = f"failed: {exc}"
    return loaded, status


def main() -> None:
    limit = int(os.environ.get("REAL_MARITIME_LIMIT", 6))
    max_side = int(os.environ.get("REAL_MARITIME_MAX_SIDE", 192))
    output_tag = os.environ.get("REAL_MARITIME_TAG", "paper").strip() or "paper"
    np_count = int(os.environ.get("REAL_MARITIME_WVF_NP", 15))
    half_width = int(os.environ.get("REAL_MARITIME_LF_HALF_WIDTH", 3))
    order = int(os.environ.get("REAL_MARITIME_ORDER", 4))
    n_orientations = int(os.environ.get("REAL_MARITIME_ORIENTATIONS", 18))
    lf_subsample = int(os.environ.get("REAL_MARITIME_LF_SUBSAMPLE", 2))

    readable_images, skipped_images = discover_readable_images(IMAGE_DIR, limit=limit, max_side=max_side)
    if not readable_images:
        out_json = RESULTS_DIR / f"real_maritime_panel_{output_tag}.json"
        out_json.write_text(
            json.dumps(
                {
                    "meta": {
                        "limit": limit,
                        "max_side": max_side,
                        "error": "No readable maritime images found",
                    },
                    "skipped_images": skipped_images,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise SystemExit(f"No readable maritime images found in {IMAGE_DIR}")

    ml_models, ml_status = load_optional_ml_models()
    ml_runners = {
        "TEED": run_teed,
        "DexiNed": run_dexined,
        "PiDiNet": run_pidinet,
    }

    per_image = []
    method_order = [
        "Sobel-15x15",
        "LoG-sigma2",
        "Canny-sigma2",
        f"WVF-Np{np_count}",
        f"LF-m{half_width}-Np{np_count}",
        "TEED",
        "DexiNed",
        "PiDiNet",
    ]

    for image_path, rgb in readable_images:
        print(f"Processing {image_path.name}")
        gray = to_gray(rgb).astype(np.float64)

        display_maps = {}
        timings_ms = {}

        for method_name, edge_map in traditional_methods(gray).items():
            display_maps[method_name] = normalize_map(edge_map)

        start = time.perf_counter()
        wvf_mag, _, _ = wvf_image(
            gray,
            np_count=np_count,
            order=order,
            n_orientations=n_orientations,
        )
        timings_ms[f"WVF-Np{np_count}"] = 1000.0 * (time.perf_counter() - start)
        display_maps[f"WVF-Np{np_count}"] = normalize_map(wvf_mag)

        start = time.perf_counter()
        lf_mag, _, _ = lf_image(
            gray,
            half_width=half_width,
            np_count=np_count,
            order=order,
            n_orientations=n_orientations,
            subsample=lf_subsample,
        )
        timings_ms[f"LF-m{half_width}-Np{np_count}"] = 1000.0 * (time.perf_counter() - start)
        display_maps[f"LF-m{half_width}-Np{np_count}"] = normalize_map(lf_mag)

        for model_name, model in ml_models.items():
            runner = ml_runners[model_name]
            start = time.perf_counter()
            try:
                pred = runner(model, rgb)
                timings_ms[model_name] = 1000.0 * (time.perf_counter() - start)
                display_maps[model_name] = normalize_map(pred)
            except Exception as exc:
                timings_ms[model_name] = -1.0
                ml_status[model_name] = f"failed during inference: {exc}"

        per_image.append(
            {
                "name": image_path.name,
                "rgb": rgb,
                "maps": display_maps,
                "timings_ms": timings_ms,
            }
        )

    available_methods = [
        method for method in method_order if any(method in item["maps"] for item in per_image)
    ]
    n_rows = len(per_image)
    n_cols = 1 + len(available_methods)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.2 * n_rows), squeeze=False)

    for row_idx, item in enumerate(per_image):
        axes[row_idx, 0].imshow(item["rgb"])
        axes[row_idx, 0].axis("off")
        if row_idx == 0:
            axes[row_idx, 0].set_title("Original", fontsize=9)
        axes[row_idx, 0].set_ylabel(item["name"], fontsize=8)

        for col_idx, method in enumerate(available_methods, start=1):
            ax = axes[row_idx, col_idx]
            edge_map = item["maps"].get(method)
            if edge_map is None:
                ax.axis("off")
                continue
            ax.imshow(1.0 - normalize_map(edge_map), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(method, fontsize=9)
            if method in item["timings_ms"] and item["timings_ms"][method] >= 0:
                ax.text(
                    0.98,
                    0.04,
                    f"{item['timings_ms'][method]:.0f} ms",
                    ha="right",
                    va="bottom",
                    fontsize=6,
                    color="#222222",
                    transform=ax.transAxes,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.0),
                )

    fig.suptitle("Real Maritime Qualitative Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_png = RESULTS_DIR / f"real_maritime_panel_{output_tag}.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    out_json = RESULTS_DIR / f"real_maritime_panel_{output_tag}.json"
    serializable = {
        "meta": {
            "limit": limit,
            "max_side": max_side,
            "wvf_np_count": np_count,
            "lf_half_width": half_width,
            "order": order,
            "n_orientations": n_orientations,
            "lf_subsample": lf_subsample,
            "ml_status": ml_status,
            "n_readable_images": len(readable_images),
        },
        "skipped_images": skipped_images,
        "images": [
            {
                "name": item["name"],
                "timings_ms": item["timings_ms"],
                "available_methods": sorted(item["maps"].keys()),
            }
            for item in per_image
        ],
    }
    out_json.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    print(f"Saved {out_png}")
    print(f"Saved {out_json}")


if __name__ == "__main__":
    main()
