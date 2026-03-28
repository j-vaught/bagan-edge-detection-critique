#!/usr/bin/env python3
"""
Sweep WVF/LF parameters toward the author settings on declared subsets.

This is meant to answer one question directly:
does moving toward Np=150/250 and larger LF widths improve the reproduced
metrics enough to change the critique, or mostly increase cost?
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmarks" / "wvf_scale_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "benchmarks"))

from benchmark_all import compute_metrics, load_bsds500_test, load_uded_test, to_gray
from wvf_lf import lf_image, wvf_image


def parse_int_list(name: str, default: str) -> list[int]:
    return [int(x.strip()) for x in os.environ.get(name, default).split(",") if x.strip()]


def parse_shape_list(name: str, default: str) -> list[tuple[int, int]]:
    shapes = []
    for raw in os.environ.get(name, default).split(","):
        raw = raw.strip().lower()
        if not raw:
            continue
        h_str, w_str = raw.split("x", 1)
        shapes.append((int(h_str), int(w_str)))
    return shapes


def dataset_loader(name: str):
    if name.upper() == "BSDS500":
        return load_bsds500_test
    if name.upper() == "UDED":
        return load_uded_test
    raise ValueError(f"Unsupported dataset {name!r}")


def resolve_worker_count(task_count: int) -> int:
    raw = (
        os.environ.get("WVF_SWEEP_WORKERS")
        or os.environ.get("SLURM_CPUS_PER_TASK")
        or os.environ.get("OMP_NUM_THREADS")
        or str(os.cpu_count() or 1)
    )
    try:
        workers = int(raw)
    except ValueError:
        workers = os.cpu_count() or 1
    return max(1, min(workers, task_count))


def run_scale_task(task: dict) -> dict:
    start = time.perf_counter()
    if task["method"] == "WVF":
        mag, _, _ = wvf_image(
            task["image"],
            np_count=task["np_count"],
            order=task["order"],
            n_orientations=task["n_orientations"],
        )
    else:
        mag, _, _ = lf_image(
            task["image"],
            half_width=task["half_width"],
            np_count=task["np_count"],
            order=task["order"],
            n_orientations=task["n_orientations"],
            subsample=task["lf_subsample"],
        )
    elapsed = time.perf_counter() - start
    if mag.max() > 0:
        mag = mag / mag.max()
    return {
        "config_key": task["config_key"],
        "image_idx": task["image_idx"],
        "mag": mag,
        "elapsed_s": elapsed,
    }


def main() -> None:
    dataset_name = os.environ.get("SWEEP_DATASET", "UDED").strip().upper()
    max_images = int(os.environ.get("SWEEP_MAX_IMAGES", 3))
    np_counts = parse_int_list("SWEEP_NP_COUNTS", "15,50,100,150")
    half_widths = parse_int_list("SWEEP_LF_HALF_WIDTHS", "3,7,14")
    resize_shapes = parse_shape_list("SWEEP_RESIZE_SHAPES", "64x64,96x96")
    n_orientations = int(os.environ.get("SWEEP_N_ORIENTATIONS", 18))
    order = int(os.environ.get("SWEEP_ORDER", 4))
    n_thresholds = int(os.environ.get("SWEEP_N_THRESHOLDS", 1001))
    match_radius = int(os.environ.get("SWEEP_MATCH_RADIUS", 3))
    lf_subsample = int(os.environ.get("SWEEP_LF_SUBSAMPLE", 2))
    output_tag = os.environ.get("SWEEP_TAG", dataset_name.lower())

    images, ground_truths, names = dataset_loader(dataset_name)()
    if not images:
        raise SystemExit(f"No images loaded for dataset {dataset_name}")
    images = images[:max_images]
    ground_truths = ground_truths[:max_images]
    names = names[:max_images]

    rows = []

    print("WVF/LF SCALE SWEEP")
    print(f"  dataset={dataset_name} images={len(images)}")
    print(f"  np_counts={np_counts}")
    print(f"  half_widths={half_widths}")
    print(f"  resize_shapes={resize_shapes}")
    print(f"  orientations={n_orientations}, thresholds={n_thresholds}")

    resized_images_by_shape = {}
    resized_gt_by_shape = {}
    for resize_shape in resize_shapes:
        resized_images = []
        resized_gt = []
        for img, gt in zip(images, ground_truths):
            resized_images.append(
                resize(to_gray(img), resize_shape, anti_aliasing=True, preserve_range=True).astype(np.float64)
            )
            resized_gt.append(
                resize(gt, resize_shape, anti_aliasing=True, preserve_range=True).astype(np.float64)
            )
        resized_images_by_shape[resize_shape] = resized_images
        resized_gt_by_shape[resize_shape] = resized_gt

    configs = []
    for resize_shape in resize_shapes:
        for np_count in np_counts:
            configs.append(
                {
                    "config_key": ("WVF", np_count, None, resize_shape),
                    "dataset": dataset_name,
                    "method": "WVF",
                    "np_count": np_count,
                    "half_width": None,
                    "resize_shape": resize_shape,
                }
            )

            for half_width in half_widths:
                configs.append(
                    {
                        "config_key": ("LF", np_count, half_width, resize_shape),
                        "dataset": dataset_name,
                        "method": "LF",
                        "np_count": np_count,
                        "half_width": half_width,
                        "resize_shape": resize_shape,
                    }
                )

    tasks = []
    for config in configs:
        resize_shape = config["resize_shape"]
        for image_idx, img in enumerate(resized_images_by_shape[resize_shape]):
            task = {
                "config_key": config["config_key"],
                "method": config["method"],
                "np_count": config["np_count"],
                "half_width": config["half_width"],
                "resize_shape": resize_shape,
                "order": order,
                "n_orientations": n_orientations,
                "lf_subsample": lf_subsample,
                "image_idx": image_idx,
                "image": img,
            }
            tasks.append(task)

    workers = resolve_worker_count(len(tasks))
    print(f"  Dispatching {len(tasks)} WVF/LF tasks across {workers} worker processes")

    grouped = defaultdict(list)
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_scale_task, task) for task in tasks]
        progress_interval = max(1, len(tasks) // 10)
        for future in as_completed(futures):
            result = future.result()
            grouped[result["config_key"]].append(result)
            completed += 1
            if completed % progress_interval == 0 or completed == len(tasks):
                print(f"  Completed {completed}/{len(tasks)} WVF/LF tasks")

    for config in configs:
        key = config["config_key"]
        resize_shape = config["resize_shape"]
        ordered = sorted(grouped[key], key=lambda item: item["image_idx"])
        preds = [item["mag"] for item in ordered]
        elapsed = sum(item["elapsed_s"] for item in ordered)
        resized_gt = resized_gt_by_shape[resize_shape]
        ods, ois, ap = compute_metrics(preds, resized_gt, n_thresholds=n_thresholds, match_radius=match_radius)
        rows.append(
            {
                "dataset": dataset_name,
                "method": config["method"],
                "np_count": config["np_count"],
                "half_width": config["half_width"],
                "resize_shape": resize_shape,
                "ods": ods,
                "ois": ois,
                "ap": ap,
                "seconds_per_image": elapsed / len(preds),
            }
        )
        if config["method"] == "WVF":
            print(
                f"  WVF np={config['np_count']:>3} resize={resize_shape[0]}x{resize_shape[1]} "
                f"ODS={ods:.4f} time={elapsed/len(preds):.2f}s/img"
            )
        else:
            print(
                f"  LF  np={config['np_count']:>3} m={config['half_width']:>2} "
                f"resize={resize_shape[0]}x{resize_shape[1]} "
                f"ODS={ods:.4f} time={elapsed/len(preds):.2f}s/img"
            )

    out_json = RESULTS_DIR / f"wvf_scale_sweep_{output_tag}.json"
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump({"rows": rows}, fh, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    colors = {"WVF": "#c28f2c", 3: "#7a0019", 7: "#1f77b4", 14: "#2a7f62"}
    markers = {(64, 64): "o", (96, 96): "s", (128, 128): "^"}

    for resize_shape in resize_shapes:
        subset = [r for r in rows if r["resize_shape"] == resize_shape and r["method"] == "WVF"]
        subset = sorted(subset, key=lambda r: r["np_count"])
        axes[0].plot(
            [r["np_count"] for r in subset],
            [r["ods"] for r in subset],
            marker=markers.get(tuple(resize_shape), "o"),
            linewidth=2,
            color=colors["WVF"],
            alpha=0.7,
            label=f"WVF {resize_shape[0]}x{resize_shape[1]}",
        )
        axes[1].plot(
            [r["np_count"] for r in subset],
            [r["seconds_per_image"] for r in subset],
            marker=markers.get(tuple(resize_shape), "o"),
            linewidth=2,
            color=colors["WVF"],
            alpha=0.7,
            label=f"WVF {resize_shape[0]}x{resize_shape[1]}",
        )

        for half_width in half_widths:
            subset = [
                r
                for r in rows
                if r["resize_shape"] == resize_shape and r["method"] == "LF" and r["half_width"] == half_width
            ]
            subset = sorted(subset, key=lambda r: r["np_count"])
            axes[0].plot(
                [r["np_count"] for r in subset],
                [r["ods"] for r in subset],
                marker=markers.get(tuple(resize_shape), "o"),
                linestyle="--",
                linewidth=1.6,
                color=colors.get(half_width, "#444444"),
                alpha=0.7,
                label=f"LF m={half_width} {resize_shape[0]}x{resize_shape[1]}",
            )
            axes[1].plot(
                [r["np_count"] for r in subset],
                [r["seconds_per_image"] for r in subset],
                marker=markers.get(tuple(resize_shape), "o"),
                linestyle="--",
                linewidth=1.6,
                color=colors.get(half_width, "#444444"),
                alpha=0.7,
                label=f"LF m={half_width} {resize_shape[0]}x{resize_shape[1]}",
            )

    axes[0].set_title(f"{dataset_name}: ODS vs Np")
    axes[0].set_xlabel("Np")
    axes[0].set_ylabel("ODS")
    axes[0].grid(alpha=0.25)

    axes[1].set_title(f"{dataset_name}: Runtime vs Np")
    axes[1].set_xlabel("Np")
    axes[1].set_ylabel("Seconds per image")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.25)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles[: min(len(handles), 10)], labels[: min(len(labels), 10)],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04), fontsize=8)
    fig.suptitle(f"WVF/LF Scale Sweep Toward Paper Settings ({dataset_name})", fontsize=13, fontweight="bold")
    fig.savefig(RESULTS_DIR / f"wvf_scale_sweep_{output_tag}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {out_json}")


if __name__ == "__main__":
    main()
