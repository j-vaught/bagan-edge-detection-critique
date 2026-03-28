#!/usr/bin/env python3
"""
Sweep WVF angle-estimation settings toward the configurations described in the papers.

Outputs:
  - JSON summary of per-configuration errors
  - Plots of spline error vs Np and runtime vs Np
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmarks" / "angle_sweep_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))

from baselines import sobel_gradients
from synthetic import create_parallel_line_image
from wvf_lf import cubic_spline_angle
from analyze_results import (
    angular_error_deg,
    sample_parallel_line_points,
    wvf_orientation_profile,
)


def parse_int_list(name: str, default: str) -> list[int]:
    return [int(x.strip()) for x in os.environ.get(name, default).split(",") if x.strip()]


def parse_float_list(name: str, default: str) -> list[float]:
    return [float(x.strip()) for x in os.environ.get(name, default).split(",") if x.strip()]


def resolve_worker_count(task_count: int) -> int:
    raw = (
        os.environ.get("ANGLE_SWEEP_WORKERS")
        or os.environ.get("SLURM_CPUS_PER_TASK")
        or os.environ.get("OMP_NUM_THREADS")
        or str(os.cpu_count() or 1)
    )
    try:
        workers = int(raw)
    except ValueError:
        workers = os.cpu_count() or 1
    return max(1, min(workers, task_count))


def run_angle_task(task: dict) -> dict:
    np_count = task["np_count"]
    n_orientations = task["n_orientations"]
    snr_levels = task["snr_levels"]
    test_angles = task["test_angles"]
    size = task["size"]
    order = task["order"]
    sample_points = task["sample_points"]

    rows = []
    start = time.perf_counter()

    for snr in snr_levels:
        sobel_errors = []
        spline_errors = []

        for angle_deg in test_angles:
            img, _, true_angle = create_parallel_line_image(
                size=size, n_lines=1, spacing=30, angle_deg=angle_deg, snr=snr
            )
            _, _, sobel_mag, sobel_angle = sobel_gradients(img)
            points = sample_parallel_line_points(
                size=size, angle_deg=angle_deg, n_points=sample_points
            )

            point_sobel_errors = []
            point_spline_errors = []
            for px, py in points:
                y0 = max(py - 1, 0)
                y1 = min(py + 2, size)
                x0 = max(px - 1, 0)
                x1 = min(px + 2, size)
                local_mag = sobel_mag[y0:y1, x0:x1]
                local_idx = np.argmax(local_mag)
                ly, lx = np.unravel_index(local_idx, local_mag.shape)
                best_x = x0 + lx
                best_y = y0 + ly

                sobel_pred_deg = np.degrees(sobel_angle[best_y, best_x])
                point_sobel_errors.append(angular_error_deg(sobel_pred_deg, true_angle))

                angles_rad, profile = wvf_orientation_profile(
                    img,
                    best_x,
                    best_y,
                    np_count=np_count,
                    order=order,
                    n_orientations=n_orientations,
                )
                spline_angle, _, _ = cubic_spline_angle(profile, angles_rad)
                spline_pred_deg = np.degrees(spline_angle)
                point_spline_errors.append(angular_error_deg(spline_pred_deg, true_angle))

            sobel_errors.append(float(np.median(point_sobel_errors)))
            spline_errors.append(float(np.median(point_spline_errors)))

        rows.append(
            {
                "np_count": np_count,
                "n_orientations": n_orientations,
                "snr": snr,
                "sobel_mean_error": float(np.mean(sobel_errors)),
                "spline_mean_error": float(np.mean(spline_errors)),
                "improvement_deg": float(np.mean(sobel_errors) - np.mean(spline_errors)),
            }
        )

    elapsed = time.perf_counter() - start
    for row in rows:
        row["elapsed_s"] = elapsed

    return {
        "np_count": np_count,
        "n_orientations": n_orientations,
        "elapsed_s": elapsed,
        "rows": rows,
    }


def main() -> None:
    np.random.seed(int(os.environ.get("ANGLE_SWEEP_SEED", 1234)))
    np_counts = parse_int_list("ANGLE_SWEEP_NP_COUNTS", "15,50,100,150")
    n_orientations_list = parse_int_list("ANGLE_SWEEP_ORIENTATIONS", "18,36")
    snr_levels = parse_float_list("ANGLE_SWEEP_SNRS", "2.0,1.0,0.75")
    test_angles = parse_float_list("ANGLE_SWEEP_TEST_ANGLES", "0,23,63.5,90,135,174")
    size = int(os.environ.get("ANGLE_SWEEP_IMAGE_SIZE", 128))
    order = int(os.environ.get("ANGLE_SWEEP_ORDER", 4))
    sample_points = int(os.environ.get("ANGLE_SWEEP_SAMPLE_POINTS", 7))
    output_tag = os.environ.get("ANGLE_SWEEP_TAG", "paper")

    rows = []

    print("ANGLE REPRODUCTION SWEEP")
    print(f"  np_counts={np_counts}")
    print(f"  orientations={n_orientations_list}")
    print(f"  snr_levels={snr_levels}")
    print(f"  size={size}, order={order}, sample_points={sample_points}")

    tasks = [
        {
            "np_count": np_count,
            "n_orientations": n_orientations,
            "snr_levels": snr_levels,
            "test_angles": test_angles,
            "size": size,
            "order": order,
            "sample_points": sample_points,
        }
        for n_orientations in n_orientations_list
        for np_count in np_counts
    ]
    workers = resolve_worker_count(len(tasks))
    print(f"  Dispatching {len(tasks)} angle tasks across {workers} worker processes")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_angle_task, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            rows.extend(result["rows"])
            print(
                f"  np={result['np_count']:>3} orient={result['n_orientations']:>2} "
                f"done in {result['elapsed_s']:.1f}s"
            )

    out_json = RESULTS_DIR / f"angle_sweep_{output_tag}.json"
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump({"rows": rows}, fh, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    color_map = {18: "#7a0019", 36: "#1f77b4", 72: "#2a7f62"}

    for n_orientations in sorted(set(r["n_orientations"] for r in rows)):
        for snr in snr_levels:
            subset = [r for r in rows if r["n_orientations"] == n_orientations and r["snr"] == snr]
            subset = sorted(subset, key=lambda r: r["np_count"])
            spline_alpha = min(0.95, 0.35 + 0.18 * snr_levels.index(snr))
            sobel_alpha = min(0.8, 0.2 + 0.15 * snr_levels.index(snr))
            axes[0].plot(
                [r["np_count"] for r in subset],
                [r["spline_mean_error"] for r in subset],
                marker="o",
                linewidth=2,
                color=color_map.get(n_orientations, "#444444"),
                alpha=spline_alpha,
                label=f"Spline {n_orientations} orient, SNR={snr:g}",
            )
            axes[0].plot(
                [r["np_count"] for r in subset],
                [r["sobel_mean_error"] for r in subset],
                marker="x",
                linestyle="--",
                linewidth=1,
                color=color_map.get(n_orientations, "#444444"),
                alpha=sobel_alpha,
            )

    seen = set()
    for r in sorted(rows, key=lambda row: (row["n_orientations"], row["np_count"])):
        key = (r["n_orientations"], r["np_count"])
        if key in seen:
            continue
        seen.add(key)
        axes[1].scatter(
            r["np_count"],
            r["elapsed_s"],
            s=70,
            color=color_map.get(r["n_orientations"], "#444444"),
            marker="o" if r["n_orientations"] == min(n_orientations_list) else "^",
        )
        axes[1].annotate(
            f"{r['n_orientations']} orient",
            (r["np_count"], r["elapsed_s"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
        )

    axes[0].set_title("Mean Angular Error vs Np")
    axes[0].set_xlabel("Np")
    axes[0].set_ylabel("Mean median angular error (deg)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)

    axes[1].set_title("Sweep Runtime vs Np")
    axes[1].set_xlabel("Np")
    axes[1].set_ylabel("Elapsed seconds per configuration")
    axes[1].grid(alpha=0.25)

    fig.suptitle("Angle Reproduction Sweep Toward Paper Settings", fontsize=13, fontweight="bold")
    fig.savefig(RESULTS_DIR / f"angle_sweep_{output_tag}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {out_json}")


if __name__ == "__main__":
    main()
