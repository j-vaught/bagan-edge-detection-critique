#!/usr/bin/env python3
"""
Generate paper-ready figures from the finished benchmark artifacts.
Run through SLURM; do not invoke manually.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BSDS_JSON = ROOT / "benchmarks" / "results" / "benchmark_results.json"
UDED_JSON = ROOT / "benchmarks" / "results" / "benchmark_uded_results.json"
MARITIME_JSON = ROOT / "benchmarks" / "maritime_results" / "maritime_results.json"
MARITIME_DIR = ROOT / "benchmarks" / "maritime_results"
RESULTS_DIR = ROOT / "results"

COLORS = {
    "traditional": "#7a0019",
    "ml": "#1f77b4",
    "wvf_lf": "#c28f2c",
}

TYPE_LABELS = {
    "traditional": "Traditional",
    "ml": "ML",
    "wvf_lf": "WVF/LF subset",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def shorten(name: str) -> str:
    return (
        name.replace("Sobel-", "Sobel ")
        .replace("Prewitt-", "Prewitt ")
        .replace("Canny-", "Canny ")
        .replace("LoG-", "LoG ")
        .replace("LF-m3-Np15", "LF m=3, Np=15")
        .replace("WVF-Np15", "WVF Np=15")
    )


def method_style(entry: dict) -> tuple[str, str | None, float]:
    color = COLORS.get(entry.get("type", "traditional"), "#666666")
    hatch = "///" if entry.get("type") == "wvf_lf" else None
    alpha = 0.9 if entry.get("type") != "wvf_lf" else 0.65
    return color, hatch, alpha


def plot_quality_panel(ax, data: dict, title: str, xlim: tuple[float, float]) -> None:
    items = sorted(data["results"].items(), key=lambda kv: kv[1]["ODS"], reverse=True)
    names = [shorten(name) for name, _ in items]
    y = np.arange(len(items))
    ods = [entry["ODS"] for _, entry in items]
    ois = [entry["OIS"] for _, entry in items]

    for idx, (_, entry) in enumerate(items):
        color, hatch, alpha = method_style(entry)
        bar = ax.barh(
            idx,
            entry["ODS"],
            color=color,
            alpha=alpha,
            edgecolor="white",
            linewidth=0.7,
        )
        if hatch:
            bar[0].set_hatch(hatch)

    ax.scatter(ois, y, color="black", s=18, zorder=4, label="OIS")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.set_xlabel("ODS bar, OIS dot", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for idx, val in enumerate(ods):
        ax.text(val + 0.007, idx, f"{val:.3f}", va="center", fontsize=7)


def fig_dataset_quality() -> None:
    bsds = load_json(BSDS_JSON)
    uded = load_json(UDED_JSON)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), constrained_layout=True)
    plot_quality_panel(axes[0], bsds, "BSDS500", (0.0, 0.62))
    plot_quality_panel(axes[1], uded, "UDED", (0.0, 0.98))

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS["traditional"], alpha=0.9, label=TYPE_LABELS["traditional"]),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["ml"], alpha=0.9, label=TYPE_LABELS["ml"]),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["wvf_lf"], alpha=0.65, hatch="///", label=TYPE_LABELS["wvf_lf"]),
        plt.Line2D([0], [0], marker="o", color="black", linestyle="None", markersize=5, label="OIS"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.suptitle("Dataset Quality Comparison Across All Reproduced Methods", fontsize=13, fontweight="bold")
    fig.savefig(FIG_DIR / "dataset_quality_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff_panel(ax, data: dict, title: str) -> None:
    items = sorted(data["results"].items(), key=lambda kv: kv[1]["ODS"], reverse=True)
    for name, entry in items:
        color, _, _ = method_style(entry)
        x = entry["avg_time_s"] * 1000.0
        y = entry["ODS"]
        marker = "s" if entry.get("type") == "wvf_lf" else ("o" if entry.get("type") == "traditional" else "^")
        facecolors = "none" if entry.get("type") == "wvf_lf" else color
        ax.scatter(x, y, s=70, marker=marker, c=facecolors, edgecolors=color, linewidths=1.4, zorder=3)
        ax.annotate(shorten(name), (x, y), textcoords="offset points", xytext=(4, 3), fontsize=7)

    ax.set_xscale("log")
    ax.grid(alpha=0.2, which="both")
    ax.set_xlabel("ms per image (log scale)", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fig_runtime_tradeoff() -> None:
    bsds = load_json(BSDS_JSON)
    uded = load_json(UDED_JSON)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), constrained_layout=True)
    plot_tradeoff_panel(axes[0], bsds, "BSDS500: Quality vs Runtime")
    plot_tradeoff_panel(axes[1], uded, "UDED: Quality vs Runtime")
    axes[0].set_ylabel("ODS", fontsize=9)

    for ax in axes:
        ax.axvline(33.3, color="#2a7f62", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.text(33.3, ax.get_ylim()[0] + 0.03, "33 ms", rotation=90, va="bottom", ha="right", fontsize=7, color="#2a7f62")

    handles = [
        plt.Line2D([0], [0], marker="o", color=COLORS["traditional"], linestyle="None", markersize=6, label=TYPE_LABELS["traditional"]),
        plt.Line2D([0], [0], marker="^", color=COLORS["ml"], linestyle="None", markersize=6, label=TYPE_LABELS["ml"]),
        plt.Line2D([0], [0], marker="s", markerfacecolor="none", markeredgecolor=COLORS["wvf_lf"], linestyle="None", markersize=6, label=TYPE_LABELS["wvf_lf"]),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.suptitle("Runtime/Quality Tradeoff Makes the Fairness Problem Visual", fontsize=13, fontweight="bold")
    fig.savefig(FIG_DIR / "runtime_tradeoff.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_maritime_summary() -> None:
    maritime = load_json(MARITIME_JSON)
    overall = maritime["overall"]
    scenes = list(maritime["by_scene"].keys())
    methods = sorted(overall.keys(), key=lambda method: overall[method]["mean"], reverse=True)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5.8),
        gridspec_kw={"width_ratios": [1.05, 1.4]},
        constrained_layout=True,
    )

    means = [overall[m]["mean"] for m in methods]
    cis = [overall[m]["ci95"] for m in methods]
    colors = [COLORS["traditional"] if "Sobel" in m or "Prewitt" in m or "Canny" in m else COLORS["ml"] for m in methods]
    y = np.arange(len(methods))
    axes[0].barh(y, means, xerr=cis, color=colors, alpha=0.9, edgecolor="white", linewidth=0.7)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([shorten(m) for m in methods], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Mean F-score with 95% CI", fontsize=9)
    axes[0].set_title("Overall Maritime Performance", fontsize=11, fontweight="bold")
    axes[0].grid(axis="x", alpha=0.2)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    heat = np.array([[maritime["by_scene"][scene][method]["mean"] for method in methods] for scene in scenes])
    im = axes[1].imshow(heat, aspect="auto", cmap="magma")
    axes[1].set_xticks(np.arange(len(methods)))
    axes[1].set_xticklabels([shorten(m) for m in methods], rotation=35, ha="right", fontsize=8)
    axes[1].set_yticks(np.arange(len(scenes)))
    axes[1].set_yticklabels(
        [
            "Horizon+Buoys",
            "Cable",
            "Wave Field",
            "Underexposed",
            "Dark Horizon",
            "Low-SNR Waves",
        ],
        fontsize=8,
    )
    axes[1].set_title("Per-Scene Mean F-score Heatmap", fontsize=11, fontweight="bold")
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Mean F-score", rotation=90, fontsize=8)

    for row in range(heat.shape[0]):
        for col in range(heat.shape[1]):
            val = heat[row, col]
            text_color = "white" if val < 0.45 else "black"
            axes[1].text(col, row, f"{val:.2f}", ha="center", va="center", fontsize=7, color=text_color)

    fig.suptitle("Repeated-Trial Maritime Comparison", fontsize=13, fontweight="bold")
    fig.savefig(FIG_DIR / "maritime_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_maritime_montage() -> None:
    files = [
        ("Horizon + Buoys", MARITIME_DIR / "Horizon_with_Boat_and_Buoys.png"),
        ("Cable in Water", MARITIME_DIR / "Cable_in_Water.png"),
        ("Wave Field", MARITIME_DIR / "Wave_Field.png"),
        ("Underexposed", MARITIME_DIR / "Underexposed_Marine_Scene.png"),
        ("Dark Horizon", MARITIME_DIR / "Dark_Horizon_(high_noise).png"),
        ("Low-SNR Waves", MARITIME_DIR / "Very_Low_SNR_Waves.png"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), constrained_layout=True)
    for ax, (title, path) in zip(axes.flat, files):
        ax.imshow(mpimg.imread(path))
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    fig.suptitle("Representative Synthetic Maritime Scenes Used in the Replication", fontsize=13, fontweight="bold")
    fig.savefig(FIG_DIR / "maritime_scene_montage.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_angle_combo() -> None:
    images = RESULTS_DIR / "angle_test_images.png"
    errors = RESULTS_DIR / "angle_error_comparison.png"

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    axes[0].imshow(mpimg.imread(images))
    axes[0].set_title("Synthetic Angle Test Inputs", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(mpimg.imread(errors))
    axes[1].set_title("Measured Angle Error", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    fig.suptitle("Angle Section Visual Summary", fontsize=13, fontweight="bold")
    fig.savefig(FIG_DIR / "angle_visual_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig_dataset_quality()
    fig_runtime_tradeoff()
    fig_maritime_summary()
    fig_maritime_montage()
    fig_angle_combo()
    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
