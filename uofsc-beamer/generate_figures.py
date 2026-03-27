"""
Generate all figures for the critique presentation slides.
Run on SLURM with GPU access for ML model comparisons.
"""

import sys
import time
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# UofSC brand colors
GARNET = '#73000a'
BLACK = '#000000'
GOLD = '#d4a017'
GRAY = '#666666'
BLUE = '#2b6cb0'
GREEN = '#2d8659'

from baselines import sobel_gradients, prewitt_gradients
from synthetic import create_multi_angle_line_image
from scipy import ndimage


# ============================================================
# Figure 1: BSDS500 ODS Bar Chart (main result)
# ============================================================
def fig_bsds500_ods():
    methods = ['Sobel\n15x15', 'Sobel\n9x9', 'DexiNed', 'Sobel\n7x7',
               'TEED', 'Sobel\n5x5', 'Prewitt\n3x3', 'Sobel\n3x3',
               'LoG', 'Canny\nσ=2', 'Canny\nσ=1', 'WVF\nNp=15']
    ods = [0.549, 0.516, 0.495, 0.493, 0.470, 0.464, 0.453, 0.449,
           0.382, 0.303, 0.201, 0.000]
    colors = [GARNET, GARNET, BLUE, GARNET, BLUE, GARNET, GARNET,
              GARNET, GRAY, GRAY, GRAY, GOLD]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(methods)), ods, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel('ODS F-Score', fontsize=12, fontweight='bold')
    ax.set_title('BSDS500 Edge Detection Benchmark (50 test images, A100 GPU)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 0.65)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add value labels
    for bar, val in zip(bars, ods):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Legend
    patches = [mpatches.Patch(color=GARNET, label='Traditional (Sobel/Prewitt)'),
               mpatches.Patch(color=BLUE, label='ML (DexiNed/TEED)'),
               mpatches.Patch(color=GOLD, label='WVF (Bagan)'),
               mpatches.Patch(color=GRAY, label='Other Traditional')]
    ax.legend(handles=patches, loc='upper right', fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'bsds500_ods.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  fig_bsds500_ods done")


# ============================================================
# Figure 2: Runtime comparison (log scale)
# ============================================================
def fig_runtime():
    methods = ['Sobel 3x3', 'Prewitt 3x3', 'Sobel 7x7', 'Sobel 15x15',
               'Canny', 'LoG', 'TEED', 'DexiNed', 'WVF Np=15\n(32x32 only)']
    times_ms = [14, 11, 8, 12, 17, 225, 74, 341, 9057]
    colors = [GARNET]*4 + [GRAY]*2 + [BLUE]*2 + [GOLD]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    y_pos = range(len(methods))
    bars = ax.barh(y_pos, times_ms, color=colors, edgecolor='white', height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=10)
    ax.set_xlabel('Time per Image (ms, log scale)', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_title('Runtime Comparison on BSDS500 (A100 GPU)', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, val in zip(bars, times_ms):
        ax.text(val * 1.1, bar.get_y() + bar.get_height()/2,
                f'{val:,}ms', va='center', fontsize=9, fontweight='bold')

    # Annotations
    ax.axvline(x=33, color=GREEN, linestyle='--', alpha=0.7, linewidth=1.5)
    ax.text(33, len(methods)-0.5, '30fps\nrealtime', ha='center',
            fontsize=8, color=GREEN, fontweight='bold')

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'runtime_comparison.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  fig_runtime done")


# ============================================================
# Figure 3: Sobel kernel size scaling
# ============================================================
def fig_kernel_scaling():
    bsds_ods = {'3': 0.449, '5': 0.464, '7': 0.493, '9': 0.516, '15': 0.549}
    uded_ods = {'3': 0.883, '5': 0.888, '7': 0.900, '9': 0.904, '15': 0.877}

    sizes = [3, 5, 7, 9, 15]
    bsds = [bsds_ods[str(s)] for s in sizes]
    uded = [uded_ods[str(s)] for s in sizes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(sizes, bsds, 'o-', color=GARNET, linewidth=2.5, markersize=10, label='Sobel')
    ax1.axhline(y=0.495, color=BLUE, linestyle='--', linewidth=1.5, label='DexiNed (0.495)')
    ax1.axhline(y=0.470, color=BLUE, linestyle=':', linewidth=1.5, label='TEED (0.470)')
    ax1.set_xlabel('Sobel Kernel Size', fontsize=11, fontweight='bold')
    ax1.set_ylabel('ODS F-Score', fontsize=11, fontweight='bold')
    ax1.set_title('BSDS500', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_xticks(sizes)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2.plot(sizes, uded, 'o-', color=GARNET, linewidth=2.5, markersize=10, label='Sobel')
    ax2.axhline(y=0.901, color=BLUE, linestyle='--', linewidth=1.5, label='DexiNed (0.901)')
    ax2.axhline(y=0.881, color=BLUE, linestyle=':', linewidth=1.5, label='TEED (0.881)')
    ax2.set_xlabel('Sobel Kernel Size', fontsize=11, fontweight='bold')
    ax2.set_ylabel('ODS F-Score', fontsize=11, fontweight='bold')
    ax2.set_title('UDED', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xticks(sizes)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle('Sobel ODS vs Kernel Size — ML Models as Horizontal Baselines',
                 fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'kernel_scaling.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  fig_kernel_scaling done")


# ============================================================
# Figure 4: Condition number analysis
# ============================================================
def fig_condition_numbers():
    from wvf_lf import analyze_condition_numbers
    np_counts = [15, 25, 50, 100, 150, 250]
    results = analyze_condition_numbers(np_counts, order=4, n_orientations=72)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: condition vs Np
    means = [results[n].mean() for n in np_counts]
    maxes = [results[n].max() for n in np_counts]
    ax1.semilogy(np_counts, means, 'o-', color=GARNET, linewidth=2, markersize=8, label='Mean')
    ax1.semilogy(np_counts, maxes, 's--', color=BLUE, linewidth=2, markersize=8, label='Max')
    ax1.fill_between(np_counts, means, maxes, alpha=0.15, color=GARNET)
    ax1.set_xlabel('$N_p$ (neighbor pixels)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Condition Number of $A^TA$', fontsize=11, fontweight='bold')
    ax1.set_title('System Conditioning vs $N_p$', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.axhline(y=1e5, color='red', linestyle=':', alpha=0.5)
    ax1.text(200, 1.3e5, 'Np=250 used\nin papers', fontsize=8, color='red', ha='center')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right: condition vs orientation at Np=250
    angles = np.linspace(0, 360, 72)
    ax2.plot(angles, results[250], color=GARNET, linewidth=1.5)
    ax2.set_xlabel('Orientation θ (degrees)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Condition Number', fontsize=11, fontweight='bold')
    ax2.set_title('Conditioning at $N_p=250$ vs Orientation', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'condition_numbers.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  fig_condition_numbers done")


# ============================================================
# Figure 5: Taylor verification
# ============================================================
def fig_taylor_verification():
    from wvf_lf import get_circular_neighbors, build_taylor_matrix

    def known_fn(x, y):
        return 3.0 + 2.0*x - 1.5*y + 0.5*x**2 + 0.3*y**2 - 0.2*x*y

    names = ['$f(0,0)$', '$f_x$', '$f_y$', '$f_{xx}$', '$f_{yy}$', '$f_{xy}$']
    true_vals = [3.0, 2.0, -1.5, 1.0, 0.6, -0.2]

    coords = get_circular_neighbors(25)
    values = np.array([known_fn(c[0], c[1]) for c in coords])
    A = build_taylor_matrix(coords, order=4)
    z = np.linalg.lstsq(A, values, rcond=None)[0]
    recovered = [z[0], z[1], z[2], z[3], z[4], z[5]]
    errors = [abs(r - t) for r, t in zip(recovered, true_vals)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    x = range(len(names))
    ax1.bar([i-0.15 for i in x], true_vals, 0.3, label='True', color=GARNET, alpha=0.8)
    ax1.bar([i+0.15 for i in x], recovered, 0.3, label='WVF Recovered', color=BLUE, alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=10)
    ax1.set_ylabel('Value', fontsize=11, fontweight='bold')
    ax1.set_title('Derivative Recovery', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2.bar(x, errors, color=GREEN, alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=10)
    ax2.set_ylabel('Absolute Error', fontsize=11, fontweight='bold')
    ax2.set_title('Recovery Error (machine precision)', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.set_ylim(1e-17, 1e-13)
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle('Taylor Expansion Verification: Known Polynomial $f(x,y) = 3 + 2x - 1.5y + 0.5x^2 + 0.3y^2 - 0.2xy$',
                 fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'taylor_verification.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  fig_taylor_verification done")


# ============================================================
# Figure 6: SNR robustness - visual edge comparison
# ============================================================
def fig_snr_comparison():
    snr_levels = [2.0, 1.0, 0.75, 0.5]
    fig, axes = plt.subplots(4, 5, figsize=(13, 10))
    col_titles = ['Original', 'Sobel 3×3', 'Sobel 15×15', 'Canny σ=2', 'LoG σ=2']

    for si, snr in enumerate(snr_levels):
        img, clean, _ = create_multi_angle_line_image(size=256, snr=snr)
        gray = img

        _, _, sobel3_mag, _ = sobel_gradients(gray)
        sigma15 = (15 - 1) / 4.0
        gx = ndimage.gaussian_filter1d(gray.astype(np.float64), sigma15, axis=1, order=1)
        gy = ndimage.gaussian_filter1d(gray.astype(np.float64), sigma15, axis=0, order=1)
        sobel15_mag = np.sqrt(gx**2 + gy**2)

        from skimage.feature import canny
        canny_edges = canny(gray / 255.0, sigma=2.0).astype(np.float64)

        log_filtered = ndimage.gaussian_laplace(gray.astype(np.float64), 2.0)
        log_edges = np.abs(log_filtered)

        panels = [gray, sobel3_mag, sobel15_mag, canny_edges, log_edges]
        for ci, panel in enumerate(panels):
            axes[si, ci].imshow(panel, cmap='gray')
            axes[si, ci].axis('off')
            if si == 0:
                axes[si, ci].set_title(col_titles[ci], fontsize=10, fontweight='bold')
        axes[si, 0].text(-10, 128, f'SNR={snr}', fontsize=11, fontweight='bold',
                          rotation=90, va='center', ha='right', color=GARNET)

    fig.suptitle('Edge Detection at Varying SNR Levels (Synthetic Multi-Line Image)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'snr_comparison.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  fig_snr_comparison done")


# ============================================================
# Figure 7: Angle detection error
# ============================================================
def fig_angle_errors():
    true_angles = [0.0, 45.0, 84.0, 90.0, 113.0, 153.5]
    sobel_pred = [52.67, 47.76, 86.67, 88.68, 110.10, 148.04]
    errors = [min(abs(p - t), abs(p - t + 180), abs(p - t - 180))
              for p, t in zip(sobel_pred, true_angles)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    x = range(len(true_angles))
    ax1.bar([i-0.15 for i in x], true_angles, 0.3, label='True Angle', color=GARNET, alpha=0.8)
    ax1.bar([i+0.15 for i in x], sobel_pred, 0.3, label='Sobel Arctan', color=BLUE, alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{a}°' for a in true_angles], fontsize=9)
    ax1.set_ylabel('Angle (degrees)', fontsize=11, fontweight='bold')
    ax1.set_title('Predicted vs True Edge Angle', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    bar_colors = [GARNET if e > 5 else GREEN for e in errors]
    ax2.bar(x, errors, color=bar_colors, alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{a}°' for a in true_angles], fontsize=9)
    ax2.set_ylabel('Angular Error (degrees)', fontsize=11, fontweight='bold')
    ax2.set_title('Sobel Arctan Error (< 1° claimed by WVF)', fontsize=12, fontweight='bold')
    ax2.axhline(y=1.0, color=GREEN, linestyle='--', linewidth=2, label='1° target')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'angle_errors.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  fig_angle_errors done")


# ============================================================
# Figure 8: Maritime scene results
# ============================================================
def fig_maritime_heatmap():
    scenes = ['Horizon', 'Cable', 'Waves', 'Underexp.', 'Dark+Noise', 'Low-SNR']
    methods = ['Sobel 3×3', 'Sobel 5×5', 'Sobel 7×7', 'Sobel 9×9',
               'Sobel 15×15', 'Prewitt', 'Canny σ=1', 'Canny σ=2']
    data = np.array([
        [0.070, 0.124, 0.230, 0.158, 0.060, 0.215],
        [0.061, 0.120, 0.237, 0.118, 0.061, 0.214],
        [0.087, 0.141, 0.283, 0.230, 0.066, 0.222],
        [0.116, 0.172, 0.353, 0.380, 0.069, 0.233],
        [0.189, 0.227, 0.488, 0.771, 0.081, 0.283],
        [0.077, 0.126, 0.236, 0.187, 0.061, 0.216],
        [0.058, 0.055, 0.327, 0.894, 0.058, 0.209],
        [0.119, 0.073, 0.000, 0.000, 0.074, 0.210],
    ])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=0.9)
    ax.set_xticks(range(len(scenes)))
    ax.set_xticklabels(scenes, fontsize=10, fontweight='bold')
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=10)
    ax.set_title('Maritime Scene F-Scores (Replicating Bagan Test Conditions)',
                 fontsize=13, fontweight='bold')

    for i in range(len(methods)):
        for j in range(len(scenes)):
            val = data[i, j]
            color = 'white' if val > 0.5 or val < 0.1 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=9, fontweight='bold', color=color)

    fig.colorbar(im, ax=ax, label='F-Score', shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'maritime_heatmap.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  fig_maritime_heatmap done")


# ============================================================
# Figure 9: UDED comparison bars
# ============================================================
def fig_uded_comparison():
    methods = ['Sobel\n9x9', 'DexiNed', 'Sobel\n7x7', 'Sobel\n5x5',
               'Prewitt', 'Sobel\n3x3', 'TEED', 'Sobel\n15x15']
    ods = [0.904, 0.901, 0.900, 0.888, 0.885, 0.883, 0.881, 0.877]
    ois = [0.915, 0.910, 0.911, 0.898, 0.897, 0.895, 0.890, 0.894]
    colors = [GARNET, BLUE, GARNET, GARNET, GARNET, GARNET, BLUE, GARNET]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(methods))
    w = 0.35
    bars1 = ax.bar(x - w/2, ods, w, label='ODS', color=colors, alpha=0.9, edgecolor='white')
    bars2 = ax.bar(x + w/2, ois, w, label='OIS', color=colors, alpha=0.5, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel('F-Score', fontsize=12, fontweight='bold')
    ax.set_title('UDED Benchmark — Traditional Filters Match ML Models', fontsize=13, fontweight='bold')
    ax.set_ylim(0.85, 0.93)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'uded_comparison.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  fig_uded_comparison done")


# ============================================================
# Figure 10: Visual edge map comparison on synthetic maritime
# ============================================================
def fig_maritime_visual():
    from synthetic import create_multi_angle_line_image

    # Generate horizon scene
    h, w = 300, 500
    img = np.zeros((h, w), dtype=np.float64)
    hy = int(h * 0.35)
    img[:hy, :] = 160 + np.random.normal(0, 5, (hy, w))
    y, x = np.mgrid[0:h-hy, 0:w].astype(np.float64)
    water = 100 + 12*np.sin(0.05*x + 0.3*y) + 6*np.sin(0.12*x - 0.1*y)
    water += np.random.normal(0, 20, (h-hy, w))
    img[hy:, :] = water
    # boat
    img[hy+30:hy+40, 300:380] = 50
    img[hy+10:hy+30, 338:342] = 50
    # buoy
    for dy in range(-6, 7):
        for dx in range(-6, 7):
            if dx**2 + dy**2 <= 36:
                img[hy+60+dy, 150+dx] = 40

    # Run methods
    _, _, sobel3, _ = sobel_gradients(img)
    sigma15 = 3.5
    gx15 = ndimage.gaussian_filter1d(img, sigma15, axis=1, order=1)
    gy15 = ndimage.gaussian_filter1d(img, sigma15, axis=0, order=1)
    sobel15 = np.sqrt(gx15**2 + gy15**2)
    from skimage.feature import canny
    canny_e = canny(img / 255.0, sigma=2.0).astype(np.float64)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for ax, data, title in zip(axes,
        [img, sobel3/sobel3.max(), sobel15/sobel15.max(), canny_e],
        ['Original Scene', 'Sobel 3×3 Edges', 'Sobel 15×15 Edges', 'Canny σ=2 Edges']):
        ax.imshow(data, cmap='gray')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')

    fig.suptitle('Synthetic Maritime Scene — Edge Detection Comparison',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'maritime_visual.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  fig_maritime_visual done")


# ============================================================
# Figure 11: Compute fairness diagram
# ============================================================
def fig_compute_fairness():
    fig, ax = plt.subplots(figsize=(10, 5))

    methods = ['Sobel 3×3', 'Sobel 15×15', 'WVF Np=250', 'DexiNed', 'TEED']
    pixels_used = [9, 177, 250, 'N/A\n(learned)', 'N/A\n(learned)']
    ods_bsds = [0.449, 0.549, None, 0.495, 0.470]
    time_ms = [14, 12, 9057, 341, 74]

    # Scatter: pixels vs ODS
    for i, (name, px, ods, t) in enumerate(zip(methods, pixels_used, ods_bsds, time_ms)):
        if isinstance(px, (int, float)) and ods is not None:
            color = GARNET if 'Sobel' in name else (GOLD if 'WVF' in name else BLUE)
            size = np.log10(t) * 100 + 50
            ax.scatter(px, ods, s=size, c=color, alpha=0.8, edgecolors='black', zorder=5)
            offset_y = 0.015 if 'WVF' not in name else -0.02
            ax.annotate(f'{name}\n({t}ms)', (px, ods + offset_y),
                        fontsize=9, ha='center', fontweight='bold')

    # ML models (no pixel count, shown on right)
    for i, (name, ods, t) in enumerate([(methods[3], ods_bsds[3], time_ms[3]),
                                         (methods[4], ods_bsds[4], time_ms[4])]):
        ax.scatter(300 + i*30, ods, s=np.log10(t)*100+50, c=BLUE, alpha=0.8,
                   edgecolors='black', zorder=5)
        ax.annotate(f'{name}\n({t}ms)', (300 + i*30, ods + 0.015),
                    fontsize=9, ha='center', fontweight='bold')

    ax.set_xlabel('Pixels Used per Gradient Estimate', fontsize=12, fontweight='bold')
    ax.set_ylabel('ODS F-Score (BSDS500)', fontsize=12, fontweight='bold')
    ax.set_title('Compute Fairness: Data Utilization vs Performance\n(bubble size ∝ log runtime)',
                 fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'compute_fairness.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  fig_compute_fairness done")


# ============================================================
# Figure 12: Summary scorecard
# ============================================================
def fig_summary_scorecard():
    categories = ['Math\nCorrectness', 'Numerical\nStability', 'Experimental\nFairness',
                  'Runtime\nAnalysis', 'Statistical\nValidity', 'Claims vs\nEvidence',
                  'Angular\nAccuracy', 'Low-SNR\nPerformance']
    scores = [10, 7, 3, 1, 2, 3, 9, 8]  # out of 10
    colors_sc = [GREEN if s >= 7 else (GOLD if s >= 4 else GARNET) for s in scores]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.barh(range(len(categories)), scores, color=colors_sc, edgecolor='white',
                   height=0.7, alpha=0.85)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=10, fontweight='bold')
    ax.set_xlabel('Score (out of 10)', fontsize=12, fontweight='bold')
    ax.set_title('WVF/LF Publication Quality Scorecard', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 11)
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, val in zip(bars, scores):
        ax.text(val + 0.2, bar.get_y() + bar.get_height()/2,
                f'{val}/10', va='center', fontsize=11, fontweight='bold')

    patches = [mpatches.Patch(color=GREEN, label='Strong (7-10)'),
               mpatches.Patch(color=GOLD, label='Moderate (4-6)'),
               mpatches.Patch(color=GARNET, label='Weak (1-3)')]
    ax.legend(handles=patches, loc='lower right', fontsize=10)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'summary_scorecard.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  fig_summary_scorecard done")


# ============================================================
def main():
    print("Generating presentation figures...")
    fig_bsds500_ods()
    fig_runtime()
    fig_kernel_scaling()
    fig_condition_numbers()
    fig_taylor_verification()
    fig_snr_comparison()
    fig_angle_errors()
    fig_maritime_heatmap()
    fig_uded_comparison()
    fig_maritime_visual()
    fig_compute_fairness()
    fig_summary_scorecard()
    print(f"\nAll figures saved to {FIG_DIR}/")
    print("Files:")
    for f in sorted(FIG_DIR.glob("*.pdf")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
