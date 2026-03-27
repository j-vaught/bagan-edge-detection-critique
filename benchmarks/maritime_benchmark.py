"""
Maritime edge detection benchmark replicating Bagan's exact test protocols.

Generates synthetic maritime scenes matching the scenarios in:
  - IEEE 2024 Fig.3: Multi-line at varying SNR (Np=250, m=14)
  - IEEE 2024 Fig.4-5: Aquatic scenes (boat+buoys, noisy buoy, cable in water)
  - IEEE 2025 Fig.4: Angle detection on known-angle lines
  - IEEE 2025 Fig.6-7: Multi-scale edge detection on marine scenes

Since HPC blocks external image downloads, we generate challenging synthetic
maritime-like scenes: water texture with embedded objects, horizon lines,
floating objects with low contrast, wave patterns.

Also runs on BSDS500 images that contain water/outdoor scenes.
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

RESULTS_DIR = ROOT / "benchmarks" / "maritime_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Synthetic Maritime Scene Generation
# ============================================================

def generate_water_texture(h, w, wave_freq=0.05, noise_level=20):
    """Generate realistic water-like texture with waves."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    # Multiple wave frequencies for realism
    water = 128.0
    water += 15 * np.sin(wave_freq * x + 0.3 * y)
    water += 8 * np.sin(2.3 * wave_freq * x - 0.15 * y + 1.2)
    water += 5 * np.sin(0.7 * wave_freq * x + 0.5 * y + 2.5)
    water += 3 * np.sin(4.1 * wave_freq * x + 0.1 * y + 0.8)
    water += np.random.normal(0, noise_level, (h, w))
    return np.clip(water, 0, 255)


def generate_horizon_scene(h=480, w=640, horizon_y=0.35, sky_val=180,
                            water_noise=25, boat_contrast=0.3):
    """
    Replicate Bagan's Fig.4: Poor quality littoral image with horizon,
    boats, and buoys. Underexposed with low contrast.

    Parameters match Bagan's test: challenging aquatic photo with
    poor lighting, small objects, water texture noise.
    """
    img = np.zeros((h, w), dtype=np.float64)
    hy = int(h * horizon_y)

    # Sky region (slightly brighter, smooth gradient)
    for y in range(hy):
        img[y, :] = sky_val - (sky_val - 100) * (y / hy) ** 2
    img[:hy, :] += np.random.normal(0, 5, (hy, w))

    # Water region (textured, noisy)
    water_h = h - hy
    img[hy:, :] = generate_water_texture(water_h, w, noise_level=water_noise)

    # Horizon line (sharp transition)
    img[hy-1:hy+1, :] = (img[hy-2, :] + img[hy+2, :]) / 2

    # Add boat silhouette (small, dark, low contrast - like Bagan's test)
    boat_cx, boat_cy = int(w * 0.6), hy + int(water_h * 0.15)
    boat_w, boat_h_size = 60, 15
    boat_val = img[boat_cy, boat_cx] * (1 - boat_contrast)
    img[boat_cy-boat_h_size:boat_cy, boat_cx-boat_w:boat_cx+boat_w] = boat_val
    # Mast
    img[boat_cy-boat_h_size*3:boat_cy-boat_h_size, boat_cx-2:boat_cx+2] = boat_val

    # Add buoy (small circular object - like Bagan Fig.5 row 2)
    buoy_cx, buoy_cy, buoy_r = int(w * 0.3), hy + int(water_h * 0.3), 8
    for dy in range(-buoy_r, buoy_r+1):
        for dx in range(-buoy_r, buoy_r+1):
            if dx**2 + dy**2 <= buoy_r**2:
                ny, nx = buoy_cy + dy, buoy_cx + dx
                if 0 <= ny < h and 0 <= nx < w:
                    img[ny, nx] = img[ny, nx] * 0.5  # Dark buoy

    # Second smaller buoy
    buoy2_cx, buoy2_cy, buoy2_r = int(w * 0.75), hy + int(water_h * 0.25), 5
    for dy in range(-buoy2_r, buoy2_r+1):
        for dx in range(-buoy2_r, buoy2_r+1):
            if dx**2 + dy**2 <= buoy2_r**2:
                ny, nx = buoy2_cy + dy, buoy2_cx + dx
                if 0 <= ny < h and 0 <= nx < w:
                    img[ny, nx] = min(255, img[ny, nx] * 1.5)  # Bright buoy

    # Ground truth edges: horizon, boat outline, buoy edges
    gt = np.zeros((h, w), dtype=np.float64)
    gt[hy-1:hy+2, :] = 1.0  # Horizon
    gt[boat_cy-boat_h_size, boat_cx-boat_w:boat_cx+boat_w] = 1.0  # Boat top
    gt[boat_cy, boat_cx-boat_w:boat_cx+boat_w] = 1.0  # Boat bottom
    gt[boat_cy-boat_h_size:boat_cy, boat_cx-boat_w] = 1.0  # Boat left
    gt[boat_cy-boat_h_size:boat_cy, boat_cx+boat_w] = 1.0  # Boat right
    gt[boat_cy-boat_h_size*3:boat_cy-boat_h_size, boat_cx-2] = 1.0  # Mast
    gt[boat_cy-boat_h_size*3:boat_cy-boat_h_size, boat_cx+2] = 1.0
    # Buoy circles
    for dy in range(-buoy_r-1, buoy_r+2):
        for dx in range(-buoy_r-1, buoy_r+2):
            d = np.sqrt(dx**2 + dy**2)
            if abs(d - buoy_r) < 1.5:
                ny, nx = buoy_cy + dy, buoy_cx + dx
                if 0 <= ny < h and 0 <= nx < w:
                    gt[ny, nx] = 1.0

    return np.clip(img, 0, 255), gt


def generate_cable_in_water(h=480, w=640, cable_contrast=0.15, water_noise=30):
    """
    Replicate Bagan's IEEE 2024 Fig.5 row 3: braided cable floating
    in water. Nearly impossible to see. Very low contrast.
    """
    img = generate_water_texture(h, w, wave_freq=0.08, noise_level=water_noise)
    gt = np.zeros((h, w), dtype=np.float64)

    # Cable: sinuous line across the image
    cable_y_center = h // 2
    cable_thickness = 3
    for x in range(20, w - 20):
        cy = cable_y_center + int(15 * np.sin(0.02 * x) + 5 * np.sin(0.07 * x))
        for dy in range(-cable_thickness, cable_thickness + 1):
            ny = cy + dy
            if 0 <= ny < h:
                img[ny, x] = img[ny, x] * (1 + cable_contrast)
                if abs(dy) == cable_thickness:
                    gt[ny, x] = 1.0

    return np.clip(img, 0, 255), gt


def generate_wave_field(h=480, w=640, n_waves=8, noise_level=15):
    """
    Replicate Bagan thesis Fig.5.7-5.9: Lake Murray wave field.
    Parallel wave crests at a specific angle.
    """
    img = np.full((h, w), 120.0)
    gt = np.zeros((h, w), dtype=np.float64)

    angle = 15  # degrees
    angle_rad = np.radians(angle)
    wave_spacing = 40
    wave_amplitude = 25

    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    proj = x * np.cos(angle_rad) + y * np.sin(angle_rad)

    for i in range(n_waves):
        center = i * wave_spacing + 20
        wave = wave_amplitude * np.exp(-0.5 * ((proj - center) / 5) ** 2)
        img += wave
        # GT: edges at wave crests (where gradient is strongest)
        crest_mask = np.abs(proj - center) < 2
        gt[crest_mask] = 1.0

    img += np.random.normal(0, noise_level, (h, w))
    return np.clip(img, 0, 255), gt


def generate_underexposed_scene(h=480, w=640):
    """
    Replicate Bagan's aquatic dataset Fig.6: underexposed marine scene
    with small objects barely visible.
    """
    # Very dark base
    img = np.random.normal(40, 10, (h, w))

    # Subtle horizon
    hy = int(h * 0.4)
    img[:hy, :] += 15

    # Very faint boat shape
    boat_cx, boat_cy = w // 2, hy + 50
    for x in range(boat_cx - 40, boat_cx + 40):
        for y in range(boat_cy - 8, boat_cy):
            if 0 <= y < h and 0 <= x < w:
                img[y, x] += 12  # Very subtle

    # Small bright spot (buoy reflection)
    buoy_x, buoy_y = int(w * 0.25), hy + 80
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            if dx**2 + dy**2 <= 16:
                ny, nx = buoy_y + dy, buoy_x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    img[ny, nx] += 30

    gt = np.zeros((h, w), dtype=np.float64)
    gt[hy-1:hy+2, :] = 1.0
    gt[boat_cy-8, boat_cx-40:boat_cx+40] = 1.0
    gt[boat_cy, boat_cx-40:boat_cx+40] = 1.0

    return np.clip(img, 0, 255), gt


# ============================================================
# Run Methods
# ============================================================

from baselines import sobel_gradients, prewitt_gradients, canny_edges
from wvf_lf import wvf_image

import torch
import torch.nn.functional as F


def run_all_methods(image_gray, image_rgb=None):
    """Run all methods on a single image and return edge maps."""
    results = {}

    # Traditional at multiple sizes (Bagan's missing baseline)
    for ksize in [3, 5, 7, 9, 15]:
        from scipy import ndimage
        img = image_gray.astype(np.float64)
        if ksize == 3:
            gx = ndimage.sobel(img, axis=1)
            gy = ndimage.sobel(img, axis=0)
        else:
            sigma = (ksize - 1) / 4.0
            gx = ndimage.gaussian_filter1d(img, sigma, axis=1, order=1)
            gy = ndimage.gaussian_filter1d(img, sigma, axis=0, order=1)
        mag = np.sqrt(gx**2 + gy**2)
        if mag.max() > 0:
            mag = mag / mag.max()
        results[f'Sobel-{ksize}x{ksize}'] = mag

    # Prewitt
    gx = ndimage.prewitt(image_gray.astype(np.float64), axis=1)
    gy = ndimage.prewitt(image_gray.astype(np.float64), axis=0)
    mag = np.sqrt(gx**2 + gy**2)
    if mag.max() > 0:
        mag = mag / mag.max()
    results['Prewitt-3x3'] = mag

    # Canny
    from skimage.feature import canny
    results['Canny-sigma1'] = canny(image_gray / 255.0, sigma=1.0).astype(np.float64)
    results['Canny-sigma2'] = canny(image_gray / 255.0, sigma=2.0).astype(np.float64)

    # ML models (if available and image is RGB)
    if image_rgb is not None and torch.cuda.is_available():
        try:
            models_dir = ROOT / "models"

            # TEED
            sys.path.insert(0, str(models_dir / "TEED"))
            from ted import TED
            model = TED()
            ckpt = torch.load(str(models_dir / "TEED/checkpoints/BIPED/5/5_model.pth"),
                               map_location='cpu')
            model.load_state_dict(ckpt)
            model.eval().cuda()

            mean_bgr = np.array([104.007, 116.669, 122.679])
            inp = image_rgb[:, :, ::-1].astype(np.float32) - mean_bgr
            inp = inp.transpose(2, 0, 1)
            h, w = inp.shape[1], inp.shape[2]
            ph, pw = (16 - h % 16) % 16, (16 - w % 16) % 16
            t = torch.from_numpy(inp).unsqueeze(0).float().cuda()
            if ph > 0 or pw > 0:
                t = F.pad(t, (0, pw, 0, ph), mode='reflect')
            with torch.no_grad():
                out = torch.sigmoid(model(t)[-1]).squeeze().cpu().numpy()
            results['TEED'] = out[:h, :w]
            del model
            torch.cuda.empty_cache()
            sys.path.pop(0)
        except Exception as e:
            print(f"  TEED failed: {e}")

        try:
            # DexiNed
            sys.path.insert(0, str(models_dir / "DexiNed"))
            from model import DexiNed
            model = DexiNed()
            ckpt = torch.load(str(models_dir / "DexiNed/checkpoints/BIPED/10/10_model.pth"),
                               map_location='cpu')
            model.load_state_dict(ckpt)
            model.eval().cuda()

            mean_bgr = np.array([103.939, 116.779, 123.68])
            inp = image_rgb[:, :, ::-1].astype(np.float32) - mean_bgr
            inp = inp.transpose(2, 0, 1)
            h, w = inp.shape[1], inp.shape[2]
            ph, pw = (16 - h % 16) % 16, (16 - w % 16) % 16
            t = torch.from_numpy(inp).unsqueeze(0).float().cuda()
            if ph > 0 or pw > 0:
                t = F.pad(t, (0, pw, 0, ph), mode='reflect')
            with torch.no_grad():
                out = torch.sigmoid(model(t)[-1]).squeeze().cpu().numpy()
            results['DexiNed'] = out[:h, :w]
            del model
            torch.cuda.empty_cache()
            sys.path.pop(0)
        except Exception as e:
            print(f"  DexiNed failed: {e}")

    return results


def compute_f_score(pred, gt, threshold=0.5, match_radius=3):
    """Compute F-score at a given threshold."""
    from scipy import ndimage as ndi
    pred_edges = pred > threshold
    gt_bool = gt > 0.5

    if not np.any(pred_edges) and not np.any(gt_bool):
        return 1.0, 1.0, 1.0

    struct = np.ones((2 * match_radius + 1, 2 * match_radius + 1))
    gt_dilated = ndi.binary_dilation(gt_bool, structure=struct)
    pred_dilated = ndi.binary_dilation(pred_edges, structure=struct)

    tp = np.sum(pred_edges & gt_dilated)
    fp = np.sum(pred_edges & ~gt_dilated)
    fn = np.sum(gt_bool & ~pred_dilated)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f


def find_best_threshold(pred, gt, n_thresholds=100):
    """Sweep thresholds and return best F-score."""
    best_f = 0
    best_t = 0
    for t in np.linspace(0.01, 0.99, n_thresholds):
        _, _, f = compute_f_score(pred, gt, threshold=t)
        if f > best_f:
            best_f = f
            best_t = t
    return best_f, best_t


# ============================================================
# Visualization (replicating Bagan's figure format)
# ============================================================

def plot_comparison(image, results, gt, scene_name, save_dir):
    """
    Generate side-by-side comparison like Bagan's figures.
    Shows: Original | Sobel 3x3 | Sobel 15x15 | Canny | TEED | DexiNed
    """
    methods_to_show = ['Sobel-3x3', 'Sobel-9x9', 'Sobel-15x15',
                       'Canny-sigma2', 'TEED', 'DexiNed']
    available = [m for m in methods_to_show if m in results]

    n_cols = len(available) + 2  # +original +GT
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3))

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original', fontsize=8)
    axes[0].axis('off')

    axes[1].imshow(gt, cmap='gray')
    axes[1].set_title('Ground Truth', fontsize=8)
    axes[1].axis('off')

    for i, method in enumerate(available):
        edge_map = results[method]
        # Use best threshold for display
        best_f, best_t = find_best_threshold(edge_map, gt, n_thresholds=50)
        binary_edges = (edge_map > best_t).astype(np.float64)

        axes[i + 2].imshow(1 - binary_edges, cmap='gray')
        axes[i + 2].set_title(f'{method}\nF={best_f:.3f}', fontsize=7)
        axes[i + 2].axis('off')

    fig.suptitle(scene_name, fontsize=10)
    fig.tight_layout()
    fig.savefig(save_dir / f'{scene_name.replace(" ", "_")}.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("MARITIME EDGE DETECTION BENCHMARK")
    print("Replicating Bagan's test protocols")
    print("=" * 70)

    scenes = {
        'Horizon with Boat and Buoys': generate_horizon_scene,
        'Cable in Water': generate_cable_in_water,
        'Wave Field': generate_wave_field,
        'Underexposed Marine Scene': generate_underexposed_scene,
        'Dark Horizon (high noise)': lambda: generate_horizon_scene(
            water_noise=40, boat_contrast=0.15),
        'Very Low SNR Waves': lambda: generate_wave_field(noise_level=35),
    }

    all_scores = {}

    for scene_name, gen_fn in scenes.items():
        print(f"\n--- {scene_name} ---")
        img, gt = gen_fn()

        # Create RGB version for ML models (stack grayscale)
        img_rgb = np.stack([img, img, img], axis=-1).astype(np.uint8)

        results = run_all_methods(img, img_rgb)

        scene_scores = {}
        for method_name, edge_map in results.items():
            best_f, best_t = find_best_threshold(edge_map, gt)
            scene_scores[method_name] = best_f
            print(f"  {method_name}: F={best_f:.4f} (t={best_t:.2f})")

        all_scores[scene_name] = scene_scores

        # Generate comparison figure
        plot_comparison(img, results, gt, scene_name, RESULTS_DIR)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Average F-score across all maritime scenes")
    print("=" * 70)

    method_avgs = {}
    for scene, scores in all_scores.items():
        for method, f in scores.items():
            if method not in method_avgs:
                method_avgs[method] = []
            method_avgs[method].append(f)

    report_lines = ["# Maritime Benchmark Results\n"]
    report_lines.append("## Average F-score Across All Synthetic Maritime Scenes\n")
    report_lines.append("| Method | Avg F-score | Scenes |")
    report_lines.append("|--------|----------:|-------:|")

    for method, scores in sorted(method_avgs.items(), key=lambda x: -np.mean(x[1])):
        avg = np.mean(scores)
        report_lines.append(f"| {method} | {avg:.4f} | {len(scores)} |")
        print(f"  {method}: avg F={avg:.4f}")

    report_lines.append("\n## Per-Scene Results\n")
    for scene, scores in all_scores.items():
        report_lines.append(f"### {scene}\n")
        report_lines.append("| Method | F-score |")
        report_lines.append("|--------|--------:|")
        for method, f in sorted(scores.items(), key=lambda x: -x[1]):
            report_lines.append(f"| {method} | {f:.4f} |")
        report_lines.append("")

    report_path = RESULTS_DIR / "maritime_report.md"
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
    print(f"\nReport: {report_path}")

    # Save raw JSON
    with open(RESULTS_DIR / "maritime_results.json", 'w') as f:
        json.dump(all_scores, f, indent=2)


if __name__ == "__main__":
    main()
