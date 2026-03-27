"""
Unified benchmark: Traditional filters vs WVF/LF vs ML models on BSDS500/UDED.

Models benchmarked:
  Traditional: Sobel (3x3, 5x5, 7x7, 9x9), Prewitt (3x3), Canny
  WVF/LF: Bagan's Wide View Filter and Line Filter
  ML: TEED, DexiNed, PiDiNet, DiffusionEdge, NBED

Datasets: BSDS500 (200 test images), UDED
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paths
ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
DATASETS_DIR = ROOT / "datasets"
RESULTS_DIR = ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F
from scipy import ndimage, io as sio
from skimage import io as skio

# ============================================================
# Dataset Loaders
# ============================================================

def load_bsds500_test():
    """Load BSDS500 test set images and ground truth."""
    img_dir = DATASETS_DIR / "BSDS500_repo" / "BSDS500" / "data" / "images" / "test"
    gt_dir = DATASETS_DIR / "BSDS500_repo" / "BSDS500" / "data" / "groundTruth" / "test"

    images = []
    ground_truths = []
    names = []

    img_files = sorted(img_dir.glob("*.jpg"))
    print(f"Loading BSDS500 test set: {len(img_files)} images")

    for img_file in img_files:
        img = skio.imread(str(img_file))
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)

        gt_file = gt_dir / f"{img_file.stem}.mat"
        if gt_file.exists():
            mat = sio.loadmat(str(gt_file))
            # BSDS500 GT is in 'groundTruth' field, multiple annotators
            gt_data = mat['groundTruth'][0]
            # Average all annotator boundaries
            boundaries = []
            for ann in gt_data:
                seg = ann[0][0][0]  # segmentation
                bdry = ann[0][0][1]  # boundaries
                boundaries.append(bdry.astype(np.float64))
            gt = np.mean(boundaries, axis=0)
            gt = (gt > 0.5).astype(np.float64)
        else:
            gt = np.zeros(img.shape[:2], dtype=np.float64)

        images.append(img)
        ground_truths.append(gt)
        names.append(img_file.stem)

    return images, ground_truths, names


def load_uded_test():
    """Load UDED test set."""
    uded_dir = DATASETS_DIR / "UDED_repo"
    img_dir = uded_dir / "imgs"
    gt_dir = uded_dir / "gt"

    images = []
    ground_truths = []
    names = []

    if not img_dir.exists():
        print("UDED dataset not found, skipping")
        return images, ground_truths, names

    img_files = sorted(img_dir.glob("*.*"))
    print(f"Loading UDED test set: {len(img_files)} images")

    for img_file in img_files:
        img = skio.imread(str(img_file))
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)

        gt_file = gt_dir / f"{img_file.stem}.png"
        if not gt_file.exists():
            gt_file = gt_dir / f"{img_file.stem}.jpg"
        if gt_file.exists():
            gt = skio.imread(str(gt_file))
            if gt.ndim == 3:
                gt = gt[:, :, 0]
            gt = gt.astype(np.float64) / 255.0
        else:
            gt = np.zeros(img.shape[:2], dtype=np.float64)

        images.append(img)
        ground_truths.append(gt)
        names.append(img_file.stem)

    return images, ground_truths, names


# ============================================================
# Traditional Filter Methods
# ============================================================

def run_sobel(image_gray, ksize=3):
    """Run Sobel filter at specified kernel size."""
    img = image_gray.astype(np.float64)
    if ksize == 3:
        gx = ndimage.sobel(img, axis=1)
        gy = ndimage.sobel(img, axis=0)
    else:
        sigma = (ksize - 1) / 4.0
        gx = ndimage.gaussian_filter1d(img, sigma, axis=1, order=1)
        gy = ndimage.gaussian_filter1d(img, sigma, axis=0, order=1)
    return np.sqrt(gx**2 + gy**2)


def run_prewitt(image_gray):
    """Run Prewitt 3x3 filter."""
    img = image_gray.astype(np.float64)
    gx = ndimage.prewitt(img, axis=1)
    gy = ndimage.prewitt(img, axis=0)
    return np.sqrt(gx**2 + gy**2)


def run_canny(image_gray, sigma=1.0):
    """Run Canny via skimage."""
    from skimage.feature import canny
    return canny(image_gray.astype(np.float64) / 255.0, sigma=sigma).astype(np.float64)


def run_log(image_gray, sigma=2.0):
    """Run Laplacian of Gaussian."""
    img = image_gray.astype(np.float64)
    filtered = ndimage.gaussian_laplace(img, sigma)
    # Zero crossings
    edges = np.zeros_like(filtered)
    for y in range(1, filtered.shape[0] - 1):
        for x in range(1, filtered.shape[1] - 1):
            neighbors = [filtered[y-1, x], filtered[y+1, x],
                         filtered[y, x-1], filtered[y, x+1]]
            if any(n * filtered[y, x] < 0 for n in neighbors):
                edges[y, x] = abs(filtered[y, x])
    return edges


def to_gray(img):
    """Convert RGB to grayscale."""
    if img.ndim == 3:
        return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    return img


# ============================================================
# ML Model Wrappers
# ============================================================

def load_teed():
    """Load TEED model."""
    teed_dir = MODELS_DIR / "TEED"
    sys.path.insert(0, str(teed_dir))
    from ted import TED
    model = TED()
    ckpt = torch.load(str(teed_dir / "checkpoints" / "BIPED" / "5" / "5_model.pth"),
                       map_location='cpu')
    model.load_state_dict(ckpt)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    sys.path.pop(0)
    return model


def run_teed(model, image):
    """Run TEED inference on a single image. Returns edge probability map."""
    mean_bgr = np.array([104.007, 116.669, 122.679])
    img = image[:, :, ::-1].astype(np.float32)  # RGB -> BGR
    img = img - mean_bgr
    img = img.transpose(2, 0, 1)  # HWC -> CHW

    # Pad to multiple of 16
    h, w = img.shape[1], img.shape[2]
    ph = (16 - h % 16) % 16
    pw = (16 - w % 16) % 16

    tensor = torch.from_numpy(img).unsqueeze(0).float()
    if ph > 0 or pw > 0:
        tensor = F.pad(tensor, (0, pw, 0, ph), mode='reflect')

    if torch.cuda.is_available():
        tensor = tensor.cuda()

    with torch.no_grad():
        outputs = model(tensor)
        edge_map = torch.sigmoid(outputs[-1]).squeeze().cpu().numpy()

    return edge_map[:h, :w]


def load_dexined():
    """Load DexiNed model."""
    dexi_dir = MODELS_DIR / "DexiNed"
    sys.path.insert(0, str(dexi_dir))
    from model import DexiNed
    model = DexiNed()
    ckpt = torch.load(str(dexi_dir / "checkpoints" / "BIPED" / "10" / "10_model.pth"),
                       map_location='cpu')
    model.load_state_dict(ckpt)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    sys.path.pop(0)
    return model


def run_dexined(model, image):
    """Run DexiNed inference. Returns edge probability map."""
    mean_bgr = np.array([103.939, 116.779, 123.68])
    img = image[:, :, ::-1].astype(np.float32)  # RGB -> BGR
    img = img - mean_bgr
    img = img.transpose(2, 0, 1)

    h, w = img.shape[1], img.shape[2]
    ph = (16 - h % 16) % 16
    pw = (16 - w % 16) % 16

    tensor = torch.from_numpy(img).unsqueeze(0).float()
    if ph > 0 or pw > 0:
        tensor = F.pad(tensor, (0, pw, 0, ph), mode='reflect')

    if torch.cuda.is_available():
        tensor = tensor.cuda()

    with torch.no_grad():
        outputs = model(tensor)
        # Use fused output (last one)
        edge_map = torch.sigmoid(outputs[-1]).squeeze().cpu().numpy()

    return edge_map[:h, :w]


def load_pidinet():
    """Load PiDiNet model."""
    pidi_dir = MODELS_DIR / "pidinet"
    sys.path.insert(0, str(pidi_dir))
    from models.convert_pidinet import convert_pidinet
    from models import pidinet

    config = [
        [16, 16, 16, 16],  # carv4 config channels
    ]
    model = pidinet(sa=True, dil=True, config='carv4')
    ckpt = torch.load(str(pidi_dir / "trained_models" / "table5_pidinet.pth"),
                       map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model = convert_pidinet(model, 'carv4')
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    sys.path.pop(0)
    return model


def run_pidinet(model, image):
    """Run PiDiNet inference. Returns edge probability map."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = image.astype(np.float32) / 255.0
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)

    tensor = torch.from_numpy(img).unsqueeze(0).float()
    if torch.cuda.is_available():
        tensor = tensor.cuda()

    with torch.no_grad():
        outputs = model(tensor)
        edge_map = torch.sigmoid(outputs[-1]).squeeze().cpu().numpy()

    return edge_map


def load_nbed():
    """Load NBED model."""
    nbed_dir = MODELS_DIR / "NBED"
    weight_path = nbed_dir / "weights" / "nbed_bsds.pth"
    if not weight_path.exists():
        return None

    sys.path.insert(0, str(nbed_dir))
    try:
        # NBED uses a custom architecture; try to import
        from inference import load_model
        model = load_model(str(weight_path))
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
    except Exception as e:
        print(f"  NBED load failed: {e}")
        return None
    finally:
        sys.path.pop(0)
    return model


def run_nbed(model, image):
    """Run NBED inference."""
    if model is None:
        return None
    # NBED preprocessing - standard ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = image.astype(np.float32) / 255.0
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)

    tensor = torch.from_numpy(img).unsqueeze(0).float()
    if torch.cuda.is_available():
        tensor = tensor.cuda()

    with torch.no_grad():
        output = model(tensor)
        if isinstance(output, (list, tuple)):
            output = output[-1]
        edge_map = torch.sigmoid(output).squeeze().cpu().numpy()

    return edge_map


# ============================================================
# Evaluation Metrics
# ============================================================

def compute_metrics(pred, gt, n_thresholds=100, match_radius=3):
    """
    Compute ODS, OIS, and average precision for a set of predictions.

    Parameters
    ----------
    pred : list of ndarray
        Predicted edge probability maps.
    gt : list of ndarray
        Ground truth binary edge maps.
    n_thresholds : int
        Number of thresholds to sweep.
    match_radius : int
        Pixel matching radius.

    Returns
    -------
    ods, ois, ap : float
    """
    max_val = max(p.max() for p in pred if p.max() > 0)
    if max_val == 0:
        return 0.0, 0.0, 0.0

    thresholds = np.linspace(0, 1, n_thresholds)

    struct = np.ones((2 * match_radius + 1, 2 * match_radius + 1))

    all_precision = np.zeros(n_thresholds)
    all_recall = np.zeros(n_thresholds)
    per_image_best_f = []

    for p, g in zip(pred, gt):
        p_norm = p / max_val if max_val > 0 else p
        g_bool = g > 0.5
        g_dilated = ndimage.binary_dilation(g_bool, structure=struct)

        best_f = 0.0
        for ti, t in enumerate(thresholds):
            pred_edges = p_norm > t
            if not np.any(pred_edges) and not np.any(g_bool):
                continue

            tp = np.sum(pred_edges & g_dilated)
            fp = np.sum(pred_edges & ~g_dilated)

            pred_dilated = ndimage.binary_dilation(pred_edges, structure=struct)
            fn = np.sum(g_bool & ~pred_dilated)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            all_precision[ti] += precision
            all_recall[ti] += recall
            best_f = max(best_f, f)

        per_image_best_f.append(best_f)

    n_images = len(pred)
    all_precision /= n_images
    all_recall /= n_images

    f_scores = np.where(
        all_precision + all_recall > 0,
        2 * all_precision * all_recall / (all_precision + all_recall),
        0
    )

    ods = float(np.max(f_scores))
    ois = float(np.mean(per_image_best_f))

    # Average precision (area under precision-recall curve)
    sort_idx = np.argsort(all_recall)
    ap = float(np.trapz(all_precision[sort_idx], all_recall[sort_idx]))

    return ods, ois, ap


# ============================================================
# Benchmark Runner
# ============================================================

def benchmark_traditional(images, ground_truths, names):
    """Benchmark all traditional methods."""
    results = {}

    methods = {
        'Sobel-3x3': lambda img: run_sobel(to_gray(img), ksize=3),
        'Sobel-5x5': lambda img: run_sobel(to_gray(img), ksize=5),
        'Sobel-7x7': lambda img: run_sobel(to_gray(img), ksize=7),
        'Sobel-9x9': lambda img: run_sobel(to_gray(img), ksize=9),
        'Sobel-15x15': lambda img: run_sobel(to_gray(img), ksize=15),
        'Prewitt-3x3': lambda img: run_prewitt(to_gray(img)),
        'Canny-sigma1': lambda img: run_canny(to_gray(img), sigma=1.0),
        'Canny-sigma2': lambda img: run_canny(to_gray(img), sigma=2.0),
        'LoG-sigma2': lambda img: run_log(to_gray(img), sigma=2.0),
    }

    for method_name, method_fn in methods.items():
        print(f"  Running {method_name}...")
        preds = []
        total_time = 0
        for img in images:
            start = time.perf_counter()
            pred = method_fn(img)
            total_time += time.perf_counter() - start
            # Normalize to 0-1
            if pred.max() > 0:
                pred = pred / pred.max()
            preds.append(pred)

        avg_time = total_time / len(images)
        ods, ois, ap = compute_metrics(preds, ground_truths)
        results[method_name] = {
            'ODS': ods, 'OIS': ois, 'AP': ap,
            'avg_time_s': avg_time, 'type': 'traditional'
        }
        print(f"    ODS={ods:.4f}  OIS={ois:.4f}  AP={ap:.4f}  "
              f"time={avg_time*1000:.1f}ms/img")

    return results


def benchmark_wvf(images, ground_truths, names, max_images=20):
    """Benchmark WVF/LF (limited images due to speed)."""
    from wvf_lf import wvf_image, lf_image

    results = {}
    subset = min(max_images, len(images))
    print(f"  Running WVF/LF on {subset}/{len(images)} images (slow method)...")

    for method_name, method_fn in [
        ('WVF-Np15', lambda img: wvf_image(to_gray(img), np_count=15, order=4, n_orientations=18)),
    ]:
        preds = []
        total_time = 0
        for i in range(subset):
            # Resize to small for tractability
            from skimage.transform import resize
            small = resize(to_gray(images[i]), (64, 64), anti_aliasing=True) * 255
            gt_small = resize(ground_truths[i], (64, 64), anti_aliasing=True)

            start = time.perf_counter()
            mag, _, _ = method_fn(small)
            total_time += time.perf_counter() - start

            if mag.max() > 0:
                mag = mag / mag.max()
            preds.append(mag)

        avg_time = total_time / subset
        gt_resized = [resize(ground_truths[i], (64, 64), anti_aliasing=True) for i in range(subset)]
        ods, ois, ap = compute_metrics(preds, gt_resized)
        results[method_name] = {
            'ODS': ods, 'OIS': ois, 'AP': ap,
            'avg_time_s': avg_time, 'type': 'wvf_lf',
            'note': f'{subset} images, resized to 64x64'
        }
        print(f"    ODS={ods:.4f}  OIS={ois:.4f}  AP={ap:.4f}  "
              f"time={avg_time:.1f}s/img")

    return results


def benchmark_ml_model(model_name, load_fn, run_fn, images, ground_truths):
    """Benchmark a single ML model."""
    print(f"  Loading {model_name}...")
    try:
        model = load_fn()
        if model is None:
            print(f"    {model_name} not available, skipping")
            return None
    except Exception as e:
        print(f"    {model_name} load failed: {e}")
        return None

    preds = []
    total_time = 0
    n_success = 0

    for i, img in enumerate(images):
        try:
            start = time.perf_counter()
            pred = run_fn(model, img)
            total_time += time.perf_counter() - start

            if pred is None:
                continue
            if pred.max() > 0:
                pred = pred / pred.max()

            # Resize pred to match GT if needed
            if pred.shape != ground_truths[i].shape:
                from skimage.transform import resize
                pred = resize(pred, ground_truths[i].shape, anti_aliasing=True)

            preds.append(pred)
            n_success += 1
        except Exception as e:
            if i == 0:
                print(f"    Error on first image: {e}")
                import traceback
                traceback.print_exc()
            preds.append(np.zeros_like(ground_truths[i]))
            n_success += 1

    if n_success == 0:
        return None

    avg_time = total_time / n_success
    ods, ois, ap = compute_metrics(preds, ground_truths[:n_success])

    result = {
        'ODS': ods, 'OIS': ois, 'AP': ap,
        'avg_time_s': avg_time, 'type': 'ml',
        'n_images': n_success
    }
    print(f"    ODS={ods:.4f}  OIS={ois:.4f}  AP={ap:.4f}  "
          f"time={avg_time*1000:.1f}ms/img  ({n_success} images)")

    # Clean up GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def benchmark_ml_models(images, ground_truths, names):
    """Benchmark all ML models."""
    results = {}

    ml_models = [
        ('TEED', load_teed, run_teed),
        ('DexiNed', load_dexined, run_dexined),
        ('PiDiNet', load_pidinet, run_pidinet),
    ]

    for model_name, load_fn, run_fn in ml_models:
        result = benchmark_ml_model(model_name, load_fn, run_fn,
                                     images, ground_truths)
        if result is not None:
            results[model_name] = result

    return results


# ============================================================
# Visualization
# ============================================================

def plot_results(all_results, dataset_name):
    """Generate comparison plots."""
    methods = []
    ods_scores = []
    ois_scores = []
    times = []
    colors = []

    color_map = {'traditional': '#4477AA', 'wvf_lf': '#EE6677', 'ml': '#228833'}

    for name, data in sorted(all_results.items(), key=lambda x: x[1]['ODS']):
        methods.append(name)
        ods_scores.append(data['ODS'])
        ois_scores.append(data['OIS'])
        times.append(data['avg_time_s'])
        colors.append(color_map.get(data['type'], '#CCBB44'))

    # ODS/OIS bar chart
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(methods))
    width = 0.35
    bars1 = ax.bar(x - width/2, ods_scores, width, label='ODS', color=colors, alpha=0.8)
    bars2 = ax.bar(x + width/2, ois_scores, width, label='OIS', color=colors, alpha=0.5)
    ax.set_xlabel('Method')
    ax.set_ylabel('F-Score')
    ax.set_title(f'Edge Detection Benchmark on {dataset_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f'{dataset_name}_ods_ois.png', dpi=150)
    plt.close(fig)

    # Runtime comparison (log scale)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.barh(methods, times, color=colors)
    ax.set_xlabel('Average Time per Image (seconds, log scale)')
    ax.set_title(f'Runtime Comparison on {dataset_name}')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f'{dataset_name}_runtime.png', dpi=150)
    plt.close(fig)

    print(f"  Plots saved to {RESULTS_DIR}/")


def generate_report(all_results, dataset_name):
    """Generate markdown results table."""
    lines = [f"# Benchmark Results: {dataset_name}\n"]
    lines.append("| Method | Type | ODS | OIS | AP | Time (ms/img) |")
    lines.append("|--------|------|----:|----:|---:|--------------:|")

    for name, data in sorted(all_results.items(), key=lambda x: -x[1]['ODS']):
        time_ms = data['avg_time_s'] * 1000
        time_str = f"{time_ms:.1f}" if time_ms < 10000 else f"{time_ms/1000:.1f}s"
        lines.append(
            f"| {name} | {data['type']} | {data['ODS']:.4f} | "
            f"{data['OIS']:.4f} | {data['AP']:.4f} | {time_str} |"
        )
        note = data.get('note', '')
        if note:
            lines[-1] += f" *{note}*"

    lines.append("")
    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("EDGE DETECTION BENCHMARK")
    print("Traditional Filters vs WVF/LF vs ML Models")
    print("=" * 70)

    # Load BSDS500
    print("\n[1/5] Loading datasets...")
    bsds_images, bsds_gt, bsds_names = load_bsds500_test()

    # Limit to manageable subset for initial run
    max_images = int(os.environ.get('BENCH_MAX_IMAGES', 50))
    if len(bsds_images) > max_images:
        print(f"  Using subset of {max_images}/{len(bsds_images)} images")
        bsds_images = bsds_images[:max_images]
        bsds_gt = bsds_gt[:max_images]
        bsds_names = bsds_names[:max_images]

    all_results = {}

    # Traditional methods
    print("\n[2/5] Benchmarking traditional filters...")
    trad_results = benchmark_traditional(bsds_images, bsds_gt, bsds_names)
    all_results.update(trad_results)

    # WVF/LF
    print("\n[3/5] Benchmarking WVF/LF...")
    wvf_results = benchmark_wvf(bsds_images, bsds_gt, bsds_names, max_images=10)
    all_results.update(wvf_results)

    # ML models
    print("\n[4/5] Benchmarking ML models...")
    ml_results = benchmark_ml_models(bsds_images, bsds_gt, bsds_names)
    all_results.update(ml_results)

    # Generate outputs
    print("\n[5/5] Generating report and plots...")
    report = generate_report(all_results, "BSDS500")
    print(report)

    plot_results(all_results, "BSDS500")

    # Save raw results
    serializable = {}
    for k, v in all_results.items():
        serializable[k] = {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                           for kk, vv in v.items()}
    with open(RESULTS_DIR / 'benchmark_results.json', 'w') as f:
        json.dump(serializable, f, indent=2)

    # Save report
    report_path = RESULTS_DIR / 'benchmark_report.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nResults saved to {RESULTS_DIR}/")

    # Also try UDED
    print("\n" + "=" * 70)
    print("UDED BENCHMARK")
    print("=" * 70)
    uded_images, uded_gt, uded_names = load_uded_test()
    if uded_images:
        uded_max = min(max_images, len(uded_images))
        uded_images = uded_images[:uded_max]
        uded_gt = uded_gt[:uded_max]
        uded_names = uded_names[:uded_max]

        uded_results = {}
        print("  Traditional filters...")
        uded_results.update(benchmark_traditional(uded_images, uded_gt, uded_names))
        print("  ML models...")
        uded_results.update(benchmark_ml_models(uded_images, uded_gt, uded_names))

        uded_report = generate_report(uded_results, "UDED")
        print(uded_report)
        with open(RESULTS_DIR / 'benchmark_uded_report.md', 'w') as f:
            f.write(uded_report)


if __name__ == "__main__":
    main()
