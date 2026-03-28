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
import socket
import platform
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from pathlib import Path

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


def env_flag(name, default=False):
    """Parse a boolean environment variable."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def parse_resize_shape(value):
    """Parse WIDTHxHEIGHT style resize strings from the environment."""
    raw = (value or "64x64").strip().lower().replace(" ", "")
    if "x" not in raw:
        raise ValueError(f"Invalid resize shape {value!r}; expected HxW")
    h_str, w_str = raw.split("x", 1)
    return int(h_str), int(w_str)


def normalize_output_tag(value):
    """Normalize output tags so benchmark campaigns do not overwrite each other."""
    if not value:
        return ""
    value = value.strip()
    return value if value.startswith("_") else f"_{value}"


def parse_dataset_list(value):
    """Parse dataset lists using comma, colon, or semicolon separators."""
    raw = (value or "BSDS500,UDED").strip()
    return {
        item.strip().upper()
        for item in re.split(r"[,;:]+", raw)
        if item.strip()
    }


def resolve_worker_count(env_name, task_count):
    """Resolve a bounded process count from environment or SLURM metadata."""
    raw = (
        os.environ.get(env_name)
        or os.environ.get("SLURM_CPUS_PER_TASK")
        or os.environ.get("OMP_NUM_THREADS")
        or str(os.cpu_count() or 1)
    )
    try:
        workers = int(raw)
    except ValueError:
        workers = os.cpu_count() or 1
    return max(1, min(workers, task_count))


def run_wvf_lf_task(task):
    """Run a single WVF or LF image task in a worker process."""
    from wvf_lf import lf_image, wvf_image

    image = task["image"]
    start = time.perf_counter()

    if task["method"] == "WVF":
        mag, _, _ = wvf_image(
            image,
            np_count=task["np_count"],
            order=task["order"],
            n_orientations=task["n_orientations"],
        )
    else:
        mag, _, _ = lf_image(
            image,
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
        "config_name": task["config_name"],
        "image_idx": task["image_idx"],
        "mag": mag,
        "elapsed_s": elapsed,
    }


def collect_runtime_metadata():
    """Capture the execution environment for provenance in reports."""
    metadata = {
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "benchmark_note": (
            "Traditional filters and WVF/LF run in NumPy/SciPy on CPU. "
            "ML models use PyTorch and only run on GPU when torch.cuda.is_available() is true."
        ),
    }
    if torch.cuda.is_available():
        metadata["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    else:
        metadata["cuda_devices"] = []
    return metadata

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

def benchmark_traditional(images, ground_truths, names, n_thresholds=100, match_radius=3):
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
        ods, ois, ap = compute_metrics(preds, ground_truths, n_thresholds=n_thresholds,
                                       match_radius=match_radius)
        results[method_name] = {
            'ODS': ods, 'OIS': ois, 'AP': ap,
            'avg_time_s': avg_time, 'type': 'traditional'
        }
        print(f"    ODS={ods:.4f}  OIS={ois:.4f}  AP={ap:.4f}  "
              f"time={avg_time*1000:.1f}ms/img")

    return results


def benchmark_wvf_lf(images, ground_truths, names, dataset_name,
                     resize_shape=(64, 64), wvf_max_images=10, lf_max_images=5,
                     wvf_np_count=15, lf_half_width=3, order=4, n_orientations=18,
                     lf_subsample=2, n_thresholds=100, match_radius=3):
    """Benchmark tractable WVF/LF subsets with explicit notes."""
    from skimage.transform import resize

    results = {}
    resize_label = f"{resize_shape[0]}x{resize_shape[1]}"

    configs = [
        (
            f'WVF-Np{wvf_np_count}',
            min(wvf_max_images, len(images)),
            f'{dataset_name} subset, Np={wvf_np_count}, {n_orientations} orientations, '
            f'resized to {resize_label}',
        ),
        (
            f'LF-m{lf_half_width}-Np{wvf_np_count}',
            min(lf_max_images, len(images)),
            f'{dataset_name} subset, m={lf_half_width}, Np={wvf_np_count}, {n_orientations} '
            f'orientations, subsample={lf_subsample}, resized to {resize_label}',
        ),
    ]

    small_images = []
    gt_resized = []
    max_subset = max((subset for _, subset, _ in configs), default=0)
    for i in range(max_subset):
        small_images.append(
            resize(to_gray(images[i]), resize_shape, anti_aliasing=True, preserve_range=True).astype(np.float64)
        )
        gt_resized.append(
            resize(ground_truths[i], resize_shape, anti_aliasing=True, preserve_range=True)
        )

    tasks = []
    for method_name, subset, note in configs:
        if subset == 0:
            continue
        print(f"  Running {method_name} on {subset}/{len(images)} images...")
        for image_idx in range(subset):
            task = {
                "config_name": method_name,
                "method": "WVF" if method_name.startswith("WVF-") else "LF",
                "image_idx": image_idx,
                "image": small_images[image_idx],
                "np_count": wvf_np_count,
                "half_width": lf_half_width,
                "order": order,
                "n_orientations": n_orientations,
                "lf_subsample": lf_subsample,
            }
            tasks.append(task)

    workers = resolve_worker_count("WVF_PARALLEL_WORKERS", len(tasks))
    print(f"  Using {workers} worker processes for WVF/LF tasks")

    grouped = defaultdict(list)
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_wvf_lf_task, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            grouped[result["config_name"]].append(result)
            completed += 1
            if completed % max(1, min(10, len(tasks))) == 0 or completed == len(tasks):
                print(f"    Completed {completed}/{len(tasks)} WVF/LF tasks")

    config_meta = {name: (subset, note) for name, subset, note in configs}
    for method_name, subset, _ in configs:
        if subset == 0:
            continue
        ordered = sorted(grouped[method_name], key=lambda item: item["image_idx"])
        preds = [item["mag"] for item in ordered]
        gt_subset = gt_resized[:subset]
        total_time = sum(item["elapsed_s"] for item in ordered)
        avg_time = total_time / subset
        ods, ois, ap = compute_metrics(preds, gt_subset, n_thresholds=n_thresholds,
                                       match_radius=match_radius)
        note = config_meta[method_name][1]
        results[method_name] = {
            'ODS': ods,
            'OIS': ois,
            'AP': ap,
            'avg_time_s': avg_time,
            'type': 'wvf_lf',
            'note': note,
            'n_images': subset,
        }
        print(
            f"    ODS={ods:.4f}  OIS={ois:.4f}  AP={ap:.4f}  "
            f"time={avg_time:.1f}s/img"
        )

    return results


def benchmark_ml_model(model_name, load_fn, run_fn, images, ground_truths,
                       n_thresholds=100, match_radius=3):
    """Benchmark a single ML model."""
    print(f"  Loading {model_name}...")
    device_label = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    ods, ois, ap = compute_metrics(preds, ground_truths[:n_success],
                                   n_thresholds=n_thresholds, match_radius=match_radius)

    result = {
        'ODS': ods, 'OIS': ois, 'AP': ap,
        'avg_time_s': avg_time, 'type': 'ml',
        'n_images': n_success,
        'note': f'PyTorch inference on {device_label}',
    }
    print(f"    ODS={ods:.4f}  OIS={ois:.4f}  AP={ap:.4f}  "
          f"time={avg_time*1000:.1f}ms/img  ({n_success} images)")

    # Clean up GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def benchmark_ml_models(images, ground_truths, names, n_thresholds=100, match_radius=3):
    """Benchmark all ML models."""
    results = {}

    ml_models = [
        ('TEED', load_teed, run_teed),
        ('DexiNed', load_dexined, run_dexined),
        ('PiDiNet', load_pidinet, run_pidinet),
    ]

    for model_name, load_fn, run_fn in ml_models:
        result = benchmark_ml_model(model_name, load_fn, run_fn,
                                     images, ground_truths,
                                     n_thresholds=n_thresholds, match_radius=match_radius)
        if result is not None:
            results[model_name] = result

    return results


# ============================================================
# Visualization
# ============================================================

def plot_results(all_results, dataset_name, output_tag=""):
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
    fig.savefig(RESULTS_DIR / f'{dataset_name}{output_tag}_ods_ois.png', dpi=150)
    plt.close(fig)

    # Runtime comparison (log scale)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.barh(methods, times, color=colors)
    ax.set_xlabel('Average Time per Image (seconds, log scale)')
    ax.set_title(f'Runtime Comparison on {dataset_name}')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f'{dataset_name}{output_tag}_runtime.png', dpi=150)
    plt.close(fig)

    print(f"  Plots saved to {RESULTS_DIR}/")


def serialize_results_dict(results):
    """Convert numpy scalars to plain Python types for JSON export."""
    serializable = {}
    for key, value in results.items():
        serializable[key] = {
            inner_key: float(inner_val) if isinstance(inner_val, (np.floating, float)) else inner_val
            for inner_key, inner_val in value.items()
        }
    return serializable


def generate_report(all_results, dataset_name, meta):
    """Generate markdown results table with provenance."""
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
    lines.append("## Provenance\n")
    lines.append(f"- Images available: {meta['images_available']}")
    lines.append(f"- Images evaluated in this report: {meta['images_used']}")
    lines.append(
        f"- Evaluation: {meta['n_thresholds']} thresholds, {meta['match_radius']}-pixel match radius"
    )
    lines.append(
        f"- Runtime environment: Python {meta['runtime']['python_version']}, "
        f"NumPy {meta['runtime']['numpy_version']}, "
        f"PyTorch {meta['runtime']['torch_version']}, "
        f"CUDA available = {meta['runtime']['cuda_available']}"
    )
    lines.append(f"- Host: {meta['runtime']['hostname']}")
    lines.append(f"- Note: {meta['runtime']['benchmark_note']}")
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

    runtime_meta = collect_runtime_metadata()

    output_tag = normalize_output_tag(os.environ.get('BENCH_OUTPUT_TAG', ''))
    n_thresholds = int(os.environ.get('BENCH_N_THRESHOLDS', 100))
    match_radius = int(os.environ.get('BENCH_MATCH_RADIUS', 3))
    max_images = int(os.environ.get('BENCH_MAX_IMAGES', 50))
    wvf_resize_shape = parse_resize_shape(os.environ.get('WVF_RESIZE_SHAPE', '64x64'))
    wvf_max_images = int(os.environ.get('WVF_MAX_IMAGES', 10))
    lf_max_images = int(os.environ.get('LF_MAX_IMAGES', 5))
    wvf_np_count = int(os.environ.get('WVF_NP_COUNT', 15))
    lf_half_width = int(os.environ.get('LF_HALF_WIDTH', 3))
    wvf_order = int(os.environ.get('WVF_ORDER', 4))
    wvf_n_orientations = int(os.environ.get('WVF_N_ORIENTATIONS', 18))
    lf_subsample = int(os.environ.get('LF_SUBSAMPLE', 2))
    run_traditional = env_flag('RUN_TRADITIONAL', True)
    run_wvf_lf = env_flag('RUN_WVF_LF', True)
    run_ml = env_flag('RUN_ML', True)
    datasets_to_run = parse_dataset_list(os.environ.get('BENCH_DATASETS', 'BSDS500,UDED'))

    print("\n[1/5] Configuration...")
    print(f"  Datasets: {sorted(datasets_to_run)}")
    print(f"  Thresholds: {n_thresholds}")
    print(f"  Match radius: {match_radius}")
    print(
        "  WVF/LF params: "
        f"Np={wvf_np_count}, m={lf_half_width}, order={wvf_order}, "
        f"orientations={wvf_n_orientations}, resize={wvf_resize_shape}, "
        f"lf_subsample={lf_subsample}"
    )
    print(
        "  Sections enabled: "
        f"traditional={run_traditional}, wvf_lf={run_wvf_lf}, ml={run_ml}"
    )

    def run_dataset(dataset_key, images, ground_truths, names,
                    results_json_name, results_report_name):
        if len(images) == 0:
            print(f"  {dataset_key}: no images loaded, skipping")
            return

        available = len(images)
        if len(images) > max_images:
            print(f"  Using subset of {max_images}/{len(images)} images for {dataset_key}")
            images = images[:max_images]
            ground_truths = ground_truths[:max_images]
            names = names[:max_images]

        all_results = {}

        dataset_prefix = dataset_key.upper()
        dataset_wvf_max = int(
            os.environ.get(
                f'{dataset_prefix}_WVF_MAX_IMAGES',
                os.environ.get('WVF_MAX_IMAGES', wvf_max_images),
            )
        )
        dataset_lf_max = int(
            os.environ.get(
                f'{dataset_prefix}_LF_MAX_IMAGES',
                os.environ.get('LF_MAX_IMAGES', lf_max_images),
            )
        )

        if run_traditional:
            print("  Traditional filters...")
            all_results.update(
                benchmark_traditional(images, ground_truths, names,
                                      n_thresholds=n_thresholds, match_radius=match_radius)
            )
        if run_wvf_lf:
            print("  WVF/LF subsets...")
            all_results.update(
                benchmark_wvf_lf(
                    images, ground_truths, names, dataset_name=dataset_key,
                    resize_shape=wvf_resize_shape,
                    wvf_max_images=dataset_wvf_max,
                    lf_max_images=dataset_lf_max,
                    wvf_np_count=wvf_np_count,
                    lf_half_width=lf_half_width,
                    order=wvf_order,
                    n_orientations=wvf_n_orientations,
                    lf_subsample=lf_subsample,
                    n_thresholds=n_thresholds,
                    match_radius=match_radius,
                )
            )
        if run_ml:
            print("  ML models...")
            all_results.update(
                benchmark_ml_models(images, ground_truths, names,
                                    n_thresholds=n_thresholds, match_radius=match_radius)
            )

        meta = {
            "dataset": dataset_key,
            "images_available": available,
            "images_used": len(images),
            "n_thresholds": n_thresholds,
            "match_radius": match_radius,
            "runtime": runtime_meta,
        }
        report = generate_report(all_results, dataset_key, meta)
        print(report)
        plot_results(all_results, dataset_key, output_tag=output_tag)

        with open(RESULTS_DIR / results_json_name, 'w') as f:
            json.dump({"meta": meta, "results": serialize_results_dict(all_results)}, f, indent=2)
        with open(RESULTS_DIR / results_report_name, 'w') as f:
            f.write(report)
        print(f"\nResults saved to {RESULTS_DIR}/")

    print("\n[2/5] Loading datasets...")
    if "BSDS500" in datasets_to_run:
        bsds_images, bsds_gt, bsds_names = load_bsds500_test()
        print("\n[3/5] BSDS500 BENCHMARK")
        print("=" * 70)
        run_dataset(
            "BSDS500", bsds_images, bsds_gt, bsds_names,
            f'benchmark_results{output_tag}.json',
            f'benchmark_report{output_tag}.md',
        )

    if "UDED" in datasets_to_run:
        print("\n" + "=" * 70)
        print("UDED BENCHMARK")
        print("=" * 70)
        uded_images, uded_gt, uded_names = load_uded_test()
        run_dataset(
            "UDED", uded_images, uded_gt, uded_names,
            f'benchmark_uded_results{output_tag}.json',
            f'benchmark_uded_report{output_tag}.md',
        )


if __name__ == "__main__":
    main()
