"""
Baseline edge detection methods for comparison against WVF/LF.

Implements Sobel, Prewitt, and Canny using scipy/numpy (no OpenCV dependency),
with optional OpenCV backends when available.
"""

import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d


def sobel_gradients(image):
    """
    Compute Sobel gradients (3x3 kernel).

    The standard Sobel operator that Bagan compares against.
    Uses the conventional kernels:
      Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
      Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    Parameters
    ----------
    image : ndarray, shape (H, W)
        Grayscale image.

    Returns
    -------
    grad_x : ndarray
        Horizontal gradient.
    grad_y : ndarray
        Vertical gradient.
    magnitude : ndarray
        Gradient magnitude.
    angle : ndarray
        Gradient angle in radians [0, pi).
    """
    img = image.astype(np.float64)
    grad_x = ndimage.sobel(img, axis=1)
    grad_y = ndimage.sobel(img, axis=0)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x) % np.pi
    return grad_x, grad_y, magnitude, angle


def prewitt_gradients(image):
    """
    Compute Prewitt gradients (3x3 kernel).

    The Prewitt operator: equal-weight alternative to Sobel.
      Gx = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
      Gy = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]

    Parameters
    ----------
    image : ndarray, shape (H, W)
        Grayscale image.

    Returns
    -------
    grad_x, grad_y, magnitude, angle : ndarrays
    """
    img = image.astype(np.float64)
    grad_x = ndimage.prewitt(img, axis=1)
    grad_y = ndimage.prewitt(img, axis=0)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x) % np.pi
    return grad_x, grad_y, magnitude, angle


def extended_sobel_gradients(image, ksize=5):
    """
    Compute extended Sobel gradients (5x5, 7x7, etc.).

    Larger Sobel-like kernels that Bagan mentions as common alternatives.
    Uses scipy's gaussian_gradient_magnitude as a proxy for larger kernels.

    Parameters
    ----------
    image : ndarray, shape (H, W)
        Grayscale image.
    ksize : int
        Kernel size (must be odd).

    Returns
    -------
    grad_x, grad_y, magnitude, angle : ndarrays
    """
    img = image.astype(np.float64)
    sigma = (ksize - 1) / 4.0

    grad_x = ndimage.gaussian_filter1d(img, sigma, axis=1, order=1)
    grad_y = ndimage.gaussian_filter1d(img, sigma, axis=0, order=1)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x) % np.pi
    return grad_x, grad_y, magnitude, angle


def canny_edges(image, low_threshold=None, high_threshold=None, sigma=1.0):
    """
    Canny edge detection using scipy (no OpenCV needed).

    Implements the standard Canny pipeline:
    1. Gaussian smoothing
    2. Sobel gradient computation
    3. Non-maximum suppression
    4. Hysteresis thresholding

    Parameters
    ----------
    image : ndarray, shape (H, W)
        Grayscale image (0-255 or 0-1).
    low_threshold : float
        Lower hysteresis threshold. Default: auto from image stats.
    high_threshold : float
        Upper hysteresis threshold. Default: auto from image stats.
    sigma : float
        Gaussian smoothing sigma.

    Returns
    -------
    edges : ndarray, shape (H, W), dtype bool
        Binary edge map.
    """
    img = image.astype(np.float64)

    if sigma > 0:
        img = ndimage.gaussian_filter(img, sigma)

    grad_x = ndimage.sobel(img, axis=1)
    grad_y = ndimage.sobel(img, axis=0)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x)

    H, W = magnitude.shape
    nms = np.zeros_like(magnitude)

    angle_deg = np.degrees(angle) % 180

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            a = angle_deg[y, x]

            if (0 <= a < 22.5) or (157.5 <= a < 180):
                n1, n2 = magnitude[y, x - 1], magnitude[y, x + 1]
            elif 22.5 <= a < 67.5:
                n1, n2 = magnitude[y - 1, x + 1], magnitude[y + 1, x - 1]
            elif 67.5 <= a < 112.5:
                n1, n2 = magnitude[y - 1, x], magnitude[y + 1, x]
            else:
                n1, n2 = magnitude[y - 1, x - 1], magnitude[y + 1, x + 1]

            if magnitude[y, x] >= n1 and magnitude[y, x] >= n2:
                nms[y, x] = magnitude[y, x]

    if high_threshold is None:
        high_threshold = np.percentile(nms[nms > 0], 90) if np.any(nms > 0) else 1.0
    if low_threshold is None:
        low_threshold = high_threshold * 0.4

    strong = nms >= high_threshold
    weak = (nms >= low_threshold) & ~strong

    edges = strong.copy()
    changed = True
    while changed:
        changed = False
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                if weak[y, x] and not edges[y, x]:
                    if np.any(edges[y - 1:y + 2, x - 1:x + 2]):
                        edges[y, x] = True
                        changed = True

    return edges


def compute_edges_at_threshold(magnitude, threshold):
    """
    Simple thresholding to produce binary edge map.

    Used for ODS/OIS computation: sweep threshold and compare to GT.

    Parameters
    ----------
    magnitude : ndarray
        Gradient magnitude image.
    threshold : float
        Threshold value.

    Returns
    -------
    edges : ndarray, dtype bool
        Binary edge map.
    """
    return magnitude > threshold


def compute_ods_ois(magnitude, ground_truth, n_thresholds=100,
                     match_radius=3):
    """
    Compute ODS and OIS metrics for edge detection evaluation.

    ODS: Best F-score across the dataset using a single global threshold.
    OIS: Best F-score per image, then averaged.

    This replicates the evaluation methodology described in the papers
    (Section IV.C of IEEE 2024, Section IV.C of IEEE 2025).

    Parameters
    ----------
    magnitude : ndarray or list of ndarray
        Gradient magnitude map(s).
    ground_truth : ndarray or list of ndarray
        Binary ground truth edge map(s).
    n_thresholds : int
        Number of threshold values to sweep (papers use 1001).
    match_radius : int
        Pixel radius for edge matching (papers use 3).

    Returns
    -------
    ods : float
        Optimal Dataset Scale F-score.
    ois : float
        Optimal Image Scale F-score.
    thresholds : ndarray
        Threshold values tested.
    f_scores : ndarray
        F-scores at each threshold.
    """
    if not isinstance(magnitude, list):
        magnitude = [magnitude]
        ground_truth = [ground_truth]

    max_val = max(m.max() for m in magnitude)
    if max_val == 0:
        return 0.0, 0.0, np.array([]), np.array([])

    thresholds = np.linspace(0, max_val, n_thresholds)

    all_tp = np.zeros(n_thresholds)
    all_fp = np.zeros(n_thresholds)
    all_fn = np.zeros(n_thresholds)
    per_image_best_f = []

    for mag, gt in zip(magnitude, ground_truth):
        gt_bool = gt > 0

        if match_radius > 1:
            gt_dilated = ndimage.binary_dilation(
                gt_bool,
                structure=np.ones((2 * match_radius + 1, 2 * match_radius + 1))
            )
            pred_check = gt_dilated
        else:
            pred_check = gt_bool

        image_f_scores = []

        for ti, t in enumerate(thresholds):
            pred = mag > t

            tp = np.sum(pred & pred_check)
            fp = np.sum(pred & ~pred_check)
            fn = np.sum(gt_bool & ~ndimage.binary_dilation(
                pred,
                structure=np.ones((2 * match_radius + 1, 2 * match_radius + 1))
            )) if match_radius > 1 else np.sum(gt_bool & ~pred)

            all_tp[ti] += tp
            all_fp[ti] += fp
            all_fn[ti] += fn

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            image_f_scores.append(f)

        per_image_best_f.append(max(image_f_scores))

    f_scores = np.zeros(n_thresholds)
    for ti in range(n_thresholds):
        p = all_tp[ti] / (all_tp[ti] + all_fp[ti]) if (all_tp[ti] + all_fp[ti]) > 0 else 0
        r = all_tp[ti] / (all_tp[ti] + all_fn[ti]) if (all_tp[ti] + all_fn[ti]) > 0 else 0
        f_scores[ti] = 2 * p * r / (p + r) if (p + r) > 0 else 0

    ods = np.max(f_scores)
    ois = np.mean(per_image_best_f)

    return ods, ois, thresholds, f_scores


def runtime_comparison(image, methods=None, n_runs=10):
    """
    Benchmark runtime of different gradient computation methods.

    This addresses a key critique: WVF/LF use vastly more pixels per
    computation than Sobel/Prewitt, so runtime comparison is essential.

    Parameters
    ----------
    image : ndarray, shape (H, W)
        Test image.
    methods : dict
        Method name -> callable. Default: Sobel, Prewitt, Extended Sobel.
    n_runs : int
        Number of repetitions for timing.

    Returns
    -------
    results : dict
        Method name -> {'mean_time': float, 'std_time': float}
    """
    import time

    if methods is None:
        methods = {
            'Sobel (3x3)': lambda img: sobel_gradients(img),
            'Prewitt (3x3)': lambda img: prewitt_gradients(img),
            'Extended Sobel (5x5)': lambda img: extended_sobel_gradients(img, ksize=5),
            'Extended Sobel (7x7)': lambda img: extended_sobel_gradients(img, ksize=7),
        }

    results = {}
    for name, func in methods.items():
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            func(image)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        results[name] = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
        }

    return results


if __name__ == "__main__":
    print("Baselines loaded. Running quick test...")
    test_img = np.random.rand(128, 128) * 255

    for name, func in [('Sobel', sobel_gradients), ('Prewitt', prewitt_gradients)]:
        gx, gy, mag, ang = func(test_img)
        print(f"  {name}: mag range [{mag.min():.2f}, {mag.max():.2f}]")

    edges = canny_edges(test_img)
    print(f"  Canny: {np.sum(edges)} edge pixels")

    print("Done.")
