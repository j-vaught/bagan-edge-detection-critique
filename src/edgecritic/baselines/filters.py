"""Classical edge detection filters: Sobel, Prewitt, Canny."""

import numpy as np
from scipy import ndimage


def sobel_gradients(image):
    """Compute Sobel gradients (3x3 kernel).

    Parameters
    ----------
    image : ndarray, shape (H, W)

    Returns
    -------
    grad_x, grad_y, magnitude, angle : ndarrays
    """
    img = image.astype(np.float64)
    grad_x = ndimage.sobel(img, axis=1)
    grad_y = ndimage.sobel(img, axis=0)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x) % np.pi
    return grad_x, grad_y, magnitude, angle


def prewitt_gradients(image):
    """Compute Prewitt gradients (3x3 kernel).

    Parameters
    ----------
    image : ndarray, shape (H, W)

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
    """Compute extended Sobel gradients (larger kernels via Gaussian derivatives).

    Parameters
    ----------
    image : ndarray, shape (H, W)
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
    """Canny edge detection using scipy.

    Parameters
    ----------
    image : ndarray, shape (H, W)
    low_threshold : float, optional
    high_threshold : float, optional
    sigma : float

    Returns
    -------
    edges : ndarray, shape (H, W), dtype bool
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
    """Simple thresholding to produce binary edge map.

    Parameters
    ----------
    magnitude : ndarray
    threshold : float

    Returns
    -------
    edges : ndarray, dtype bool
    """
    return magnitude > threshold
