"""Runtime benchmarking utilities."""

import time

import numpy as np

from edgecritic.baselines.filters import (
    extended_sobel_gradients,
    prewitt_gradients,
    sobel_gradients,
)


def runtime_comparison(image, methods=None, n_runs=10):
    """Benchmark runtime of different gradient computation methods.

    Parameters
    ----------
    image : ndarray, shape (H, W)
        Test image.
    methods : dict, optional
        Method name -> callable. Default: Sobel, Prewitt, Extended Sobel.
    n_runs : int
        Number of repetitions for timing.

    Returns
    -------
    results : dict
        Method name -> {'mean_time': float, 'std_time': float}
    """
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
