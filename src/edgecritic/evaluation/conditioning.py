"""Condition number and orientation profile analysis."""

import numpy as np

from edgecritic.core.taylor import (
    build_taylor_matrix,
    get_circular_neighbors,
    rotate_coordinates,
)
from edgecritic.wvf._cpu import wvf_single_pixel


def analyze_condition_numbers(np_counts, order=4, n_orientations=36):
    """Analyze how condition number of A^T A varies with Np and orientation.

    Parameters
    ----------
    np_counts : list of int
        Different Np values to test.
    order : int
        Taylor expansion order.
    n_orientations : int
        Number of orientations.

    Returns
    -------
    results : dict
        Maps Np -> array of condition numbers across orientations.
    """
    results = {}
    angles = np.linspace(0, 2 * np.pi, n_orientations, endpoint=False)

    for np_count in np_counts:
        conds = []
        global_coords = get_circular_neighbors(np_count)
        for theta in angles:
            local_coords = rotate_coordinates(global_coords, theta)
            A = build_taylor_matrix(local_coords, order=order)
            cond = np.linalg.cond(A.T @ A)
            conds.append(cond)
        results[np_count] = np.array(conds)

    return results


def wvf_orientation_profile(image, x, y, np_count=15, order=4, n_orientations=18):
    """Evaluate WVF derivative magnitude at one pixel across orientations.

    Parameters
    ----------
    image : ndarray, shape (H, W)
    x, y : int
        Pixel coordinates.
    np_count : int
    order : int
    n_orientations : int

    Returns
    -------
    angles : ndarray, shape (N,)
        Orientation angles in radians.
    profile : ndarray, shape (N,)
        |f_x| at each orientation.
    """
    angles = np.linspace(0, np.pi, n_orientations, endpoint=False)
    profile = []
    for theta in angles:
        fx, _, _ = wvf_single_pixel(image, x, y, np_count=np_count, order=order, theta=theta)
        profile.append(abs(fx))
    return angles, np.array(profile)
