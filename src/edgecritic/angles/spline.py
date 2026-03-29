"""Cubic spline angle estimation for edge orientation."""

import numpy as np
from scipy.interpolate import CubicSpline


def cubic_spline_angle(gradients, angles):
    """Compute precise gradient orientation using periodic cubic splines.

    Fits a periodic cubic spline to gradient magnitudes vs orientation,
    then finds the angle at the global maximum.

    Parameters
    ----------
    gradients : ndarray, shape (N,)
        Gradient magnitudes at each sampled orientation.
    angles : ndarray, shape (N,)
        Orientation angles in radians (0 to pi).

    Returns
    -------
    best_angle : float
        Estimated edge angle in radians.
    max_gradient : float
        Maximum gradient value from spline.
    spline_func : CubicSpline
        The fitted spline for visualization.
    """
    extended_angles = np.concatenate([angles - np.pi, angles, angles + np.pi])
    extended_grads = np.concatenate([gradients, gradients, gradients])

    sort_idx = np.argsort(extended_angles)
    extended_angles = extended_angles[sort_idx]
    extended_grads = extended_grads[sort_idx]

    cs = CubicSpline(extended_angles, extended_grads)

    fine_angles = np.linspace(0, np.pi, 1000)
    fine_grads = cs(fine_angles)

    best_idx = np.argmax(fine_grads)
    best_angle = fine_angles[best_idx]
    max_gradient = fine_grads[best_idx]

    return best_angle, max_gradient, cs
