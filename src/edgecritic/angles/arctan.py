"""Traditional arctan-based angle computation."""

import numpy as np


def arctan_angle(grad_x, grad_y):
    """Traditional arctan-based angle computation.

    The conventional method used by Canny and others.

    Parameters
    ----------
    grad_x : float or ndarray
        Horizontal gradient component.
    grad_y : float or ndarray
        Vertical gradient component.

    Returns
    -------
    angle : float or ndarray
        Gradient angle in radians [0, pi).
    """
    angle = np.arctan2(grad_y, grad_x)
    angle = angle % np.pi
    return angle
