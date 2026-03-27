"""
Re-implementation of the Wide View Filter (WVF) and Line Filter (LF)
from Bagan & Wang (USC, 2023-2025) for independent verification.

Based on:
  - Thesis: "Wide View and Line Filter for Enhanced Image Gradient
    Computation and Edge Determination" (2023)
  - IEEE OCEANS 2024: "Wide View and Line Filter for Enhanced Gradient
    Computation in Aquatic Environment"
  - IEEE OCEANS 2025: "Multi-Scale and Multi-Domain Edge Determination
    with Accurate Gradient Orientation Computation in Aquatic Environment"

References equations from the IEEE papers (Eq. 1-6).
"""

import numpy as np
from scipy.interpolate import CubicSpline


def build_taylor_matrix(coords, order=4):
    """
    Build the design matrix A for the 2D Taylor expansion (Eq. 1-2).

    Given neighbor pixel positions (x_i, y_i) in the local coordinate system,
    construct the Vandermonde-like matrix whose columns correspond to each
    monomial term up to the specified order.

    For order=4, the expansion has 15 derivative coefficients:
      f, f_x, f_y, f_xx/2, f_yy/2, f_xy, f_xxx/6, f_yyy/6, f_xxy/2,
      f_xyy/2, f_xxxx/24, f_yyyy/24, f_xxxy/6, f_xxyy/4, f_xyyy/6

    Parameters
    ----------
    coords : ndarray, shape (Np, 2)
        Local coordinates (x_i, y_i) of neighbor pixels.
    order : int
        Maximum order of Taylor expansion (default 4).

    Returns
    -------
    A : ndarray, shape (Np, num_coefficients)
        Design matrix for the least-squares system.
    """
    x = coords[:, 0]
    y = coords[:, 1]
    n = len(x)

    columns = [np.ones(n)]  # f^0 (constant term)

    if order >= 1:
        columns.append(x)                          # f_x * x
        columns.append(y)                          # f_y * y

    if order >= 2:
        columns.append(x**2 / 2)                   # f_xx * x^2/2
        columns.append(y**2 / 2)                   # f_yy * y^2/2
        columns.append(x * y)                      # f_xy * xy

    if order >= 3:
        columns.append(x**3 / 6)                   # f_xxx * x^3/6
        columns.append(y**3 / 6)                   # f_yyy * y^3/6
        columns.append(x**2 * y / 2)               # f_xxy * x^2*y/2
        columns.append(x * y**2 / 2)               # f_xyy * x*y^2/2

    if order >= 4:
        columns.append(x**4 / 24)                  # f_xxxx * x^4/24
        columns.append(y**4 / 24)                  # f_yyyy * y^4/24
        columns.append(x**3 * y / 6)               # f_xxxy * x^3*y/6
        columns.append(x**2 * y**2 / 4)            # f_xxyy * x^2*y^2/4
        columns.append(x * y**3 / 6)               # f_xyyy * x*y^3/6

    if order >= 5:
        columns.append(x**5 / 120)
        columns.append(y**5 / 120)
        columns.append(x**4 * y / 24)
        columns.append(x**3 * y**2 / 12)
        columns.append(x**2 * y**3 / 12)
        columns.append(x * y**4 / 24)

    A = np.column_stack(columns)
    return A


def get_circular_neighbors(np_count, radius=None):
    """
    Get pixel positions within a circular neighborhood of given count.

    The WVF uses Np pixels in a circular region around the origin pixel.
    We select the Np closest integer-coordinate pixels to the origin
    within a circle.

    Parameters
    ----------
    np_count : int
        Number of neighbor pixels (Np).
    radius : float, optional
        If given, use this radius. Otherwise, auto-compute to get ~np_count pixels.

    Returns
    -------
    coords : ndarray, shape (Np, 2)
        Integer (x, y) coordinates relative to origin.
    """
    if radius is None:
        radius = int(np.ceil(np.sqrt(np_count / np.pi))) + 1

    candidates = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            dist = np.sqrt(dx**2 + dy**2)
            if dist <= radius:
                candidates.append((dx, dy, dist))

    candidates.sort(key=lambda c: c[2])
    selected = candidates[:np_count]
    coords = np.array([(c[0], c[1]) for c in selected], dtype=np.float64)
    return coords


def rotate_coordinates(coords, theta):
    """
    Rotate coordinates into the local coordinate system at angle theta.

    The WVF operates in a local (x, y) system rotated by theta from the
    global (X, Y) system. The normal direction is along x, tangential along y.

    Parameters
    ----------
    coords : ndarray, shape (N, 2)
        Coordinates in global frame.
    theta : float
        Rotation angle in radians.

    Returns
    -------
    rotated : ndarray, shape (N, 2)
        Coordinates in local frame.
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R = np.array([[cos_t, sin_t],
                  [-sin_t, cos_t]])
    return coords @ R.T


def compute_wvf_pseudoinverse(coords, order=4):
    """
    Compute the pseudo-inverse A* = (A^T A)^{-1} A^T (Eq. 3).

    Parameters
    ----------
    coords : ndarray, shape (Np, 2)
        Local coordinates of neighbor pixels.
    order : int
        Taylor expansion order.

    Returns
    -------
    A_pinv : ndarray
        Pseudo-inverse matrix.
    cond_number : float
        Condition number of A^T A (for numerical stability analysis).
    """
    A = build_taylor_matrix(coords, order=order)
    ATA = A.T @ A
    cond = np.linalg.cond(ATA)
    A_pinv = np.linalg.pinv(A)
    return A_pinv, cond


def wvf_single_pixel(image, X0, Y0, np_count=15, order=4, theta=0.0):
    """
    Apply the Wide View Filter at a single pixel (X0, Y0) at orientation theta.

    Implements the core WVF algorithm:
    1. Select Np neighbor pixels in circular region
    2. Rotate to local coordinate system at angle theta
    3. Build Taylor expansion design matrix
    4. Solve least-squares for derivative coefficients
    5. Extract normal (f_x) and tangential (f_y) derivatives

    Parameters
    ----------
    image : ndarray, shape (H, W)
        Grayscale image.
    X0, Y0 : int
        Global pixel coordinates of target pixel.
    np_count : int
        Number of neighbor pixels.
    order : int
        Taylor expansion order.
    theta : float
        Orientation angle in radians.

    Returns
    -------
    f_x : float
        Normal derivative (gradient along x in local frame).
    f_y : float
        Tangential derivative (gradient along y in local frame).
    cond : float
        Condition number of the system.
    """
    H, W = image.shape
    global_coords = get_circular_neighbors(np_count)

    rotated_local = rotate_coordinates(global_coords, theta)

    global_pixel_coords = global_coords.astype(int) + np.array([X0, Y0])

    valid = ((global_pixel_coords[:, 0] >= 0) &
             (global_pixel_coords[:, 0] < W) &
             (global_pixel_coords[:, 1] >= 0) &
             (global_pixel_coords[:, 1] < H))

    if np.sum(valid) < order * (order + 1) // 2 + 3:
        return 0.0, 0.0, np.inf

    local_coords = rotated_local[valid]
    pixel_values = image[global_pixel_coords[valid, 1],
                         global_pixel_coords[valid, 0]]

    A_pinv, cond = compute_wvf_pseudoinverse(local_coords, order=order)

    z = A_pinv @ pixel_values

    f_x = z[1] if len(z) > 1 else 0.0
    f_y = z[2] if len(z) > 2 else 0.0

    return f_x, f_y, cond


def wvf_image(image, np_count=15, order=4, n_orientations=36):
    """
    Apply WVF to entire image at multiple orientations.

    For each pixel, run WVF at n_orientations angles and keep the
    orientation with the maximum normal derivative magnitude.

    Parameters
    ----------
    image : ndarray, shape (H, W)
        Grayscale image (float, 0-1 or 0-255).
    np_count : int
        Number of neighbor pixels per WVF application.
    order : int
        Taylor expansion order.
    n_orientations : int
        Number of orientations to sample (Ns in the papers).

    Returns
    -------
    gradient_mag : ndarray, shape (H, W)
        Maximum gradient magnitude at each pixel.
    gradient_angle : ndarray, shape (H, W)
        Angle (radians) at which maximum gradient occurs.
    condition_numbers : ndarray, shape (H, W)
        Condition number at each pixel (for stability analysis).
    """
    H, W = image.shape
    img = image.astype(np.float64)

    gradient_mag = np.zeros((H, W))
    gradient_angle = np.zeros((H, W))
    condition_numbers = np.zeros((H, W))

    angles = np.linspace(0, 2 * np.pi, n_orientations, endpoint=False)

    global_neighbors = get_circular_neighbors(np_count)

    border = int(np.max(np.abs(global_neighbors))) + 1

    for y in range(border, H - border):
        for x in range(border, W - border):
            max_mag = 0.0
            best_angle = 0.0
            best_cond = np.inf

            pixel_values_cache = {}
            for ni in range(len(global_neighbors)):
                gx = int(global_neighbors[ni, 0]) + x
                gy = int(global_neighbors[ni, 1]) + y
                pixel_values_cache[ni] = img[gy, gx]

            pixel_vals = np.array([pixel_values_cache[i]
                                   for i in range(len(global_neighbors))])

            for theta in angles:
                local_coords = rotate_coordinates(global_neighbors, theta)
                A = build_taylor_matrix(local_coords, order=order)
                try:
                    z = np.linalg.lstsq(A, pixel_vals, rcond=None)[0]
                    f_x = z[1]
                    mag = abs(f_x)
                    if mag > max_mag:
                        max_mag = mag
                        best_angle = theta
                        cond = np.linalg.cond(A.T @ A)
                        best_cond = cond
                except np.linalg.LinAlgError:
                    continue

            gradient_mag[y, x] = max_mag
            gradient_angle[y, x] = best_angle
            condition_numbers[y, x] = best_cond

    return gradient_mag, gradient_angle, condition_numbers


def line_filter_single_pixel(image, X0, Y0, half_width=7, np_count=15,
                              order=4, theta=0.0, use_weights=True):
    """
    Apply Line Filter at a single pixel (Eq. 4-6).

    The LF chains (2m+1) WVF applications along a line centered on (X0, Y0)
    at orientation theta. Virtual pixels along the line serve as expansion
    points for individual WVF applications. Results are combined as a
    weighted average.

    Parameters
    ----------
    image : ndarray, shape (H, W)
        Grayscale image.
    X0, Y0 : int
        Target pixel coordinates.
    half_width : int
        Half-width m of the line filter. Total length = 2m+1.
    np_count : int
        Neighbor count per WVF application.
    order : int
        Taylor expansion order.
    theta : float
        Line orientation in radians.
    use_weights : bool
        Whether to use Gaussian distance weighting.

    Returns
    -------
    f_x : float
        Weighted normal derivative.
    f_y : float
        Weighted tangential derivative.
    """
    H, W = image.shape
    img = image.astype(np.float64)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    total_fx = 0.0
    total_fy = 0.0
    total_weight = 0.0

    sigma = half_width / 2.0 if use_weights else 1.0

    for j in range(-half_width, half_width + 1):
        vx = X0 + j * cos_t
        vy = Y0 + j * sin_t

        vx_int = int(round(vx))
        vy_int = int(round(vy))

        if vx_int < 0 or vx_int >= W or vy_int < 0 or vy_int >= H:
            continue

        if use_weights:
            w = np.exp(-0.5 * (j / sigma) ** 2)
        else:
            w = 1.0

        fx, fy, _ = wvf_single_pixel(img, vx_int, vy_int,
                                       np_count=np_count, order=order,
                                       theta=theta)
        total_fx += w * fx
        total_fy += w * fy
        total_weight += w

    if total_weight > 0:
        total_fx /= total_weight
        total_fy /= total_weight

    return total_fx, total_fy


def lf_image(image, half_width=7, np_count=15, order=4, n_orientations=36,
             use_weights=True, subsample=1):
    """
    Apply Line Filter to entire image at multiple orientations.

    Parameters
    ----------
    image : ndarray, shape (H, W)
        Grayscale image.
    half_width : int
        LF half-width (m).
    np_count : int
        Neighbor count per WVF.
    order : int
        Taylor expansion order.
    n_orientations : int
        Number of orientations (Ns).
    use_weights : bool
        Gaussian weighting of virtual pixels.
    subsample : int
        Process every Nth pixel (for speed during testing).

    Returns
    -------
    gradient_mag : ndarray, shape (H, W)
        Maximum gradient magnitude.
    gradient_angle : ndarray, shape (H, W)
        Angle of maximum gradient.
    all_gradients : ndarray, shape (H, W, n_orientations)
        Gradient at each orientation (for cubic spline analysis).
    """
    H, W = image.shape
    img = image.astype(np.float64)

    gradient_mag = np.zeros((H, W))
    gradient_angle = np.zeros((H, W))
    all_gradients = np.zeros((H, W, n_orientations))

    angles = np.linspace(0, np.pi, n_orientations, endpoint=False)

    border = half_width + int(np.ceil(np.sqrt(np_count / np.pi))) + 2

    for y in range(border, H - border, subsample):
        for x in range(border, W - border, subsample):
            max_mag = 0.0
            best_angle = 0.0

            for ai, theta in enumerate(angles):
                fx, fy = line_filter_single_pixel(
                    img, x, y, half_width=half_width,
                    np_count=np_count, order=order,
                    theta=theta, use_weights=use_weights
                )
                mag = abs(fx)
                all_gradients[y, x, ai] = mag

                if mag > max_mag:
                    max_mag = mag
                    best_angle = theta

            gradient_mag[y, x] = max_mag
            gradient_angle[y, x] = best_angle

    return gradient_mag, gradient_angle, all_gradients


def cubic_spline_angle(gradients, angles):
    """
    Compute precise gradient orientation using periodic cubic splines.

    This is the angle detection method from Paper 3 (Section III.A).
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


def arctan_angle(grad_x, grad_y):
    """
    Traditional arctan-based angle computation for comparison.

    This is the conventional method used by Canny and others,
    which Bagan critiques as inaccurate.

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


def analyze_condition_numbers(np_counts, order=4, n_orientations=36):
    """
    Analyze how condition number of A^T A varies with Np and orientation.

    This tests a key concern: as Np grows large, does the least-squares
    system remain well-conditioned?

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


if __name__ == "__main__":
    print("WVF/LF implementation loaded successfully.")
    print("Testing basic functionality...")

    cond_results = analyze_condition_numbers([15, 50, 100, 250], order=4)
    for np_count, conds in cond_results.items():
        print(f"  Np={np_count}: condition number range "
              f"[{conds.min():.2e}, {conds.max():.2e}], "
              f"mean={conds.mean():.2e}")

    test_img = np.random.rand(64, 64) * 255
    fx, fy, cond = wvf_single_pixel(test_img, 32, 32, np_count=15, theta=0.0)
    print(f"  WVF single pixel test: fx={fx:.4f}, fy={fy:.4f}, cond={cond:.2e}")
    print("Done.")
