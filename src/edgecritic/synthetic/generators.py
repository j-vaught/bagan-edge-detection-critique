"""Synthetic test image generation for verifying WVF/LF behavior."""

import numpy as np
from pathlib import Path


def create_multi_angle_line_image(size=256, line_width=2, angles_deg=None,
                                  snr=2.0, background=128):
    """Create a synthetic image with lines at multiple known angles.

    Parameters
    ----------
    size : int
        Image size (square).
    line_width : int
        Width of each line in pixels.
    angles_deg : list of float
        Line angles in degrees. Default: [0, 23, 63.5, 90, 135, 174].
    snr : float
        Signal-to-noise ratio. SNR = signal_amplitude / noise_std.
    background : float
        Background intensity (0-255).

    Returns
    -------
    image : ndarray, shape (size, size)
        Synthetic image with noise.
    clean_image : ndarray, shape (size, size)
        Image without noise (ground truth).
    angle_map : ndarray, shape (size, size)
        True edge angle at each pixel (-1 for non-edge pixels).
    """
    if angles_deg is None:
        angles_deg = [0, 23, 63.5, 90, 135, 174]

    clean_image = np.full((size, size), background, dtype=np.float64)
    angle_map = np.full((size, size), -1.0)

    cx, cy = size // 2, size // 2
    signal_amplitude = background * 0.8

    for angle_deg in angles_deg:
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        for t in np.linspace(-size, size, size * 4):
            px = cx + t * cos_a
            py = cy + t * sin_a

            for dw in range(-line_width // 2, line_width // 2 + 1):
                lx = int(round(px - dw * sin_a))
                ly = int(round(py + dw * cos_a))

                if 0 <= lx < size and 0 <= ly < size:
                    clean_image[ly, lx] = background + signal_amplitude
                    edge_normal = (angle_deg + 90) % 180
                    angle_map[ly, lx] = edge_normal

    if snr > 0:
        noise_std = signal_amplitude / snr
        noise = np.random.normal(0, noise_std, (size, size))
        image = clean_image + noise
        image = np.clip(image, 0, 255)
    else:
        image = clean_image.copy()

    return image, clean_image, angle_map


def create_parallel_line_image(size=256, n_lines=5, spacing=30,
                                angle_deg=90, line_width=2,
                                snr=2.0, background=128):
    """Create image with parallel lines at a single angle.

    Parameters
    ----------
    size : int
        Image size.
    n_lines : int
        Number of parallel lines.
    spacing : int
        Pixel spacing between lines.
    angle_deg : float
        Orientation of all lines.
    line_width : int
        Width of each line.
    snr : float
        Signal-to-noise ratio.
    background : float
        Background intensity.

    Returns
    -------
    image : ndarray
        Noisy image.
    clean_image : ndarray
        Clean image.
    true_angle : float
        Known edge normal angle in degrees.
    """
    clean_image = np.full((size, size), background, dtype=np.float64)
    signal_amplitude = background * 0.8

    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    cx, cy = size // 2, size // 2

    for i in range(n_lines):
        offset = (i - n_lines // 2) * spacing

        base_x = cx + offset * sin_a
        base_y = cy - offset * cos_a

        for t in np.linspace(-size, size, size * 4):
            px = base_x + t * cos_a
            py = base_y + t * sin_a

            for dw in range(-line_width // 2, line_width // 2 + 1):
                lx = int(round(px - dw * sin_a))
                ly = int(round(py + dw * cos_a))

                if 0 <= lx < size and 0 <= ly < size:
                    clean_image[ly, lx] = background + signal_amplitude

    if snr > 0:
        noise_std = signal_amplitude / snr
        noise = np.random.normal(0, noise_std, (size, size))
        image = clean_image + noise
        image = np.clip(image, 0, 255)
    else:
        image = clean_image.copy()

    true_angle = (angle_deg + 90) % 180
    return image, clean_image, true_angle


def create_step_edge_image(size=256, edge_angle_deg=45, snr=2.0,
                            high_val=200, low_val=50):
    """Create a simple step edge for gradient magnitude testing.

    Parameters
    ----------
    size : int
        Image size.
    edge_angle_deg : float
        Angle of the edge.
    snr : float
        Signal-to-noise ratio.
    high_val, low_val : float
        Intensity values on each side of the edge.

    Returns
    -------
    image : ndarray
        Noisy step edge image.
    clean_image : ndarray
        Clean step edge.
    true_angle : float
        True edge normal angle in degrees.
    """
    clean_image = np.full((size, size), low_val, dtype=np.float64)

    angle_rad = np.radians(edge_angle_deg)

    cx, cy = size // 2, size // 2

    for y in range(size):
        for x in range(size):
            dx = x - cx
            dy = y - cy
            proj = dx * np.cos(angle_rad) + dy * np.sin(angle_rad)
            if proj > 0:
                clean_image[y, x] = high_val

    signal_amplitude = high_val - low_val
    if snr > 0:
        noise_std = signal_amplitude / snr
        noise = np.random.normal(0, noise_std, (size, size))
        image = clean_image + noise
        image = np.clip(image, 0, 255)
    else:
        image = clean_image.copy()

    true_angle = edge_angle_deg % 180
    return image, clean_image, true_angle


def generate_all_test_images(output_dir="results"):
    """Generate the full suite of synthetic test images.

    Parameters
    ----------
    output_dir : str
        Directory to save generated images as .npy files.
    """
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    test_angles = [0, 23, 63.5, 90, 135, 174]
    snr_levels = [0.5, 0.75, 1.0, 2.0]

    print("Generating multi-angle line images...")
    for snr in snr_levels:
        img, clean, angle_map = create_multi_angle_line_image(
            size=256, angles_deg=test_angles, snr=snr
        )
        np.save(out / f"multiline_snr{snr:.2f}.npy", img)
        np.save(out / f"multiline_snr{snr:.2f}_clean.npy", clean)
        np.save(out / f"multiline_snr{snr:.2f}_angles.npy", angle_map)
        print(f"  SNR={snr}: saved ({img.shape})")

    print("Generating parallel line images...")
    for angle in [0, 45, 90, 135]:
        img, clean, true_angle = create_parallel_line_image(
            size=256, angle_deg=angle, snr=2.0
        )
        np.save(out / f"parallel_{angle}deg.npy", img)
        print(f"  angle={angle}deg: true edge normal={true_angle}deg")

    print("Generating step edge images...")
    for angle in [0, 30, 45, 60, 90, 120, 150]:
        img, clean, true_angle = create_step_edge_image(
            size=256, edge_angle_deg=angle, snr=2.0
        )
        np.save(out / f"step_edge_{angle}deg.npy", img)
        print(f"  edge angle={angle}deg")

    print("All test images generated.")
