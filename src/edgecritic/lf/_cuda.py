"""GPU-accelerated Line Filter using PyTorch CUDA.

Fuses all (2*half_width+1) line offsets into a single batched matmul
per orientation, fully saturating the GPU.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch

from edgecritic.core.taylor import get_circular_neighbors
from edgecritic.wvf._cuda import _precompute_pseudoinverses

logger = logging.getLogger("edgecritic")


def lf_image_cuda(
    image: np.ndarray,
    half_width: int = 7,
    np_count: int = 15,
    order: int = 4,
    n_orientations: int = 36,
    device: str | torch.device = "cuda",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GPU-accelerated Line Filter.

    Parameters
    ----------
    image : ndarray, shape (H, W)
        Grayscale image.
    half_width : int
        Half-width m of the line filter. Total line length = 2m+1.
    np_count : int
        Number of neighbor pixels per WVF application.
    order : int
        Taylor expansion order.
    n_orientations : int
        Number of orientations to sweep.
    device : str
        PyTorch device.

    Returns
    -------
    gradient_mag : ndarray, shape (H, W)
    gradient_angle : ndarray, shape (H, W)
    condition_numbers : ndarray, shape (H, W)
    """
    device = torch.device(device)
    H, W = image.shape
    img_t = torch.tensor(image, dtype=torch.float32, device=device)

    global_neighbors = get_circular_neighbors(np_count)
    offsets = global_neighbors.astype(int)
    neighbor_radius = int(np.max(np.abs(offsets)))
    border = neighbor_radius + half_width + 1

    angles = np.linspace(0, 2 * np.pi, n_orientations, endpoint=False)

    t0 = time.perf_counter()
    P_all = _precompute_pseudoinverses(global_neighbors, angles, order, device)
    P_fx = P_all[:, 1, :]

    L = 2 * half_width + 1
    sigma = half_width / 2.0
    j_offsets = np.arange(-half_width, half_width + 1, dtype=np.float64)
    weights = np.exp(-0.5 * (j_offsets / sigma) ** 2)
    weights /= weights.sum()
    weights_t = torch.tensor(weights, dtype=torch.float32, device=device)

    ys = torch.arange(border, H - border, device=device)
    xs = torch.arange(border, W - border, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    flat_y = grid_y.reshape(-1).float()
    flat_x = grid_x.reshape(-1).float()
    N = int(flat_y.shape[0])

    dx = torch.tensor(offsets[:, 0], dtype=torch.long, device=device)
    dy = torch.tensor(offsets[:, 1], dtype=torch.long, device=device)

    t1 = time.perf_counter()

    best_response = torch.zeros(N, device=device)
    best_angle_idx = torch.zeros(N, dtype=torch.long, device=device)

    for oi, theta in enumerate(angles):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        line_dx = torch.tensor(
            [round(j * cos_t) for j in range(-half_width, half_width + 1)],
            dtype=torch.long, device=device,
        )
        line_dy = torch.tensor(
            [round(j * sin_t) for j in range(-half_width, half_width + 1)],
            dtype=torch.long, device=device,
        )

        flat_y_long = flat_y.long()
        flat_x_long = flat_x.long()

        virtual_y = flat_y_long.unsqueeze(0) + line_dy.unsqueeze(1)
        virtual_x = flat_x_long.unsqueeze(0) + line_dx.unsqueeze(1)

        nb_y = virtual_y.unsqueeze(2) + dy.unsqueeze(0).unsqueeze(0)
        nb_x = virtual_x.unsqueeze(2) + dx.unsqueeze(0).unsqueeze(0)

        F_all = img_t[nb_y, nb_x]

        F_flat = F_all.reshape(L * N, -1)

        fx_all = F_flat @ P_fx[oi]

        fx_lines = fx_all.reshape(L, N)
        weighted_fx = (weights_t.unsqueeze(1) * fx_lines).sum(dim=0)
        response = torch.abs(weighted_fx)

        improved = response > best_response
        best_response = torch.where(improved, response, best_response)
        best_angle_idx = torch.where(improved, torch.tensor(oi, device=device), best_angle_idx)

    t2 = time.perf_counter()

    gradient_mag = np.zeros((H, W), dtype=np.float64)
    gradient_angle = np.zeros((H, W), dtype=np.float64)
    condition_numbers = np.zeros((H, W), dtype=np.float64)

    flat_y_np = flat_y.long().cpu().numpy()
    flat_x_np = flat_x.long().cpu().numpy()
    gradient_mag[flat_y_np, flat_x_np] = best_response.cpu().numpy()
    gradient_angle[flat_y_np, flat_x_np] = angles[best_angle_idx.cpu().numpy()]

    t3 = time.perf_counter()

    logger.debug(
        "lf_image_cuda: precompute=%.3fs compute=%.3fs transfer=%.3fs "
        "total=%.3fs pixels=%d orientations=%d line_length=%d",
        t1 - t0, t2 - t1, t3 - t2, t3 - t0, N, n_orientations, L,
    )

    return gradient_mag, gradient_angle, condition_numbers
