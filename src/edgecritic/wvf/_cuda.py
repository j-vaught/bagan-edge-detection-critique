"""GPU-accelerated Wide View Filter using PyTorch CUDA.

Precomputes the pseudoinverse P_theta = (A^T A)^{-1} A^T for each
orientation, then gathers all neighbor intensities as a batched tensor
and runs one batched matmul per orientation.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch

from edgecritic.core.taylor import (
    build_taylor_matrix,
    get_circular_neighbors,
    rotate_coordinates,
)

logger = logging.getLogger("edgecritic")


def _precompute_pseudoinverses(
    global_neighbors: np.ndarray,
    angles: np.ndarray,
    order: int,
    device: torch.device,
) -> torch.Tensor:
    """Precompute P_theta for each orientation. Returns (n_orientations, M, Np)."""
    pinvs = []
    for theta in angles:
        local_coords = rotate_coordinates(global_neighbors, theta)
        A = build_taylor_matrix(local_coords, order=order)
        P = np.linalg.pinv(A)
        pinvs.append(P)
    return torch.tensor(np.stack(pinvs), dtype=torch.float32, device=device)


def wvf_image_cuda(
    image: np.ndarray,
    np_count: int = 15,
    order: int = 4,
    n_orientations: int = 36,
    device: str | torch.device = "cuda",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GPU-accelerated WVF.

    Parameters
    ----------
    image : ndarray, shape (H, W)
        Grayscale image.
    np_count : int
        Number of neighbor pixels.
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
    border = int(np.max(np.abs(offsets))) + 1

    angles = np.linspace(0, 2 * np.pi, n_orientations, endpoint=False)

    t0 = time.perf_counter()
    P_all = _precompute_pseudoinverses(global_neighbors, angles, order, device)

    ys = torch.arange(border, H - border, device=device)
    xs = torch.arange(border, W - border, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    flat_y = grid_y.reshape(-1)
    flat_x = grid_x.reshape(-1)
    N = flat_y.shape[0]

    dx = torch.tensor(offsets[:, 0], dtype=torch.long, device=device)
    dy = torch.tensor(offsets[:, 1], dtype=torch.long, device=device)

    neighbor_y = flat_y.unsqueeze(1) + dy.unsqueeze(0)
    neighbor_x = flat_x.unsqueeze(1) + dx.unsqueeze(0)
    F = img_t[neighbor_y, neighbor_x]

    t1 = time.perf_counter()

    P_fx = P_all[:, 1, :]
    responses = torch.abs(P_fx @ F.T)

    t2 = time.perf_counter()

    best_mag, best_idx = responses.max(dim=0)

    gradient_mag = np.zeros((H, W), dtype=np.float64)
    gradient_angle = np.zeros((H, W), dtype=np.float64)
    condition_numbers = np.zeros((H, W), dtype=np.float64)

    flat_y_np = flat_y.cpu().numpy()
    flat_x_np = flat_x.cpu().numpy()
    gradient_mag[flat_y_np, flat_x_np] = best_mag.cpu().numpy()
    gradient_angle[flat_y_np, flat_x_np] = angles[best_idx.cpu().numpy()]

    t3 = time.perf_counter()

    logger.debug(
        "wvf_image_cuda: precompute=%.3fs gather=%.3fs matmul=%.3fs "
        "transfer=%.3fs total=%.3fs pixels=%d orientations=%d",
        t1 - t0, t1 - t0, t2 - t1, t3 - t2, t3 - t0, N, n_orientations,
    )

    return gradient_mag, gradient_angle, condition_numbers


def wvf_image_cuda_batched(
    image: np.ndarray,
    np_count: int = 15,
    order: int = 4,
    n_orientations: int = 36,
    device: str | torch.device = "cuda",
    pixel_batch_size: int = 500_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Memory-safe GPU WVF that processes pixels in batches."""
    device = torch.device(device)
    H, W = image.shape
    img_t = torch.tensor(image, dtype=torch.float32, device=device)

    global_neighbors = get_circular_neighbors(np_count)
    offsets = global_neighbors.astype(int)
    border = int(np.max(np.abs(offsets))) + 1

    angles = np.linspace(0, 2 * np.pi, n_orientations, endpoint=False)
    P_all = _precompute_pseudoinverses(global_neighbors, angles, order, device)
    P_fx = P_all[:, 1, :]

    ys = torch.arange(border, H - border, device=device)
    xs = torch.arange(border, W - border, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    flat_y = grid_y.reshape(-1)
    flat_x = grid_x.reshape(-1)
    N = flat_y.shape[0]

    dx = torch.tensor(offsets[:, 0], dtype=torch.long, device=device)
    dy = torch.tensor(offsets[:, 1], dtype=torch.long, device=device)

    gradient_mag = np.zeros((H, W), dtype=np.float64)
    gradient_angle = np.zeros((H, W), dtype=np.float64)
    condition_numbers = np.zeros((H, W), dtype=np.float64)

    t0 = time.perf_counter()
    for start in range(0, N, pixel_batch_size):
        end = min(start + pixel_batch_size, N)
        by = flat_y[start:end]
        bx = flat_x[start:end]

        neighbor_y = by.unsqueeze(1) + dy.unsqueeze(0)
        neighbor_x = bx.unsqueeze(1) + dx.unsqueeze(0)
        F = img_t[neighbor_y, neighbor_x]

        responses = torch.abs(P_fx @ F.T)
        best_mag, best_idx = responses.max(dim=0)

        by_np = by.cpu().numpy()
        bx_np = bx.cpu().numpy()
        gradient_mag[by_np, bx_np] = best_mag.cpu().numpy()
        gradient_angle[by_np, bx_np] = angles[best_idx.cpu().numpy()]

    t1 = time.perf_counter()
    logger.debug(
        "wvf_image_cuda_batched: total=%.3fs pixels=%d orientations=%d batches=%d",
        t1 - t0, N, n_orientations, (N + pixel_batch_size - 1) // pixel_batch_size,
    )

    return gradient_mag, gradient_angle, condition_numbers
