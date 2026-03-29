"""GPU-accelerated Line Filter using PyTorch CUDA.

Fuses all (2*half_width+1) line offsets into a single batched matmul
per orientation. Supports pixel batching to fit within a VRAM budget.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch

from edgecritic.core.taylor import get_circular_neighbors
from edgecritic.wvf._cuda import _precompute_pseudoinverses

logger = logging.getLogger("edgecritic")


def _estimate_batch_size(L: int, Np: int, vram_bytes: int) -> int:
    """Estimate how many pixels fit in one batch given a VRAM budget.

    Per batch of B pixels, peak memory is roughly:
      - index tensors: 2 * L * B * Np * 8  (int64 nb_y, nb_x)
      - virtual coords: 2 * L * B * 8      (int64 virtual_y, virtual_x)
      - gather tensor: L * B * Np * 4       (float32 F_all)
      - flat tensor:   L * B * Np * 4       (float32 F_flat, may coexist)
      - matmul result: L * B * 4            (float32 fx_all)
    Total ≈ B * L * (Np * 24 + 20)

    We use 50% of the budget to account for PyTorch allocator overhead,
    fragmentation, and the image tensor itself.
    """
    per_pixel = L * (Np * 24 + 20)
    batch = int(vram_bytes * 0.5 / max(per_pixel, 1))
    return max(batch, 256)  # minimum 256 pixels per batch


def lf_image_cuda(
    image: np.ndarray,
    half_width: int = 7,
    np_count: int = 15,
    order: int = 4,
    n_orientations: int = 36,
    device: str | torch.device = "cuda",
    max_vram_gb: float | None = None,
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
    max_vram_gb : float, optional
        Maximum VRAM to use in GB. If None, auto-detects from the device
        (uses 80% of free memory). Set this to control memory usage
        explicitly, e.g. ``max_vram_gb=30`` for a 40 GB GPU.

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
    flat_y = grid_y.reshape(-1).long()
    flat_x = grid_x.reshape(-1).long()
    N = int(flat_y.shape[0])

    dx = torch.tensor(offsets[:, 0], dtype=torch.long, device=device)
    dy = torch.tensor(offsets[:, 1], dtype=torch.long, device=device)

    # Determine batch size from VRAM budget
    if max_vram_gb is not None:
        vram_budget = int(max_vram_gb * 1e9)
    else:
        vram_budget = int(torch.cuda.mem_get_info(device)[0] * 0.8)

    batch_size = _estimate_batch_size(L, np_count, vram_budget)
    n_batches = (N + batch_size - 1) // batch_size

    t1 = time.perf_counter()
    if n_batches > 1:
        logger.debug(
            "lf_image_cuda: batching %d pixels into %d batches of %d "
            "(L=%d, Np=%d, budget=%.1fGB)",
            N, n_batches, batch_size, L, np_count, vram_budget / 1e9,
        )

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

        for bi in range(n_batches):
            s = bi * batch_size
            e = min(s + batch_size, N)
            by = flat_y[s:e]
            bx = flat_x[s:e]
            B = e - s

            # (L, B) virtual pixel positions along the line
            virtual_y = by.unsqueeze(0) + line_dy.unsqueeze(1)  # (L, B)
            virtual_x = bx.unsqueeze(0) + line_dx.unsqueeze(1)  # (L, B)

            # (L, B, Np) neighbor positions
            nb_y = virtual_y.unsqueeze(2) + dy.unsqueeze(0).unsqueeze(0)
            nb_x = virtual_x.unsqueeze(2) + dx.unsqueeze(0).unsqueeze(0)

            # Gather intensities and compute f_x
            F_all = img_t[nb_y, nb_x]           # (L, B, Np)
            del nb_y, nb_x, virtual_y, virtual_x
            F_flat = F_all.reshape(L * B, -1)   # (L*B, Np)
            del F_all
            fx_all = F_flat @ P_fx[oi]           # (L*B,)
            del F_flat

            # Weighted sum along the line
            fx_lines = fx_all.reshape(L, B)      # (L, B)
            del fx_all
            weighted_fx = (weights_t.unsqueeze(1) * fx_lines).sum(dim=0)  # (B,)
            del fx_lines
            response = torch.abs(weighted_fx)

            # Update best for this batch
            improved = response > best_response[s:e]
            best_response[s:e] = torch.where(improved, response, best_response[s:e])
            best_angle_idx[s:e] = torch.where(
                improved, torch.tensor(oi, device=device), best_angle_idx[s:e]
            )

    t2 = time.perf_counter()

    gradient_mag = np.zeros((H, W), dtype=np.float64)
    gradient_angle = np.zeros((H, W), dtype=np.float64)
    condition_numbers = np.zeros((H, W), dtype=np.float64)

    flat_y_np = flat_y.cpu().numpy()
    flat_x_np = flat_x.cpu().numpy()
    gradient_mag[flat_y_np, flat_x_np] = best_response.cpu().numpy()
    gradient_angle[flat_y_np, flat_x_np] = angles[best_angle_idx.cpu().numpy()]

    t3 = time.perf_counter()

    logger.debug(
        "lf_image_cuda: precompute=%.3fs compute=%.3fs transfer=%.3fs "
        "total=%.3fs pixels=%d orientations=%d line_length=%d batches=%d "
        "vram_budget=%.1fGB",
        t1 - t0, t2 - t1, t3 - t2, t3 - t0, N, n_orientations, L,
        n_batches, vram_budget / 1e9,
    )

    return gradient_mag, gradient_angle, condition_numbers
