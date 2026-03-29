"""Classical edge detection baselines."""

from edgecritic.baselines.filters import (
    canny_edges,
    compute_edges_at_threshold,
    extended_sobel_gradients,
    prewitt_gradients,
    sobel_gradients,
)

__all__ = [
    "sobel_gradients",
    "prewitt_gradients",
    "extended_sobel_gradients",
    "canny_edges",
    "compute_edges_at_threshold",
]
