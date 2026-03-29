"""edgecritic: Wide View Filter and Line Filter for edge detection research.

Usage::

    from edgecritic import wvf_image, lf_image
    result = wvf_image(image, np_count=50, backend="auto")
    print(result.gradient_mag.max(), result.backend)
"""

__version__ = "0.1.0"

from edgecritic._types import EdgeResult
from edgecritic.angles import arctan_angle, cubic_spline_angle
from edgecritic.baselines import (
    canny_edges,
    compute_edges_at_threshold,
    extended_sobel_gradients,
    prewitt_gradients,
    sobel_gradients,
)
from edgecritic.evaluation import (
    angular_error_deg,
    compute_ods_ois,
    runtime_comparison,
)
from edgecritic.lf import lf_image
from edgecritic.synthetic import (
    create_multi_angle_line_image,
    create_parallel_line_image,
    create_step_edge_image,
)
from edgecritic.wvf import wvf_image

__all__ = [
    "EdgeResult",
    "wvf_image",
    "lf_image",
    "sobel_gradients",
    "prewitt_gradients",
    "extended_sobel_gradients",
    "canny_edges",
    "compute_edges_at_threshold",
    "compute_ods_ois",
    "angular_error_deg",
    "runtime_comparison",
    "cubic_spline_angle",
    "arctan_angle",
    "create_multi_angle_line_image",
    "create_parallel_line_image",
    "create_step_edge_image",
]
