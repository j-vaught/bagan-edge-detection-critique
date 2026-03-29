"""Angle estimation methods for edge orientation."""

from edgecritic.angles.arctan import arctan_angle
from edgecritic.angles.spline import cubic_spline_angle

__all__ = ["cubic_spline_angle", "arctan_angle"]
