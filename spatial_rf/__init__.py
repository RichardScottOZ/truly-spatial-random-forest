"""
Truly Spatial Random Forests (SRF) for geoscience data analysis.

Implementation of the algorithm described in:
    Talebi, H., Peeters, L.J.M., Otto, A. & Tolosana-Delgado, R. (2022).
    A Truly Spatial Random Forests Algorithm for Geoscience Data Analysis
    and Modelling. Mathematical Geosciences, 54, 1–22.
    https://doi.org/10.1007/s11004-021-09946-w
"""

from spatial_rf.classifier import SpatialRandomForestClassifier
from spatial_rf.regressor import SpatialRandomForestRegressor
from spatial_rf.spatial_features import extract_spatial_features

__all__ = [
    "SpatialRandomForestClassifier",
    "SpatialRandomForestRegressor",
    "extract_spatial_features",
]

__version__ = "0.1.0"
