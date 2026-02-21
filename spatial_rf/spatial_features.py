"""
Spatial feature extraction for the Truly Spatial Random Forest algorithm.

Computes neighborhood summary statistics (mean, variance, skewness, etc.)
to capture local spatial patterns as described in Talebi et al. (2022).
"""

import numpy as np
from scipy import stats as scipy_stats
from joblib import Parallel, delayed

from spatial_rf.utils import (
    find_neighbors_knn,
    find_neighbors_radius,
    validate_coordinates,
)


_DEFAULT_STATS = ("mean", "std", "min", "max")

_SUPPORTED_STATS = ("mean", "std", "min", "max", "median", "skew", "kurtosis", "count")


def _compute_stats_for_point(X, neighbor_idx, stat_funcs):
    """Compute summary statistics for a single point's neighborhood."""
    if len(neighbor_idx) == 0:
        return np.full(X.shape[1] * len(stat_funcs), np.nan)

    neighborhood = X[neighbor_idx]
    features = []
    for func in stat_funcs:
        features.append(func(neighborhood, axis=0))
    return np.concatenate(features)


def _get_stat_func(name):
    """Return a callable for the given statistic name."""
    if name == "mean":
        return lambda arr, axis: np.nanmean(arr, axis=axis)
    elif name == "std":
        return lambda arr, axis: np.nanstd(arr, axis=axis)
    elif name == "min":
        return lambda arr, axis: np.nanmin(arr, axis=axis)
    elif name == "max":
        return lambda arr, axis: np.nanmax(arr, axis=axis)
    elif name == "median":
        return lambda arr, axis: np.nanmedian(arr, axis=axis)
    elif name == "skew":
        return lambda arr, axis: scipy_stats.skew(arr, axis=axis, nan_policy="omit")
    elif name == "kurtosis":
        return lambda arr, axis: scipy_stats.kurtosis(
            arr, axis=axis, nan_policy="omit"
        )
    elif name == "count":
        return lambda arr, axis: np.sum(~np.isnan(arr), axis=axis).astype(float)
    else:
        raise ValueError(
            f"Unknown statistic '{name}'. Supported: {_SUPPORTED_STATS}"
        )


def extract_spatial_features(
    X,
    coordinates,
    neighbor_mode="knn",
    k=10,
    radius=None,
    stats=None,
    include_original=True,
    n_jobs=1,
):
    """Extract spatial neighborhood features for each sample.

    For each sample, finds its spatial neighbors and computes summary
    statistics over the neighborhood for each original feature. This
    implements the spatial feature vectorization from Talebi et al. (2022).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Original feature matrix.
    coordinates : array-like of shape (n_samples, n_dims)
        Spatial coordinates of each sample.
    neighbor_mode : {'knn', 'radius'}, default='knn'
        Method for defining neighborhoods:
        - 'knn': k-nearest neighbors
        - 'radius': all neighbors within a fixed radius
    k : int, default=10
        Number of nearest neighbors (used when neighbor_mode='knn').
    radius : float, optional
        Search radius (used when neighbor_mode='radius').
    stats : tuple of str, optional
        Summary statistics to compute. Supported values:
        'mean', 'std', 'min', 'max', 'median', 'skew', 'kurtosis', 'count'.
        Default is ('mean', 'std', 'min', 'max').
    include_original : bool, default=True
        Whether to prepend the original features to the spatial features.
    n_jobs : int, default=1
        Number of parallel jobs for feature computation.
        Use -1 for all available cores.

    Returns
    -------
    X_spatial : ndarray
        Extended feature matrix with spatial summary statistics appended.
    feature_names : list of str
        Names of the generated spatial features.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    coordinates = validate_coordinates(coordinates, X)

    if stats is None:
        stats = _DEFAULT_STATS
    for s in stats:
        if s not in _SUPPORTED_STATS:
            raise ValueError(
                f"Unknown statistic '{s}'. Supported: {_SUPPORTED_STATS}"
            )

    # Find neighbors
    if neighbor_mode == "knn":
        neighbors = find_neighbors_knn(coordinates, k=k)
    elif neighbor_mode == "radius":
        if radius is None:
            raise ValueError("radius must be specified when neighbor_mode='radius'")
        neighbors = find_neighbors_radius(coordinates, radius=radius)
    else:
        raise ValueError(
            f"Unknown neighbor_mode '{neighbor_mode}'. Use 'knn' or 'radius'."
        )

    stat_funcs = [_get_stat_func(s) for s in stats]

    # Compute spatial features
    if n_jobs == 1:
        spatial_features = np.array(
            [_compute_stats_for_point(X, nb, stat_funcs) for nb in neighbors]
        )
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_compute_stats_for_point)(X, nb, stat_funcs) for nb in neighbors
        )
        spatial_features = np.array(results)

    # Build feature names
    n_features = X.shape[1]
    feature_names = []
    if include_original:
        feature_names.extend([f"feat_{i}" for i in range(n_features)])
    for stat_name in stats:
        for i in range(n_features):
            feature_names.append(f"spatial_{stat_name}_feat_{i}")

    # Combine original features with spatial features
    if include_original:
        X_spatial = np.hstack([X, spatial_features])
    else:
        X_spatial = spatial_features

    return X_spatial, feature_names
