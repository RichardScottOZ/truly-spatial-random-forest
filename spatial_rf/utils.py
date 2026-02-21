"""
Utility functions for spatial neighbor search and validation.
"""

import numpy as np
from scipy.spatial import cKDTree


def find_neighbors_radius(coordinates, radius):
    """Find neighbors within a radius for each point using a KD-Tree.

    Parameters
    ----------
    coordinates : array-like of shape (n_samples, n_dims)
        Spatial coordinates of each sample.
    radius : float
        Search radius for neighbor lookup.

    Returns
    -------
    neighbors : list of arrays
        For each point, an array of indices of its neighbors (excluding itself).
    """
    tree = cKDTree(coordinates)
    neighbors = tree.query_ball_point(coordinates, r=radius)
    # Remove self from neighbors
    for i in range(len(neighbors)):
        neighbors[i] = [j for j in neighbors[i] if j != i]
    return neighbors


def find_neighbors_knn(coordinates, k):
    """Find k-nearest neighbors for each point using a KD-Tree.

    Parameters
    ----------
    coordinates : array-like of shape (n_samples, n_dims)
        Spatial coordinates of each sample.
    k : int
        Number of nearest neighbors.

    Returns
    -------
    neighbors : list of arrays
        For each point, an array of indices of its k-nearest neighbors.
    """
    tree = cKDTree(coordinates)
    # k+1 because query includes self
    _, indices = tree.query(coordinates, k=k + 1)
    # Remove self (first column)
    neighbors = [indices[i, 1:] for i in range(len(indices))]
    return neighbors


def validate_coordinates(coordinates, X):
    """Validate that coordinates and features have compatible shapes.

    Parameters
    ----------
    coordinates : array-like
        Spatial coordinates.
    X : array-like
        Feature matrix.

    Returns
    -------
    coordinates : ndarray of shape (n_samples, n_dims)
    """
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim == 1:
        coordinates = coordinates.reshape(-1, 1)
    if coordinates.shape[0] != X.shape[0]:
        raise ValueError(
            f"coordinates has {coordinates.shape[0]} samples but X has "
            f"{X.shape[0]} samples."
        )
    return coordinates
