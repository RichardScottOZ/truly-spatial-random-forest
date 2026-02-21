"""Tests for spatial utility functions."""

import numpy as np
import pytest

from spatial_rf.utils import (
    find_neighbors_knn,
    find_neighbors_radius,
    validate_coordinates,
)


class TestFindNeighborsKnn:
    def test_correct_count(self):
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        neighbors = find_neighbors_knn(coords, k=2)
        for nb in neighbors:
            assert len(nb) == 2

    def test_self_excluded(self):
        coords = np.array([[0.0], [1.0], [2.0]])
        neighbors = find_neighbors_knn(coords, k=1)
        for i, nb in enumerate(neighbors):
            assert i not in nb


class TestFindNeighborsRadius:
    def test_correct_neighbors(self):
        coords = np.array([[0.0], [1.0], [2.0], [10.0]])
        neighbors = find_neighbors_radius(coords, radius=1.5)
        # Point 0: neighbors within 1.5 => point 1
        assert 1 in neighbors[0]
        assert 2 not in neighbors[0]
        # Point 10: no neighbors within 1.5
        assert len(neighbors[3]) == 0

    def test_self_excluded(self):
        coords = np.array([[0.0], [1.0], [2.0]])
        neighbors = find_neighbors_radius(coords, radius=5.0)
        for i, nb in enumerate(neighbors):
            assert i not in nb


class TestValidateCoordinates:
    def test_valid_2d(self):
        X = np.random.rand(10, 3)
        coords = np.random.rand(10, 2)
        result = validate_coordinates(coords, X)
        assert result.shape == (10, 2)

    def test_valid_1d(self):
        X = np.random.rand(10, 3)
        coords = np.random.rand(10)
        result = validate_coordinates(coords, X)
        assert result.shape == (10, 1)

    def test_mismatch_raises(self):
        X = np.random.rand(10, 3)
        coords = np.random.rand(5, 2)
        with pytest.raises(ValueError, match="samples"):
            validate_coordinates(coords, X)
