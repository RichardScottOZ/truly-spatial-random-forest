"""Tests for SpatialRandomForestRegressor."""

import numpy as np
import pytest

from spatial_rf import SpatialRandomForestRegressor


@pytest.fixture
def regression_data():
    """Create a simple spatial regression dataset."""
    rng = np.random.RandomState(42)
    n_samples = 100
    coords = rng.rand(n_samples, 2) * 100
    X = rng.rand(n_samples, 3)
    # Target with spatial trend
    y = coords[:, 0] * 0.1 + X[:, 0] * 2 + rng.normal(0, 0.1, n_samples)
    return X, y, coords


class TestSpatialRandomForestRegressor:
    def test_fit_predict(self, regression_data):
        X, y, coords = regression_data
        reg = SpatialRandomForestRegressor(
            n_estimators=10, k=5, random_state=42
        )
        reg.fit(X, y, coordinates=coords)
        predictions = reg.predict(X, coordinates=coords)
        assert predictions.shape == y.shape

    def test_reasonable_predictions(self, regression_data):
        X, y, coords = regression_data
        reg = SpatialRandomForestRegressor(
            n_estimators=50, k=5, random_state=42
        )
        reg.fit(X, y, coordinates=coords)
        predictions = reg.predict(X, coordinates=coords)
        # R^2 on training data should be positive
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.5

    def test_feature_importances(self, regression_data):
        X, y, coords = regression_data
        reg = SpatialRandomForestRegressor(
            n_estimators=10, k=5, random_state=42
        )
        reg.fit(X, y, coordinates=coords)
        importances = reg.feature_importances_
        assert len(importances) == len(reg.feature_names_)

    def test_feature_names_stored(self, regression_data):
        X, y, coords = regression_data
        reg = SpatialRandomForestRegressor(
            n_estimators=10, k=5, stats=("mean", "std"), random_state=42
        )
        reg.fit(X, y, coordinates=coords)
        assert hasattr(reg, "feature_names_")
        # 3 original + 2 stats * 3 features = 9
        assert len(reg.feature_names_) == 9

    def test_radius_mode(self, regression_data):
        X, y, coords = regression_data
        reg = SpatialRandomForestRegressor(
            n_estimators=10, neighbor_mode="radius", radius=25.0, random_state=42
        )
        reg.fit(X, y, coordinates=coords)
        predictions = reg.predict(X, coordinates=coords)
        assert predictions.shape == y.shape

    def test_predict_before_fit_raises(self, regression_data):
        X, y, coords = regression_data
        reg = SpatialRandomForestRegressor()
        with pytest.raises(Exception):
            reg.predict(X, coordinates=coords)

    def test_without_original_features(self, regression_data):
        X, y, coords = regression_data
        reg = SpatialRandomForestRegressor(
            n_estimators=10, k=5, include_original=False, random_state=42
        )
        reg.fit(X, y, coordinates=coords)
        # Default stats: mean, std, min, max => 4 * 3 = 12
        assert reg.n_spatial_features_ == 12
