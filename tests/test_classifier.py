"""Tests for SpatialRandomForestClassifier."""

import numpy as np
import pytest

from spatial_rf import SpatialRandomForestClassifier


@pytest.fixture
def classification_data():
    """Create a simple spatial classification dataset."""
    rng = np.random.RandomState(42)
    n_samples = 100
    coords = rng.rand(n_samples, 2) * 100
    X = rng.rand(n_samples, 4)
    # Label based on spatial location (left vs right half)
    y = (coords[:, 0] > 50).astype(int)
    return X, y, coords


class TestSpatialRandomForestClassifier:
    def test_fit_predict(self, classification_data):
        X, y, coords = classification_data
        clf = SpatialRandomForestClassifier(
            n_estimators=10, k=5, random_state=42
        )
        clf.fit(X, y, coordinates=coords)
        predictions = clf.predict(X, coordinates=coords)
        assert predictions.shape == y.shape
        assert set(predictions).issubset({0, 1})

    def test_predict_proba(self, classification_data):
        X, y, coords = classification_data
        clf = SpatialRandomForestClassifier(
            n_estimators=10, k=5, random_state=42
        )
        clf.fit(X, y, coordinates=coords)
        proba = clf.predict_proba(X, coordinates=coords)
        assert proba.shape == (100, 2)
        np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0)

    def test_feature_importances(self, classification_data):
        X, y, coords = classification_data
        clf = SpatialRandomForestClassifier(
            n_estimators=10, k=5, random_state=42
        )
        clf.fit(X, y, coordinates=coords)
        importances = clf.feature_importances_
        assert len(importances) == len(clf.feature_names_)
        assert all(imp >= 0 for imp in importances)

    def test_feature_names_stored(self, classification_data):
        X, y, coords = classification_data
        clf = SpatialRandomForestClassifier(
            n_estimators=10, k=5, stats=("mean",), random_state=42
        )
        clf.fit(X, y, coordinates=coords)
        assert hasattr(clf, "feature_names_")
        # 4 original + 1 stat * 4 features = 8
        assert len(clf.feature_names_) == 8

    def test_radius_mode(self, classification_data):
        X, y, coords = classification_data
        clf = SpatialRandomForestClassifier(
            n_estimators=10, neighbor_mode="radius", radius=25.0, random_state=42
        )
        clf.fit(X, y, coordinates=coords)
        predictions = clf.predict(X, coordinates=coords)
        assert predictions.shape == y.shape

    def test_predict_before_fit_raises(self, classification_data):
        X, y, coords = classification_data
        clf = SpatialRandomForestClassifier()
        with pytest.raises(Exception):
            clf.predict(X, coordinates=coords)

    def test_n_spatial_features(self, classification_data):
        X, y, coords = classification_data
        clf = SpatialRandomForestClassifier(
            n_estimators=10, k=5, stats=("mean", "std"), random_state=42
        )
        clf.fit(X, y, coordinates=coords)
        # 2 stats * 4 features = 8 spatial features
        assert clf.n_spatial_features_ == 8

    def test_custom_rf_kwargs(self, classification_data):
        X, y, coords = classification_data
        clf = SpatialRandomForestClassifier(
            n_estimators=10, k=5, random_state=42, max_depth=3
        )
        clf.fit(X, y, coordinates=coords)
        assert clf.rf_.max_depth == 3
