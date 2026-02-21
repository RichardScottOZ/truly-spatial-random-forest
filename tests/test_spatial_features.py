"""Tests for spatial feature extraction."""

import numpy as np
import pytest

from spatial_rf.spatial_features import extract_spatial_features


@pytest.fixture
def sample_data():
    """Create simple test data with known spatial structure."""
    rng = np.random.RandomState(42)
    n_samples = 50
    n_features = 3
    X = rng.rand(n_samples, n_features)
    coords = rng.rand(n_samples, 2) * 100
    return X, coords


class TestExtractSpatialFeatures:
    def test_output_shape_knn(self, sample_data):
        X, coords = sample_data
        stats = ("mean", "std")
        X_spatial, names = extract_spatial_features(
            X, coords, neighbor_mode="knn", k=5, stats=stats
        )
        # Original features + 2 stats * 3 features = 3 + 6 = 9
        assert X_spatial.shape == (50, 9)
        assert len(names) == 9

    def test_output_shape_radius(self, sample_data):
        X, coords = sample_data
        stats = ("mean",)
        X_spatial, names = extract_spatial_features(
            X, coords, neighbor_mode="radius", radius=30.0, stats=stats
        )
        # Original 3 features + 1 stat * 3 features = 6
        assert X_spatial.shape == (50, 6)

    def test_output_shape_no_original(self, sample_data):
        X, coords = sample_data
        stats = ("mean", "std")
        X_spatial, names = extract_spatial_features(
            X, coords, k=5, stats=stats, include_original=False
        )
        # Only spatial: 2 stats * 3 features = 6
        assert X_spatial.shape == (50, 6)

    def test_default_stats(self, sample_data):
        X, coords = sample_data
        X_spatial, names = extract_spatial_features(X, coords, k=5)
        # Default: mean, std, min, max => 3 + 4*3 = 15
        assert X_spatial.shape == (50, 15)

    def test_feature_names(self, sample_data):
        X, coords = sample_data
        stats = ("mean",)
        X_spatial, names = extract_spatial_features(X, coords, k=5, stats=stats)
        assert "feat_0" in names
        assert "spatial_mean_feat_0" in names
        assert "spatial_mean_feat_2" in names

    def test_original_features_preserved(self, sample_data):
        X, coords = sample_data
        X_spatial, _ = extract_spatial_features(X, coords, k=5, stats=("mean",))
        # First 3 columns should be original features
        np.testing.assert_array_equal(X_spatial[:, :3], X)

    def test_spatial_mean_is_correct(self):
        """Verify spatial mean is computed correctly for a simple case."""
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        coords = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
        X_spatial, _ = extract_spatial_features(
            X, coords, k=2, stats=("mean",)
        )
        # For point at coord 2 (value=3), 2 nearest neighbors are at 1,3
        # with values 2,4 => mean=3.0
        assert X_spatial[2, 1] == pytest.approx(3.0)

    def test_invalid_stat_raises(self, sample_data):
        X, coords = sample_data
        with pytest.raises(ValueError, match="Unknown statistic"):
            extract_spatial_features(X, coords, k=5, stats=("invalid",))

    def test_invalid_neighbor_mode_raises(self, sample_data):
        X, coords = sample_data
        with pytest.raises(ValueError, match="Unknown neighbor_mode"):
            extract_spatial_features(X, coords, neighbor_mode="invalid")

    def test_radius_without_value_raises(self, sample_data):
        X, coords = sample_data
        with pytest.raises(ValueError, match="radius must be specified"):
            extract_spatial_features(X, coords, neighbor_mode="radius")

    def test_mismatched_shapes_raises(self, sample_data):
        X, coords = sample_data
        with pytest.raises(ValueError, match="samples"):
            extract_spatial_features(X, coords[:10], k=5)

    def test_parallel_matches_sequential(self, sample_data):
        X, coords = sample_data
        X_seq, _ = extract_spatial_features(X, coords, k=5, n_jobs=1)
        X_par, _ = extract_spatial_features(X, coords, k=5, n_jobs=2)
        np.testing.assert_array_almost_equal(X_seq, X_par)

    def test_all_supported_stats(self, sample_data):
        X, coords = sample_data
        stats = ("mean", "std", "min", "max", "median", "skew", "kurtosis", "count")
        X_spatial, names = extract_spatial_features(X, coords, k=5, stats=stats)
        # 3 original + 8 stats * 3 features = 27
        assert X_spatial.shape == (50, 27)
        assert len(names) == 27

    def test_1d_features(self):
        """Test with a single feature column."""
        X = np.random.rand(20)
        coords = np.random.rand(20, 2)
        X_spatial, names = extract_spatial_features(X, coords, k=3, stats=("mean",))
        assert X_spatial.shape == (20, 2)
