"""
Spatial Random Forest Classifier.

Wraps scikit-learn's RandomForestClassifier with automatic spatial
feature extraction as described in Talebi et al. (2022).
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted

from spatial_rf.spatial_features import extract_spatial_features


class SpatialRandomForestClassifier(BaseEstimator, ClassifierMixin):
    """Random Forest classifier with spatial feature augmentation.

    Implements the Truly Spatial Random Forest (SRF) algorithm for
    classification. For each sample, spatial neighborhood statistics are
    computed and appended to the original features before training.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    neighbor_mode : {'knn', 'radius'}, default='knn'
        Method for defining spatial neighborhoods.
    k : int, default=10
        Number of nearest neighbors (when neighbor_mode='knn').
    radius : float, optional
        Search radius (when neighbor_mode='radius').
    stats : tuple of str, optional
        Summary statistics to compute over neighborhoods.
        Default is ('mean', 'std', 'min', 'max').
    include_original : bool, default=True
        Whether to include original features alongside spatial features.
    n_jobs_features : int, default=1
        Number of parallel jobs for spatial feature extraction.
    n_jobs : int, default=None
        Number of parallel jobs for the Random Forest (passed to sklearn).
    random_state : int or None, default=None
        Random state for reproducibility.
    **rf_kwargs
        Additional keyword arguments passed to
        ``sklearn.ensemble.RandomForestClassifier``.

    Attributes
    ----------
    rf_ : RandomForestClassifier
        The fitted scikit-learn Random Forest classifier.
    feature_names_ : list of str
        Names of features used during training (original + spatial).
    n_spatial_features_ : int
        Number of spatial features generated.

    References
    ----------
    Talebi, H., Peeters, L.J.M., Otto, A. & Tolosana-Delgado, R. (2022).
    A Truly Spatial Random Forests Algorithm for Geoscience Data Analysis
    and Modelling. Mathematical Geosciences, 54, 1–22.

    Examples
    --------
    >>> import numpy as np
    >>> from spatial_rf import SpatialRandomForestClassifier
    >>> X = np.random.rand(100, 3)
    >>> coords = np.random.rand(100, 2) * 100
    >>> y = (X[:, 0] > 0.5).astype(int)
    >>> clf = SpatialRandomForestClassifier(k=5, random_state=42)
    >>> clf.fit(X, y, coordinates=coords)
    SpatialRandomForestClassifier(k=5, random_state=42)
    >>> predictions = clf.predict(X, coordinates=coords)
    """

    def __init__(
        self,
        n_estimators=100,
        neighbor_mode="knn",
        k=10,
        radius=None,
        stats=None,
        include_original=True,
        n_jobs_features=1,
        n_jobs=None,
        random_state=None,
        **rf_kwargs,
    ):
        self.n_estimators = n_estimators
        self.neighbor_mode = neighbor_mode
        self.k = k
        self.radius = radius
        self.stats = stats
        self.include_original = include_original
        self.n_jobs_features = n_jobs_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.rf_kwargs = rf_kwargs

    def fit(self, X, y, coordinates):
        """Fit the Spatial Random Forest classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Target class labels.
        coordinates : array-like of shape (n_samples, n_dims)
            Spatial coordinates of training samples.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        X_spatial, self.feature_names_ = extract_spatial_features(
            X,
            coordinates,
            neighbor_mode=self.neighbor_mode,
            k=self.k,
            radius=self.radius,
            stats=self.stats,
            include_original=self.include_original,
            n_jobs=self.n_jobs_features,
        )

        n_orig = X.shape[1] if self.include_original else 0
        self.n_spatial_features_ = X_spatial.shape[1] - n_orig

        self.rf_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            **self.rf_kwargs,
        )
        self.rf_.fit(X_spatial, y)
        return self

    def predict(self, X, coordinates):
        """Predict class labels for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        coordinates : array-like of shape (n_samples, n_dims)
            Spatial coordinates of samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self, "rf_")
        X_spatial, _ = extract_spatial_features(
            X,
            coordinates,
            neighbor_mode=self.neighbor_mode,
            k=self.k,
            radius=self.radius,
            stats=self.stats,
            include_original=self.include_original,
            n_jobs=self.n_jobs_features,
        )
        return self.rf_.predict(X_spatial)

    def predict_proba(self, X, coordinates):
        """Predict class probabilities for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        coordinates : array-like of shape (n_samples, n_dims)
            Spatial coordinates of samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self, "rf_")
        X_spatial, _ = extract_spatial_features(
            X,
            coordinates,
            neighbor_mode=self.neighbor_mode,
            k=self.k,
            radius=self.radius,
            stats=self.stats,
            include_original=self.include_original,
            n_jobs=self.n_jobs_features,
        )
        return self.rf_.predict_proba(X_spatial)

    @property
    def feature_importances_(self):
        """Feature importances from the underlying Random Forest."""
        check_is_fitted(self, "rf_")
        return self.rf_.feature_importances_
