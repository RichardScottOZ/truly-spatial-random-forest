# Truly Spatial Random Forest

A Python implementation of the Truly Spatial Random Forests (SRF) algorithm for geoscience data analysis.

Based on: Talebi, H., Peeters, L.J.M., Otto, A. & Tolosana-Delgado, R. (2022).
*A Truly Spatial Random Forests Algorithm for Geoscience Data Analysis and Modelling.*
Mathematical Geosciences, 54, 1–22.
[https://doi.org/10.1007/s11004-021-09946-w](https://link.springer.com/article/10.1007/s11004-021-09946-w)

## Overview

Standard random forests treat each observation independently, ignoring spatial
relationships. The SRF algorithm extends random forests by extracting
**neighbourhood summary statistics** (mean, standard deviation, skewness, etc.)
for each sample's spatial neighbourhood and appending them as additional features.
This enables the model to capture local spatial patterns, spatial dependencies,
and intrinsic heterogeneity in geoscience datasets.

### Key Features

- **Spatial feature augmentation** — automatically computes neighbourhood statistics
  (mean, std, min, max, median, skew, kurtosis, count) for each sample
- **Flexible neighbourhood definitions** — k-nearest neighbours or fixed-radius search
- **Classification and regression** — `SpatialRandomForestClassifier` and
  `SpatialRandomForestRegressor` wrap scikit-learn's random forests
- **Parallel feature extraction** — built-in `joblib` parallelism via `n_jobs_features`
- **scikit-learn compatible** — follows the estimator API (with an extra `coordinates`
  argument on `fit`/`predict`)

## Installation

```bash
pip install -e .
```

With optional parallelisation backends:

```bash
pip install -e ".[dask]"   # Dask support
pip install -e ".[ray]"    # Ray support
pip install -e ".[all]"    # Both
pip install -e ".[dev]"    # Development/test dependencies
```

## Quick Start

### Classification

```python
import numpy as np
from spatial_rf import SpatialRandomForestClassifier

X = np.random.rand(500, 4)
coords = np.random.rand(500, 2) * 1000  # easting, northing
y = (coords[:, 0] > 500).astype(int)    # spatial class boundary

clf = SpatialRandomForestClassifier(
    n_estimators=100,
    k=10,                 # 10 nearest neighbours
    stats=("mean", "std", "min", "max"),
    random_state=42,
)
clf.fit(X, y, coordinates=coords)
predictions = clf.predict(X, coordinates=coords)
probabilities = clf.predict_proba(X, coordinates=coords)
```

### Regression

```python
from spatial_rf import SpatialRandomForestRegressor

reg = SpatialRandomForestRegressor(
    n_estimators=100,
    neighbor_mode="radius",
    radius=50.0,
    stats=("mean", "std"),
    random_state=42,
)
reg.fit(X, y_continuous, coordinates=coords)
predictions = reg.predict(X, coordinates=coords)
```

### Spatial Feature Extraction Only

```python
from spatial_rf import extract_spatial_features

X_augmented, feature_names = extract_spatial_features(
    X, coords, neighbor_mode="knn", k=15,
    stats=("mean", "std", "skew"),
    n_jobs=4,  # parallel feature computation
)
```

## Parallelisation at Scale

Spatial feature extraction and random forest training are independently
parallelisable. At scale (millions of samples, hundreds of features),
the built-in `joblib` parallelism may not be enough. Below are
recommended approaches.

### Built-in joblib Parallelism

The simplest option — works out of the box:

```python
clf = SpatialRandomForestClassifier(
    n_jobs_features=4,  # parallel spatial feature extraction
    n_jobs=-1,          # parallel RF training (all cores)
)
```

### Dask / Dask-Gateway

[Dask](https://www.dask.org/) distributes computation across multiple
workers or a cluster. [Dask-Gateway](https://gateway.dask.org/) provides
managed, multi-tenant Dask clusters on Kubernetes or HPC.

```python
from dask.distributed import Client
from dask import delayed
import numpy as np
from spatial_rf import extract_spatial_features
from sklearn.ensemble import RandomForestClassifier

# Connect to a Dask cluster (local, SLURM, Kubernetes, or Dask-Gateway)
client = Client("tcp://scheduler:8786")
# Or via Dask-Gateway:
# from dask_gateway import Gateway
# gateway = Gateway("http://dask-gateway.example.com")
# cluster = gateway.new_cluster()
# cluster.scale(20)
# client = cluster.get_client()

# Partition data into spatial tiles
tile_size = len(X) // n_tiles
tiles = [
    (X[i:i+tile_size], coords[i:i+tile_size])
    for i in range(0, len(X), tile_size)
]

# Extract features in parallel across tiles
@delayed
def extract_tile(X_tile, coords_tile, all_coords, all_X):
    return extract_spatial_features(
        X_tile, coords_tile, k=10, stats=("mean", "std")
    )[0]

futures = [extract_tile(Xt, ct, coords, X) for Xt, ct in tiles]
results = client.compute(futures)
X_augmented = np.vstack(client.gather(results))

# Train with Dask-ML or standard sklearn
clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
clf.fit(X_augmented, y)
```

### Ray

[Ray](https://www.ray.io/) provides distributed computing with minimal
code changes and excellent scaling on multi-node clusters.

```python
import ray
import numpy as np
from spatial_rf import extract_spatial_features

ray.init()  # or ray.init(address="auto") for an existing cluster

@ray.remote
def extract_partition(X_part, coords_part, k, stats):
    X_aug, names = extract_spatial_features(
        X_part, coords_part, k=k, stats=stats
    )
    return X_aug

# Split data into partitions
n_parts = 16
splits = np.array_split(np.arange(len(X)), n_parts)
futures = [
    extract_partition.remote(X[s], coords[s], k=10, stats=("mean", "std"))
    for s in splits
]
X_augmented = np.vstack(ray.get(futures))
```

### Choosing a Parallelisation Strategy

| Scale | Recommendation |
|-------|---------------|
| < 100k samples | `n_jobs_features=-1` (built-in joblib) |
| 100k – 1M samples | Dask `LocalCluster` or Ray on a single machine |
| > 1M samples | Dask-Gateway on Kubernetes/HPC or Ray cluster |
| Cloud / multi-tenant | Dask-Gateway for managed cluster lifecycle |

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
