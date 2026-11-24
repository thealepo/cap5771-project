import numpy as np


# Continuous numerical attributes
continuous_cols = [
    'age',
    'fnlwgt',
    'capital_gain',
    'capital_loss',
    'hours_per_week'
]


N_BINS = 3


def equal_width_edges(x, k):
    """Split the range [min, max] into k same-sized slices."""

    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()

    return np.array([
        lo + (hi - lo) * i / k
        for i in range(1, k)
    ])


def equal_depth_edges(x, k):
    """Cut at percentiles so each bin holds ~1/k of the rows."""

    x = np.asarray(x, dtype=float)

    edges = np.quantile(
        x,
        [i / k for i in range(1, k)]
    )

    # Duplicate cuts collapse (e.g., lots of zeros)
    return np.unique(edges)


def kmeans_1d_edges(x, k, max_iter=100):
    """Find k centers, then cut halfway between centers."""

    x = np.asarray(x, dtype=float)

    # Start with evenly spaced centers
    centers = np.linspace(x.min(), x.max(), k)

    for _ in range(max_iter):

        # Assign each value to its closest center
        labels = np.argmin(
            np.abs(x[:, None] - centers[None, :]),
            axis=1
        )

        # Move each center to the average of its group
        new_centers = centers.copy()

        for j in range(k):
            if np.any(labels == j):
                new_centers[j] = x[labels == j].mean()

        # Stop when the centers stop moving
        if np.allclose(new_centers, centers):
            break

        centers = new_centers

    centers = np.sort(centers)

    return (centers[:-1] + centers[1:]) / 2


def apply_edges(x, edges):
    """Turn numerical values into ordered bin numbers."""

    return np.digitize(
        np.asarray(x, dtype=float),
        edges,
        right=True
    )


# Available discretization methods
methods = {
    'equal_width': equal_width_edges,
    'equal_depth': equal_depth_edges,
    'kmeans': kmeans_1d_edges,
}


def discretize_data(train, test, n_bins=N_BINS):
    """Fit bin edges on train and apply them to both datasets."""

    binned_train = {}
    binned_test = {}
    all_edges = {}

    for name, edge_func in methods.items():

        tr = train.copy()
        te = test.copy()

        all_edges[name] = {}

        for col in continuous_cols:

            # Learn bin edges using training data only
            edges = edge_func(train[col], n_bins)

            all_edges[name][col] = edges

            # Apply the same edges to both datasets
            tr[col] = apply_edges(train[col], edges)
            te[col] = apply_edges(test[col], edges)

        binned_train[name] = tr
        binned_test[name] = te

    return binned_train, binned_test, all_edges