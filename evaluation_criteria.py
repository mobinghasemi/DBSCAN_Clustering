# Import necessary libraries: numpy for numerical operations and pandas for data handling
import numpy as np
import pandas as pd


def pairwise_distancess(X, Y=None):
    # If Y is not provided, set it to be the same as X
    if Y is None:
        Y = X

    # Compute the pairwise distances between vectors in X and Y using broadcasting.
    # X[:, np.newaxis] adds an extra dimension to X to make it (n, 1, m) and
    # Y[np.newaxis, :] makes it (1, p, m) where n and p are the number of vectors
    # in X and Y respectively, and m is the number of features. This allows for
    # broadcasting to compute pairwise differences, resulting in an (n, p, m) array.
    # Then, we compute the norm along the last dimension (axis=2) to get pairwise distances.
    dist_matrix = np.linalg.norm(X[:, np.newaxis] - Y[np.newaxis, :], axis=2)

    return dist_matrix


def davies_bouldin_index(data_path, labels):
    """Compute the Davies-Bouldin index for clustering results."""

    # Load data from the CSV file located at the provided path
    X = pd.read_csv(data_path)

    # For each unique cluster label, compute the centroid (mean) of all points with that label
    centroids = [np.mean(X[labels == label], axis=0) for label in np.unique(labels)]

    # # For each cluster, compute the average distance between points within that cluster and its centroid
    intra_cluster_dists = [
        np.mean(pairwise_distancess(X[labels == label].values, np.array(centroid).reshape(1, -1)))
        for label, centroid in zip(np.unique(labels), centroids)
    ]



    # Compute pairwise distances between the centroids of the clusters
    centroid_dists = pairwise_distancess(np.array(centroids))

    # For each cluster, compute the Davies-Bouldin score by comparing its intra-cluster distance
    # and the distances to centroids of other clusters. The Davies-Bouldin index is the average
    # of these scores for all clusters.
    n_clusters = len(centroids)
    db_index = np.mean([
        max([
            (intra_cluster_dists[i] + intra_cluster_dists[j]) / centroid_dists[i, j]
            for j in range(n_clusters) if i != j
        ])
        for i in range(n_clusters)
    ])

    return db_index
