import numpy as np


def knn_median(dist_matrix, k, axis=1):
    """
    Compute the median of each point's k-th nearest neighbor distance.

    Parameters
    ----------
    dist_matrix : np.ndarray, shape (n, n)
        Pairwise distance matrix.
    k : int
        Number of nearest neighbors (excludes self if diagonal is zero).
    axis : int, optional
        Axis along which to find neighbors. Default is 1.

    Returns
    -------
    numpy.floating
        Median of the k-th NN distance over all points (excluding self).
    """
    knn_indices = np.argpartition(dist_matrix, k, axis=axis)[:, :k+1]
    knn_distances = np.take_along_axis(dist_matrix, knn_indices, axis=axis)
    kth_nn = np.max(knn_distances, axis=axis)
    return np.median(kth_nn)

