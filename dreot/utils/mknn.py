import numpy as np


def find_mutual_knn(W, k):
    """
    Find mutual k-nearest neighbor pairs between two point sets X and Y.

    A pair (i, j) is mutual if j is among X_i's top-k neighbors in Y
    AND i is among Y_j's top-k neighbors in X.

    Parameters
    ----------
    W : ndarray of shape (n_X, n_Y)
        Pairwise similarity matrix (higher = more similar).
    k : int
        Number of nearest neighbors.

    Returns
    -------
    (row_indices, col_indices) : tuple of ndarrays
        Indices of mutual kNN pairs, as returned by np.where.
    """
    knn_indices_X_to_Y = np.argpartition(W, -k, axis=1)[:, -k:]
    knn_indices_Y_to_X = np.argpartition(W, -k, axis=0)[-k:, :]

    mutual_knn = np.zeros_like(W, dtype=int)
    mutual_knn[np.arange(W.shape[0])[:, None], knn_indices_X_to_Y] += 1
    mutual_knn[knn_indices_Y_to_X.T, np.arange(W.shape[1])[:, None]] += 1

    return np.where(mutual_knn == 2)
