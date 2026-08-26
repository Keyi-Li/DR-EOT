import numpy as np
import pandas as pd


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


def aggregate_by_label(scores, labels):
    """
    Compute mean score per label plus an ungrouped overall mean.

    Parameters
    ----------
    scores : array-like of shape (n,)
    labels : array-like of shape (n,)

    Returns
    -------
    pd.Series
        Index = sorted unique labels + "overall".
    """
    scores = np.asarray(scores).ravel()
    labels = np.asarray(labels)
    unique_labels = np.sort(np.unique(labels))
    result = {ct: scores[labels == ct].mean() for ct in unique_labels}
    result["overall"] = scores.mean()
    return pd.Series(result)


def topk_MKNN(W, k_tuple):
    """
    For each X cell, find its ordered top-K mutual KNN partners in Y by
    expanding both neighborhoods symmetrically until K mutual KNNs accumulate.

    Pair (i, j) is a mutual KNN at level H if j is in i's top-H Y neighbors
    AND i is in j's top-H X neighbors (the same H threshold on both sides).
    The smallest H at which (i, j) qualifies is max(row_rank[i,j], col_rank[i,j]).

    The returned list is ordered by this discovery level, with proximity to i
    (row_rank) as a tie-breaker within the same level.

    Parameters
    ----------
    W : ndarray of shape (n_X, n_Y)
        Pairwise similarity matrix (higher = more similar).
    k_tuple : sequence of int
        Target mutual-KNN counts per cell. Only max(k_tuple) controls how
        many columns are returned. Individual values mark evaluation
        breakpoints: result[:, :k] gives the top-k mutual KNNs for each k.

    Returns
    -------
    ndarray of shape (n_X, max(k_tuple)), dtype int
        result[i, r] is the Y-cell index of X cell i's (r+1)-th mutual KNN,
        ordered by discovery level (then by proximity to i within the same level).
    """
    max_k = max(k_tuple)
    n_X, n_Y = W.shape

    # 1-indexed rank of j in i's Y-neighbor list (row perspective)
    row_rank = np.argsort(np.argsort(-W, axis=1), axis=1) + 1  # (n_X, n_Y)

    # 1-indexed rank of i in j's X-neighbor list (col perspective)
    col_rank = np.argsort(np.argsort(-W, axis=0), axis=0) + 1  # (n_X, n_Y)

    # mutual_level[i, j] = smallest H at which (i, j) is a mutual KNN pair
    mutual_level = np.maximum(row_rank, col_rank)  # (n_X, n_Y)

    result = np.full((n_X, max_k), -1, dtype=int)
    for i in range(n_X):
        # primary sort: discovery level; tie-break: proximity to i (row_rank)
        order = np.lexsort((row_rank[i], mutual_level[i]))
        result[i] = order[:max_k]
    return result


def pct_mknn(W, k, labels_X, labels_Y, mode="diversity"):
    """
    Compute the label-concordant mutual kNN score between two point sets X and Y.

    Parameters
    ----------
    W : ndarray of shape (n_X, n_Y)
        Pairwise similarity matrix (higher = more similar).
    k : int
        Number of nearest neighbors.
    labels_X : ndarray of shape (n_X,)
    labels_Y : ndarray of shape (n_Y,)
    mode : {"diversity", "accuracy", "both"}, default "diversity"
        "diversity" : divide by min(n_X, n_Y) * k (expected pairs under
                     perfect alignment).
        "accuracy"  : divide by total mutual pairs found.
        "both"      : return (diversity_score, accuracy_score).

    Returns
    -------
    float or tuple of float
    """
    mutual_pairs = find_mutual_knn(W, k)
    n_correct = np.sum(labels_X[mutual_pairs[0]] == labels_Y[mutual_pairs[1]])

    if mode == "diversity":
        return n_correct / (np.min(W.shape) * k)
    elif mode == "accuracy":
        return n_correct / len(mutual_pairs[0])
    elif mode == "both":
        return n_correct / (np.min(W.shape) * k), n_correct / len(mutual_pairs[0])
    else:
        raise ValueError(f"mode must be 'diversity', 'accuracy', or 'both', got {mode!r}")


def pct_mknn_per(W, k, labels_X, labels_Y, mode="diversity"):
    """
    Per-cell label-concordant mutual kNN scores.

    Parameters
    ----------
    W : ndarray of shape (n_X, n_Y)
    k : int
    labels_X : ndarray of shape (n_X,)
    labels_Y : ndarray of shape (n_Y,)
    mode : {"diversity", "accuracy", "both"}, default "diversity"
        "diversity" : per-cell score = correct mutual kNN / k.
        "accuracy"  : per-cell score = correct mutual kNN / total mutual kNN
                      for that cell (0 if no mutual kNN).
        "both"      : return ((div_row, div_col), (acc_row, acc_col)).

    Returns
    -------
    Depending on mode:
        "diversity" : (div_row, div_col) — ndarrays of shape (n_X,), (n_Y,)
        "accuracy"  : (acc_row, acc_col)
        "both"      : ((div_row, div_col), (acc_row, acc_col))
    """
    mutual_pairs = find_mutual_knn(W, k)
    rows, cols = mutual_pairs[0], mutual_pairs[1]
    match = (labels_X[rows] == labels_Y[cols])
    n_X, n_Y = W.shape

    mnn_per_row     = np.bincount(rows, minlength=n_X)
    correct_per_row = np.bincount(rows[match], minlength=n_X)
    mnn_per_col     = np.bincount(cols, minlength=n_Y)
    correct_per_col = np.bincount(cols[match], minlength=n_Y)

    if mode == "diversity":
        return correct_per_row / k, correct_per_col / k
    elif mode == "accuracy":
        acc_row = np.divide(correct_per_row, mnn_per_row,
                            out=np.zeros(n_X, dtype=float), where=mnn_per_row > 0)
        acc_col = np.divide(correct_per_col, mnn_per_col,
                            out=np.zeros(n_Y, dtype=float), where=mnn_per_col > 0)
        return acc_row, acc_col
    elif mode == "both":
        div_row = correct_per_row / k
        div_col = correct_per_col / k
        acc_row = np.divide(correct_per_row, mnn_per_row,
                            out=np.zeros(n_X, dtype=float), where=mnn_per_row > 0)
        acc_col = np.divide(correct_per_col, mnn_per_col,
                            out=np.zeros(n_Y, dtype=float), where=mnn_per_col > 0)
        return (div_row, div_col), (acc_row, acc_col)
    else:
        raise ValueError(f"mode must be 'diversity', 'accuracy', or 'both', got {mode!r}")
