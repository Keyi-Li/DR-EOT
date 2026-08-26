import numpy as np
import pandas as pd
from tqdm import tqdm


def knn_intersection(W_dict, W_ref, k_list, axis=1, mode="overall",
                     labels_X=None, labels_Y=None):
    """
    For each W in W_dict and each k, compute KNN intersection sizes with W_ref.

    Parameters
    ----------
    W_dict : dict {str: ndarray}
    W_ref : ndarray
    k_list : list of int
    axis : {0, 1}, default 1
    mode : {"overall", "detailed"}
        "overall"  : total intersection count summed over all rows/columns.
        "detailed" : mean per label class + macro "overall" row.
    labels_X : array-like, optional  (needed for axis=1, mode="detailed")
    labels_Y : array-like, optional  (needed for axis=0, mode="detailed")

    Returns
    -------
    pd.DataFrame (mode="overall") or dict of pd.DataFrame (mode="detailed").
    """
    def _knn_indices(W, k):
        if axis == 1:
            return np.argpartition(W, -k, axis=1)[:, -k:]
        else:
            return np.argpartition(W, -k, axis=0)[-k:, :].T

    ref_idx_cache = {k: _knn_indices(W_ref, k) for k in k_list}
    n = W_ref.shape[0] if axis == 1 else W_ref.shape[1]
    cols = [f"k={k}" for k in k_list]

    def _per_cell_counts(W):
        counts = np.zeros((n, len(k_list)), dtype=int)
        for ki, k in enumerate(k_list):
            W_idx = _knn_indices(W, k)
            counts[:, ki] = [
                len(set(w_row) & set(r_row))
                for w_row, r_row in zip(W_idx, ref_idx_cache[k])
            ]
        return counts

    if mode == "overall":
        result = {}
        for name, W in tqdm(W_dict.items()):
            result[name] = _per_cell_counts(W).sum(axis=0)
        return pd.DataFrame(result, index=cols).T

    elif mode == "detailed":
        labels = labels_X if axis == 1 else labels_Y
        unique_labels = np.sort(np.unique(labels))
        all_keys = list(unique_labels) + ["overall"]
        records = {lbl: {} for lbl in all_keys}

        for name, W in tqdm(W_dict.items()):
            counts = _per_cell_counts(W)
            per_label = {
                lbl: counts[labels == lbl].mean(axis=0) for lbl in unique_labels
            }
            per_label["overall"] = np.stack(list(per_label.values())).mean(axis=0)
            for lbl in all_keys:
                records[lbl][name] = per_label[lbl]

        return {
            lbl: pd.DataFrame(records[lbl], index=cols).T.round(4)
            for lbl in all_keys
        }
    else:
        raise ValueError(f"mode must be 'overall' or 'detailed', got {mode!r}")
