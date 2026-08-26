import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

from .mknn import find_mutual_knn, pct_mknn, pct_mknn_per, aggregate_by_label, topk_MKNN


# ── Alignment error ────────────────────────────────────────────────────────────

def knn_celltype(W, adata_tgt, adata_src, k=None, mode="weight", col_name="celltype"):
    """
    For each source cell, identify the most likely matching target cell type
    based on an affinity/transport matrix W.

    Operates row-wise: each row of W corresponds to a source cell, each column
    to a target cell. To perform column-wise analysis, transpose W and swap
    adata_src / adata_tgt.

    Intuitively, W[i, j] measures how strongly source cell i is coupled to
    target cell j. For each source cell, the function finds its top-k target
    neighbors and scores each target cell type either by summing coupling
    weights ("weight") or by counting neighbors ("count"). The predicted
    match, "max_align", is the target cell type with the highest score — e.g.
    if a source T-cell's top neighbors are mostly target T-cells, max_align="T-cell",
    indicating good alignment; if they are mostly B-cells, it flags a mismatch.

    Parameters
    ----------
    W : np.ndarray of shape (n_src, n_tgt)
    adata_tgt : AnnData
    adata_src : AnnData
    k : int, optional
        Number of top neighbors per source cell. None = all target cells.
    mode : {"weight", "count"}
        "weight" : sum transport weights toward each cell type.
        "count"  : count top-k neighbors per cell type.
    col_name : str, default "celltype"

    Returns
    -------
    pd.DataFrame of shape (n_src, n_unique_celltypes + 2)
        Columns: "celltype", one "{ct}_{mode}" score column per target cell type,
        "max_align".
    """
    tgt_celltypes = adata_tgt.obs[col_name].values
    unique_celltypes = np.unique(tgt_celltypes)

    if k is None:
        k = adata_tgt.shape[0]
        print("Warning: with k = None, all neighbors are considered.")

    knn_indices = np.argpartition(W, -k, axis=1)[:, -k:]

    if mode == "count":
        knn_celltypes = tgt_celltypes[knn_indices]
        ct_stats = np.stack(
            [(knn_celltypes == ct).sum(axis=1) for ct in unique_celltypes], axis=1
        )
    else:
        W_use = np.zeros_like(W)
        np.put_along_axis(
            W_use, knn_indices,
            np.take_along_axis(W, knn_indices, axis=1),
            axis=1,
        )
        ct_stats = np.stack(
            [W_use[:, tgt_celltypes == ct].sum(axis=1) for ct in unique_celltypes], axis=1
        )

    result = pd.DataFrame(
        {f"{ct}_{mode}": ct_stats[:, i] for i, ct in enumerate(unique_celltypes)}
    )
    result.insert(0, "celltype", adata_src.obs[col_name].values)
    result.index = adata_src.obs.index
    result["max_align"] = unique_celltypes[np.argmax(ct_stats, axis=1)]
    return result


def impurity(res, k, metric="impurity"):
    """
    Add same-type kNN impurity or purity columns to the result DataFrame.

    For each source cell of type T, impurity = (k - count of T-type target
    neighbors) / k — the fraction of mismatched neighbors. Purity = 1 - impurity.

    Parameters
    ----------
    res : pd.DataFrame
        Output of knn_celltype(..., mode="count"). Must have 'celltype' and
        '{ct}_count' columns.
    k : int
    metric : {"impurity", "purity"}, default "impurity"
        "impurity" : adds 'error' and 'error_rate' columns (fraction of
                     mismatched neighbors).
        "purity"   : adds 'correct' and 'purity_rate' columns (fraction of
                     matched neighbors).

    Returns
    -------
    pd.DataFrame
        The input DataFrame modified in-place with added metric columns.
    """
    if metric not in ("impurity", "purity"):
        raise ValueError(f"metric must be 'impurity' or 'purity', got {metric!r}")

    correct = np.zeros(len(res), dtype=float)
    for ct in res["celltype"].unique():
        mask = res["celltype"] == ct
        correct[mask.values] = res.loc[mask, f"{ct}_count"].values

    if metric == "impurity":
        res["error"] = k - correct
        res["error_rate"] = res["error"] / k
    else:
        res["correct"] = correct
        res["purity_rate"] = correct / k
    return res


# def error_rate(res, forbidden_pairs, k):
#     """
#     Compute the forbidden-pair error rate for each source cell.

#     Parameters
#     ----------
#     res : pd.DataFrame
#         Output of knn_celltype(..., mode="count").
#     forbidden_pairs : list of (str, str)
#         (src_celltype, tgt_celltype) pairs that are biologically invalid.
#     k : int

#     Returns
#     -------
#     pd.DataFrame
#         The input DataFrame modified in-place with added 'error' and
#         'error_rate' columns.
#     """
#     res["error"] = 0
#     for (src, tgt) in forbidden_pairs:
#         res.loc[res["celltype"] == src, "error"] += res[f"{tgt}_count"]
#     res["error_rate"] = res["error"] / k
#     return res


def summarize_error_stats(df, celltype_col="celltype"):
    """
    Per-celltype means plus an 'Overall' row for all numeric columns.

    The 'Overall' row is a macro-average: the mean of per-celltype means,
    giving equal weight to each cell type regardless of its size. For example,
    if B-cell=0.12, NK-cell=0.31, T-cell=0.08, then Overall=(0.12+0.31+0.08)/3=0.17.

    Parameters
    ----------
    df : pd.DataFrame
    celltype_col : str, default "celltype"

    Returns
    -------
    pd.DataFrame
        Rows = cell types + "Overall" (macro-average), columns = numeric metrics.
    """
    error_cols = df.select_dtypes(include=np.number).columns.tolist()
    groupby_mean = df.groupby(celltype_col)[error_cols].mean()
    overall_mean = groupby_mean.mean().rename("Overall").to_frame().T
    summary = pd.concat([groupby_mean, overall_mean])
    summary.index.name = "group"
    logger.info(
        "'Overall' row is a macro-average (mean of per-celltype means), "
        "giving equal weight to each cell type regardless of group size."
    )
    return summary


# ── kNN intersection ───────────────────────────────────────────────────────────

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


# ── evaluate pct_mknn across W_dict ───────────────────────────────────────────

def evaluate_pct_mknn(W_dict, k_list, labels_X, labels_Y, mode="diversity"):
    """
    Benchmark multiple transport plans on label-concordant mutual kNN scores.

    Given several candidate W matrices (e.g. from different methods) and a
    range of k values, this produces a comparison table where each row is a
    method and each column is a k value. The score measures how often mutual
    nearest neighbors across X and Y share the same cell-type label — higher
    is better alignment.

    In both modes only label-concordant mutual pairs (same cell type on both
    sides) are counted in the numerator. The modes differ in the denominator:
        "diversity" : divided by min(n_X, n_Y) * k, the theoretical maximum
                      number of correct mutual pairs if every cell found k
                      mutual neighbors all of the same type. Penalises methods
                      that produce few mutual pairs overall.
        "accuracy"  : divided by the total number of mutual pairs found —
                      purely asks "of the pairs that matched, what fraction
                      shared the same label?" Insensitive to how many pairs
                      were found.
        "both"      : returns both tables as (div_df, acc_df).

    Parameters
    ----------
    W_dict : dict {str: ndarray}
        Named transport/affinity matrices to compare.
    k_list : list of int
        k values to evaluate at.
    labels_X, labels_Y : ndarray
        Cell-type labels for the X and Y datasets.
    mode : {"diversity", "accuracy", "both"}, default "diversity"

    Returns
    -------
    pd.DataFrame with rows = method names, columns = "k={k}".
    Returns (div_df, acc_df) when mode="both".
    """
    if mode == "both":
        result_div, result_acc = {}, {}
        for name, W in tqdm(W_dict.items()):
            result_div[name], result_acc[name] = {}, {}
            for k in k_list:
                div, acc = pct_mknn(W, k, labels_X, labels_Y, mode="both")
                result_div[name][k] = div
                result_acc[name][k] = acc
        fmt = lambda r: pd.DataFrame(r).T.rename(columns=lambda k: f"k={k}").round(4)
        return fmt(result_div), fmt(result_acc)
    else:
        result = {}
        for name, W in tqdm(W_dict.items()):
            result[name] = {
                k: pct_mknn(W, k, labels_X, labels_Y, mode=mode) for k in k_list
            }
        return pd.DataFrame(result).T.rename(columns=lambda k: f"k={k}").round(4)


def evaluate_pct_mknn_per(W, k_list, labels_X, labels_Y, mode="diversity"):
    """
    Per-cell MNN scores across k values for a single W matrix.

    Parameters
    ----------
    W : ndarray
    k_list : list of int
    labels_X, labels_Y : ndarray
    mode : {"diversity", "accuracy", "both"}

    Returns
    -------
    (df_X, df_Y) or ((div_X, div_Y), (acc_X, acc_Y)) when mode="both".
    Each df has rows = label classes + "overall", columns = "k={k}".
    """
    if mode not in ("diversity", "accuracy", "both"):
        raise ValueError(f"mode must be 'diversity', 'accuracy', or 'both', got {mode!r}")

    rows_X_div, rows_X_acc = {}, {}
    rows_Y_div, rows_Y_acc = {}, {}

    for k in k_list:
        col = f"k={k}"
        if mode == "diversity":
            div_row, div_col = pct_mknn_per(W, k, labels_X, labels_Y, mode="diversity")
            rows_X_div[col] = aggregate_by_label(div_row, labels_X)
            rows_Y_div[col] = aggregate_by_label(div_col, labels_Y)
        elif mode == "accuracy":
            acc_row, acc_col = pct_mknn_per(W, k, labels_X, labels_Y, mode="accuracy")
            rows_X_acc[col] = aggregate_by_label(acc_row, labels_X)
            rows_Y_acc[col] = aggregate_by_label(acc_col, labels_Y)
        else:
            (div_row, div_col), (acc_row, acc_col) = pct_mknn_per(
                W, k, labels_X, labels_Y, mode="both"
            )
            rows_X_div[col] = aggregate_by_label(div_row, labels_X)
            rows_Y_div[col] = aggregate_by_label(div_col, labels_Y)
            rows_X_acc[col] = aggregate_by_label(acc_row, labels_X)
            rows_Y_acc[col] = aggregate_by_label(acc_col, labels_Y)

    if mode == "diversity":
        return pd.DataFrame(rows_X_div).round(4), pd.DataFrame(rows_Y_div).round(4)
    elif mode == "accuracy":
        return pd.DataFrame(rows_X_acc).round(4), pd.DataFrame(rows_Y_acc).round(4)
    else:
        return (
            (pd.DataFrame(rows_X_div).round(4), pd.DataFrame(rows_Y_div).round(4)),
            (pd.DataFrame(rows_X_acc).round(4), pd.DataFrame(rows_Y_acc).round(4)),
        )


def evaluate_pct_mknn_per_Wdict(W_dict, k_list, labels_X, labels_Y, mode="diversity"):
    """
    Per-cell MNN scores across multiple W matrices and k values.

    Parameters
    ----------
    W_dict : dict {str: ndarray}
    k_list : list of int
    labels_X, labels_Y : ndarray
    mode : {"diversity", "accuracy", "both"}

    Returns
    -------
    dict {str: pd.DataFrame} with a ("W", "dataset") MultiIndex per DataFrame,
    or (div_dict, acc_dict) when mode="both".
    """
    if mode not in ("diversity", "accuracy", "both"):
        raise ValueError(f"mode must be 'diversity', 'accuracy', or 'both', got {mode!r}")

    celltypes = list(np.sort(np.unique(np.concatenate([labels_X, labels_Y])))) + ["overall"]
    records_div = {ct: {} for ct in celltypes}
    records_acc = {ct: {} for ct in celltypes}

    for name, W in tqdm(W_dict.items(), desc="W matrices"):
        if mode == "diversity":
            df_X, df_Y = evaluate_pct_mknn_per(W, k_list, labels_X, labels_Y, mode="diversity")
        elif mode == "accuracy":
            df_X, df_Y = evaluate_pct_mknn_per(W, k_list, labels_X, labels_Y, mode="accuracy")
        else:
            (div_X, div_Y), (acc_X, acc_Y) = evaluate_pct_mknn_per(
                W, k_list, labels_X, labels_Y, mode="both"
            )

        for ct in celltypes:
            if mode in ("diversity", "both"):
                records_div[ct][(name, "X")] = (div_X if mode == "both" else df_X).loc[ct]
                records_div[ct][(name, "Y")] = (div_Y if mode == "both" else df_Y).loc[ct]
            if mode in ("accuracy", "both"):
                records_acc[ct][(name, "X")] = (acc_X if mode == "both" else df_X).loc[ct]
                records_acc[ct][(name, "Y")] = (acc_Y if mode == "both" else df_Y).loc[ct]

    def assemble(records):
        result = {}
        for ct in celltypes:
            df = pd.DataFrame(records[ct]).T
            df.index = pd.MultiIndex.from_tuples(df.index, names=["W", "dataset"])
            result[ct] = df.round(4)
        return result

    if mode == "diversity":
        return assemble(records_div)
    elif mode == "accuracy":
        return assemble(records_acc)
    else:
        return assemble(records_div), assemble(records_acc)


# ── Comprehensive evaluation ───────────────────────────────────────────────────

_ALL_METRICS = ["purity", "knn_int", "cosine_sim", "mnn_div", "mnn_acc"]
_TOPKMKNN_METRICS = ["purity", "mnn_int"]


def comprehensive_evaluation(
    W_dict,
    k_list,
    labels_X,
    labels_Y,
    metrics=None,
    batch_effect=False,
    W_ref_knn_intersection=None,
    save=False,
    save_path=None,
):
    """
    Evaluate multiple W matrices across a grid of k values, separately for the
    X→Y and Y→X directions.

    Available metrics (all in [0, 1], higher = better):
        purity     — fraction of top-k neighbors with the same label.
        knn_int    — fraction of top-k neighbors shared with W_ref (÷ k).
                     Excluded automatically when batch_effect=True.
        cosine_sim — row-wise (X→Y) or column-wise (Y→X) cosine similarity
                     between W and W_ref, aggregated by cell type.
                     k-independent: the same value is reported for all k.
                     Excluded automatically when batch_effect=True.
        mnn_div    — label-concordant mutual kNN count ÷ k (diversity).
        mnn_acc    — label-concordant mutual kNN fraction of all mutual pairs
                     (accuracy; 0 for cells with no mutual pairs).

    X→Y: scores from each X cell's perspective (row-wise in W).
    Y→X: scores from each Y cell's perspective (column-wise in W).
    "overall": macro-average over per-celltype means.

    Parameters
    ----------
    W_dict : dict {str: ndarray of shape (n_X, n_Y)}
    k_list : list of int
    labels_X : array-like of shape (n_X,)
    labels_Y : array-like of shape (n_Y,)
    metrics : list of str, optional
        Subset of ["purity", "knn_int", "cosine_sim", "mnn_div", "mnn_acc"].
        Default (None) uses all metrics compatible with batch_effect.
    batch_effect : bool, default False
        If True, knn_int and cosine_sim are excluded automatically, because
        comparing against a reference W is not meaningful when datasets live
        in different spaces due to batch effects.
    W_ref_knn_intersection : ndarray of shape (n_X, n_Y), optional
        Reference matrix for knn_int and cosine_sim. Required when either
        of those metrics is active.
    save : bool, default False
    save_path : str, optional
        Directory for saved figures. Required when save=True.

    Returns
    -------
    XY_per_W : dict {metric: DataFrame(W_names × k_cols)}
    XY_per_ct : dict {metric: {ct: DataFrame(W_names × k_cols)}}
    YX_per_W : dict {metric: DataFrame(W_names × k_cols)}
    YX_per_ct : dict {metric: {ct: DataFrame(W_names × k_cols)}}
    """
    if save and save_path is None:
        raise ValueError("save_path must be specified when save=True.")

    # Resolve active metrics
    if metrics is None:
        active = list(_ALL_METRICS)
    else:
        unknown = set(metrics) - set(_ALL_METRICS)
        if unknown:
            raise ValueError(f"Unknown metrics: {unknown}. Choose from {_ALL_METRICS}.")
        active = [m for m in _ALL_METRICS if m in metrics]  # preserve canonical order

    if batch_effect:
        for m in ("knn_int", "cosine_sim"):
            if m in active:
                active.remove(m)
        logger.info("batch_effect=True: knn_int and cosine_sim excluded.")

    need_ref     = "knn_int" in active or "cosine_sim" in active
    need_knn_int = "knn_int" in active
    need_cos_sim = "cosine_sim" in active
    need_mnn     = "mnn_div" in active or "mnn_acc" in active

    if need_ref and W_ref_knn_intersection is None:
        raise ValueError(
            "W_ref_knn_intersection is required when knn_int or cosine_sim is in metrics."
        )

    labels_X = np.asarray(labels_X)
    labels_Y = np.asarray(labels_Y)
    k_list   = list(k_list)
    k_cols   = [f"k={k}" for k in k_list]

    cts_X     = sorted(set(labels_X))
    cts_Y     = sorted(set(labels_Y))
    all_cts_X = cts_X + ["overall"]
    all_cts_Y = cts_Y + ["overall"]

    def _agg(scores, labels, cts):
        d = {ct: float(scores[labels == ct].mean()) for ct in cts}
        d["overall"] = float(np.mean(list(d.values())))
        return d

    # xy_store[W_name][metric][ct][k_col] = value
    xy_store = {n: {m: {ct: {} for ct in all_cts_X} for m in active} for n in W_dict}
    yx_store = {n: {m: {ct: {} for ct in all_cts_Y} for m in active} for n in W_dict}

    if need_knn_int:
        ref_xy_idx = {k: np.argpartition(W_ref_knn_intersection, -k, axis=1)[:, -k:]
                      for k in k_list}
        ref_yx_idx = {k: np.argpartition(W_ref_knn_intersection, -k, axis=0)[-k:, :].T
                      for k in k_list}

    if need_cos_sim:
        _ref = W_ref_knn_intersection
        _ref_row_norm = np.linalg.norm(_ref, axis=1)   # (n_X,)
        _ref_col_norm = np.linalg.norm(_ref, axis=0)   # (n_Y,)

    for name, W in tqdm(W_dict.items(), desc="Evaluating W matrices"):

        # ── cosine_sim (k-independent: compute once, broadcast to all k) ──────
        if need_cos_sim:
            row_norm = np.linalg.norm(W, axis=1)                          # (n_X,)
            denom_xy = row_norm * _ref_row_norm
            cos_xy_per_cell = np.divide(
                (W * _ref).sum(axis=1), denom_xy,
                out=np.zeros(W.shape[0]), where=denom_xy > 0,
            )
            cos_xy = _agg(cos_xy_per_cell, labels_X, cts_X)

            col_norm = np.linalg.norm(W, axis=0)                          # (n_Y,)
            denom_yx = col_norm * _ref_col_norm
            cos_yx_per_cell = np.divide(
                (W * _ref).sum(axis=0), denom_yx,
                out=np.zeros(W.shape[1]), where=denom_yx > 0,
            )
            cos_yx = _agg(cos_yx_per_cell, labels_Y, cts_Y)

            for col in k_cols:                                             # same value for all k
                for ct in all_cts_X:
                    xy_store[name]["cosine_sim"][ct][col] = cos_xy[ct]
                for ct in all_cts_Y:
                    yx_store[name]["cosine_sim"][ct][col] = cos_yx[ct]

        for k in k_list:
            col = f"k={k}"

            # ── purity ────────────────────────────────────────────────────────
            knn_xy = np.argpartition(W, -k, axis=1)[:, -k:]
            knn_yx = np.argpartition(W, -k, axis=0)[-k:, :].T  # (n_Y, k)

            if "purity" in active:
                pur_xy = _agg(
                    (labels_Y[knn_xy] == labels_X[:, None]).sum(axis=1).astype(float) / k,
                    labels_X, cts_X,
                )
                pur_yx = _agg(
                    (labels_X[knn_yx] == labels_Y[:, None]).sum(axis=1).astype(float) / k,
                    labels_Y, cts_Y,
                )
                for ct in all_cts_X:
                    xy_store[name]["purity"][ct][col] = pur_xy[ct]
                for ct in all_cts_Y:
                    yx_store[name]["purity"][ct][col] = pur_yx[ct]

            # ── knn_int (÷ k → [0,1]) ─────────────────────────────────────────
            if need_knn_int:
                kint_xy = _agg(
                    np.array([len(set(a) & set(b))
                              for a, b in zip(knn_xy, ref_xy_idx[k])], dtype=float) / k,
                    labels_X, cts_X,
                )
                kint_yx = _agg(
                    np.array([len(set(a) & set(b))
                              for a, b in zip(knn_yx, ref_yx_idx[k])], dtype=float) / k,
                    labels_Y, cts_Y,
                )
                for ct in all_cts_X:
                    xy_store[name]["knn_int"][ct][col] = kint_xy[ct]
                for ct in all_cts_Y:
                    yx_store[name]["knn_int"][ct][col] = kint_yx[ct]

            # ── MNN (div + acc, both perspectives) ────────────────────────────
            if need_mnn:
                rows, cols = find_mutual_knn(W, k)
                match   = labels_X[rows] == labels_Y[cols]
                n_X, n_Y = W.shape
                mnn_row     = np.bincount(rows, minlength=n_X)
                mnn_col     = np.bincount(cols, minlength=n_Y)
                correct_row = np.bincount(rows[match], minlength=n_X)
                correct_col = np.bincount(cols[match], minlength=n_Y)

                if "mnn_div" in active:
                    mnn_div_xy = _agg(correct_row.astype(float) / k, labels_X, cts_X)
                    mnn_div_yx = _agg(correct_col.astype(float) / k, labels_Y, cts_Y)
                    for ct in all_cts_X:
                        xy_store[name]["mnn_div"][ct][col] = mnn_div_xy[ct]
                    for ct in all_cts_Y:
                        yx_store[name]["mnn_div"][ct][col] = mnn_div_yx[ct]

                if "mnn_acc" in active:
                    mnn_acc_xy = _agg(
                        np.divide(correct_row, mnn_row, out=np.zeros(n_X), where=mnn_row > 0),
                        labels_X, cts_X,
                    )
                    mnn_acc_yx = _agg(
                        np.divide(correct_col, mnn_col, out=np.zeros(n_Y), where=mnn_col > 0),
                        labels_Y, cts_Y,
                    )
                    for ct in all_cts_X:
                        xy_store[name]["mnn_acc"][ct][col] = mnn_acc_xy[ct]
                    for ct in all_cts_Y:
                        yx_store[name]["mnn_acc"][ct][col] = mnn_acc_yx[ct]

    # ── Assemble DataFrames ────────────────────────────────────────────────────
    def _assemble(store, all_cts):
        per_W  = {}
        per_ct = {}
        for m in active:
            per_W[m] = pd.DataFrame(
                {n: pd.Series(store[n][m]["overall"]) for n in W_dict}
            ).T.reindex(columns=k_cols).round(4)
            per_W[m].index.name = "W"

            per_ct[m] = {}
            for ct in all_cts:
                per_ct[m][ct] = pd.DataFrame(
                    {n: pd.Series(store[n][m][ct]) for n in W_dict}
                ).T.reindex(columns=k_cols).round(4)
                per_ct[m][ct].index.name = "W"
        return per_W, per_ct

    XY_per_W, XY_per_ct = _assemble(xy_store, all_cts_X)
    YX_per_W, YX_per_ct = _assemble(yx_store, all_cts_Y)

    # ── Plotting ───────────────────────────────────────────────────────────────
    def _make_plot(per_W, per_ct, all_cts, direction):
        ct_ordered = ["overall"] + [ct for ct in all_cts if ct != "overall"]
        n_m, n_c = len(active), len(ct_ordered)

        fig, axes = plt.subplots(
            n_m, n_c,
            figsize=(n_c * 3.5, n_m * 2.8),
            sharey="row", sharex=True,
            squeeze=False,
        )

        for mi, m in enumerate(active):
            for ci, ct in enumerate(ct_ordered):
                ax = axes[mi, ci]
                df = per_W[m] if ct == "overall" else per_ct[m][ct]
                for w_name in W_dict:
                    ax.plot(k_list, df.loc[w_name, k_cols].values,
                            "o-", label=w_name, markersize=3, linewidth=1)
                if mi == 0:
                    title = ct if len(ct) <= 20 else ct[:18] + "…"
                    ax.set_title(title, fontsize=8)
                if ci == 0:
                    ax.set_ylabel(m, fontsize=9)
                if mi == n_m - 1:
                    ax.set_xlabel("k", fontsize=8)
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)

        handles, leg_labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, leg_labels, loc="upper right",
                   bbox_to_anchor=(1.02, 1.0), fontsize=8, framealpha=0.9)
        fig.suptitle(f"Comprehensive Evaluation  ({direction})", fontsize=12, y=1.01)
        plt.tight_layout()
        if save:
            import os
            fig.savefig(os.path.join(save_path, f"comprehensive_{direction.replace('→','_')}.pdf"),
                        bbox_inches="tight")
        plt.show()

    _make_plot(XY_per_W, XY_per_ct, all_cts_X, "X→Y")
    _make_plot(YX_per_W, YX_per_ct, all_cts_Y, "Y→X")

    return XY_per_W, XY_per_ct, YX_per_W, YX_per_ct


def comprehensive_evaluation_topkmknn(
    W_dict,
    k_list,
    labels_X,
    labels_Y,
    metrics=None,
    batch_effect=False,
    W_ref=None,
    save=False,
    save_path=None,
):
    """
    Evaluate multiple W matrices using topk_MKNN scores across a grid of k values.

    For each W, topk_MKNN is run in both the X→Y and Y→X directions. At each k,
    result[:, :k] gives each cell's top-k mutual KNN partners, and two metrics
    are computed on that set.

    Available metrics (all in [0, 1], higher = better):
        purity   — true positive rate among top-k mutual KNNs: fraction that
                   share the same cell-type label as the query cell.
        mnn_int  — fraction of top-k mutual KNNs shared with W_ref's top-k
                   mutual KNNs (÷ k). Requires W_ref; excluded when
                   batch_effect=True.

    X→Y: scores from each X cell's perspective (row-wise in W).
    Y→X: scores from each Y cell's perspective; topk_MKNN is run on W.T.
    "overall": macro-average over per-celltype means.

    Parameters
    ----------
    W_dict : dict {str: ndarray of shape (n_X, n_Y)}
    k_list : list of int
    labels_X : array-like of shape (n_X,)
    labels_Y : array-like of shape (n_Y,)
    metrics : list of str, optional
        Subset of ["purity", "mnn_int"]. Default (None) uses all metrics
        compatible with batch_effect.
    batch_effect : bool, default False
        If True, mnn_int is excluded.
    W_ref : ndarray of shape (n_X, n_Y), optional
        Reference matrix for mnn_int.
    save : bool, default False
    save_path : str, optional

    Returns
    -------
    XY_per_W : dict {metric: DataFrame(W_names × k_cols)}
    XY_per_ct : dict {metric: {ct: DataFrame(W_names × k_cols)}}
    YX_per_W : dict {metric: DataFrame(W_names × k_cols)}
    YX_per_ct : dict {metric: {ct: DataFrame(W_names × k_cols)}}
    """
    if save and save_path is None:
        raise ValueError("save_path must be specified when save=True.")

    if metrics is None:
        active = list(_TOPKMKNN_METRICS)
    else:
        unknown = set(metrics) - set(_TOPKMKNN_METRICS)
        if unknown:
            raise ValueError(f"Unknown metrics: {unknown}. Choose from {_TOPKMKNN_METRICS}.")
        active = [m for m in _TOPKMKNN_METRICS if m in metrics]

    if batch_effect and "mnn_int" in active:
        active.remove("mnn_int")
        logger.info("batch_effect=True: mnn_int excluded.")

    if "mnn_int" in active and W_ref is None:
        raise ValueError("W_ref is required when mnn_int is in metrics.")

    labels_X = np.asarray(labels_X)
    labels_Y = np.asarray(labels_Y)
    k_list   = list(k_list)
    k_cols   = [f"k={k}" for k in k_list]

    cts_X     = sorted(set(labels_X))
    cts_Y     = sorted(set(labels_Y))
    all_cts_X = cts_X + ["overall"]
    all_cts_Y = cts_Y + ["overall"]

    def _agg(scores, labels, cts):
        d = {ct: float(scores[labels == ct].mean()) for ct in cts}
        d["overall"] = float(np.mean(list(d.values())))
        return d

    xy_store = {n: {m: {ct: {} for ct in all_cts_X} for m in active} for n in W_dict}
    yx_store = {n: {m: {ct: {} for ct in all_cts_Y} for m in active} for n in W_dict}

    if "mnn_int" in active:
        ref_mknn_xy = topk_MKNN(W_ref, k_list)    # (n_X, max_k)
        ref_mknn_yx = topk_MKNN(W_ref.T, k_list)  # (n_Y, max_k)

    for name, W in tqdm(W_dict.items(), desc="Evaluating W matrices"):
        mknn_xy = topk_MKNN(W, k_list)    # (n_X, max_k)
        mknn_yx = topk_MKNN(W.T, k_list)  # (n_Y, max_k)

        for k in k_list:
            col = f"k={k}"

            # ── purity: label concordance among top-k mutual KNNs ─────────────
            if "purity" in active:
                mknn_k_xy = mknn_xy[:, :k]  # (n_X, k) — Y-cell indices
                pur_xy = _agg(
                    (labels_Y[mknn_k_xy] == labels_X[:, None]).sum(axis=1).astype(float) / k,
                    labels_X, cts_X,
                )
                mknn_k_yx = mknn_yx[:, :k]  # (n_Y, k) — X-cell indices
                pur_yx = _agg(
                    (labels_X[mknn_k_yx] == labels_Y[:, None]).sum(axis=1).astype(float) / k,
                    labels_Y, cts_Y,
                )
                for ct in all_cts_X:
                    xy_store[name]["purity"][ct][col] = pur_xy[ct]
                for ct in all_cts_Y:
                    yx_store[name]["purity"][ct][col] = pur_yx[ct]

            # ── mnn_int: top-k mutual KNN overlap with reference ──────────────
            if "mnn_int" in active:
                int_xy = _agg(
                    np.array([len(set(a) & set(b))
                              for a, b in zip(mknn_xy[:, :k], ref_mknn_xy[:, :k])],
                             dtype=float) / k,
                    labels_X, cts_X,
                )
                int_yx = _agg(
                    np.array([len(set(a) & set(b))
                              for a, b in zip(mknn_yx[:, :k], ref_mknn_yx[:, :k])],
                             dtype=float) / k,
                    labels_Y, cts_Y,
                )
                for ct in all_cts_X:
                    xy_store[name]["mnn_int"][ct][col] = int_xy[ct]
                for ct in all_cts_Y:
                    yx_store[name]["mnn_int"][ct][col] = int_yx[ct]

    # ── Assemble DataFrames ────────────────────────────────────────────────────
    def _assemble(store, all_cts):
        per_W  = {}
        per_ct = {}
        for m in active:
            per_W[m] = pd.DataFrame(
                {n: pd.Series(store[n][m]["overall"]) for n in W_dict}
            ).T.reindex(columns=k_cols).round(4)
            per_W[m].index.name = "W"

            per_ct[m] = {}
            for ct in all_cts:
                per_ct[m][ct] = pd.DataFrame(
                    {n: pd.Series(store[n][m][ct]) for n in W_dict}
                ).T.reindex(columns=k_cols).round(4)
                per_ct[m][ct].index.name = "W"
        return per_W, per_ct

    XY_per_W, XY_per_ct = _assemble(xy_store, all_cts_X)
    YX_per_W, YX_per_ct = _assemble(yx_store, all_cts_Y)

    # ── Plotting ───────────────────────────────────────────────────────────────
    def _make_plot(per_W, per_ct, all_cts, direction):
        ct_ordered = ["overall"] + [ct for ct in all_cts if ct != "overall"]
        n_m, n_c = len(active), len(ct_ordered)

        fig, axes = plt.subplots(
            n_m, n_c,
            figsize=(n_c * 3.5, n_m * 2.8),
            sharey="row", sharex=True,
            squeeze=False,
        )
        for mi, m in enumerate(active):
            for ci, ct in enumerate(ct_ordered):
                ax = axes[mi, ci]
                df = per_W[m] if ct == "overall" else per_ct[m][ct]
                for w_name in W_dict:
                    ax.plot(k_list, df.loc[w_name, k_cols].values,
                            "o-", label=w_name, markersize=3, linewidth=1)
                if mi == 0:
                    title = ct if len(ct) <= 20 else ct[:18] + "…"
                    ax.set_title(title, fontsize=8)
                if ci == 0:
                    ax.set_ylabel(m, fontsize=9)
                if mi == n_m - 1:
                    ax.set_xlabel("k", fontsize=8)
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)

        handles, leg_labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, leg_labels, loc="upper right",
                   bbox_to_anchor=(1.02, 1.0), fontsize=8, framealpha=0.9)
        fig.suptitle(f"TopK-MKNN Evaluation  ({direction})", fontsize=12, y=1.01)
        plt.tight_layout()
        if save:
            import os
            fig.savefig(
                os.path.join(save_path, f"topkmknn_{direction.replace('→', '_')}.pdf"),
                bbox_inches="tight",
            )
        plt.show()

    _make_plot(XY_per_W, XY_per_ct, all_cts_X, "X→Y")
    _make_plot(YX_per_W, YX_per_ct, all_cts_Y, "Y→X")

    return XY_per_W, XY_per_ct, YX_per_W, YX_per_ct
