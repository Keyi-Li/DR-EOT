import numpy as np
import matplotlib.pyplot as plt

from .mknn import find_mutual_knn

def golden_bandwidth_mnn(
    dist,
    K,
    bd_lo,
    bd_hi,
    sinkhorn_balance_fn,
    sinkhorn_kwargs=None,
    returns_plan=False,
    max_iter=20,
    min_interval=0.5,
    plotting=False,
):
    """
    Find the bandwidth maximising mutual kNN pairs by golden-section search.

    Assumes MNN(bd) is unimodal over [bd_lo, bd_hi]. After 2 initial
    evaluations each iteration reuses one probe point, requiring only 1 new
    Sinkhorn solve. Interval shrinks by the golden ratio (~0.618) per step.

    Parameters
    ----------
    dist : ndarray, shape (n_X, n_Y)
    K : int
    bd_lo, bd_hi : float
    sinkhorn_balance_fn : callable
        Signature: sinkhorn_balance_fn(dist, bd, **sinkhorn_kwargs).
        Returns the transport plan W (ndarray) if returns_plan=True, otherwise
        returns (row_scale, col_scale).
    sinkhorn_kwargs : dict, optional
    returns_plan : bool, default False
        If True, sinkhorn_balance_fn is expected to return the transport plan W
        directly. If False, it should return (row_scale, col_scale) scaling
        vectors from which W is reconstructed internally.
    max_iter : int, default 20
    min_interval : float, default 0.5
    plotting : bool, default False

    Returns
    -------
    best_bd : float
    best_mnn : int
    """
    if sinkhorn_kwargs is None:
        sinkhorn_kwargs = {}

    def _mnn_count(bd):
        try:
            result = sinkhorn_balance_fn(dist, bd, **sinkhorn_kwargs)
            W = result if returns_plan else result[0] * np.exp(-dist / bd) * result[1].T
            return len(find_mutual_knn(W, K)[0])
        except Exception:
            return 0

    cache = {}

    def eval_at(bd):
        if bd not in cache:
            cache[bd] = _mnn_count(bd)
        return cache[bd]

    RATIO = (np.sqrt(5) - 1) / 2  # ≈ 0.618
    lo, hi = bd_lo, bd_hi
    c = lo + (1 - RATIO) * (hi - lo)
    d = lo + RATIO * (hi - lo)
    f_c, f_d = eval_at(c), eval_at(d)
    iter_log = []

    for it in range(max_iter):
        if hi - lo < min_interval:
            break
        iter_log.append((it + 1, lo, hi, c, f_c, d, f_d))

        if f_c < f_d:
            lo = c
            c, f_c = d, f_d
            d = lo + RATIO * (hi - lo)
            f_d = eval_at(d)
        elif f_c > f_d:
            hi = d
            d, f_d = c, f_c
            c = lo + (1 - RATIO) * (hi - lo)
            f_c = eval_at(c)
        else:
            lo, hi = c, d
            c = lo + (1 - RATIO) * (hi - lo)
            d = lo + RATIO * (hi - lo)
            f_c, f_d = eval_at(c), eval_at(d)

    trajectory = list(cache.items())
    history = sorted(trajectory)
    best_bd, best_mnn = max(trajectory, key=lambda x: x[1])

    if plotting:
        header = f"{'Iter':>4}  {'lo':>10}  {'hi':>10}  {'c':>10}  {'f(c)':>6}  {'d':>10}  {'f(d)':>6}"
        print(header)
        for row in iter_log:
            print(f"{row[0]:4d}  {row[1]:10.4g}  {row[2]:10.4g}  {row[3]:10.4g}  {row[4]:6}  {row[5]:10.4g}  {row[6]:6}")
        fig, ax = plt.subplots(figsize=(8, 4))
        bds, mnns = zip(*history)
        ax.plot(bds, mnns, color="steelblue", marker="o", linewidth=1.2, label="evaluated")
        ax.set_xlabel("Bandwidth (bd)")
        ax.set_ylabel(f"# MNN pairs  (K={K})")
        ax.set_title("Golden-section search: MNN vs bandwidth")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return best_bd, best_mnn
