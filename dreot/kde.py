from __future__ import annotations
"""
Bootstrap-Lepski Bandwidth Selection for Kernel Density Estimation.

Implements Algorithm 4 from the reference: adaptive bandwidth selection
using Lepski's method with bootstrap calibration of the noise threshold.

The core idea:
    - For a grid of bandwidths h_1 < h_2 < ... < h_m, estimate scaled KDEs.
    - Use bootstrap resampling to build a null distribution of the difference
      between KDE estimates at two different bandwidths.
    - The Lepski rule selects the largest bandwidth h* such that, for all
      finer bandwidths h_i < h*, the observed difference is explainable
      by sampling variability alone (i.e., bias has not yet emerged from
      the noise floor).

References:
    Algorithm 4 — Bootstrap-Lepski Bandwidth Selection.
"""

import numpy as np
from sklearn.metrics import pairwise_distances
from loguru import logger
from dataclasses import dataclass
from typing import Optional


@dataclass
class LepskiResult:
    """Container for the result of bandwidth selection at a single point."""

    point_index: int
    optimal_h_index: int
    optimal_h: float
    density_estimate: float


@dataclass
class GlobalLepskiResult:
    """Result of global bandwidth selection (one bandwidth for all points)."""
 
    optimal_h_index: int
    optimal_h: float
    density_estimates: np.ndarray  # shape (n,)


class BootstrapLepski:
    """Bootstrap-Lepski adaptive bandwidth selector for kernel density estimation.

    The bandwidth grid is always geometric as required by the algorithm:
        h_1 < h_2 < ... < h_m,   where  h_{j+1} = h_j / beta,  beta in (0, 1).
    Since beta < 1, dividing by beta *increases* h, so h_1 is the smallest
    (finest) bandwidth and h_m is the largest (smoothest).

    Parameters
    ----------
    X : np.ndarray, shape (n, D)
        Dataset of n points in R^D.
    h_range : tuple[float, float]
        Range of bandwidths (h1, h2) where 0 < h1 < h2.
    n_bandwidths : int
        Number of bandwidths m in the grid.
    n_bootstrap : int
        Number of bootstrap resamples (k in the algorithm).
    alpha : float
        Confidence level in (0, 1). The threshold is the (1 - alpha)-quantile
        of the bootstrap null distribution.
    C : float, optional
        Multiplicative constant in front of the quantile threshold. Default 1.0
        recovers the algorithm exactly; values > 1 give a more conservative
        (smoother) selection.
    seed : int or None, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        X: np.ndarray,
        h_range: tuple[float, float],
        n_bandwidths: int,
        n_bootstrap: int,
        alpha: float,
        C: float = 1.0,
        n_jobs: Optional[int] = 1,
        seed: Optional[int] = 42,
    ):
        # ── Validate inputs ──────────────────────────────────────────────
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}.")
        if h_range[0] < 0 or h_range[1] < 0 or h_range[0] > h_range[1]:
            raise ValueError(f"h_range must be a tuple of two positive numbers with the first less than the second, got {h_range}.")
        if n_bandwidths < 2:
            raise ValueError(f"n_bandwidths must be >= 2, got {n_bandwidths}.")
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        if n_bootstrap < 1:
            raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}.")

        self.X = X
        self.n, self.dim = X.shape
        self.bandwidth_grid = np.geomspace(*h_range, num=n_bandwidths)
        self.n_bandwidths = n_bandwidths
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.quantile_level = (1 - alpha) * 100
        self.C = C
        self.n_jobs = n_jobs
        self.rng = np.random.default_rng(seed)

        # ── Precomputed quantities (populated by _precompute) ────────────
        self._sq_dist: Optional[np.ndarray] = None          # (n, n)
        self._sq_dist_boot: list[np.ndarray] = []           # list of (n, n_bootstrap) arrays
        self._f_hat: Optional[np.ndarray] = None             # (n, n_bandwidths)
        self._normalizers: Optional[np.ndarray] = None       # (n_bandwidths,)
        self._f_hat_boot: Optional[np.ndarray] = None        # (n, n_bandwidths, n_bootstrap)

        self._precompute()

    # ════════════════════════════════════════════════════════════════════
    #  Precomputation pipeline
    # ════════════════════════════════════════════════════════════════════

    def _precompute(self) -> None:
        """Run the full precomputation: bootstrap, distances, density estimates."""
        logger.info("Starting precomputation pipeline.")
        boot_samples = self._generate_bootstrap_samples()
        self._compute_distances(boot_samples)
        self._compute_density_estimates()
        self._compute_bootstrap_density_estimates()
        logger.info("Precomputation complete.")

    def _generate_bootstrap_samples(self) -> list[np.ndarray]:
        """Step 1: Generate n_bootstrap bootstrap datasets by resampling rows of X."""
        logger.info(f"Generating {self.n_bootstrap} bootstrap samples.")
        samples = []
        for _ in range(self.n_bootstrap):
            indices = self.rng.choice(self.n, size=self.n, replace=True)
            samples.append(self.X[indices,:])
        return samples

    def _compute_distances(self, boot_samples: list[np.ndarray]) -> None:
        """Step 2: Compute squared Euclidean distance matrices.

        - D(X, X)          for the original dataset      [Eq. 3.27]
        - D(X, X_boot^(l))  for each bootstrap replicate  [Eq. 3.28]
        """
        logger.info("Computing squared distance matrices.")
        self._sq_dist = pairwise_distances(self.X, self.X, metric="sqeuclidean", n_jobs=self.n_jobs)
        self._sq_dist_boot = [
            pairwise_distances(self.X, X_b, metric="sqeuclidean", n_jobs=self.n_jobs)
            for X_b in boot_samples
        ]

    # ── Density estimation helpers ───────────────────────────────────────

    @staticmethod
    def _gaussian_kernel_sum(sq_dist: np.ndarray, h: float) -> np.ndarray:
        """Compute row-wise mean of the Gaussian kernel: (1/n) sum_j exp(-D_pj / 2h^2).

        Parameters
        ----------
        sq_dist : np.ndarray, shape (n_query, n_ref)
            Squared Euclidean distances.
        h : float
            Bandwidth.

        Returns
        -------
        np.ndarray, shape (n_query,)
            Unnormalized density values at each query point.
        """
        return np.mean(np.exp(-sq_dist / (2.0 * h**2)), axis=1)

    def _compute_density_estimates(self) -> None:
        """Step 3a: Compute scaled density estimates f_hat_h(x_p) for all h.  [Eq. 3.29]

        For each bandwidth h, the scaled density is:
            f_hat_h(x_p) = [sum_j K_h(x_p, x_j)] / [(1/n) sum_i sum_j K_h(x_i, x_j)]

        where K_h(x, y) = exp(-||x - y||^2 / 2h^2).
        """
        logger.info("Computing density estimates on original data.")
        n_h = self.n_bandwidths
        f_hat = np.empty((self.n, n_h))
        normalizers = np.empty(n_h)

        for j, h in enumerate(self.bandwidth_grid):
            unnormed = self._gaussian_kernel_sum(self._sq_dist, h)  # (n,)
            normalizers[j] = np.mean(unnormed)
            f_hat[:, j] = unnormed / normalizers[j]

        self._f_hat = f_hat
        self._normalizers = normalizers

    def _compute_bootstrap_density_estimates(self) -> None:
        """Step 3b: Compute bootstrap density estimates f_hat_h^(l)(x_p).

        Uses D^(l) in the numerator and D in the denominator (via stored normalizers),
        as specified below Eq. (3.29).
        """
        logger.info("Computing bootstrap density estimates.")
        n_h = self.n_bandwidths
        # Preallocate: (n_points, n_bandwidths, n_bootstrap)
        f_boot = np.empty((self.n, n_h, self.n_bootstrap))

        for ell, sq_dist_b in enumerate(self._sq_dist_boot):
            for j, h in enumerate(self.bandwidth_grid):
                unnormed = self._gaussian_kernel_sum(sq_dist_b, h)
                f_boot[:, j, ell] = unnormed / self._normalizers[j]

        self._f_hat_boot = f_boot

    # ════════════════════════════════════════════════════════════════════
    #  Lepski bandwidth selection
    # ════════════════════════════════════════════════════════════════════

    def _bootstrap_null_pointwise(
        self, point_idx: int, h_large_idx: int, h_small_idx: int
    ) -> np.ndarray:
        """Compute the bootstrap null distribution for the deviation.  [Eq. 3.31–3.32]

        For a given point x_p and bandwidth pair (h_j, h_i) with h_i < h_j:
            V_ij^(l) = f_hat_{h_j}^(l)(x_p) - f_hat_{h_i}^(l)(x_p)
            V_bar_ij = mean_l [ V_ij^(l) ]
            tilde{V}_ij^(l) = | V_ij^(l) - V_bar_ij |

        Returns
        -------
        np.ndarray, shape (n_bootstrap,)
            The centered absolute deviations {tilde{V}_ij^(l)}.
        """
        # V_ij^(l) for each bootstrap replicate
        v_boot = (
            self._f_hat_boot[point_idx, h_large_idx, :]
            - self._f_hat_boot[point_idx, h_small_idx, :]
        )
        # Center and take absolute value
        v_mean = np.mean(v_boot)
        return np.abs(v_boot - v_mean)

    def _threshold_pointwise(
        self, point_idx: int, h_large_idx: int, h_small_idx: int
    ) -> float:
        """Compute the Lepski threshold T_ij(x_p).  [Eq. 3.33]

        T_ij = C * q_{1-alpha}( tilde{V}_ij^(1), ..., tilde{V}_ij^(k) )
        """
        null_dist = self._bootstrap_null_pointwise(
            point_idx, h_large_idx, h_small_idx
        )
        return self.C * np.percentile(null_dist, self.quantile_level, method="nearest")

    def _lepski_test_pointwise(
        self, point_idx: int, h_large_idx: int, h_small_idx: int
    ) -> bool:
        """Check whether the Lepski condition holds for one (i, j) pair at one point.

        The condition (pointwise version of Part 5) is:
            |V_ij(x_p)| = |f_hat_{h_j}(x_p) - f_hat_{h_i}(x_p)| <= T_ij(x_p)
        """
        observed_diff = abs(
            self._f_hat[point_idx, h_large_idx] - self._f_hat[point_idx, h_small_idx]
        )
        threshold = self._threshold_pointwise(point_idx, h_large_idx, h_small_idx)
        return observed_diff <= threshold

    def select_bandwidth_pointwise(self, point_idx: int) -> LepskiResult:
        """Select the optimal bandwidth for a single point x_p using the Lepski rule.

        Iterates from the smallest bandwidth upward. For each h_j, checks whether
        all comparisons to finer bandwidths h_i < h_j pass. Returns the largest
        h_j for which all comparisons pass.

        Parameters
        ----------
        point_idx : int
            Index of the evaluation point in X.

        Returns
        -------
        LepskiResult
            Contains the selected bandwidth index, bandwidth value, and
            corresponding density estimate.
        """
        optimal_idx = 0  # fallback: smallest bandwidth

        for j in range(1, self.n_bandwidths):
            all_pass = True
            for i in range(j):
                if not self._lepski_test_pointwise(point_idx, h_large_idx=j, h_small_idx=i):
                    logger.debug(
                        f"Point {point_idx}: Lepski test failed at "
                        f"h[{j}]={self.bandwidth_grid[j]:.4e} vs h[{i}]={self.bandwidth_grid[i]:.4e}"
                    )
                    all_pass = False
                    break  # no need to check other i's for this j
            if not all_pass:
                break
            optimal_idx = j  # this j passed all comparisons, update candidate

        
        # np.fill_diagonal(self._sq_dist, np.inf)
        # optimal_estimate = self._gaussian_kernel_sum(self._sq_dist, self.bandwidth_grid[optimal_idx])
        # optimal_estimate /= np.mean(optimal_estimate)

        return LepskiResult(
            point_index=point_idx,
            optimal_h_index=optimal_idx,
            optimal_h=self.bandwidth_grid[optimal_idx],
            density_estimate=self._f_hat[point_idx,optimal_idx],
        )

    def select_bandwidth_all_points(self) -> list[LepskiResult]:
        """Run pointwise Lepski bandwidth selection for every point in X.

        Returns
        -------
        list[LepskiResult]
            One result per data point.
        """
        logger.info(f"Running pointwise Lepski selection for {self.n} points.")
        results = [self.select_bandwidth_pointwise(p) for p in range(self.n)]
        logger.info("Pointwise selection complete.")
        return results
    

class GlobalBootstrapLepski(BootstrapLepski):
    """Global Bootstrap-Lepski: selects a single bandwidth for all points.
 
    Implements Part 5 of the algorithm:
 
        h* = max { h_j : sum_p |V_ij(x_p)| <= sum_p T_ij(x_p),  for all i < j }
 
    Two aggregation modes control how the threshold sum_p T_ij(x_p) is built
    from the bootstrap null distribution (shape (n, k)):
 
        ind=True  (sum-of-quantiles, matches the algorithm):
            For each point p, take the (1-alpha)-quantile over the k bootstrap
            replicates, then sum over points.
                threshold = C * sum_p  q_{1-alpha}( tilde{V}_ij^(1)(x_p), ..., tilde{V}_ij^(k)(x_p) )
 
        ind=False (quantile-of-sums, alternative):
            For each bootstrap replicate l, sum the deviations over points,
            then take a single (1-alpha)-quantile of those k sums.
                threshold = C * q_{1-alpha}( sum_p tilde{V}_ij^(1)(x_p), ..., sum_p tilde{V}_ij^(k)(x_p) )
 
    The two are generally NOT equal.  ind=True is more conservative (larger
    threshold) because the sum of quantiles >= quantile of sums by
    subadditivity.
 
    All precomputation is inherited from BootstrapLepski.
 
    Parameters
    ----------
    X, h1, beta, n_bandwidths, n_bootstrap, alpha, C, seed :
        See BootstrapLepski for details.
    ind : bool, default True
        Aggregation mode for the threshold:
            True  — sum of pointwise quantiles (matches the algorithm).
            False — quantile of pointwise sums  (alternative).
    """
 
    def __init__(
        self,
        X: np.ndarray,
        h_range: tuple[float, float],
        n_bandwidths: int,
        n_bootstrap: int,
        alpha: float,
        C: float = 1.0,
        n_jobs: Optional[int] =1,
        seed: Optional[int] = 42,
        ind: bool = True,
    ):
        super().__init__(X, h_range, n_bandwidths, n_bootstrap, alpha, C, n_jobs, seed)
        self.ind = ind
 
    # ── Bootstrap null distribution (vectorized over all points) ─────────
 
    def _bootstrap_null_all_points(
        self, h_large_idx: int, h_small_idx: int
    ) -> np.ndarray:
        """Bootstrap null distribution across ALL points simultaneously.  [Eq. 3.31–3.32]
 
        Computes the centered absolute deviations for every point at once:
            V_ij^(l)(x_p)        = f_hat_{h_j}^(l)(x_p) - f_hat_{h_i}^(l)(x_p)
            bar{V}_ij(x_p)       = (1/k) sum_l V_ij^(l)(x_p)
            tilde{V}_ij^(l)(x_p) = |V_ij^(l)(x_p) - bar{V}_ij(x_p)|
 
        Returns
        -------
        np.ndarray, shape (n, k)
            Centered absolute deviations for each (point, bootstrap replicate).
        """
        v_boot = (
            self._f_hat_boot[:, h_large_idx, :]
            - self._f_hat_boot[:, h_small_idx, :]
        )  # (n, k)
        v_mean = np.mean(v_boot, axis=1, keepdims=True)  # (n, 1)
        return np.abs(v_boot - v_mean)  # (n, k)
 
    # ── Global threshold and test ────────────────────────────────────────
 
    def _global_threshold(self, h_large_idx: int, h_small_idx: int) -> float:
        """Aggregated threshold for the global Lepski test.
 
        Uses self.ind to select the aggregation mode:
            True  — sum of pointwise quantiles (matches algorithm).
            False — quantile of pointwise sums  (alternative).
        """
        null_all = self._bootstrap_null_all_points(h_large_idx, h_small_idx)  # (n, k)
 
        if self.ind:
            # Per-point quantile over bootstrap axis, then sum over points.
            pointwise_quantiles = np.percentile(
                null_all, self.quantile_level, axis=1, method="nearest"
            )  # (n,)
            return self.C * np.sum(pointwise_quantiles)
        else:
            # Sum over points for each bootstrap replicate, then single quantile.
            bootstrap_sums = np.sum(null_all, axis=0)  # (k,)
            return self.C * np.percentile(
                bootstrap_sums, self.quantile_level, method="nearest"
            )
 
    def _global_observed(self, h_large_idx: int, h_small_idx: int) -> float:
        """Aggregated observed statistic:  sum_p |V_ij(x_p)|.  [Eq. 3.30 + Part 5]"""
        return np.sum(np.abs(
            self._f_hat[:, h_large_idx] - self._f_hat[:, h_small_idx]
        ))
 
    def _lepski_test_global(self, h_large_idx: int, h_small_idx: int) -> bool:
        """Check the global Lepski condition for one (i, j) pair.
 
            sum_p |V_ij(x_p)| <= threshold
        """
        return self._global_observed(h_large_idx, h_small_idx) <= \
               self._global_threshold(h_large_idx, h_small_idx)
 
    # ── Selection routine ────────────────────────────────────────────────
 
    def select_bandwidth_global(self) -> GlobalLepskiResult:
        """Select the optimal global bandwidth using the Lepski rule (Part 5).
 
        Iterates from the smallest bandwidth upward. For each h_j, checks
        whether the aggregated condition holds for all finer h_i < h_j.
        Returns the largest h_j for which all comparisons pass.
 
        Returns
        -------
        GlobalLepskiResult
            Contains the selected bandwidth index, value, and density
            estimates at all points using that bandwidth.
        """
        mode_str = "sum-of-quantiles" if self.ind else "quantile-of-sums"
        logger.info(f"Running global Lepski selection (mode: {mode_str}).")
        optimal_idx = 0
 
        for j in range(1, self.n_bandwidths):
            all_pass = True
            for i in range(j):
                if not self._lepski_test_global(h_large_idx=j, h_small_idx=i):
                    logger.debug(
                        f"Global: failed at "
                        f"h[{j}]={self.bandwidth_grid[j]:.4e} vs "
                        f"h[{i}]={self.bandwidth_grid[i]:.4e}"
                    )
                    all_pass = False
                    break
            if not all_pass:
                break
            optimal_idx = j
 
        logger.info(
            f"Global optimal ({mode_str}): "
            f"h[{optimal_idx}] = {self.bandwidth_grid[optimal_idx]:.4e}"
        )

                
        # np.fill_diagonal(self._sq_dist, np.inf)
        # optimal_estimate = self._gaussian_kernel_sum(self._sq_dist, self.bandwidth_grid[optimal_idx])
        # optimal_estimate /= np.mean(optimal_estimate)
        
        return GlobalLepskiResult(
            optimal_h_index=optimal_idx,
            optimal_h=self.bandwidth_grid[optimal_idx],
            density_estimates= self._f_hat[:,optimal_idx].copy(),
        )

