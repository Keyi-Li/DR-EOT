
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class SinkhornNumericalError(RuntimeError):
    """
    Raised when a numerical issue (NaN, overflow, underflow) is detected
    during Sinkhorn computation. Catch this to handle bad inputs or
    ill-conditioned problems gracefully.

    Example
    -------
    try:
        row_scale, col_scale = sinkhorn_balance(D, bandwidth, row_sum, col_sum)
    except SinkhornNumericalError as e:
        print(f"Sinkhorn failed: {e}")
    """
    pass


def _compute_kernel(D, bandwidth, mu=None, rho=None):
    """Compute normalized kernel matrix."""
    if bandwidth <= 0:
        raise SinkhornNumericalError(
            f"bandwidth={bandwidth} is non-positive. "
        )

    K = np.exp(-D / bandwidth)

    if np.isnan(K).any():
        raise SinkhornNumericalError(
            "NaN in kernel matrix after exp(-D/bandwidth). "
        )
    if np.isinf(K).any():
        raise SinkhornNumericalError(
            "Inf in kernel matrix after exp(-D/bandwidth). "
        )

    zero_rows = np.flatnonzero(np.sum(K, axis=1) == 0)
    zero_cols = np.flatnonzero(np.sum(K, axis=0) == 0)
    if zero_rows.size:
        raise SinkhornNumericalError(
            f"{zero_rows.size} zero row sums detected at indices, e.g., {zero_rows[:min(5, zero_rows.size)].tolist()}. "
        )
    if zero_cols.size:
        raise SinkhornNumericalError(
            f"{zero_cols.size} zero column sums detected at indices, e.g., {zero_cols[:min(5, zero_cols.size)].tolist()}. "
        )

    if mu is not None and rho is not None:
        bad_mu = np.flatnonzero(mu.ravel() <= 0)
        bad_rho = np.flatnonzero(rho.ravel() <= 0)
        if bad_mu.size:
            raise SinkhornNumericalError(
                f"mu contains {bad_mu.size} non-positive value(s) at indices, e.g., {bad_mu[:min(5, bad_mu.size)].tolist()}. "
            )
        if bad_rho.size:
            raise SinkhornNumericalError(
                f"rho contains {bad_rho.size} non-positive value(s) at indices, e.g., {bad_rho[:min(5, bad_rho.size)].tolist()}. "
            )
        K = K / (mu * rho.T)
        if np.isnan(K).any():
            raise SinkhornNumericalError(
                "NaN detected after K / (mu * rho.T). "
            )
        if np.isinf(K).any():
            raise SinkhornNumericalError(
                "Inf detected after K / (mu * rho.T). "
            )
    return K


def _check_convergence(K_final, row_sum, col_sum, delta, prev_diffs=None, stagnation_tol=1e-3, verbose=False, iteration=None):
    """
    Check convergence and stagnation. Raises SinkhornNumericalError on NaN.

    Returns:
        (status, max_diff_row, max_diff_col)
        status: "converged", "stagnated", or None (keep iterating)
    """
    if np.isnan(K_final).any():
        iter_str = f" at iteration {iteration}" if iteration is not None else ""
        raise SinkhornNumericalError(
            f"NaN detected in the transport plan{iter_str}. "
        )
    max_diff_row = np.abs(np.sum(K_final, axis=1, keepdims=True) - row_sum).max()
    max_diff_col = np.abs(np.sum(K_final, axis=0, keepdims=True).T - col_sum).max()
    if verbose:
        iter_str = f"iter {iteration}: " if iteration is not None else ""
        print(f"{iter_str}max_diff_row={max_diff_row:.6e}, max_diff_col={max_diff_col:.6e}")
    if max_diff_row < delta and max_diff_col < delta:
        return "converged", max_diff_row, max_diff_col
    if prev_diffs is not None:
        prev_row, prev_col = prev_diffs
        row_change = abs(max_diff_row - prev_row) 
        col_change = abs(max_diff_col - prev_col)
        if row_change < stagnation_tol and col_change < stagnation_tol:
            return "stagnated", max_diff_row, max_diff_col
    return None, max_diff_row, max_diff_col


def _check_scales(row_scale, col_scale, iteration):
    """Raise SinkhornNumericalError if scaling vectors contain NaN, Inf, or zero."""
    iter_str = f" at iteration {iteration}" if iteration is not None else ""
    for name, scale in [("row_scale", row_scale), ("col_scale", col_scale)]:
        if np.isnan(scale).any():
            raise SinkhornNumericalError(
                f"NaN in {name}{iter_str}. "
            )
        if np.isinf(scale).any():
            raise SinkhornNumericalError(
                f"Inf in {name}{iter_str}. "
            )
        if np.any(scale == 0):
            raise SinkhornNumericalError(
                f"Zero values in {name}{iter_str}. "
            )


def _check_normalization_scale(scale, iteration):
    """Raise SinkhornNumericalError if the per-iteration balance factor is degenerate."""
    iter_str = f" at iteration {iteration}"
    if np.isnan(scale):
        raise SinkhornNumericalError(
            f"Normalization scaler sqrt(mean(row_scale)/mean(col_scale)) is NaN{iter_str}. "
        )
    if np.isinf(scale):
        raise SinkhornNumericalError(
            f"Normalization scaler sqrt(mean(row_scale)/mean(col_scale)) is Inf{iter_str}. "
        )
    if scale == 0:
        raise SinkhornNumericalError(
            f"Normalization scaler sqrt(mean(row_scale)/mean(col_scale)) is zero{iter_str}. "
        )


def _check_bad_convergence(plan, row_sum, col_sum, bad_convergence_tol, context_str,
                           raise_on_bad=True):
    """
    Compute max relative marginal errors.
    - Within tolerance: always logs a warning and returns.
    - Exceeded tolerance: raises SinkhornNumericalError if raise_on_bad=True,
      otherwise logs a warning and returns so the caller still gets the scales.
    """
    row_errs = np.abs(np.sum(plan, axis=1) - row_sum.ravel())
    col_errs = np.abs(np.sum(plan, axis=0) - col_sum.ravel())
    rel_row = float((row_errs / row_sum.ravel()).max())
    rel_col = float((col_errs / col_sum.ravel()).max())
    if rel_row < bad_convergence_tol and rel_col < bad_convergence_tol:
        logger.warning(
            f"{context_str}: rel_row={rel_row:.4e}, rel_col={rel_col:.4e} "
            f"both within bad_convergence_tol={bad_convergence_tol:.4e}. Returning."
        )
    elif raise_on_bad:
        raise SinkhornNumericalError(
            f"{context_str} with unacceptable relative errors: "
            f"row rel_error={rel_row:.4e}, col rel_error={rel_col:.4e}. "
            f"Exceeded bad_convergence_tol={bad_convergence_tol:.4e}."
        )
    else:
        logger.warning(
            f"{context_str} with unacceptable relative errors: "
            f"row rel_error={rel_row:.4e}, col rel_error={rel_col:.4e}. "
            f"Exceeded bad_convergence_tol={bad_convergence_tol:.4e}. Returning anyway."
        )


def sinkhorn_balance(D, bandwidth, row_sum, col_sum, delta=1e-12, max_iter=1000000, check_freq=100, stagnation_tol=1e-3, bad_convergence_tol=0.05, raise_on_bad_convergence=True, verbose=False):
    """
    Standard Sinkhorn with scaling factors balanced.

    Args:
        D: Distance matrix (n x m)
        bandwidth: Temperature parameter
        row_sum: Target row sums (n x 1)
        col_sum: Target column sums (m x 1)
        delta: Convergence tolerance
        max_iter: Maximum iterations
        check_freq: Convergence check frequency
        stagnation_tol: Relative change threshold below which both errors are
            considered stagnated (default 1e-3). Set to 0 to disable.
        bad_convergence_tol: Relative marginal error threshold for classifying
            convergence quality on stagnation or max_iter exit. Default 0.05.
        raise_on_bad_convergence: If True (default), raise SinkhornNumericalError
            when relative errors exceed bad_convergence_tol. If False, log a
            warning and return row_scale, col_scale regardless.

    Returns:
        row_scale, col_scale: Scaling factors

    Raises:
        SinkhornNumericalError: on NaN, overflow, or underflow; and on bad
            convergence when raise_on_bad_convergence=True.
    """
    K = _compute_kernel(D, bandwidth)

    row_scale = row_sum / np.sum(K, axis=1, keepdims=True)
    col_scale = col_sum / np.sum(K * row_scale, axis=0, keepdims=True).T
    _check_scales(row_scale, col_scale, iteration="initialization")

    prev_diffs = None
    for i in tqdm(range(max_iter), desc="Standard Sinkhorn"):
        row_scale = row_sum / np.sum(K * col_scale.T, axis=1, keepdims=True)
        col_scale = col_sum / np.sum(K * row_scale, axis=0, keepdims=True).T

        # Normalize to prevent numerical overflow
        scale = np.sqrt(np.mean(row_scale) / np.mean(col_scale))
        _check_normalization_scale(scale, i)
        row_scale /= scale
        col_scale *= scale

        if not i % check_freq:
            _check_scales(row_scale, col_scale, iteration=i)
            K_final = row_scale * K * col_scale.T
            status, max_diff_row, max_diff_col = _check_convergence(
                K_final, row_sum, col_sum, delta,
                prev_diffs=prev_diffs, stagnation_tol=stagnation_tol,
                verbose=verbose, iteration=i
            )
            prev_diffs = (max_diff_row, max_diff_col)
            if status == "converged":
                logger.info(f"Converged at iteration {i}")
                break
            elif status == "stagnated":
                _check_bad_convergence(K_final, row_sum, col_sum, bad_convergence_tol,
                                       f"Stagnation at iteration {i}",
                                       raise_on_bad=raise_on_bad_convergence)
                break
    else:
        K_final = row_scale * K * col_scale.T
        _check_bad_convergence(K_final, row_sum, col_sum, bad_convergence_tol,
                               f"max_iter={max_iter} reached without convergence",
                               raise_on_bad=raise_on_bad_convergence)

    return row_scale, col_scale


def sinkhorn_density_adjusted(D, bandwidth, mu, rho, alpha=1, delta=1e-12, max_iter=1000000, check_freq=100, stagnation_tol=1e-3, bad_convergence_tol=0.05, raise_on_bad_convergence=True, verbose=False):
    """
    Sinkhorn with density re-weighting.

    Args:
        D: Distance matrix (n x m)
        bandwidth: Temperature parameter
        mu: Density scaling (n x 1)
        rho: Density scaling (m x 1)
        alpha: Density exponent
        delta: Convergence tolerance
        max_iter: Maximum iterations
        check_freq: Convergence check frequency
        stagnation_tol: Relative change threshold below which both errors are
            considered stagnated (default 1e-3). Set to 0 to disable.
        bad_convergence_tol: On stagnation, the marginal error for the worst
            row/col is divided by its target sum. If the resulting relative
            error exceeds this threshold, SinkhornNumericalError is raised.
            Default 0.01 (1%).

    Returns:
        row_scale, col_scale: Scaling factors

    Raises:
        SinkhornNumericalError: on NaN, overflow, or underflow.
            Bad convergence (stagnation or max_iter) is reported via
            logger.warning and row_scale, col_scale are returned regardless.
    """
    mu = mu.reshape(-1, 1)
    rho = rho.reshape(-1, 1)

    bad_mu = np.flatnonzero(mu.ravel() <= 0)
    bad_rho = np.flatnonzero(rho.ravel() <= 0)
    if bad_mu.size:
        raise SinkhornNumericalError(
            f"mu contains {bad_mu.size} non-positive value(s) at indices {bad_mu[:5].tolist()}. "
        )
    if bad_rho.size:
        raise SinkhornNumericalError(
            f"rho contains {bad_rho.size} non-positive value(s) at indices {bad_rho[:5].tolist()}. "
        )

    mu  = np.power(mu, alpha)
    rho = np.power(rho, alpha)

    if np.isnan(mu).any() or np.isnan(rho).any():
        raise SinkhornNumericalError(
            f"NaN in mu or rho after np.power(..., alpha={alpha}). "
        )

    M = _compute_kernel(D, bandwidth, mu, rho)
    m, n = M.shape

    mean_inv_mu = np.mean(1 / mu)
    mean_inv_rho = np.mean(1 / rho)
    if np.isinf(mean_inv_mu):
        raise SinkhornNumericalError(
            "mean(1/mu) is Inf after applying alpha. "
        )
    if np.isinf(mean_inv_rho) or mean_inv_rho == 0:
        raise SinkhornNumericalError(
            f"mean(1/rho) = {mean_inv_rho} is degenerate (Inf or zero) after applying alpha. "
        )

    s = np.sqrt(mean_inv_mu / mean_inv_rho)
    if np.isnan(s) or np.isinf(s) or s == 0:
        raise SinkhornNumericalError(
            f"Normalization scale s = sqrt(mean(1/mu)/mean(1/rho)) = {s} is degenerate. "
        )

    row_sum = np.ones((m, 1)) * n / mu / s
    col_sum = np.ones((n, 1)) * m / rho * s

    row_scale = row_sum / np.sum(M, axis=1, keepdims=True)
    col_scale = col_sum / np.sum(M * row_scale, axis=0, keepdims=True).T
    _check_scales(row_scale, col_scale, iteration="initialization")

    prev_diffs = None
    for i in tqdm(range(max_iter), desc=f"Sinkhorn density adjusted (alpha={alpha})"):
        row_scale = row_sum / np.sum(M * col_scale.T, axis=1, keepdims=True)
        col_scale = col_sum / np.sum(M * row_scale, axis=0, keepdims=True).T

        # Normalize to prevent numerical overflow
        scale = np.sqrt(np.mean(row_scale) / np.mean(col_scale))
        _check_normalization_scale(scale, i)
        row_scale /= scale
        col_scale *= scale

        if i % check_freq == 0:
            _check_scales(row_scale, col_scale, iteration=i)
            M_final = row_scale * M * col_scale.T
            status, max_diff_row, max_diff_col = _check_convergence(
                M_final, row_sum, col_sum, delta,
                prev_diffs=prev_diffs, stagnation_tol=stagnation_tol,
                verbose=verbose, iteration=i
            )
            prev_diffs = (max_diff_row, max_diff_col)
            if status == "converged":
                logger.info(f"Converged at iteration {i}")
                break
            elif status == "stagnated":
                _check_bad_convergence(M_final, row_sum, col_sum, bad_convergence_tol,
                                       f"Stagnation at iteration {i}",
                                       raise_on_bad=raise_on_bad_convergence)
                break
    else:
        M_final = row_scale * M * col_scale.T
        _check_bad_convergence(M_final, row_sum, col_sum, bad_convergence_tol,
                               f"max_iter={max_iter} reached without convergence",
                               raise_on_bad=raise_on_bad_convergence)

    return row_scale, col_scale