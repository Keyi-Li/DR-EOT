"""
dreot — Density-Reweighted Entropic Optimal Transport

Core API
--------
sinkhorn_eot(D, eps, row_sum, col_sum, ...)
    Standard EOT via balanced Sinkhorn-Knopp.

sinkhorn_dreot(D, eps, mu, nu, theta, ...)
    Density-reweighted EOT (Algorithm 1 in the paper).

GlobalBootstrapLepski
    Bootstrap-Lepski adaptive bandwidth selector for KDE.

Supporting utilities are in dreot.utils.
"""

from .sinkhorn import sinkhorn_balance as sinkhorn_eot
from .sinkhorn import sinkhorn_density_adjusted as sinkhorn_dreot
from .sinkhorn import SinkhornNumericalError
from .kde import GlobalBootstrapLepski, BootstrapLepski

__version__ = "0.1.0"
__all__ = [
    "sinkhorn_eot",
    "sinkhorn_dreot",
    "SinkhornNumericalError",
    "GlobalBootstrapLepski",
    "BootstrapLepski",
]
