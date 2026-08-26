# DR-EOT — Density-Reweighted Entropic Optimal Transport

Code for the paper:

> **Density-Reweighted Entropic Optimal Transport: Decoupling Geometry from Sampling Density**  
> Keyi Li, Yuval Kluger, Boris Landa  
> Program of Applied & Computational Mathematics, Yale University  
> [arXiv:2608.16506](https://arxiv.org/abs/2608.16506)

---

## Problem

Standard EOT aligns datasets by minimizing transport cost subject to **uniform empirical marginals**. When two datasets share similar geometry but differ in sampling density, EOT matches points by relative density rather than geometric proximity — producing geometrically misleading correspondences.

**DR-EOT** addresses this by reweighting the kernel and marginals by the local sampling density, controlled by a parameter $\theta \in [0, 1]$:
- $\theta = 0$: standard EOT (density-sensitive)
- $\theta = 1$: geometry-driven alignment (density-discounted)
- Intermediate $\theta$: interpolation between the two regimes

## Installation

```bash
git clone https://github.com/Keyi-Li/DR-EOT.git
cd DR-EOT
pip install -e .

# For the UOT baseline in Figure 4:
pip install pot
```

**Dependencies:** numpy, scipy, scikit-learn, loguru, tqdm, matplotlib, pandas

## Quick Start

```python
import numpy as np
from sklearn.metrics import pairwise_distances
from dreot import sinkhorn_eot, sinkhorn_dreot, GlobalBootstrapLepski

# Two datasets X (m points) and Y (n points) in R^D
X, Y = ...   # your data, shape (m, D) and (n, D)

D = pairwise_distances(X, Y, metric="sqeuclidean")
eps = 0.1   # entropic regularization

# --- Option A: known densities ---
mu = ...  # density estimates at X, shape (m,)
nu = ...  # density estimates at Y, shape (n,)

row_s, col_s = sinkhorn_dreot(
    D, eps,
    mu.reshape(-1, 1), nu.reshape(-1, 1),
    alpha=1,   # theta = 1: fully geometry-driven
    delta=1e-6, max_iter=5000,
)
W = row_s * np.exp(-D / eps) * col_s.T   # (m, n) transport plan

# --- Option B: standard EOT (baseline) ---
row_s, col_s = sinkhorn_eot(
    D, eps,
    np.ones((m, 1)) * n,
    np.ones((n, 1)) * m,
    delta=1e-6, max_iter=5000,
)
W_eot = row_s * np.exp(-D / eps) * col_s.T
```

### Adaptive KDE bandwidth (Bootstrap-Lepski)

```python
from dreot import GlobalBootstrapLepski
from dreot.utils import knn_median

dist_XX = pairwise_distances(X, X, metric="sqeuclidean")
h_lo = 0.5 * np.sqrt(knn_median(dist_XX, 5))
h_hi = 0.5 * np.sqrt(knn_median(dist_XX, len(X) // 5))

lepski = GlobalBootstrapLepski(
    X,
    h_range=(h_lo, h_hi),
    n_bandwidths=50,
    n_bootstrap=20,
    alpha=0.05,
    C=1.0,
    seed=42,
)
result = lepski.select_bandwidth_global()
mu_hat = result.density_estimates   # shape (m,)
```

## Reproducing Paper Figures

Each figure has a dedicated notebook in `notebooks/`:

| Notebook | Figure | Description |
|---|---|---|
| `fig1_demo_mismatch.ipynb` | Fig 1 | Standard EOT vs DR-EOT on line+curve example |
| `fig2_theta_sweep.ipynb` | Fig 2 | Effect of varying θ ∈ {0, 1/3, 2/3, 1} |
| `fig3_convergence.ipynb` | Fig 3 | Empirical convergence rate of DR-EOT |
| `fig4_simulation.ipynb` | Fig 4 | Benchmark vs UOT on arc manifolds |
| `demo.ipynb` | — | Quick-start demo (smaller data, all steps) |

Run from the `notebooks/` directory:

```bash
cd notebooks
jupyter notebook fig1_demo_mismatch.ipynb
```

> **Note:** Figure 4 (`fig4_simulation.ipynb`) is compute-intensive (≈hours for 10 replicates). Set `N_SIMS = 1` for a quick sanity check.

## Package Structure

```
dreot/
├── sinkhorn.py      # Core algorithm: sinkhorn_eot, sinkhorn_dreot
├── kde.py           # Bootstrap-Lepski KDE: GlobalBootstrapLepski
├── models.py        # Data-generating models: Arc, Manifold, WrappedGaussianMixture
└── utils/
    ├── bandwidth.py # knn_median, self_normalized_kde
    ├── bd_search.py # golden_bandwidth_mnn (ε selection via MKNN)
    ├── evaluation.py # knn_intersection, knn_celltype, ...
    ├── mknn.py      # find_mutual_knn, pct_mknn, ...
    └── plotting.py  # highlight_minmax, plot_pct_mknn, ...
```

## Algorithm Summary

Given datasets $X = \{x_i\}_{i=1}^m$ and $Y = \{y_j\}_{j=1}^n$, density estimates $\hat{f}_i$ and $\hat{g}_j$, regularization $\varepsilon > 0$, and discounting factor $\theta \in [0,1]$:

1. **Scaling constant:**
$$S^{(\theta)} = \frac{\frac{1}{m}\sum_i \hat{f}_i^{-\theta}}{\frac{1}{n}\sum_j \hat{g}_j^{-\theta}}$$

2. **Adjusted kernel:**
$$M_{ij}^{(\theta)} = \frac{\exp\!\left(-\|x_i - y_j\|^2 / \varepsilon\right)}{\hat{f}_i^{\theta}\,\hat{g}_j^{\theta}}$$

3. **Sinkhorn scaling:** find $\alpha \in \mathbb{R}^m_+$, $\beta \in \mathbb{R}^n_+$ such that marginal constraints hold

4. **Output plan:**
$$W_{ij}^{(\theta)} = \alpha_i \exp\!\left(-\|x_i - y_j\|^2 / \varepsilon\right) \beta_j$$

## Citation

```bibtex
@misc{li2026dreot,
  title={Density-Reweighted Entropic Optimal Transport: Decoupling Geometry from Sampling Density},
  author={Li, Keyi and Kluger, Yuval and Landa, Boris},
  year={2026},
  eprint={2608.16506},
  archivePrefix={arXiv},
  primaryClass={stat.ML},
  url={https://arxiv.org/abs/2608.16506}
}
```
