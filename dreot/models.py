import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import norm
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple


class WrappedGaussianMixture:
    """Mixture of wrapped Gaussians on theta in [a, a+L]."""

    def __init__(self, mus, sigmas, weights=None, a=0.0, L=2*np.pi, n_wrap=10):
        self.mus    = np.asarray(mus,    dtype=float)
        self.sigmas = np.asarray(sigmas, dtype=float)
        self.K      = len(self.mus)
        self.a      = float(a)
        self.L      = float(L)
        self.ks     = np.arange(-n_wrap, n_wrap + 1)
        raw_w       = np.ones(self.K) if weights is None else np.asarray(weights, dtype=float)
        self.weights = raw_w / raw_w.sum()

    def pdf(self, theta):
        theta = np.asarray(theta, dtype=float) - self.a
        out = np.zeros_like(theta)
        for w, mu, sigma in zip(self.weights, self.mus, self.sigmas):
            for k in self.ks:
                out += w * norm.pdf(theta, mu + k * self.L, sigma)
        return out

    def sample(self, n):
        idx     = np.random.choice(self.K, size=n, p=self.weights)
        samples = (np.random.normal(self.mus[idx], self.sigmas[idx], size=n)) % self.L + self.a
        return np.sort(samples)

    def plot_pdf(self, ax=None, num_points=1000, **plot_kwargs):
        if ax is None:
            ax = plt.gca()
        thetas = np.linspace(self.a, self.a + self.L, num_points)
        ax.plot(thetas, self.pdf(thetas), **plot_kwargs)
        ax.set_xlabel('theta')
        ax.set_ylabel('Density')


class UniformDistribution:
    """Uniform distribution on [a, a+L]."""

    def __init__(self, a=0.0, L=2*np.pi, **kwargs):
        self.a = float(a)
        self.L = float(L)

    def pdf(self, theta):
        theta = np.asarray(theta, dtype=float)
        return np.where((theta >= self.a) & (theta <= self.a + self.L), 1.0 / self.L, 0.0)

    def sample(self, n):
        return np.sort(np.random.uniform(self.a, self.a + self.L, size=n))

    def plot_pdf(self, ax=None, num_points=1000, **plot_kwargs):
        if ax is None:
            ax = plt.gca()
        thetas = np.linspace(self.a, self.a + self.L, num_points)
        ax.plot(thetas, self.pdf(thetas), **plot_kwargs)
        ax.set_xlabel('theta')
        ax.set_ylabel('Density')


@dataclass
class Arc:
    center: Tuple[float, float]
    radius: float
    density: Any  # UniformDistribution or WrappedGaussianMixture
    name: str = ""

    def __post_init__(self):
        self.theta_range = (self.density.a, self.density.a + self.density.L)

    @property
    def arc_length(self) -> float:
        return self.radius * (self.theta_range[1] - self.theta_range[0])

    def theta_to_xy(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta).ravel()
        cx, cy = self.center
        return np.column_stack([
            cx + self.radius * np.cos(theta),
            cy + self.radius * np.sin(theta)
        ])

    def sample_theta(self, n: int) -> np.ndarray:
        thetas = self.density.sample(n)
        return np.clip(thetas, *self.theta_range)

    def sample(self, n: int) -> np.ndarray:
        thetas = self.sample_theta(n)
        pts = self.theta_to_xy(thetas)
        return thetas, pts

    def density_at_theta(self, theta: np.ndarray) -> np.ndarray:
        return self.density.pdf(np.asarray(theta))

    def density_xy(self, theta: np.ndarray) -> np.ndarray:
        return self.density.pdf(np.asarray(theta)) / self.radius


class Manifold:
    def __init__(self, arcs: List[Arc], arc_weights: Optional[np.ndarray] = None):
        self.arcs = arcs
        K = len(arcs)
        if arc_weights is None:
            arc_weights = np.ones(K)
        arc_weights = np.asarray(arc_weights, dtype=float)
        self.arc_weights = arc_weights / arc_weights.sum()

    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        counts = np.random.multinomial(n, self.arc_weights)
        thetas_list, pts_list, ids_list = [], [], []
        for k, (arc, cnt) in enumerate(zip(self.arcs, counts)):
            if cnt > 0:
                thetas, pts = arc.sample(cnt)
                thetas_list.append(thetas)
                pts_list.append(pts)
                ids_list.append(np.full(cnt, k, dtype=int))
        thetas  = np.concatenate(thetas_list)
        pts     = np.vstack(pts_list)
        arc_ids = np.concatenate(ids_list)
        return thetas, pts, arc_ids

    def pdf_xy(self, thetas, arc_ids) -> np.ndarray:
        total_density = np.zeros(thetas.shape[0])
        for k, (arc, pw) in enumerate(zip(self.arcs, self.arc_weights)):
            theta_k = thetas[arc_ids == k]
            if theta_k.size > 0:
                total_density[arc_ids == k] = pw * arc.density_xy(theta_k)
        return total_density

    def get_density_range(self, num_points=1000):
        density = np.hstack([w * arc.density_xy(np.linspace(*arc.theta_range, num_points))
                             for arc, w in zip(self.arcs, self.arc_weights)])
        return density.min(), density.max()

    def plot_arcs(self, ax=None, num_points=1000, add_colorbar=True, **plot_kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        thetas  = np.vstack([np.linspace(*arc.theta_range, num_points) for arc in self.arcs])
        pts     = np.vstack([arc.theta_to_xy(thetas[k]) for k, arc in enumerate(self.arcs)])
        density = np.hstack([w * arc.density_xy(thetas[k])
                             for k, (arc, w) in enumerate(zip(self.arcs, self.arc_weights))])
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=density, cmap='viridis', **plot_kwargs)
        if add_colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(sc, cax=cax, label='Density')
        ax.set_aspect('equal')
        return sc
