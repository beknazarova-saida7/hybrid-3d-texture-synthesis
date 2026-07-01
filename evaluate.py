"""
Stage 1: Procedural base texture generation.
Implements fractal Brownian motion (fBM) blended with Voronoi cellular
noise, corresponding to Eq. (2)-(3) in the paper.
"""
import numpy as np


def _value_noise(shape, res, seed):
    """Bilinear value noise at a given lattice resolution."""
    rng = np.random.default_rng(seed)
    grid = rng.random((res + 1, res + 1))
    x = np.linspace(0, res, shape[0], endpoint=False)
    y = np.linspace(0, res, shape[1], endpoint=False)
    xi = x.astype(int)
    yi = y.astype(int)
    xf = x - xi
    yf = y - yi
    top = grid[xi][:, yi] * (1 - xf)[:, None] + grid[xi + 1][:, yi] * xf[:, None]
    bot = grid[xi][:, yi + 1] * (1 - xf)[:, None] + grid[xi + 1][:, yi + 1] * xf[:, None]
    return top * (1 - yf)[None, :] + bot * yf[None, :]


def fbm(shape, octaves=6, seed=0):
    """
    Fractal Brownian motion, Eq. (2):
        T_base(x, y) = sum_i (1/2^i) * N(2^i x, 2^i y),  i = 0..n-1
    """
    total = np.zeros(shape)
    amp = 1.0
    amp_sum = 0.0
    for i in range(octaves):
        res = 2 ** (i + 1)
        total += amp * _value_noise(shape, res, seed + i)
        amp_sum += amp
        amp *= 0.5
    return total / amp_sum


def voronoi(shape, n_sites=30, seed=1):
    """
    Voronoi cellular noise, distance to nearest site:
        V(x, y) = min_i || p - p_i ||
    """
    rng = np.random.default_rng(seed)
    sites = rng.random((n_sites, 2)) * np.array(shape)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    coords = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    d = np.sqrt(((coords[:, None, :] - sites[None, :, :]) ** 2).sum(-1)).min(1)
    d = d.reshape(shape)
    return d / d.max()


def generate_base_texture(shape=(256, 256), alpha=0.5, seed=0):
    """
    Eq. (3): T_base = alpha * fBM(x,y) + (1 - alpha) * V(x,y)
    alpha in [0, 1]:
        alpha = 1.0 -> wood-like patterns
        alpha = 0.5 -> stone
        alpha = 0.0 -> organic cellular textures
    """
    f = fbm(shape, octaves=6, seed=seed)
    v = voronoi(shape, n_sites=30, seed=seed + 100)
    base = alpha * f + (1 - alpha) * v
    return (base - base.min()) / (base.max() - base.min() + 1e-12)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, a, name in zip(axes, [1.0, 0.5, 0.0], ["wood (a=1.0)", "stone (a=0.5)", "cellular (a=0.0)"]):
        img = generate_base_texture(alpha=a, seed=42)
        ax.imshow(img, cmap="gray")
        ax.set_title(name)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("../results/stage1_examples.png", dpi=150)
    print("Saved results/stage1_examples.png")
