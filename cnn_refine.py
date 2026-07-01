"""
Stage 2: B-spline UV geometry adaptation.
Implements Eq. (4)-(7): a tensor-product bicubic B-spline displacement
field, fitted on a sparse 8x8 grid of control points with SciPy's
RectBivariateSpline, applied to the procedural base texture.
"""
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import map_coordinates


def bspline_displacement_field(shape, grid_n=8, sigma=0.05, seed=0):
    """
    Fits a degree-3 (bicubic) B-spline interpolant to a sparse grid_n x grid_n
    grid of displacement control points {d_ij} ~ N(0, sigma^2), producing
    dense displacement fields Delta_u(x, y) and Delta_v(x, y) over the full
    texture resolution (Eq. 4-6).
    """
    rng = np.random.default_rng(seed)
    ctrl_u = rng.normal(0, sigma, (grid_n, grid_n))
    ctrl_v = rng.normal(0, sigma, (grid_n, grid_n))

    cx = np.linspace(0, shape[0] - 1, grid_n)
    cy = np.linspace(0, shape[1] - 1, grid_n)
    spline_u = RectBivariateSpline(cx, cy, ctrl_u, kx=3, ky=3)
    spline_v = RectBivariateSpline(cx, cy, ctrl_v, kx=3, ky=3)

    xs = np.arange(shape[0])
    ys = np.arange(shape[1])
    du = spline_u(xs, ys) * shape[0]
    dv = spline_v(xs, ys) * shape[1]
    return du, dv


def warp_texture(texture, du=None, dv=None, grid_n=8, sigma=0.05, seed=0):
    """
    Eq. (7): T_warped(x, y) = T_base(x + Delta_u(x,y), y + Delta_v(x,y))
    O(n^2) time for an n x n texture; zero neural parameters.
    """
    shape = texture.shape
    if du is None or dv is None:
        du, dv = bspline_displacement_field(shape, grid_n=grid_n, sigma=sigma, seed=seed)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    coords = np.stack([
        np.clip(yy + dv, 0, shape[0] - 1),
        np.clip(xx + du, 0, shape[1] - 1),
    ])
    return map_coordinates(texture, coords, order=1, mode="reflect")


def warp_from_curvature(texture, curvature_map, scale=0.05):
    """
    For real 3D mesh validation (Section III-B / IV-E): derives the
    displacement field directly from a mesh's principal-curvature map
    (e.g., computed via Cohen-Steiner & Morvan's method) instead of
    random control points, coupling the warp to actual surface geometry.

    curvature_map: array of shape (H, W, 2) giving (du, dv) curvature-
    derived displacement per texel, already resampled to texture resolution.
    """
    du = curvature_map[..., 0] * scale * texture.shape[0]
    dv = curvature_map[..., 1] * scale * texture.shape[1]
    shape = texture.shape
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    coords = np.stack([
        np.clip(yy + dv, 0, shape[0] - 1),
        np.clip(xx + du, 0, shape[1] - 1),
    ])
    return map_coordinates(texture, coords, order=1, mode="reflect")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from procedural import generate_base_texture

    base = generate_base_texture(alpha=0.5, seed=42)
    warped = warp_texture(base, seed=42)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(base, cmap="gray"); axes[0].set_title("T_base"); axes[0].axis("off")
    axes[1].imshow(warped, cmap="gray"); axes[1].set_title("T_warped"); axes[1].axis("off")
    plt.tight_layout()
    plt.savefig("../results/stage2_example.png", dpi=150)
    print("Saved results/stage2_example.png")
