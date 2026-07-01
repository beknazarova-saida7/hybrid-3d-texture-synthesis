"""
Evaluation metrics used in Section IV: SSIM, PSNR, LPIPS, FID, and
Wilcoxon signed-rank significance testing.

Requires: scikit-image, scipy, lpips (pip install lpips),
          pytorch-fid or a local FID implementation for the FID metric.
"""
import numpy as np
from scipy.stats import wilcoxon
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr


def compute_ssim(img_a, img_b):
    return sk_ssim(img_a, img_b, data_range=1.0)


def compute_psnr(img_a, img_b):
    return sk_psnr(img_a, img_b, data_range=1.0)


def compute_lpips(img_a, img_b, loss_fn=None, device="cpu"):
    """
    Requires the `lpips` package:  pip install lpips
    loss_fn should be lpips.LPIPS(net='alex') created once and reused
    across the evaluation loop for efficiency.
    """
    import torch
    if loss_fn is None:
        import lpips
        loss_fn = lpips.LPIPS(net="alex").to(device)

    def to_tensor(img):
        t = torch.tensor(img, dtype=torch.float32, device=device)[None, None]
        t = t.repeat(1, 3, 1, 1) * 2 - 1  # LPIPS expects [-1, 1], 3-channel
        return t

    with torch.no_grad():
        d = loss_fn(to_tensor(img_a), to_tensor(img_b))
    return float(d.item())


def compute_fid(features_a, features_b):
    """
    Standard Frechet Inception Distance between two sets of Inception
    pool3 features (each shape [N, 2048]), following Heusel et al. [26].
    Feature extraction itself should be done once per method with
    pytorch-fid or torchmetrics.image.fid.FrechetInceptionDistance and
    passed in here as numpy arrays.
    """
    from scipy import linalg

    mu_a, mu_b = features_a.mean(0), features_b.mean(0)
    cov_a = np.cov(features_a, rowvar=False)
    cov_b = np.cov(features_b, rowvar=False)

    diff = mu_a - mu_b
    covmean, _ = linalg.sqrtm(cov_a @ cov_b, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(cov_a + cov_b - 2 * covmean)
    return float(fid)


def significance_test(scores_method, scores_baseline, alpha=0.05):
    """
    Wilcoxon signed-rank test comparing paired per-texture scores between
    a candidate method and a baseline, as used throughout Section IV.
    Returns (statistic, p_value, is_significant).
    """
    stat, p = wilcoxon(scores_method, scores_baseline)
    return stat, p, p < alpha


def summarize(scores):
    """Returns (mean, std) formatted as reported in Tables II-IV."""
    arr = np.asarray(scores)
    return float(arr.mean()), float(arr.std())


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    a = rng.random((256, 256)).astype(np.float32)
    b = np.clip(a + rng.normal(0, 0.05, a.shape), 0, 1).astype(np.float32)
    print("SSIM:", compute_ssim(a, b))
    print("PSNR:", compute_psnr(a, b))
