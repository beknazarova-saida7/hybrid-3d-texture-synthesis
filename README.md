"""
Main experiment driver, reproducing Tables II-IV of the paper.

Intended to run in Google Colab on the Describable Textures Dataset
(DTD) [23], on an NVIDIA T4 GPU, following the experimental protocol of
Section IV-A: 75 textures (15 x 5 material categories: woven, braided,
stratified, knitted, marbled), 3 random seeds, SSIM/PSNR/LPIPS/FID with
Wilcoxon signed-rank significance testing at alpha = 0.05.

Usage (Colab):
    !pip install lpips pytorch-fid
    from src.run_experiment import run_pipeline_on_texture, run_full_ablation
    ...

This script provides the full pipeline call and the ablation-study
loop; baseline methods (image-based quilting, GAN Gram-matrix
optimization, Stable Diffusion v1.5) should be wired in via
`baselines.py` (stub included) once dataset access is configured.
"""
import time
import numpy as np

from procedural import generate_base_texture
from bspline_warp import warp_texture
from evaluate import compute_ssim, compute_psnr, significance_test, summarize


def run_pipeline_on_texture(reference_texture, alpha=0.5, seed=0,
                              n_cnn_steps=150, device="cpu", use_cnn=True):
    """
    Full hybrid pipeline (Eq. 1): P(x,y) -> W(.) -> R(.)
    reference_texture: HxW numpy array in [0,1], used only as the CNN's
    style/content target during per-texture optimization (Section III-C).
    """
    t0 = time.time()
    base = generate_base_texture(reference_texture.shape, alpha=alpha, seed=seed)
    warped = warp_texture(base, seed=seed)

    if use_cnn:
        from cnn_refine import refine_texture
        final, n_params = refine_texture(warped, n_steps=n_cnn_steps, device=device)
    else:
        final, n_params = warped, 0

    elapsed_ms = (time.time() - t0) * 1000
    return final, elapsed_ms, n_params


def run_full_ablation(dataset_textures, seeds=(0, 1, 2), device="cpu"):
    """
    Reproduces Table IV: four variants (fBM-only, fBM+warp, fBM+CNN,
    full pipeline) evaluated against each reference texture, aggregated
    across seeds, with Wilcoxon significance vs. the fBM-only baseline.
    """
    variants = {
        "fbm_only": dict(use_warp=False, use_cnn=False),
        "fbm_warp": dict(use_warp=True, use_cnn=False),
        "fbm_cnn": dict(use_warp=False, use_cnn=True),
        "full": dict(use_warp=True, use_cnn=True),
    }

    results = {name: {"ssim": [], "psnr": [], "time_ms": []} for name in variants}

    for ref in dataset_textures:
        for seed in seeds:
            base = generate_base_texture(ref.shape, alpha=0.5, seed=seed)
            for name, cfg in variants.items():
                t0 = time.time()
                tex = base.copy()
                if cfg["use_warp"]:
                    tex = warp_texture(tex, seed=seed)
                if cfg["use_cnn"]:
                    from cnn_refine import refine_texture
                    tex, _ = refine_texture(tex, n_steps=150, device=device)
                elapsed = (time.time() - t0) * 1000

                results[name]["ssim"].append(compute_ssim(ref, tex))
                results[name]["psnr"].append(compute_psnr(ref, tex))
                results[name]["time_ms"].append(elapsed)

    # summarize + significance vs. fbm_only
    summary = {}
    for name in variants:
        ssim_mean, ssim_std = summarize(results[name]["ssim"])
        psnr_mean, psnr_std = summarize(results[name]["psnr"])
        time_mean, time_std = summarize(results[name]["time_ms"])
        entry = {
            "ssim": f"{ssim_mean:.3f}±{ssim_std:.3f}",
            "psnr": f"{psnr_mean:.2f}±{psnr_std:.2f}",
            "time_ms": f"{time_mean:.0f}±{time_std:.0f}",
        }
        if name != "fbm_only":
            _, p_ssim, sig = significance_test(results[name]["ssim"], results["fbm_only"]["ssim"])
            entry["ssim_p_value"] = p_ssim
            entry["ssim_significant"] = sig
        summary[name] = entry

    return summary, results


if __name__ == "__main__":
    # Minimal smoke test with synthetic references (replace with DTD samples)
    rng = np.random.default_rng(0)
    fake_refs = [rng.random((64, 64)).astype(np.float32) for _ in range(3)]
    summary, _ = run_full_ablation(fake_refs, seeds=(0,), device="cpu")
    for k, v in summary.items():
        print(k, v)
