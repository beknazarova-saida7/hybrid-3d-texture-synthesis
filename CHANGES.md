# Changes in release v2.0-revision

The files of the previous version of this repository were uploaded under wrong names and could not be executed, and the code for the baselines and for the mesh experiments was missing. During the revision the following errors of the original implementation were found and corrected:

1. **Timing.** The reported 26 ms covered only the U-Net forward pass; the method performed 150 optimization steps per texture (≈ 5.7 s end-to-end). All methods are now timed end-to-end under identical conditions.
2. **Style target.** The CNN style target was the procedural input itself instead of the DTD exemplar. Fixed.
3. **VGG19 layers.** Indices 3/8/13 (relu1_2/relu2_2/relu3_2) were used instead of relu1_1/relu2_1/relu3_1. Fixed (indices 1/6/11).
4. **Border artifacts.** Coordinate clipping in the warp and zero padding in the U-Net. Replaced by periodic (wrap-around) sampling and reflection padding.
5. **B-spline.** `RectBivariateSpline` interpolation replaced by an explicit periodic tensor-product cubic B-spline (Eq. (4)–(7) of the paper).
6. **Seeds.** PyTorch/CUDA generators were not seeded; now all generators are seeded and cuDNN is deterministic.
7. **Statistics.** Wilcoxon tests now use paired per-texture means (n = 75 / 30) instead of treating seeds as independent samples; exact p-values with Holm correction are reported.
8. **Method.** Stage 3 was redesigned as a feed-forward AdaIN decoder trained once (Hybrid-FF); the corrected per-texture optimization is kept as Hybrid-Opt.
9. **Removed.** The real-mesh validation (former Table 5) could not be reproduced and was removed from the paper.


## Third run (v4)

- The method is reported in two modes: fidelity mode (without the B-spline stage, main method) and variation mode (with it).
- Added Stable Diffusion v1.5 + IP-Adapter as an exemplar-conditioned generative baseline.
- Added a white-noise + feed-forward ablation (same decoder architecture and training).
- Validation grid for the style weight extended to w = 1 (optimum w = 3 now inside the grid).
- Added output-diversity evaluation (mean pairwise LPIPS between seeds; secondary Inception/SSIM measures).
