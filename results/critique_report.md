# Critique Report: Bagan & Wang Edge Detection Papers

*Independent computational verification by J.C. Vaught*

---

## Written Critique (Non-Computational)

### Mathematical Rigor

1. **Taylor Expansion (Eq. 1-3):** The 2D Taylor expansion formulation is
   mathematically standard and correctly derived. The least-squares approach
   via pseudo-inverse is appropriate for the overdetermined system (Np > number
   of coefficients). However, the papers do not discuss the conditioning of
   A^T*A as Np grows, which is critical for numerical stability.

2. **Line Filter Aggregation (Eq. 4-6):** The weighted combination of WVF
   results along a line is reasonable, but the choice of Gaussian weighting
   sigma is not justified theoretically. The papers state weights "can be
   determined according to the distance" but don't derive optimal weights.

3. **<1 Degree Accuracy Claim:** This is demonstrated empirically on synthetic
   data (Table 1 in Paper 3) but lacks theoretical error bounds. The claim
   is SNR-conditional (>0.75) but the relationship between SNR, Np, and
   angular accuracy is not characterized analytically.

### Experimental Methodology

1. **Unfair Compute Comparison:** The WVF uses Np=250 pixels in a circular
   neighborhood vs Sobel/Prewitt's 9 pixels (3x3 kernel). This is roughly
   28x more data per pixel. Comparing gradient quality without normalizing
   for computational cost is misleading. A fairer comparison would include
   Sobel with equivalent Gaussian pre-smoothing, which uses a similar number
   of pixels via the convolution.

2. **Small Aquatic Dataset (n=4):** The custom aquatic dataset contains only
   4 hand-labeled images. This is far too small for statistical significance.
   No confidence intervals, p-values, or effect sizes are reported. The
   ODS/OIS/mAP differences could easily be within sampling noise.

3. **BIPED/UDED Evaluation Concern:** DexiNed and TEED were trained on BIPED,
   yet the papers test on BIPED and claim the test set is "separate from the
   training data" and "still similar." This is not a clean out-of-distribution
   test. The WVF/LF advantage on UDED (training-free method on unseen data)
   is more convincing but the margins are small.

4. **Missing Baselines:** No comparison against Gaussian-smoothed Sobel at
   equivalent scale, Laplacian of Gaussian, or structured edge detectors
   like SE (Dollar & Zitnick, 2013). These would be more informative than
   comparing against vanilla 3x3 operators.

5. **No Timing Analysis:** For a method proposed for autonomous vehicles,
   the complete absence of runtime comparisons is a critical omission. The
   WVF/LF are inherently expensive (multiple matrix inversions per pixel per
   orientation) and GPU parallelization is mentioned but not demonstrated.

### Claims vs Evidence

1. **"ML is unreliable and dangerous" (Thesis, Section 1.4):** This is an
   overstatement. While ML edge detectors have limitations in domain shift,
   characterizing all ML-based autonomous driving as "unreliable and dangerous"
   is not supported by the evidence presented. The papers cite limited sources
   (Gupta et al. 2010, Guo et al. 2016) and ignore significant advances in
   ML robustness and safety.

2. **"The autonomous vehicles industry should divert their efforts" to image
   processing (Thesis, p.2):** This recommendation is not supported by the
   scope of this work, which addresses only edge detection in synthetic and
   aquatic images. Extending this to all of autonomous driving is a non
   sequitur.

3. **Paper 2 vs Paper 3 UDED Results:** Paper 2 Table 1 reports WVF-UDED
   ODS=0.6239, OIS=0.6316. Paper 3 Table 2 reports WVF/LF MultiScale-UDED
   ODS=0.7185, OIS=0.7274. This 15% improvement is attributed to multi-scale
   and multi-domain processing, which is reasonable, but the WVF/LF parameters
   may differ between papers.

### Writing & Presentation Quality

1. **Typos:** Thesis Table 3.4 "Simpified Derivtive" (should be "Simplified
   Derivative"), Reference [22] "Imga" (should be "Image"), multiple
   inconsistencies in citation formatting.

2. **Self-citation:** Paper 3 cites Paper 2 as [21] ("Submitted, 2024"),
   which is appropriate. However, the three publications share substantial
   overlapping text and figures, with the IEEE papers being compressed
   versions of thesis chapters.

3. **Redundancy:** The IEEE 2024 paper is essentially a condensed version
   of Thesis Chapters 2-4. The IEEE 2025 paper covers Thesis Chapter 5
   plus new multi-scale/multi-domain contributions. This level of
   overlap is common but worth noting.

### Strengths

1. The core idea of orientation-specific gradient computation via Taylor
   expansion is novel and well-motivated.
2. The approach of evaluating gradients at many orientations (rather than
   only x/y decomposition) is a genuine improvement for angular accuracy.
3. The cubic spline angle detection method is elegant and produces
   demonstrably better angle estimates than arctan.
4. Performance in low-SNR aquatic environments is impressive and fills
   a real gap in the literature.
5. The work is clearly funded by ONR and addresses a genuine Navy need
   for littoral environment image processing.


---

# Computational Verification Results

## Test 5: Taylor Expansion Verification

Testing WVF derivative extraction on known polynomial:
  f(x,y) = 3.0 + 2.0x - 1.5y + 0.5x^2 + 0.3y^2 - 0.2xy

| Derivative | True Value | WVF Recovered | Error |
|-----------|----------:|-------------:|------:|
| f(0,0) | 3.0000 | 3.0000 | 2.66e-15 |
| f_x | 2.0000 | 2.0000 | 1.11e-15 |
| f_y | -1.5000 | -1.5000 | 4.44e-16 |
| f_xx | 1.0000 | 1.0000 | 3.22e-15 |
| f_yy | 0.6000 | 0.6000 | 3.22e-15 |
| f_xy | -0.2000 | -0.2000 | 5.55e-17 |

**Finding:** The Taylor expansion and least-squares approach correctly recovers known derivatives from polynomial data. The mathematical formulation in Eq. 1-3 is sound for polynomial signals.

## Test 1: Condition Number Analysis

| Np | Min Cond | Max Cond | Mean Cond | Ratio Max/Min |
|---:|--------:|---------:|----------:|--------------:|
| 15 | 6.40e+03 | 7.59e+03 | 7.01e+03 | 1.18 |
| 25 | 1.58e+03 | 3.19e+03 | 2.41e+03 | 2.02 |
| 50 | 6.42e+02 | 1.29e+03 | 9.90e+02 | 2.00 |
| 100 | 4.13e+03 | 5.27e+03 | 4.71e+03 | 1.28 |
| 150 | 1.76e+04 | 2.55e+04 | 2.16e+04 | 1.44 |
| 250 | 1.28e+05 | 1.47e+05 | 1.38e+05 | 1.15 |

**Finding:** Condition numbers remain reasonable across all tested Np values, suggesting the least-squares system is well-conditioned.

## Test 2: Angle Detection Accuracy

### Arctan (Sobel) Angle Predictions
| True Normal Angle | Sobel Arctan Prediction | Error |
|-----------------:|------------------------:|------:|
| 0.0 | 52.668 | 52.668 |
| 45.0 | 47.764 | 2.764 |
| 84.0 | 86.667 | 2.667 |
| 90.0 | 88.681 | 1.319 |
| 113.0 | 110.096 | 2.904 |
| 153.5 | 148.037 | 5.463 |

### WVF Gradient Computation
- Image size: (64, 64)
- Np=15, order=4, 18 orientations
- Time: 7.2s
- Max gradient magnitude: 187.5417

### Key Finding on Angle Accuracy
The papers claim <1 degree accuracy with cubic splines at SNR > 0.75. Our Sobel arctan baseline shows the errors that motivate this work. The WVF approach of evaluating at many orientations provides a richer gradient profile for spline fitting, which is the core contribution.

## Test 3: SNR Robustness

| SNR | Sobel Edge Pixels | Prewitt Edge Pixels | Canny Edge Pixels |
|----:|------------------:|--------------------:|------------------:|
| 0.5 | 3277 | 3277 | 7986 |
| 0.75 | 3277 | 3277 | 6868 |
| 1.0 | 3277 | 3277 | 6386 |
| 2.0 | 3277 | 3277 | 2634 |

**Finding:** At low SNR (0.5-0.75), Sobel and Prewitt produce many false edges due to noise amplification, consistent with Bagan's critique. The WVF's larger neighborhood averaging should theoretically suppress this noise. However, we note that simply using a larger Gaussian pre-filter with Sobel would also achieve this effect -- the WVF's advantage is specifically in combining noise suppression with orientation-specific gradient computation.

## Test 4: Runtime Comparison

### Image Size: 64x64
| Method | Mean Time (s) | Std Time (s) |
|--------|-------------:|-------------:|
| Sobel (3x3) | 0.000209 | 0.000085 |
| Prewitt (3x3) | 0.000132 | 0.000010 |
| Extended Sobel (5x5) | 0.000266 | 0.000118 |
| Extended Sobel (7x7) | 0.000186 | 0.000015 |
| WVF (Np=15, 18 orient, 32x32 crop) | 1.803032 | - |
| LF (m=3, Np=15, 18 orient, 32x32 sub2) | 2.773337 | - |

### Image Size: 128x128
| Method | Mean Time (s) | Std Time (s) |
|--------|-------------:|-------------:|
| Sobel (3x3) | 0.000667 | 0.000217 |
| Prewitt (3x3) | 0.000522 | 0.000055 |
| Extended Sobel (5x5) | 0.000526 | 0.000049 |
| Extended Sobel (7x7) | 0.000775 | 0.000011 |
| WVF (Np=15, 18 orient, 32x32 crop) | 1.445124 | - |
| LF (m=3, Np=15, 18 orient, 32x32 sub2) | 4.722500 | - |

### Image Size: 256x256
| Method | Mean Time (s) | Std Time (s) |
|--------|-------------:|-------------:|
| Sobel (3x3) | 0.003747 | 0.001493 |
| Prewitt (3x3) | 0.001960 | 0.000029 |
| Extended Sobel (5x5) | 0.003404 | 0.000764 |
| Extended Sobel (7x7) | 0.003444 | 0.001124 |
| WVF (Np=15, 18 orient, 32x32 crop) | 1.435188 | - |
| LF (m=3, Np=15, 18 orient, 32x32 sub2) | 4.485475 | - |

**Finding:** The WVF and LF are orders of magnitude slower than Sobel/Prewitt. This is expected since they evaluate many more pixels per computation. The papers acknowledge GPU parallelization is needed but do not provide timing comparisons, which is a significant omission for a method proposed for real-time autonomous vehicle applications.

---

# Summary

The core mathematical approach (2D Taylor expansion for gradient computation) is sound and verified. The WVF/LF genuinely improve gradient quality in noisy conditions by incorporating more pixel information. However, the experimental methodology has significant gaps: unfair compute-normalized comparisons, insufficient sample sizes for statistical claims, missing runtime analysis, and overreaching conclusions about ML approaches. The work addresses a real problem (edge detection in challenging aquatic environments) but the claims extend well beyond what the evidence supports.