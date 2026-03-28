# Benchmark Results: UDED

| Method | Type | ODS | OIS | AP | Time (ms/img) |
|--------|------|----:|----:|---:|--------------:|
| Sobel-9x9 | traditional | 0.9038 | 0.9148 | 0.9541 | 15.0 |
| DexiNed | ml | 0.9010 | 0.9101 | 0.9052 | 518.3 | *PyTorch inference on cpu*
| Sobel-7x7 | traditional | 0.9003 | 0.9105 | 0.9488 | 16.1 |
| Sobel-5x5 | traditional | 0.8876 | 0.8984 | 0.9350 | 18.1 |
| Prewitt-3x3 | traditional | 0.8847 | 0.8968 | 0.9311 | 19.4 |
| Sobel-3x3 | traditional | 0.8828 | 0.8947 | 0.9296 | 14.2 |
| TEED | ml | 0.8814 | 0.8898 | 0.8501 | 291.1 | *PyTorch inference on cpu*
| Sobel-15x15 | traditional | 0.8770 | 0.8937 | 0.9423 | 19.0 |
| LoG-sigma2 | traditional | 0.8585 | 0.8682 | 0.9070 | 356.9 |
| Canny-sigma1 | traditional | 0.8211 | 0.8034 | 0.3428 | 32.0 |
| Canny-sigma2 | traditional | 0.8047 | 0.7927 | 0.3252 | 22.6 |
| WVF-Np15 | wvf_lf | 0.0580 | 0.0979 | 0.0071 | 7323.8 | *UDED subset, Np=15, 18 orientations, resized to 64x64*
| LF-m3-Np15 | wvf_lf | 0.0526 | 0.0526 | 0.0030 | 27.0s | *UDED subset, m=3, Np=15, 18 orientations, subsample=2, resized to 64x64*

## Provenance

- Images available: 30
- Images evaluated in this report: 30
- Evaluation: 100 thresholds, 3-pixel match radius
- Runtime environment: Python 3.12.4, NumPy 1.26.4, PyTorch 2.3.1+cu121, CUDA available = False
- Host: node031
- Note: Traditional filters and WVF/LF run in NumPy/SciPy on CPU. ML models use PyTorch and only run on GPU when torch.cuda.is_available() is true.
