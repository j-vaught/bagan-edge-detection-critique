# Benchmark Results: UDED

| Method | Type | ODS | OIS | AP | Time (ms/img) |
|--------|------|----:|----:|---:|--------------:|
| Sobel-9x9 | traditional | 0.9039 | 0.9151 | 0.9543 | 26.2 |
| DexiNed | ml | 0.9010 | 0.9102 | 0.9068 | 215.3 | *PyTorch inference on cuda*
| Sobel-7x7 | traditional | 0.9004 | 0.9109 | 0.9489 | 22.7 |
| Sobel-5x5 | traditional | 0.8877 | 0.8987 | 0.9351 | 25.6 |
| Prewitt-3x3 | traditional | 0.8848 | 0.8972 | 0.9314 | 31.0 |
| Sobel-3x3 | traditional | 0.8828 | 0.8951 | 0.9299 | 28.9 |
| TEED | ml | 0.8814 | 0.8958 | 0.8787 | 210.1 | *PyTorch inference on cuda*
| Sobel-15x15 | traditional | 0.8770 | 0.8939 | 0.9425 | 34.7 |
| LoG-sigma2 | traditional | 0.8587 | 0.8689 | 0.9073 | 661.6 |
| Canny-sigma1 | traditional | 0.8211 | 0.8034 | 0.3428 | 90.4 |
| Canny-sigma2 | traditional | 0.8047 | 0.7927 | 0.3252 | 55.6 |

## Provenance

- Images available: 30
- Images evaluated in this report: 30
- Evaluation: 1001 thresholds, 3-pixel match radius
- Runtime environment: Python 3.12.4, NumPy 1.26.4, PyTorch 2.3.1+cu121, CUDA available = True
- Host: node034
- Note: Traditional filters and WVF/LF run in NumPy/SciPy on CPU. ML models use PyTorch and only run on GPU when torch.cuda.is_available() is true.
