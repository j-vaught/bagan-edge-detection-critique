# Benchmark Results: BSDS500

| Method | Type | ODS | OIS | AP | Time (ms/img) |
|--------|------|----:|----:|---:|--------------:|
| Sobel-15x15 | traditional | 0.5491 | 0.5550 | 0.4682 | 8.2 |
| Sobel-9x9 | traditional | 0.5159 | 0.5247 | 0.4394 | 7.1 |
| DexiNed | ml | 0.4950 | 0.5058 | 0.3990 | 37.0 | *PyTorch inference on cuda*
| Sobel-7x7 | traditional | 0.4926 | 0.5001 | 0.4133 | 6.7 |
| TEED | ml | 0.4710 | 0.4747 | 0.3598 | 45.0 | *PyTorch inference on cuda*
| Sobel-5x5 | traditional | 0.4641 | 0.4681 | 0.3786 | 6.4 |
| Prewitt-3x3 | traditional | 0.4534 | 0.4613 | 0.3589 | 6.4 |
| Sobel-3x3 | traditional | 0.4495 | 0.4587 | 0.3583 | 6.4 |
| LoG-sigma2 | traditional | 0.3826 | 0.3864 | 0.2829 | 213.2 |
| Canny-sigma2 | traditional | 0.3034 | 0.2898 | 0.0862 | 13.8 |
| Canny-sigma1 | traditional | 0.2012 | 0.1939 | 0.0556 | 14.0 |

## Provenance

- Images available: 200
- Images evaluated in this report: 50
- Evaluation: 1001 thresholds, 3-pixel match radius
- Runtime environment: Python 3.12.4, NumPy 1.26.4, PyTorch 2.3.1+cu121, CUDA available = True
- Host: node035
- Note: Traditional filters and WVF/LF run in NumPy/SciPy on CPU. ML models use PyTorch and only run on GPU when torch.cuda.is_available() is true.
