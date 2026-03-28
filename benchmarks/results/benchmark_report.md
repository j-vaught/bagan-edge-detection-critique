# Benchmark Results: BSDS500

| Method | Type | ODS | OIS | AP | Time (ms/img) |
|--------|------|----:|----:|---:|--------------:|
| Sobel-15x15 | traditional | 0.5489 | 0.5539 | 0.4679 | 19.1 |
| Sobel-9x9 | traditional | 0.5159 | 0.5238 | 0.4392 | 7.7 |
| DexiNed | ml | 0.4950 | 0.5041 | 0.3958 | 438.3 | *PyTorch inference on cpu*
| Sobel-7x7 | traditional | 0.4926 | 0.4990 | 0.4132 | 10.8 |
| TEED | ml | 0.4701 | 0.4714 | 0.3331 | 98.7 | *PyTorch inference on cpu*
| Sobel-5x5 | traditional | 0.4641 | 0.4668 | 0.3784 | 11.3 |
| Prewitt-3x3 | traditional | 0.4529 | 0.4600 | 0.3587 | 8.3 |
| Sobel-3x3 | traditional | 0.4493 | 0.4576 | 0.3581 | 13.8 |
| LoG-sigma2 | traditional | 0.3822 | 0.3850 | 0.2829 | 219.1 |
| Canny-sigma2 | traditional | 0.3034 | 0.2898 | 0.0862 | 14.4 |
| Canny-sigma1 | traditional | 0.2012 | 0.1939 | 0.0556 | 32.2 |
| WVF-Np15 | wvf_lf | 0.0000 | 0.0000 | 0.0000 | 7348.7 | *BSDS500 subset, Np=15, 18 orientations, resized to 64x64*
| LF-m3-Np15 | wvf_lf | 0.0000 | 0.0000 | 0.0000 | 26.4s | *BSDS500 subset, m=3, Np=15, 18 orientations, subsample=2, resized to 64x64*

## Provenance

- Images available: 200
- Images evaluated in this report: 50
- Evaluation: 100 thresholds, 3-pixel match radius
- Runtime environment: Python 3.12.4, NumPy 1.26.4, PyTorch 2.3.1+cu121, CUDA available = False
- Host: node031
- Note: Traditional filters and WVF/LF run in NumPy/SciPy on CPU. ML models use PyTorch and only run on GPU when torch.cuda.is_available() is true.
