# Benchmark Results: UDED

| Method | Type | ODS | OIS | AP | Time (ms/img) |
|--------|------|----:|----:|---:|--------------:|
| Sobel-9x9 | traditional | 0.9038 | 0.9148 | 0.9541 | 12.2 |
| DexiNed | ml | 0.9010 | 0.9101 | 0.9052 | 485.0 |
| Sobel-7x7 | traditional | 0.9003 | 0.9105 | 0.9488 | 15.1 |
| Sobel-5x5 | traditional | 0.8876 | 0.8984 | 0.9350 | 14.6 |
| Prewitt-3x3 | traditional | 0.8847 | 0.8968 | 0.9311 | 13.6 |
| Sobel-3x3 | traditional | 0.8828 | 0.8947 | 0.9296 | 19.6 |
| TEED | ml | 0.8814 | 0.8898 | 0.8501 | 135.2 |
| Sobel-15x15 | traditional | 0.8770 | 0.8937 | 0.9423 | 19.0 |
| LoG-sigma2 | traditional | 0.8585 | 0.8682 | 0.9070 | 354.4 |
| Canny-sigma1 | traditional | 0.8211 | 0.8034 | 0.3428 | 29.6 |
| Canny-sigma2 | traditional | 0.8047 | 0.7927 | 0.3252 | 26.2 |
