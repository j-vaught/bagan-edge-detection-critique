# Benchmark Results: BSDS500

| Method | Type | ODS | OIS | AP | Time (ms/img) |
|--------|------|----:|----:|---:|--------------:|
| Sobel-15x15 | traditional | 0.5489 | 0.5539 | 0.4679 | 11.9 |
| Sobel-9x9 | traditional | 0.5159 | 0.5238 | 0.4392 | 8.9 |
| DexiNed | ml | 0.4950 | 0.5041 | 0.3958 | 340.5 |
| Sobel-7x7 | traditional | 0.4926 | 0.4990 | 0.4132 | 7.8 |
| TEED | ml | 0.4701 | 0.4714 | 0.3331 | 74.0 |
| Sobel-5x5 | traditional | 0.4641 | 0.4668 | 0.3784 | 13.5 |
| Prewitt-3x3 | traditional | 0.4529 | 0.4600 | 0.3587 | 10.8 |
| Sobel-3x3 | traditional | 0.4493 | 0.4576 | 0.3581 | 13.8 |
| LoG-sigma2 | traditional | 0.3822 | 0.3850 | 0.2829 | 225.2 |
| Canny-sigma2 | traditional | 0.3034 | 0.2898 | 0.0862 | 21.4 |
| Canny-sigma1 | traditional | 0.2012 | 0.1939 | 0.0556 | 17.1 |
| WVF-Np15 | wvf_lf | 0.0000 | 0.0000 | 0.0000 | 9056.7 | *10 images, resized to 64x64*
