# Maritime Benchmark Results

## Benchmark Design

- Scene families: 6
- Trials per scene family: 10
- Base seed: 1234
- Each report value below is an aggregate across repeated random draws of the synthetic scene generator.

## Average F-score Across All Synthetic Maritime Scenes

| Method | Avg F-score | Std | 95% CI | Samples |
|--------|------------:|----:|--------:|--------:|
| Sobel-15x15 | 0.3364 | 0.2232 | 0.0565 | 60 |
| Canny-sigma1 | 0.2540 | 0.2735 | 0.0692 | 60 |
| Sobel-9x9 | 0.2128 | 0.1124 | 0.0285 | 60 |
| Sobel-7x7 | 0.1645 | 0.0799 | 0.0202 | 60 |
| Prewitt-3x3 | 0.1416 | 0.0661 | 0.0167 | 60 |
| Sobel-3x3 | 0.1363 | 0.0660 | 0.0167 | 60 |
| Sobel-5x5 | 0.1318 | 0.0701 | 0.0177 | 60 |
| Canny-sigma2 | 0.0702 | 0.0608 | 0.0154 | 60 |

## Per-Scene Results

### Horizon with Boat and Buoys

| Method | Mean F-score | Std | 95% CI | Trials |
|--------|-------------:|----:|--------:|-------:|
| Sobel-15x15 | 0.1578 | 0.0317 | 0.0196 | 10 |
| Canny-sigma2 | 0.0980 | 0.0158 | 0.0098 | 10 |
| Sobel-9x9 | 0.0977 | 0.0225 | 0.0139 | 10 |
| Sobel-7x7 | 0.0751 | 0.0134 | 0.0083 | 10 |
| Prewitt-3x3 | 0.0693 | 0.0118 | 0.0073 | 10 |
| Sobel-3x3 | 0.0662 | 0.0088 | 0.0055 | 10 |
| Sobel-5x5 | 0.0601 | 0.0019 | 0.0012 | 10 |
| Canny-sigma1 | 0.0568 | 0.0012 | 0.0007 | 10 |

### Cable in Water

| Method | Mean F-score | Std | 95% CI | Trials |
|--------|-------------:|----:|--------:|-------:|
| Sobel-15x15 | 0.2112 | 0.0161 | 0.0100 | 10 |
| Sobel-9x9 | 0.1537 | 0.0126 | 0.0078 | 10 |
| Sobel-7x7 | 0.1301 | 0.0096 | 0.0060 | 10 |
| Prewitt-3x3 | 0.1151 | 0.0090 | 0.0056 | 10 |
| Sobel-3x3 | 0.1130 | 0.0103 | 0.0064 | 10 |
| Sobel-5x5 | 0.1108 | 0.0065 | 0.0040 | 10 |
| Canny-sigma2 | 0.0718 | 0.0027 | 0.0017 | 10 |
| Canny-sigma1 | 0.0547 | 0.0008 | 0.0005 | 10 |

### Wave Field

| Method | Mean F-score | Std | 95% CI | Trials |
|--------|-------------:|----:|--------:|-------:|
| Sobel-15x15 | 0.4980 | 0.0065 | 0.0040 | 10 |
| Sobel-9x9 | 0.3595 | 0.0064 | 0.0040 | 10 |
| Canny-sigma1 | 0.3237 | 0.0069 | 0.0043 | 10 |
| Sobel-7x7 | 0.2860 | 0.0062 | 0.0038 | 10 |
| Sobel-5x5 | 0.2347 | 0.0042 | 0.0026 | 10 |
| Prewitt-3x3 | 0.2309 | 0.0042 | 0.0026 | 10 |
| Sobel-3x3 | 0.2275 | 0.0034 | 0.0021 | 10 |
| Canny-sigma2 | 0.0000 | 0.0000 | 0.0000 | 10 |

### Underexposed Marine Scene

| Method | Mean F-score | Std | 95% CI | Trials |
|--------|-------------:|----:|--------:|-------:|
| Canny-sigma1 | 0.8225 | 0.0398 | 0.0246 | 10 |
| Sobel-15x15 | 0.7469 | 0.0186 | 0.0115 | 10 |
| Sobel-9x9 | 0.3465 | 0.0265 | 0.0164 | 10 |
| Sobel-7x7 | 0.2000 | 0.0166 | 0.0103 | 10 |
| Prewitt-3x3 | 0.1535 | 0.0140 | 0.0087 | 10 |
| Sobel-3x3 | 0.1317 | 0.0134 | 0.0083 | 10 |
| Sobel-5x5 | 0.1049 | 0.0062 | 0.0038 | 10 |
| Canny-sigma2 | 0.0000 | 0.0000 | 0.0000 | 10 |

### Dark Horizon (high noise)

| Method | Mean F-score | Std | 95% CI | Trials |
|--------|-------------:|----:|--------:|-------:|
| Sobel-15x15 | 0.1229 | 0.0799 | 0.0495 | 10 |
| Sobel-9x9 | 0.0833 | 0.0314 | 0.0195 | 10 |
| Canny-sigma2 | 0.0768 | 0.0069 | 0.0043 | 10 |
| Sobel-7x7 | 0.0713 | 0.0160 | 0.0099 | 10 |
| Prewitt-3x3 | 0.0641 | 0.0109 | 0.0067 | 10 |
| Sobel-3x3 | 0.0625 | 0.0082 | 0.0051 | 10 |
| Sobel-5x5 | 0.0615 | 0.0025 | 0.0015 | 10 |
| Canny-sigma1 | 0.0577 | 0.0011 | 0.0007 | 10 |

### Very Low SNR Waves

| Method | Mean F-score | Std | 95% CI | Trials |
|--------|-------------:|----:|--------:|-------:|
| Sobel-15x15 | 0.2814 | 0.0056 | 0.0035 | 10 |
| Sobel-9x9 | 0.2361 | 0.0027 | 0.0017 | 10 |
| Sobel-7x7 | 0.2244 | 0.0024 | 0.0015 | 10 |
| Sobel-5x5 | 0.2187 | 0.0028 | 0.0017 | 10 |
| Sobel-3x3 | 0.2167 | 0.0030 | 0.0019 | 10 |
| Prewitt-3x3 | 0.2167 | 0.0025 | 0.0016 | 10 |
| Canny-sigma1 | 0.2087 | 0.0009 | 0.0005 | 10 |
| Canny-sigma2 | 0.1745 | 0.0188 | 0.0116 | 10 |
