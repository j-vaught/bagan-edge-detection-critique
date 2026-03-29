# Datasets

## BSDS500

The Berkeley Segmentation Dataset and Benchmark (BSDS500). Contains 200 test images (481x321 or 321x481) with human-annotated boundary ground truth from 5 annotators per image. Ground truth is stored as `.mat` files with `Boundaries` and `Segmentation` fields.

- **Images:** `BSDS500/data/images/test/` (200 JPGs)
- **Ground truth:** `BSDS500/data/groundTruth/test/` (200 MATs)
- **Source:** https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html
- **Paper:** P. Arbelaez et al., "Contour Detection and Hierarchical Image Segmentation," IEEE TPAMI, 2011.

## UDED

The Underwater Dataset for Edge Detection. Contains 30 underwater/aquatic images with binary edge ground truth annotations. Used in Bagan & Wang's papers as the primary evaluation dataset for the Wide View Filter.

- **Images:** `UDED/imgs/` (30 images)
- **Ground truth:** `UDED/gt/` (30 binary edge maps)
- **Source:** https://github.com/xavysp/UDED
- **Paper:** X. Soria et al., "LDC: Lightweight Dense CNN for Edge Detection," IEEE Access, 2022.
