# Independent Paper Packet

Each agent should write a full standalone IEEE-style paper in LaTeX, not a section fragment.

Paper requirements:
- Use `IEEEtran` conference format unless there is a compelling reason not to.
- Write a complete paper with title, abstract, introduction, related/source-paper context, evidence, limitations, conclusion, and bibliography.
- Use the local repo artifacts as the primary evidence base.
- Also browse the web for supporting context or official comparison sources when useful. Prefer primary sources, official docs, dataset pages, and published papers.
- Be explicit about what is reproduced locally versus what is inferred from the source papers or external references.
- Keep each paper focused on its assigned goal, but make it self-contained.

Local source-paper PDFs:
- `Wide View and Line Filter for Enhanced Image Gradient Computation.pdf`
- `Wide_View_and_Line_Filter_for_Enhanced_Gradient_Computation_in_Aquatic_Environment.pdf`
- `Multi-Scale_and_Multi-Domain_Edge_Determination_with_Accurate_Gradient_Orientation_Computation_in_Aquatic_Environment.pdf`

Current main manuscript and prior critique artifacts:
- `paper/critique.tex`
- `results/source_paper_gap_review.md`
- `results/critique_report.md`

Exact A100 benchmark artifacts:
- `benchmarks/results/benchmark_results_exact1001_a100.json`
- `benchmarks/results/benchmark_report_exact1001_a100.md`
- `benchmarks/results/BSDS500_exact1001_a100_ods_ois.png`
- `benchmarks/results/BSDS500_exact1001_a100_runtime.png`
- `benchmarks/results/benchmark_uded_results_exact1001_uded_a100.json`
- `benchmarks/results/benchmark_uded_report_exact1001_uded_a100.md`
- `benchmarks/results/UDED_exact1001_uded_a100_ods_ois.png`
- `benchmarks/results/UDED_exact1001_uded_a100_runtime.png`

Angle-study artifacts:
- `benchmarks/angle_sweep_results/angle_sweep_paperlike_v2.json`
- `benchmarks/angle_sweep_results/angle_sweep_paperlike_v2.png`
- `results/angle_error_comparison.png`
- `results/angle_test_images.png`

Maritime and visual critique artifacts:
- `paper/figures/maritime_scene_montage.png`
- `paper/figures/maritime_summary.png`
- `results/condition_numbers.png`
- `paper/figures/runtime_tradeoff.png`
- `paper/figures/dataset_quality_comparison.png`

Important current local facts:
- BSDS500 exact run: 50 images used out of 200 available, 1001 thresholds, 3-pixel match radius, CUDA available on `node035`, GPU `NVIDIA A100-SXM4-40GB`.
- UDED exact run: 30 images used out of 30 available, 1001 thresholds, 3-pixel match radius, CUDA available on `node034`, GPU `NVIDIA A100-SXM4-40GB`.
- BSDS500 exact numbers:
  - `Sobel-15x15`: ODS 0.5491, OIS 0.5550, AP 0.4682, 8.2 ms/img
  - `Sobel-9x9`: ODS 0.5159, OIS 0.5247, AP 0.4394, 7.1 ms/img
  - `DexiNed`: ODS 0.4950, OIS 0.5058, AP 0.3990, 37.0 ms/img
  - `TEED`: ODS 0.4710, OIS 0.4747, AP 0.3598, 45.0 ms/img
  - `LoG-sigma2`: ODS 0.3826, OIS 0.3864, AP 0.2829, 213.2 ms/img
- UDED exact numbers:
  - `Sobel-9x9`: ODS 0.9039, OIS 0.9151, AP 0.9543, 26.2 ms/img
  - `DexiNed`: ODS 0.9010, OIS 0.9102, AP 0.9068, 215.3 ms/img
  - `Sobel-7x7`: ODS 0.9004, OIS 0.9109, AP 0.9489, 22.7 ms/img
  - `TEED`: ODS 0.8814, OIS 0.8958, AP 0.8787, 210.1 ms/img
  - `Sobel-15x15`: ODS 0.8770, OIS 0.8939, AP 0.9425, 34.7 ms/img
  - `LoG-sigma2`: ODS 0.8587, OIS 0.8689, AP 0.9073, 661.6 ms/img
- Angle sweep highlights:
  - Best overall improvement in the current sweep: `N_p=50`, 18 orientations, SNR 0.75, improvement 21.99 deg, runtime 1.70 s
  - Best SNR 2.0 row: `N_p=100`, 72 orientations, Sobel 16.16 deg, spline 8.83 deg, improvement 7.33 deg, runtime 24.05 s
  - Worst overall row: `N_p=15`, 72 orientations, SNR 2.0, improvement -8.69 deg
  - Mean improvement by `N_p`: 15 -> -4.16 deg, 50 -> +5.08 deg, 100 -> +2.80 deg, 150 -> -0.06 deg, 250 -> +1.47 deg
- WVF/LF scale jobs currently running on full 112-core `defq` nodes:
  - `348233` on `node005`
  - `348234` on `node009`
- Real maritime image folder currently contains zero-byte placeholders, so only synthetic maritime visuals are usable right now.
- `datasets/BIPED` is empty.
- The 2025 paper's multi-scale / multi-domain / GMM fusion pipeline is not implemented in this repo.

What each paper should make explicit:
- Which claims are directly reproduced locally.
- Which claims remain untested because data or code is missing.
- Which comparisons are fair and which are not, especially around support size and runtime.
- Which statements depend on exact A100 runs, and which depend on provisional or ongoing CPU sweeps.
