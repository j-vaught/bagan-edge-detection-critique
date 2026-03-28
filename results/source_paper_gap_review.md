# Source Paper Gap Review

This note compares the three source papers against the current reproduced work in this repository.

## Source Artifacts Reviewed

- `Wide View and Line Filter for Enhanced Image Gradient Computation.pdf` (thesis)
- `Wide_View_and_Line_Filter_for_Enhanced_Gradient_Computation_in_Aquatic_Environment.pdf` (conference paper)
- `Multi-Scale_and_Multi-Domain_Edge_Determination_with_Accurate_Gradient_Orientation_Computation_in_Aquatic_Environment.pdf` (later conference paper)
- `benchmarks/results/benchmark_results.json`
- `benchmarks/results/benchmark_uded_results.json`
- `benchmarks/maritime_results/maritime_results.json`

## Where The Current Repo Does Not Yet Match The Authors

### 1. WVF/LF parameter scale

- The thesis and conference paper emphasize large WVF/LF settings such as `Np=150`, `Np=250`, and line-filter length `29 px` (`m=14`).
- The checked-in reproduced runs are still small-setting runs: `WVF-Np15` and `LF-m3-Np15`, usually at `18` orientations and with WVF/LF evaluated only on resized subsets.
- This is the largest implementation gap between the papers and the current repo.

### 2. Datasets

- The source papers report numerical results on `UDED` and `BIPED`.
- The current repo has `UDED`, but `datasets/BIPED` is empty.
- That means we have not reproduced any BIPED benchmark tables from the papers.
- The conference papers also use a very small hand-labeled aquatic dataset of four images.
- That four-image dataset is not present in this repo, so none of those aquatic ODS/OIS/mAP tables are exact reproductions.
- The local `datasets/maritime/images` directory currently contains zero-byte placeholder files, so it cannot support a real-image qualitative comparison until those assets are populated.

### 3. Multi-scale / multi-domain method

- The later conference paper adds a multi-scale, multi-domain pipeline with GMM fusion.
- That pipeline is not implemented in the repo.
- Our current work evaluates single-scale WVF/LF subset runs plus classical and ML baselines, not the authors' full final method.

### 4. Angle-validation protocol

- The later conference paper uses two angle tests:
- a synthetic line-angle study claiming spline estimates stay within roughly one degree
- a real-image BIPED non-maximum-suppression comparison using spline angles versus arctan
- The repo currently has a direct synthetic angle test, which is useful, but it does not reproduce the real-image BIPED edge-generation comparison because the BIPED data is missing and the full spline-driven NMS pipeline is not wired up.

### 5. Threshold protocol

- The later conference paper says it evaluates thresholds from `0` to `1` in `0.001` steps, i.e. `1001` thresholds, with forward/backward matching radius `3 px`.
- The checked-in benchmark artifacts currently use `100` thresholds and radius `3`.
- So the current published repo artifacts are still a simplified evaluation relative to the conference-paper protocol.

### 6. ML baseline coverage

- The first conference paper discusses `DexiNed`, `TEED`, `BDCN`, and `EDTER` in the narrative and qualitative aquatic figures.
- The current unified benchmark pipeline is solid for `DexiNed`, `TEED`, and partially `PiDiNet`, but it does not yet reproduce the paper's broader ML comparison set in a single evaluated table.
- `BDCN`, `EDTER`, `EdgeNAT`, and the final multi-scale/domain system are still outside the reproduced benchmark path.

### 7. Runtime claims and compute environment

- The source papers do not present the kind of runtime audit that this repo now does.
- The current repo already improved on that by recording runtime provenance, but the completed benchmark jobs on `gpu-A100` still reported `torch.cuda.is_available() == False`.
- So the existing reproduced ML numbers are CPU-path numbers even when they were launched on GPU partitions.

## What We Did Add That The Papers Did Not

- A much stronger baseline audit with large-support Sobel and LoG.
- Runtime/quality tradeoff analysis.
- Repeated-seed maritime stress tests with uncertainty instead of one-shot synthetic scenes.
- A more explicit claims-vs-evidence framing in the critique paper.

## Data We Still Need For A Stronger Critique

- Exact-threshold (`1001`) BSDS500 and UDED baselines.
- WVF/LF sweeps toward `Np=150/250` and larger LF widths.
- A direct angle sweep that tests whether moving toward thesis settings actually improves spline accuracy enough to match the paper claims.
- Real maritime qualitative panels on the local image set.
- Ideally, BIPED data or the authors' four-image aquatic set. Without one of those, the paper-to-paper reproduction remains partial.
