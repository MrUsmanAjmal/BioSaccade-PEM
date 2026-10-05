# Pareto-frontier add-on: descriptive analysis added AFTER the paper run

**Status:** descriptive, post hoc. Added on 2026-10-05, after the locked paper run
`local_01_91e457e261dc` had finished (2026-10-05 10:00). It is **not** part of the locked protocol, was not
prespecified, and has no multiplicity-adjusted hypothesis tests. It must be reported as a descriptive
analysis.

## What was done
* The paper experiment folder, its protocol, locks and results were **only read, never modified**.
* Code: the paper run's own modules (`<RESULTS_DIR>\src`); their SHA-256 hashes were checked against
  `inputs.lock.json` before running.
* Models and settings: the paper run's fitted `models.joblib` and the development-selected settings
  (`development_selection.json`). Nothing was refitted or reselected.
* Clips: exactly the 45 clips of the paper's kappa and certificate sweeps (15 DAVIS, 15 REDS val,
  15 REDS train), nominal sensor, draw 0, same geometry (256x448, x4, 64-px fovea, 100 frames). Each
  clip's frames were verified against the content hash recorded in `protocol.lock.json`.
* New runs: `PEM_uniform`, `PEM_greedy`, `PPAT2026_M1`, `PEM_audit_only`. Because preview noise, patch
  noise and gaze uniforms are seeded by clip, regime and draw, these points are paired with the paper's
  sweep points on identical inputs.
* Reproducibility check: `PEM_R9_TwinAudit` at the selected settings was re-run as well; its rows were
  compared with the paper's `PEM_R9_k2` rows (identical configuration). Result in `reproducibility_check.json`.
* TwinAudit curves are taken from the paper run: kappa sweep (κ = 1, 2, 4, 8) and certificate sweep
  (cκ = 1.05, 1.25, 1.5, 2, 4, plus the selected cκ = 2.5, which is the κ = 2 point).

## Statistics
Means over source clusters (each clip is one cluster), 95% percentile bootstrap over clusters
(6000 resamples, shared across all points, so comparisons are paired). PSNR is the per-sequence mean
frame PSNR averaged over clusters (as in the paper tables). "PSNR gap at equal PQV" interpolates each
TwinAudit curve linearly in log PQV at a baseline's mean PQV; it is undefined when the baseline lies
outside the curve's PQV range.

## Figure
x-axis: mean audit PQV (log scale; lower = more reliable audit). y-axis: **paired PSNR gain over uniform
gaze** (dB; per-clip PSNR minus uniform gaze's PSNR on the same clip, noise and draw, averaged over source
clusters). Absolute PSNR is not plotted because its intervals are dominated by between-clip differences
(about ±1 dB), which hides the paired differences between methods; absolute means are in `pareto_points.csv`
(column `psnr`). Panels: all 45 clips and each dataset (15 clips each).

## Files
* `pareto_addon.py`, `pareto_analysis.py`, `run_addon.cmd`: code
* `checkpoints/`, `addon_sequence_results.csv`: raw add-on results
* `pareto_points.csv`: point estimates and intervals; `pareto_crossing.csv`: curve-vs-baseline gaps
* `pareto_frontier.pdf|svg|png`: figure; `reproducibility_check.json`

## Package copy
In this package copy, local folder paths in the scripts and in this note were replaced by the placeholders `<RESULTS_DIR>`, `<DAVIS_ROOT>` and `<REDS_ROOT>`. The raw per-clip checkpoint files (identical content to `addon_sequence_results.csv`) and the Windows launcher script are not included.
