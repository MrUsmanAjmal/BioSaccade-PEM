# BioSaccade-PEM (TwinAudit): code and data availability

This package contains the complete, executable experiment for the manuscript
*BioSaccade-PEM: self-auditing foveated active vision* (submitted to *Image and Vision Computing*),
together with the locked protocol, the full numerical results, the tables and the figures of the
reported paper run.

**The datasets are not included.** DAVIS 2017 and REDS must be downloaded from their official
sources (links below) and used under their own licence terms.

## Contents

| Path | What it is |
|---|---|
| `BioSaccade_PEM.ipynb` | The single notebook that writes the source modules, checks them, and runs the locked, resumable protocol. Outputs cleared; local paths are placeholders. |
| `source/*.py` | The nine modules exactly as executed in the paper run (byte-identical to the module code inside the notebook; SHA-256 hashes are recorded in `results/protocol/inputs.lock.json`). |
| `requirements.txt` | Exact package versions of the environment used for the paper run (Python 3.14.6, Windows 11, CPU only). |
| `results/protocol/` | `inputs.lock.json` (data content hashes, code hashes, configuration, package versions), `protocol.lock.json` (every planned job, the fitted-model provenance, development selection, disclosures, including the post-pilot amendment), development selection scores, baseline sanity gate, data integrity report, runtime and readiness reports. |
| `results/csv/` | Per-sequence results (`sequence_results.csv`, 8,355 rows), source-cluster summaries, all paired comparisons, the same-backbone gaze comparisons, same-gaze ablation and coverage summaries. |
| `results/tables/` | Journal tables (CSV and booktabs LaTeX). |
| `results/figures/` | Figures as PDF and SVG (vector) and PNG (raster), plus `captions.txt`. Figure 06 (example video frames) is not included; see Disclosures. |
| `results/evidence/` | The evidence archive produced by the run (results, source, model provenance). See `EVIDENCE_NOTE.json`: the only change from the original archive is that local dataset folder paths in `data_integrity.json` were replaced by placeholders. |
| `post_hoc_pareto_analysis/` | **Descriptive analysis added after the locked paper run** (not part of the protocol): reconstruction vs audit-reliability frontier. See the section below and its `NOTE.md`. |

## What the notebook does

1. Writes the Python modules (`video_core`, `r9_core`, `r9_data`, `r9_neural`, `r9_vsr_arch`,
   `r9_engine`, `r9_stats`, `r9_checks`, `r9_runner`) into `<RESULTS_DIR>/src`.
2. Runs implementation checks (estimator unbiasedness, gradient and solver checks, dataset readers,
   information-boundary tests, confidence-sequence coverage simulation, causality of the video-SR wrapper).
3. Verifies the datasets, hashes every frame, and fixes the data manifest.
4. Fits the predictive memory and the auditor only on DAVIS *fit* sources (no clean pixel targets),
   selects controller settings on DAVIS *development* sources with a prespecified rule, and runs a
   behavioural sanity gate (every pretrained SR baseline must beat bicubic upsampling).
5. Locks the protocol (`protocol.lock.json`) and executes every job with checkpoints. Re-running the
   notebook resumes; finished jobs are never repeated, and changed inputs are refused.
6. Computes source-cluster paired statistics (bootstrap intervals, sign-flip tests, Holm adjustment) and
   writes tables, figures and the evidence archive.

### Studies in the paper profile

| Study | Data | Purpose |
|---|---|---|
| `main` | REDS val (30 clips x 100 frames) + DAVIS 480p legacy test (40 clips), nominal and drifting sensor, 2 draws | All controlled gaze policies and ablations, plus the secondary variant TwinAudit-strict (κw = 1) |
| `reds_train_full` | All 240 REDS `train_sharp` clips | Scale: greedy, previous version, PPAT, TwinAudit |
| `reds_train_detail` | Fixed 60-clip REDS train subset, 2 draws | All controlled policies |
| `neural_native` / `neural_sensor` | 30 REDS val + 10 DAVIS + 20 REDS train | Published SR / video-SR backbones (NTIRE 2025 ESR: SPAN-F, NanoSR, SCMSR; BasicVSR++ causal and offline non-causal reference) under clean MATLAB-bicubic input and under the paper's sensor model |
| `timing` | 10 REDS val clips, serial | The only study used for latency claims |
| sensitivity | 15 REDS val + 15 DAVIS + 15 REDS train | κ, certificate κ, write κ, preview factor and kernel, assumption-violating negative control |

No model in this study is trained on REDS: the PEM memory and auditor are fitted on DAVIS fit sources,
the BasicVSR++ checkpoint is the official Vimeo-90K BI checkpoint, and the NTIRE 2025 models are
still-image checkpoints. REDS train and val are therefore used only as unseen test data.

The pretrained baseline weights are **downloaded automatically** at run time from pinned sources and
verified by checksum: the NTIRE 2025 ESR repository (`github.com/Amazingren/NTIRE2025_ESR`, commit
`b69fb668c0362deb696eecbfdebfaa5c2fcdfcb4`) and the OpenMMLab model zoo
(`basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bi_20210305-4ef437e2.pth`). They are not redistributed here.

## Installation

Tested with Python 3.14.6 on Windows 11 (CPU only, no GPU). Other operating systems should work but
were not tested.

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m ipykernel install --sys-prefix --name biosaccade-venv --display-name "Python (BioSaccade venv)"
```

`requirements.txt` already points pip at the CPU-only PyTorch wheels. On Linux/macOS use
`.venv/bin/python` instead of `.venv\Scripts\python`.

## Datasets (download separately)

| Dataset | Official source | What to download | Expected folder |
|---|---|---|---|
| DAVIS 2017 | https://davischallenge.org/davis2017/code.html | *TrainVal - Images and Annotations - 480p* (`DAVIS-2017-trainval-480p.zip`) | `<DAVIS_ROOT>/JPEGImages/480p/<clip>/*.jpg` (90 clips) |
| REDS | https://seungjunnah.github.io/Datasets/reds.html | `train_sharp` and `val_sharp` | `<REDS_ROOT>/train_sharp/000 ... 239/*.png` and `<REDS_ROOT>/val_sharp/000 ... 029/*.png` (100 frames each) |

Keep `train_sharp` and `val_sharp` in separate folders; both start at `000`, and the code registers them
as two datasets (`REDS_train`, `REDS`) so clip identities never collide. The notebook refuses to run if
a split is incomplete. The exact frame content used for the paper is identified by the per-dataset
content digests in `results/protocol/data_integrity.json`.

## Running

1. Open `BioSaccade_PEM.ipynb` and, in the first code cell, replace the placeholders
   `<RESULTS_DIR>`, `<DAVIS_ROOT>` and `<REDS_ROOT>` with your local folders.
2. Set `PROFILE` and run all cells, either in Jupyter (kernel "Python (BioSaccade venv)") or headless:

```bash
.venv\Scripts\python -m nbconvert --to notebook --execute BioSaccade_PEM.ipynb --ExecutePreprocessor.kernel_name=biosaccade-venv --ExecutePreprocessor.timeout=-1 --output executed.ipynb
```

| Profile | What it runs | Measured time (Intel Core i7 13th gen, 16 GB RAM, no GPU) |
|---|---|---|
| `smoke` | Synthetic frames, every code path including all neural baselines | about 3-5 minutes (plus a one-time download of the baseline weights) |
| `laptop` | Pilot on real data: 26 DAVIS + 10 REDS val + 10 REDS train clips, 32 frames | about 7.7 hours (measured before neural predictions were parallelised; expect less now) |
| `paper` | Full protocol: 1,815 jobs, 8,355 method-sequence runs, 759,350 frame-method evaluations | **40.4 hours** (5 CPU workers, 3 neural workers x 2 PyTorch threads) |

Resource settings (`WORKERS`, `THREADS`, `NEURAL_WORKERS`, `NEURAL_THREADS`) affect speed only and are
not part of the locked protocol. On a 16 GB machine keep `NEURAL_WORKERS` at 3 or lower: each neural
worker needs about 1.5 GB of RAM on 100-frame clips. If a run stops, run the notebook again with the same
settings; it resumes from the checkpoints.

## Disclosures

* **Post-pilot amendment** (recorded in `protocol.lock.json` before the paper run): same-backbone gaze
  comparisons (backbone + TwinAudit gaze vs backbone + memory / + patch) and the secondary variant
  TwinAudit-strict (κw = 1) were added after the laptop pilot. TwinAudit with the development-selected
  κw = 2 remains the primary method and the target of all paired comparisons. Speed-only changes:
  `THREADS = 8` and parallel neural workers (old and new execution verified to agree to floating-point
  rounding).
* **SCMSR** samples `gumbel_softmax` even at inference in its official code; the PyTorch random generator
  is therefore seeded per clip, condition and model before each baseline prediction.
* **Development selection** in the paper run used the prespecified fallback (no candidate met the
  "MSE within 5% of greedy" condition); see `results/protocol/development_selection.json`.
* The 40 legacy DAVIS test sources were seen in earlier revisions of the manuscript. REDS train and val
  were never used for design, fitting or selection.
* Offline BasicVSR++ reads future frames and is reported only as a non-causal reference.
* `biased_stress` is a deliberately assumption-violating negative control; no coverage guarantee applies.
* Latency is reported only from the serial `timing` study.
* **No dataset images are distributed.** Figure `06_example` (a reference frame, its preview and its
  reconstruction) and the raw example frame `example.npz` contain DAVIS/REDS image content, so they were
  removed from `results/figures/` and from the evidence archive (see `results/evidence/EVIDENCE_NOTE.json`).
  Running the notebook regenerates them locally.

## Post-hoc descriptive analysis: reconstruction vs audit-reliability frontier

Folder: `post_hoc_pareto_analysis/` (details in its `NOTE.md`).

This analysis was **added after the locked paper run had finished**. It is descriptive only: it was not
prespecified, is not part of `protocol.lock.json`, and has no multiplicity-adjusted hypothesis tests. It must
not be read as confirmatory evidence.

* **Question:** do the TwinAudit sensitivity sweeps (κ and certificate κ) lie above or below the baseline gaze
  policies when reconstruction quality is plotted against audit reliability?
* **What was run:** Uniform, Greedy, PPAT (2026) and Audit-only gaze on exactly the 45 clips of the paper's
  sensitivity sweeps (15 DAVIS, 15 REDS val, 15 REDS train; nominal sensor; draw 0). The run reused the paper
  run's code (hash-checked), fitted models, selected settings and geometry. Nothing was refitted or reselected,
  and the locked paper results were not modified. TwinAudit curves are taken from the paper run.
* **Pairing check:** TwinAudit at the selected settings was re-run and matched the paper's κ = 2 rows
  bit for bit (2,745 of 2,745 values; `reproducibility_check.json`).
* **Statistics:** means over source clusters with 95% paired source-cluster bootstrap intervals (6,000 resamples).
  The figure's y-axis is the paired PSNR gain over uniform gaze on the same clips. Absolute PSNR is in
  `pareto_points.csv`.
* **Finding (descriptive):** the certificate-κ sweep lies above PPAT (about +0.21 dB at equal audit PQV) and,
  only at its most reliable setting, slightly above Audit-only gaze (about +0.08 dB). The κ sweep lies only
  slightly above PPAT (about +0.09 dB) and does not reach Audit-only's audit reliability. The selected TwinAudit
  setting itself has higher audit PQV than PPAT and Audit-only, as in the main tables. Greedy gaze has the
  highest PSNR but about 40x higher audit PQV.
* **Files:** `pareto_frontier.pdf|svg|png`, `pareto_points.csv`, `pareto_crossing.csv`,
  `addon_sequence_results.csv`, `reproducibility_check.json`, `pareto_addon.py`, `pareto_analysis.py`.

## Licence

The code, protocol files, results, tables and figures in this repository are released under the MIT Licence
(see `LICENSE`), copyright (c) 2026 Muhammad Usman and Mirza Adnan Baig. Third-party files recorded inside the
evidence archive (`official_models/`: the NTIRE 2025 model definitions, with their original `LICENSE`) remain
under their original licences. The DAVIS 2017 and REDS datasets and the pretrained model weights are not
included and are subject to their own terms.
