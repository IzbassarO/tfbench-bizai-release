# TrafficFlowBench 2026 (IEEE Big Data Cup) — team BizAI

Izbassar Orynbassar, Sevgi Erdogan (Syracuse University).

This package reproduces our two selected final submissions from the competition release with **one command**:

| final | file | public LB | sha256 of the submitted csv |
|---|---|---|---|
| no. 1 (`--final v3`) | `cand_combo_v3c` | 0.88369 | `341a64b3ddeff7cb8205037c1df00e005129b1ecf4f2247e3d173a5b060aea76` |
| no. 2 (`--final v1`) | `cand_combo_v1c` | 0.88317 | `15aceedd6acff579f952936709d82b24f03d693fb2657368f6534782254c8c01` |

```bash
python run.py --final v3 --data /path/to/kaggle_public      # full run: train every model, predict, assemble
python run.py --final v3 --skip-train                       # inference from the shipped weights
python run.py --final v3 --from-components                  # shipped member predictions -> byte-identical csv
python scripts/verify.py outputs/submission_v3.csv <byte-identical rebuild>.csv   # row diff + score bound
```
Output: `outputs/submission_<final>.csv` (Kaggle `sample_submission.csv` layout: state, queue and odme rows of
validation **and** private), its `.sha256`, `outputs/run.log` and `outputs/run_log.tsv` (stage, seconds, peak RSS).

## 1. Method in one table

| task (weight) | model | code |
|---|---|---|
| Task 1 state (0.35) | four members on the template's hidden cells: pooled LightGBM (speed + per-lane flow residuals to a profile-anchored interpolation; 8 transductive re-masks of the observed test cells; seeds 42/7/123), MLP 512×2 (random_12, huber, OneCycle, seeds 42/7/123), 2-D U-Net on time × link (width 32, 30 epochs, seeds 42/7/123/2031/137), U-Net2 (width 48, 50 epochs, 576-step window, seeds 42/7/123) | `tfb.t1.train`, `tfb.t1.mlp_colab`, `tfb.t1.unet` |
| state mix | v3: weights per traffic regime (4 bins of v / v_f of the 0.25/0.25/0.5 LGBM/MLP/U-Net mix, separate for speed and flow); v1: 0.5 · (0.25 LGBM + 0.25 MLP + 0.5 U-Net) + 0.5 U-Net2 | `tfb.finals`, `configs/finals.yaml` |
| Task 3 physics (0.15) | pp2: soft fundamental-diagram correction of flow on target cells, per-panel alphas | `tfb.t3.apply` |
| Task 2 queue (0.30) | onset: decode30(0.5 family A v4 + 0.5 family B noloc), both trained on hybrid labels (released truth, clean-imputer speed on hidden train cells); ongoing: exact expected-IoU top-m of 0.5 family A v2 + 0.5 t2b v2l over the 6 horizons. No Task 2 fit sees validation/private data: labels are released train truth or the clean imputer (train rows only), and every training-window selector reads those labels | `tfb.t2`, `tfb.t2b` |
| Task 4 ODME (0.20) | split-local projection min ‖f − b‖ s.t. A f = c, f ≥ 0 (split-local prior and counts) | `tfb.t4.project` |

Constants that were chosen on our train-split holdouts (last 30 days, autumn) are **pinned** in `configs/finals.yaml`
and `configs/alphas_v3.json` / `reports/physics_alphas_pp2.json`: the regime weights, the U-Net2 share c = 0.5 and
the pp2 alphas. A run trains the members; it does not repeat that selection.

## 2. Environment

- Python 3.12, `pip install -r requirements.txt` (exact versions; torch 2.10.0 — on Linux the PyPI wheel includes CUDA 12.8).
- CPU: LightGBM uses `num_threads` 8 (Task 1) and 2 (family A v4); its deterministic mode is fixed per thread count, so
  these numbers are part of the recipe. GPU (CUDA) for the MLP and both U-Nets; they also run on a CPU, much slower.
- Disk ≈ 60 GB (data 12 GB, cache, features, outputs). RAM: ≥ 32 GB sequential, ≥ 48 GB with `--jobs 3`.
- Seeds are fixed in every module; `PYTHONHASHSEED` is not used by the code.

## 3. Data and artifacts

- Competition data: unzip the Kaggle release so that `--data` points at the folder holding `config/`, `corridors/`,
  `task1/` … `submission_key.csv`.
- Artifacts (assets of the GitHub release `v1.0` of this repository; only for `--skip-train` / `--from-components`; a
  full run needs neither):
  ```bash
  B=https://github.com/IzbassarO/tfbench-bizai-release/releases/download/v1.0
  curl -LO $B/tfb_release_components.zip -LO $B/tfb_release_weights.zip -LO $B/SHA256SUMS
  sha256sum -c SHA256SUMS && unzip -q tfb_release_components.zip && unzip -q tfb_release_weights.zip   # -> artifacts/
  ```
  `tfb_release_components.zip` (308 MB): the four Task 1 member predictions on the validation/private template + the
  final queue file. `tfb_release_weights.zip` (693 MB): LightGBM boosters (speed and per-lane flow, 3 seeds), MLP x3,
  U-Net x5 and U-Net2 x3 weights, and the Task 2 members' final probabilities on the released windows (the Task 2
  boosters are cheap to retrain and were not kept; `--skip-train` decodes these probabilities).

## 4. Stages, time and memory

`run.py` runs each stage as its own process and records wall time and peak RSS; a finished stage is skipped on the
next call (resume after a lost session). `--jobs 3` runs the three independent lanes (Task 2 · LightGBM · GPU) at once.

| lane | stage | what | measured (where) | peak RAM |
|---|---|---|---|---|
| prep | cache | released csv/parquet → per-panel npz | ~10 min (M1 Pro) | < 4 GB |
| prep | t4 | Task 4 projection | < 1 min | < 2 GB |
| t2 | t2_codex | family A v1 dataset, v2 (ongoing member), v4 (onset member) | ~4.5 h together with t2b_base (M1 Pro) | ~4 GB |
| t2 | t2b_base | t2b windows + noloc members | (in the line above) | |
| t2 | imputer | clean LightGBM imputer (train rows only), hybrid labels | TBD | |
| t2 | onset | v4 / noloc on hybrid labels | ~55 min (M1 Pro) | ~5 GB |
| t2 | ongoing | selector windows + v2l (3 seeds) | TBD | |
| lgbm | lgbm | LightGBM 8× re-masks, 3 seeds, sequential | 12.3–14.1 h (M1 Pro, 8 threads) | ~18 GB |
| gpu | mlp | feature export + 3 seeds | 0.5 h training (L4) + export TBD | |
| gpu | unet | U-Net × 5 seeds | 4.9 h (L4: 3440–3565 s per seed) | |
| gpu | unet2 | U-Net2 × 3 seeds | 9.9 h (L4: 11 780–12 030 s per seed) | |
| final | final_<v> | mix, pp2, assembly, format checks | ~1.5 min | ~8 GB |

A100 40 GB timings of the rehearsal: TBD.

## 5. What is identical and what is not

- `--from-components`: byte-identical to the submitted csv (sha256 checked by `run.py`).
- Full run and `--skip-train`: the CPU members (LightGBM, Task 2, Task 4) are deterministic for the pinned versions
  and thread counts; the GPU members (MLP, U-Nets) differ at float-rounding level across GPU types and drivers (the
  U-Net's CPU vs GPU inference differs by ≤ 0.01 km/h on the same weights). `scripts/verify.py` compares the result with
  the byte-identical rebuild row by row and bounds the score change without the hidden truth: by the Minkowski
  inequality |RMSE(a, y) − RMSE(b, y)| ≤ RMSE(a, b) per panel × regime for S_state; a changed queue window moves its IoU
  by at most 1. It fails if the bound exceeds 1 % of the public score.

## 6. Task 2 online rule

Every Task 2 feature uses only data stamped at or before the forecast origin T of its window (the row at T included).
`python scripts/check_online.py` rebuilds every member's per-window inputs from the released data and from a copy
where every row after T is replaced by random values, and requires identical inputs (exit 1 otherwise);
`--control` replaces the row at T too and requires every member to change (a positive control showing the check can
fail). On all 8 Task 2 panels: 160 windows, 0 member-windows changed; under `--control` every member changed in every
window. `pytest tests` runs the feature-level leakage tests.

Fitting side (rule: no data after T, also not through fitting): Task 2 models, priors, calibration and window selectors
use the training split only; the hybrid labels on hidden train cells come from a LightGBM imputer fitted on train rows
only (`tfb.t1.imputer_clean`). Task 1 (offline reconstruction, transductive re-masks of the observed test cells) never
feeds Task 2.
