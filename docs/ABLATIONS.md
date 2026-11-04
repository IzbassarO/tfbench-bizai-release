# Ablations and rule audit

S_total = 0.35 S_state + 0.30 S_queue + 0.15 S_physics + 0.20 S_ODME. Every step below changed one factor against the
previous best; "local" is our judge on train-split holdouts (Task 1/3: last 30 days and autumn, three seeds; Task 2:
the official per-window IoU on every admissible train window).

## Public leaderboard milestones

| date | change | public S_total |
|---|---|---|
| 10-24 | LightGBM state, persistence queue | 0.73460 |
| 10-24 | learned Task 2 queue | 0.86172 |
| 10-25 | transductive re-masks 1× → 3× | 0.87097 |
| 10-25 | location-free family B onset | 0.87309 |
| 10-25 | hybrid-label onset, pp2, ongoing blend | 0.87841 |
| 10-26 | LightGBM 8× re-masks, 3 seeds | 0.87894 |
| 10-29 | + MLP (random_12, 3 seeds) | 0.88017 |
| 10-30 | rule (B): clean-imputer onset | 0.87889 |
| 10-30 | + U-Net × 3 | 0.88086 |
| 10-31 | U-Net × 5, regime weights | 0.88184 |
| 11-02 | + U-Net2; v2l ongoing | 0.88327 |
| 11-03 | regime mix with U-Net2 | 0.88378 |
| 11-04 | rule (B): clean v2l windows — **final no. 1** (`--final v3`) | **0.88369** |
| 11-04 | the same fix on the 11-02 state — **final no. 2** (`--final v1`) | **0.88317** |

## Local judge vs public (same base)

Queue rows: expected ΔS_total = 0.30 × 0.5 × ΔIoU (one of the two queue conditions). State rows: ΔJ, J = 0.35 S_state + 0.15 S_physics.

| change | local | public |
|---|---|---|
| converged + re-masks 1× | +0.0017 | +0.00683 |
| re-masks 1× → 3× | +0.0004 | +0.00098 |
| MLP × 3 seeds | +0.0002 | +0.00019 |
| MLP random_12 | +0.00037 | +0.00028 |
| U-Net2 (c = 0.5) | +0.0013 | +0.0015 |
| regime mix with U-Net2 (chained) | ≈ +0.0004 | +0.00051 |
| state ensemble rank 1 | +0.00062 | +0.00027 |
| clean-imputer onset (CV +0.0021 IoU) | +0.0003 | −0.00128 |
| v2l ongoing (+0.0061 IoU) | +0.0009 | +0.00036 |

## Levers that did not survive

| lever | result |
|---|---|
| occupancy at hidden cells (oracle) | −0.03 … −0.06 J |
| characteristic features (free-flow / wave) | 0.00 p.p. RMSE |
| connected-run decoders (onset / ongoing) | −0.077 / −0.151 IoU |
| isotonic calibration before top-m | −0.0081 ± 0.0028 IoU |
| onset without the family A member, Q2 features (local +0.0100 IoU) | public −0.0027, −0.0049 |
| weighted onset w37 (CV +0.004 / +0.019) | public ΔS_onset ≈ −0.013 |
| physics-feature onset members G / G2 | public −0.00057 / −0.00074 |
| test-month location priors | public −0.00562 |
| site adaptation | public −0.00049 |
| raw space-time patch ongoing member | −0.072 IoU alone |
| two extra U-Net2 seeds | ≈ −0.00005 vs the final state |
| size-calibrated onset decoder | no trend in the true onset set size (−0.004 ± 0.011 cells / month) |

Why the remaining gap is small: at every Task 1 target the release blanks speed, flow, occupancy and density; our
residual in vehicle counts at hidden cells is 3.38 % against a noise-floor bound of ≤ 3.2 %, so reaching the floor would
add ΔJ ≤ +0.002. The public queue score uses 80 windows with a per-window IoU standard deviation of 0.375 and resolves
only |ΔS_total| ≳ 0.005, so Task 2 decisions were taken on the local judge.

## Rule (B) audit (no data after T, also not through fitting)

Every Task 2 path of both finals was audited: features, labels, training-window selectors, priors, calibration,
decoders and tuned weights. Features read only data stamped ≤ T (`scripts/check_online.py`: 160 released windows, 0
member inputs changed when everything after T is randomised; under `--control`, which also replaces the row at T, every
member changes in every window). Labels are released train truth or a LightGBM imputer fitted on train rows only;
priors and calibration come from the training split; the decoder is parameter-free. One path was not clean: the extra
training windows of the ongoing member v2l had been selected on hybrid labels from an earlier imputer that was fitted
with transductive validation/private rows. We re-selected those windows on the clean imputer's labels (3 356 of 3 503
windows unchanged) and refitted v2l. This changed 64 queue cells in 37 of the 80 released ongoing windows and nothing
else; public score −0.00009 / −0.00010. The ongoing weight 0.5 / 0.5, chosen on the local judge, still wins on the
clean-selected windows (+0.0063 ± 0.0020 IoU vs +0.0061 ± 0.0019 before). Task 1 (offline reconstruction with
transductive re-masks of the observed test cells) never feeds Task 2.
