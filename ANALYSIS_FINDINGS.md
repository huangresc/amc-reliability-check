# Re-analysis of the frozen AMC experiment — verified numbers

Source of truth: `Desktop/论文实验数据汇总.pdf` + `lmr_cnn_project/tables/*.csv`.
All contrasts below recomputed from per-seed tables with paired two-sided Wilcoxon,
n=10 seeds, following the pre-registration in `REGISTRATION.md`.
Scripts: `_stats.py`, `_stats2.py`.

## 0. Corrections to the summary PDF

- `dead_frac_c1` is NOT the first *convolutional* layer. From `src/models.py`, the
  classifier is `Flatten -> Linear -> ReLU -> Dropout -> Linear`. `classifier[1]` is the
  first fully-connected layer, 64 units. Total count 16384 = 64 units x 256 batch.
  So "16384/16384 dead" = all 64 units silent on the whole fixed 256-sample probe batch.
- The probe batch is the *first* minibatch of the training loader (fixed per seed via
  `set_seed`), not the full dataset. Sampling caveat to declare in Methods.

## 1. Architectures (from src/models.py)

| variant | trunk | head | first FC |
|---|---|---|---|
| M2_GAPHead | 2 conv + relu, maxpool(1,2) | AdaptiveAvgPool2d(1) | Linear(32 -> 64) |
| M4_FlattenFC | identical trunk | Flatten | Linear(1024 -> 64) |

Same trunk, same classifier width (64). Only the head differs: pooled 32-dim vs
flattened 1024-dim. PyTorch default `nn.Linear` init draws bias from U(-1/sqrt(fan_in), ...):
M2 bound = 1/sqrt(32) = 0.177, M4 bound = 1/sqrt(1024) = 0.03125.
Observed `preact_min` for the collapsed M4 run = **-0.0328**, i.e. essentially the
M4 init bound. All 16384 pre-activations sit inside a ~0.03-wide negative band:
the layer is a near-constant negative map, so the ReLU output is identically zero.

## 2. Primary contrast M2 - M4, no normalization (pp)

Full sample (n=10) vs successful subsample (n=9, drop seed 2021).

| epoch | full sample | subsample |
|---|---|---|
| 20  | -0.851  p=0.232  ns | **-3.949  p=0.039  SIG** |
| 40  | +2.162  p=0.695  ns | -1.365  p=0.301  ns |
| 80  | +3.934  p=0.375  ns | +0.291  p=0.652  ns |
| 120 | +5.327  p=0.037  SIG | +1.616  p=0.074  ns |
| 160 | +5.483  p=0.037  SIG | +1.648  p=0.074  ns |
| 240 | +5.533  p=0.014  SIG | +1.665  p=0.027  SIG |
| 320 | +5.941  p=0.006  SIG | +2.046  p=0.012  SIG |
| 480 | +6.479  p=0.004  SIG | +2.515  p=0.008  SIG |
| 640 | +6.661  p=0.002  SIG | +2.734  p=0.004  SIG |

**The single collapsed run converts a significant -3.95 pp at E=20 into a null
result (-0.85 pp, p=0.232), and inflates the final architecture gap from 2.73 pp
to 6.66 pp (+144 %).**

Crossover (subsample) between ep40 (-1.365) and ep80 (+0.291): **E ~ 60-80**.
Crossover (full sample) between ep20 and ep40: **E ~ 25-30**. One failed run
moves the reported crossover point by roughly a factor of two.

## 3. Robustness of the E=20 reversal — WEAK

Leave-one-out on the 9 surviving seeds (subsample baseline -3.949 pp):

- sign is always negative (-3.30 to -4.78 pp) — stable
- but p<0.05 only for 3 of 9 drop-one configurations; the rest give p=0.078.

=> Report as "M4 leads at E=20 by ~3.9 pp, borderline significance, sign stable
under leave-one-out". Do NOT write "highly significant reversal".

## 4. Normalization ablation at E=640, paired (pp)

| arm | gain from global std | gain from per-sample |
|---|---|---|
| M2  | +3.695 (SE 0.24, p=0.002) | +2.402 (SE 0.18, p=0.002) |
| M4  | +6.237 (SE 4.07, p=0.010) | +4.765 (SE 4.05, p=0.049) |

Difference-in-differences (M4 gain - M2 gain): global **+2.541, p=0.375 ns**;
per-sample **+2.362, p=0.232 ns**.
=> The interaction is NOT statistically established. Point estimates suggest M4
benefits more, but the CI spans zero because M4's gain is carried by the collapsed
seed. Do NOT claim an architecture-dependent normalization benefit.

## 5. Protocol can erase the architecture effect (E=640)

| protocol pairing | M2 - M4 | p |
|---|---|---|
| both no-norm | +6.661 | 0.002 SIG |
| both global std | +4.120 | 0.002 SIG |
| both per-sample | +4.299 | 0.002 SIG |
| **M2 no-norm vs M4 global std** | **+0.425** | **0.275 ns** |

=> Two papers using the same two architectures can report a 6.7 pp gap or no gap,
depending only on the preprocessing protocol. The protocol effect for M4 (+6.24 pp)
is the same size as the whole architecture effect (+6.66 pp).

## 6. Augmentation arm — the best-powered budget result

M2 (no aug) - M5 (continuous phase aug), pp:

| epoch | diff | p |
|---|---|---|
| 20 | +12.310 | 0.004 SIG |
| 40 | +9.657 | 0.004 SIG |
| 80 | +5.446 | 0.006 SIG |
| 120 | +5.124 | 0.004 SIG |
| 160 | +3.630 | 0.004 SIG |
| 240 | +2.558 | 0.002 SIG |
| 320 | +1.983 | 0.002 SIG |
| 480 | +1.376 | 0.002 SIG |
| 640 | +1.348 | 0.002 SIG |

Effect decays ~9x, monotone, no saturation, every point significant, no failed runs.
M5b (4-fold discrete phase aug) is the same shape: +12.967 pp at E=20 to +0.990 pp at
E=640, and non-significant across ep80-ep480.

=> This is the cleanest evidence for budget-dependent effect size, because it is not
contaminated by a collapse. It also gives a second, independent instance of the
phenomenon in the same study.

## 7. Collapse fingerprint (seed 2021, M4, no norm, E=640)

- dead: 16384/16384 = 100.00 %
- logit_std = 0.0071 (healthy M4n_g: 66.5421) -> ratio 9372x
- train_loss = 2.3979; **ln(11) = 2.3979** (11-class uniform prediction, exact to 4 dp)
- test_acc = 9.0205 %; chance 1/11 = 9.0909 %
- preact_min = -0.0328 ~ -1/sqrt(1024), the M4 default-init bias bound

## 8. Escape test — stronger than the PDF states

`logs_escape_test.txt`, LR raised 0.001 -> 0.1 (100x), 5 epochs, from the ep640 checkpoint:

| epoch | lr | acc | dead | preact_min |
|---|---|---|---|---|
| 0 (before) | 0.001 | 9.02 | 16384/16384 | -0.0328 |
| 1 | 0.1 | 9.14 | 16384/16384 | -0.0328 |
| 2 | 0.1 | 9.27 | 16384/16384 | -0.0328 |
| 3 | 0.1 | 9.11 | 16384/16384 | -0.0328 |
| 4 | 0.1 | 9.05 | 16384/16384 | -0.0328 |
| 5 | 0.1 | 9.27 | 16384/16384 | -0.0328 |

`preact_min` is bit-identical across all five epochs. Because every pre-activation is
strictly negative, the ReLU derivative is zero everywhere, so the subgradient of the
loss w.r.t. every parameter at or below that layer is identically zero. The trunk is
provably frozen: no first-order method at any learning rate can move it. Only the
final bias stays adjustable, and with an identically zero feature map its optimum is
the class prior. The residual accuracy wobble (9.02-9.27 %) has binomial SE 0.137 pp
at n=44000, i.e. it is within ~2 SE of a constant predictor.

=> This is not an anecdote about "a big LR didn't help"; it demonstrates an exact
absorbing state. `preact_min` invariance is the proof.

## 9. Dead fraction vs performance — a cliff, not a slope

| run | dead % | live units (of 64) | test acc |
|---|---|---|---|
| M4 no-norm | 100.00 | 0 | 9.02 |
| M4n_g | 86.55 | ~8.6 | 51.56 (seed 2021; mean 51.34) |
| M2n_g | 81.79 | ~11.6 | 55.94 (seed 2021; mean 55.46) |

Dead fractions here are from the 256-sample probe batch used in the original
probe, not the full test set. The accuracies are ep640 values from
`normalization_ablation_*_per_seed.csv`; the number in the cell is the seed-2021
value, and the mean over the ten seeds is given in parentheses.

86.55 % dead is survivable. The failure threshold is exactly zero live units.
The region between 0 and ~8.6 live units is unsampled, so the shape there is unknown.
Monitoring dead fraction as a graded health metric is not supported by these data;
the binary event "no live unit on the probe batch" is.

## 10. Early trajectory (10 M4 seeds, dead count of 16384)

| seed | ep0 | ep1 | ep2 | ep3 | ep7 | ep15 |
|---|---|---|---|---|---|---|
| 42 | 9207 | 13512 | 13530 | 12638 | 12944 | 13771 |
| 123 | 7491 | 13480 | 13727 | 13931 | 14676 | 14821 |
| 456 | 7424 | 13477 | 13199 | 13125 | 14066 | 14494 |
| 789 | 8192 | 13312 | 13409 | 14253 | 14761 | 14869 |
| 1011 | 10146 | 13824 | 14336 | 14046 | 14910 | 15219 |
| 1213 | 8960 | 13448 | 13477 | 13882 | 14818 | 15065 |
| 1415 | 8711 | 12697 | 13431 | 13567 | 14303 | 14729 |
| 1617 | 8221 | 14616 | 14671 | 14889 | 15205 | 14956 |
| 1819 | 9159 | 13669 | 12919 | 13408 | 14173 | 14906 |
| **2021** | 8448 | 12800 | 14336 | **15360** | **16384** | **16384** |

All seeds start at 45-62 % dead and rise. Healthy seeds saturate at 84-93 %.
seed 2021 is the maximum at ep3 (93.75 %) and saturates at 100 % by ep7.
n_fail = 1, so this identifies a candidate early-warning time (E ~ 3-7) but cannot
calibrate a threshold.

## 12. Escape / repair sweep — full results (escape_repair_sweep.json)

Ran after the frozen analysis above; supersedes the 5-epoch escape test in §8 and
resolves the two loose ends left by it (n=1 LR multiplier, single-batch probe).

### 12.1 P1 — collapse fingerprint over the full test set (n = 44 000)

| ReLU layer | dead fraction | alive | pre-activation range |
|---|---|---|---|
| `features.1` (conv1) | 62.50 % | 67 584 044 / 180 224 000 | [−0.3007, +0.3956] |
| `features.4` (conv2) | 99.99997 % | **31** / 90 112 000 | [−0.1472, +0.0223] |
| `classifier.2` (fc1) | 100.00 % | **0** / 2 816 000 | [−0.033019, −0.000484] |

logit_std 0.007062; test acc 9.0205 %; loss 2.398188 (ln 11 = 2.397895).

Gradient at the checkpoint, one forward/backward, before any step:

| parameter | grad norm |
|---|---|
| trunk (`features.*`) | **exactly 0.0** |
| `classifier.1.weight` | 0.0 |
| `classifier.1.bias` | 0.0 |
| `classifier.4.weight` | 0.0 |
| `classifier.4.bias` | 0.0560205 |

The only parameter in the network with nonzero gradient is the final bias.

### 12.2 The terminal state is exactly a constant predictor

| quantity | collapsed M4 (2021) | healthy M2 (2021) |
|---|---|---|
| distinct predicted labels over 44 000 test samples | **1** (label 9) | 11 |
| test accuracy | 9.0205 % | 51.02 % |
| prevalence of the emitted label | 9.0205 % | — |
| \|accuracy − prevalence\| | **0.0000 pp** | — |

The collapsed model emits one label for every input; its accuracy therefore equals
that label's prevalence *by construction*, to four decimals. This closes the account
of the terminal state: chance-level accuracy is not "poor learning", it is the
deterministic output of a constant map.

### 12.3 P2 — no learning rate escapes (10 epochs from the collapsed checkpoint)

| LR | dead fc1 | logit_std | test acc |
|---|---|---|---|
| 1e-3 (×1) | 100 % | 0.00634 | 9.018 % |
| 1e-2 (×10) | 100 % | 0.01326 | 9.052 % |
| 1e-1 (×100) | 100 % | 0.09606 | 9.264 % |
| 1e0 (×1000) | **100 %** | 0.21757 | 9.143 % |

A 1000× learning rate moves the logits (logit_std ×34) without restoring a single
unit. Only `classifier.4.bias`, the sole parameter with nonzero gradient, responds.

### 12.4 P2b — the same terminal state is reachable in the opposite direction

Healthy M2 (no normalization, seed 2021, 51.03 %) re-trained at raised LR:

| LR | conv1 dead | conv2 dead | fc1 dead | fc1 pre-act range | acc | logit_std |
|---|---|---|---|---|---|---|
| ×1 | 76.91 % | 68.09 % | 83.08 % | [−21.71, +3.17] | 51.03 % | 25.40 |
| ×10 | 86.00 % | 74.09 % | 87.59 % | [−33.02, +3.56] | 48.57 % | 26.23 |
| ×100 | 93.10 % | 59.37 % | **100 %** | **[−2637.4, −6.65]** | **9.27 %** | 0.1344 |

At ×100 a *healthy* run becomes a constant predictor (9.27 %), i.e. the identical
observable terminal state — but reached with fc1 pre-activations exploded to −2637
rather than frozen at 0.03. The two routes are therefore distinct:

- **starvation** (architecture route): pre-activations remain inside the initial bias
  support, parameters never move, gradient is exactly zero;
- **divergence** (LR route): pre-activations explode, the layer saturates negatively.

Both yield "fc1 dead → constant predictor". A dead-fraction monitor cannot separate
them; the pre-activation range does, and the remedies are opposite.

Note the architecture route also kills conv2 (99.99997 %) whereas the LR route leaves
conv2 at 59 % dead. The invariant across both is fc1 = 100 % dead.

### 12.5 P3 — re-initialization repairs the run (gradients alone do not)

From the collapsed checkpoint, `nn.Linear.reset_parameters()` on the dead layer,
then 40 epochs at the *original* LR 1e-3:

| variant | at repair (ep 0) | after 40 epochs |
|---|---|---|
| reinit `classifier.1` | dead 50.0 %, acc 9.05 % | dead 93.10 %, **acc 40.79 %**, logit_std 12.36 |
| reinit `classifier.1` + `classifier.4` | dead 50.0 %, acc 9.14 % | dead 89.04 %, acc 35.05 %, logit_std 3.70 |

Re-initialization restores half the units (a fresh bias straddles zero) and returns
the run from 9.02 % to 40.79 %. It does not reach the healthy level (M2 55.94 %,
M4 global 51.56 %) because conv2 remains dead and 40 epochs is short. Recovery is
therefore possible but partial, and requires a parameter perturbation, not a
gradient step.

### 12.6 P4 — cascade timeline, M4 seed 2021 trained from scratch

| epoch | conv1 | conv2 | fc1 | test acc | logit_std |
|---|---|---|---|---|---|
| 0 | 0.6250 | 0.4077 | 0.5156 | 9.27 % | 0.0751 |
| 1 | 0.6250 | **0.99951** | 0.7344 | 9.26 % | 0.0139 |
| 2 | 0.6250 | 0.99951 | 0.8281 | 8.96 % | 0.0096 |
| 3 | 0.6250 | 0.9999997 | 0.9375 | 8.96 % | 0.0102 |
| 5 | 0.6250 | 0.9999998 | 0.9688 | 8.96 % | 0.0090 |
| 7 | 0.5794 | 0.9999942 | **1.0000** | 9.02 % | 0.0075 |
| 20 | 0.5794 | 0.9999942 | 1.0000 | 9.14 % | 0.0053 |

conv2 collapses within one epoch; fc1 follows and is fully dead by epoch 7; conv1
stays in the 58–63 % band throughout and never signals anything. The loss is pinned
at ln K from epoch 1, so no progress occurs after the first epoch out of 640.

**Operational consequence:** the run is identifiable at epoch 1 of 640 from the
convolutional trunk. Monitoring the classifier — the layer the original probe
watched — first saturates far too late and, in the healthy control, sits at 83 %
dead without harming accuracy.

## 13. What the evidence does and does not support

STRONG (well-powered, pre-registered):
- effect size, not just rank, is budget-dependent; decay up to 9x, monotone, no
  saturation by E=640 (augmentation arm, n=10, all p<0.01)
- protocol swap can erase the architecture effect (M2-raw vs M4-standardized: +0.4 pp, ns)
- the terminal state is exactly characterised: constant predictor, accuracy = emitted
  label's prevalence to 0.0000 pp, loss = ln K, logit_std ~ 0.007
- the state is an exact absorbing state: trunk gradient identically 0.0, only the
  final bias trainable, and *four* LR multipliers up to 1000x leave it unchanged
- the same terminal state is reachable from the opposite direction (100x LR on a
  healthy run), which separates starvation from divergence
- the cascade ordering is directly observed: conv2 dies at epoch 1, fc1 by epoch 7
- recovery by re-initialization (+31.8 pp) while gradient steps cannot recover it

MODERATE (correct arithmetic, weak power):
- one failed run shifts the architecture gap by +144 % and nulls a significant
  low-budget contrast; moves the crossover point ~2x
- the one-epoch early-warning signal: demonstrated on 1 seed, needs calibration

WEAK / must be labelled as case study:
- both mechanism routes rest on single seeds (2021 for M4, 2021 for the M2 control)
- the repair result uses one re-initialization scheme and 40 epochs
- n_fail = 1 gives no power for a failure-rate claim (Fisher exact 1/10 vs 0/10, p=1.0)

NOT SUPPORTED by these data:
- architecture-dependent normalization benefit (DiD ns)
- any graded dead-fraction -> accuracy relationship
- "the ranking reverses at E=20" as a strong claim (borderline p, fragile to LOO)
- that the failed run is specific to the flatten head: the LR control shows an
  identical terminal state in the GAP head, so the head sets the *route*, not the
  *possibility*, of collapse

## 14. Revision-round results (official split, repair sweep, baseline)

Supersedes and extends §12. Produced by the `revision/` scripts; distilled
numbers in `revision/analysis_results/`. All values below match the manuscript.

### 14.1 Partition provenance (rc_partition_fingerprint.py)

The legacy `train_idx_sha1` field is `sha1(list(range(n_train)))`: it encodes
only the training-set *size*, not which samples were chosen. Because the two
historical partitions differ in size (176 000 vs 110 000), the legacy value does
distinguish them in this study, but it is not a partition fingerprint.
`partition_fingerprints.json` records the true fingerprints of both partitions
and maps the legacy values to them. The manuscript (Section III-A) reports the
two partitions separately rather than pooling them.

### 14.2 Failure rate, de-pooled

- re-split (176 000/44 000), 10 seeds: **1 failure** (seed 2021), 10.0 %,
  Wilson 95 % [1.8 %, 40.4 %].
- official stratified (110 000/110 000), 30 seeds: **2 failures** (one of them
  seed 42), 6.7 %, Wilson 95 % [1.8 %, 21.3 %].

The intervals overlap almost completely, so the data do not resolve a
difference in failure rate between the partitions; the two failed runs share
the same mechanism (the constant predictor). The partition is treated as an
axis of the draw, not a separate population.

### 14.3 Repair sweep (20 draws x 2 runs = 40 attempts)

23/40 escape (57.5 %, Wilson [42.2 %, 71.5 %]); 10/20 on the re-split, 13/20 on
the official partition (intervals overlap). Escape is all-or-nothing: escaping
attempts land in 33.3–45.5 %, failed attempts return exactly to the constant
predictor (8.96–9.27 %).

Two earlier claims, overturned by the full 20 draws (now negative results):

- the epoch-0 dead-count threshold from the first five draws (a 32-unit cutoff)
  does not survive: Spearman rho = +0.22 (p = 0.35), Mann–Whitney AUC = 0.64
  (p = 0.38). Withdrawn.
- "the verdict belongs to the draw alone": 13/20 agreement, binomial p = 0.26;
  McNemar p = 0.45. Too strong; now reported as shared but not established.

### 14.4 From-scratch baseline (exp5_scratch_baseline.py)

A 40-epoch run from random initialisation on the official split (seed 42)
reaches **44.97 %**, against an escape range of 39.6–45.5 %. A successful repair
therefore *matches* a short fresh run rather than beating it; repair is cheaper
only against the full 640-epoch schedule.

### 14.5 Weight-space comparison (rc_weights*.py)

The collapsed run's fc1 weight matrix moved by a relative Frobenius displacement
of **0.282** by epoch 20 and is bit-identical from epoch 20 to 640. The
starvation signature is a scale, not proof of immobility.

### 14.6 Detectability, corrected

Across the 30 official-split seeds, 19 of the 28 runs that recover are still at
chance at epoch 1 — the failure is *not* distinguishable there. Accuracy
separates the two outcomes from epoch 4, by a usable 8.41 pp at epoch 5.
