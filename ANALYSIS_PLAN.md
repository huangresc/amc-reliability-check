# Analysis plan — pre-specified, not pre-registered

This file records the analysis plan that was fixed before the confirmatory run
outcomes were inspected. It is **not** a pre-registration: the plan was never
deposited with an independent registry, and no registration timestamp is
claimed for it. The manuscript (Section III-D, "Analysis Plan and Statistics")
describes the same plan as *pre-specified* rather than *pre-registered*.

## Analysis unit
All seeds in the grid, n = 10 per variant. No run is excluded from any primary number.

## Primary statistic
Paired per-seed differences at each budget E in {20,40,80,120,160,240,320,480,640},
tested with two-sided Wilcoxon signed-rank. Mean ± SE reported descriptively;
median [IQR] reported alongside, because a failed run makes the distribution
bimodal and invalidates normal-theory intervals.

## First-class outcome: optimization failure rate
A run is declared FAILED iff its final-epoch accuracy < 15.0 %
(= chance 9.09 % + 5.8 pp). Reported per variant as k/10.
This rule is applied symmetrically to M2, M4, M5 and M5b.

## Secondary, conditional analysis (pre-declared, not post hoc)
"Optimization-successful subsample" = all runs not declared FAILED by the rule
above. Reported as conditional on successful optimization, and never alone.

## Reporting rule
Wherever a paired contrast is reported, the full-sample value and the
successful-subsample value are reported side by side.

## Out-of-sample extrapolation test (already executed)
Models fitted on E ∈ {20,...,320} only, then used to predict E = 640.
No refitting after observing E = 640.

## Added after the confirmatory results (labelled *Exploratory* in the paper)

The thirty-seed arm on the official split (SEED30), the twenty-draw repair
sweep, the from-scratch baseline, and the weight-space comparisons were run
after the confirmatory results were seen. They are labelled *(Exploratory)* in
the manuscript and are documented in `ANALYSIS_FINDINGS.md` and the
`revision/` directory, not as part of this pre-specified plan.
