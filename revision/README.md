# Revision-round experiments

This directory holds the code and distilled data for the experiments added
during revision. All numbers below match the manuscript.

| Experiment | Scripts | Results |
|---|---|---|
| Thirty-seed arm, official split (SEED30) | `rc_train.py`, `rc_inference.py`, `rc_recompute_headline.py`, `rc_failrate.py`, `rc_detect.py` | `analysis_results/rc_inference.json`, `rc_headline_official.json` |
| Twenty-draw repair sweep (40 attempts) | `rc_repair_perturbation.py`, `rc_repair_stats.py`, `rc_repair_leads.py` | `analysis_results/rc_repair_perturbation.json` |
| From-scratch baseline | `exp5_scratch_baseline.py` | `analysis_results/exp5_scratch_baseline.json` |
| Weight-space comparison (Frobenius displacement) | `rc_weights.py`, `rc_weights2.py`, `rc_weights3.py`, `rc_weights_2021.py`, `rc_w0_check.py` | `analysis_results/rc_weights*.json` |
| Partition provenance / fingerprints | `rc_partition_fingerprint.py`, `rc_verify_split.py`, `rc_crosscheck_partition.py` | `analysis_results/partition_fingerprints.json` |
| Mechanism probes | `exp1_static_analysis.py`, `exp2_cascade_trace.py`, `exp3_lr_escape.py`, `exp4_repair.py` | `analysis_results/exp*.json` |
| Batch-normalisation ablation | `rc_bn_ablation.py` | `analysis_results/rc_bn_ablation.json` |
| Residual-connection ablation | `rc_residual_ablation.py` | `analysis_results/rc_residual_ablation.json` |
| Escape loss-landscape (Hessian) | `rc_escape_landscape.py` | `analysis_results/rc_escape_landscape.json` |
| Paper figures 1–3 | `make_budget_figure.py`, `make_repair_figure.py`, `../utils/make_paper_figures.py` | — |

## Running

`src/` here is the **updated** library whose `data.py` builds the official
110 000/110 000 stratified split (the repository-root `src/` builds the
176 000/44 000 re-split). Run from inside this directory so that
`from src.data ...` resolves to this `src/`.

`tables/` here mirrors the repository-root per-seed CSVs (re-split results).

## Not shipped

- `runs/` — per-seed model checkpoints and `metrics.json` for the official
  split (~800 MB). The scripts above read them; the distilled numbers are in
  `analysis_results/`.
- `rml2016_data/` — the RML2016.10a dataset (publicly available from DeepSig).

The statistics reported in the paper (paired Wilcoxon, Holm correction,
cluster bootstrap, Wilson intervals) are recomputed from the JSONs in
`analysis_results/` and the CSVs in `tables/`.
