# AMC Reliability Check

Code and data for the paper:

> **Training budget and preprocessing protocol confound a head comparison in
> lightweight modulation classification: a pre-deployment reliability check**

This repository holds the per-seed tables and the scripts producing every
statistic in the paper. Per-seed model checkpoints are **not** included (they
are ~800 MB); the distilled per-seed results are in `tables/` and
`revision/analysis_results/`.

## Repository structure

| Directory | Contents |
|-----------|----------|
| `src/` | Core library for the **original re-split (176 000/44 000) experiments**: data loading, models, training loop, provenance, table generation |
| `grids/` | Experiment configuration files (YAML) |
| `runners/` | Batch training scripts that invoke `src/train.py` |
| `probes/` | Mechanism probe scripts (dying ReLU, early cascade, escape test) |
| `utils/` | Analysis and plotting utilities |
| `tables/` | Per-seed result tables (CSV) for the re-split experiments |
| `revision/` | **Revision-round code and data** (see `revision/README.md`) |
| `ANALYSIS_PLAN.md` | The analysis plan, pre-specified (not pre-registered) |
| `ANALYSIS_FINDINGS.md` | Verified numbers and findings |
| `LITERATURE_TABLE.md` | Survey of training budgets in the AMC literature |

## `revision/` — the revision-round experiments

The revision added a thirty-seed arm on the official stratified split
(110 000/110 000), a twenty-draw repair sweep, a from-scratch baseline, and a
weight-space comparison. These live in `revision/`, which is self-contained
(its own `src/`, which builds the **official** split). See `revision/README.md`.

## Reproducing the results

### 1. Dataset

Download RML2016.10a (from DeepSig) and place the pickle at the path the
scripts expect (`rml2016_data/RML2016.10a_dict.pkl`). The dataset is not
redistributed here.

### 2. Original re-split experiments

Run the scripts in `runners/` with the configs in `grids/`, using the `src/`
at the repository root (which builds the 176 000/44 000 re-split).

### 3. Revision-round (official split) experiments

`cd revision` and run the `rc_*.py` / `exp*.py` scripts against
`revision/src/`. The official-split per-seed metrics are read from `runs/`,
which is not shipped (size); the distilled results are in
`revision/analysis_results/`.

## Notes

- The analysis plan is **pre-specified, not pre-registered**: no independent
  registry deposit and no timestamp are claimed (see `ANALYSIS_PLAN.md`).
- RML2016.10a is publicly available from DeepSig (O'Shea et al.).
