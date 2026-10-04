# AMC Reliability Check

Code, data, and pre-registration for the paper:

> **Training budget and preprocessing protocol confound architecture
> comparisons in lightweight modulation classification: a pre-deployment
> reliability check**

## Repository structure

| Directory | Contents |
|-----------|----------|
| `src/` | Core library: data loading, models, training loop, provenance, table generation |
| `grids/` | Experiment configuration files (YAML) |
| `runners/` | Batch training scripts that invoke `src/train.py` |
| `probes/` | Mechanism probe scripts (dying ReLU, early cascade, escape test) |
| `utils/` | Analysis and plotting utilities |
| `tables/` | All result tables (10 CSV files: 5 main + 5 per-seed) |
| `REGISTRATION.md` | Pre-registered analysis plan |
| `ANALYSIS_FINDINGS.md` | Summary of key findings |
| `LITERATURE_TABLE.md` | Survey of training budgets in the AMC literature |

## Reproducing the results

### 1. Dataset

Download RML2016.10a from Kaggle or DeepSig and place the pickle file at:
