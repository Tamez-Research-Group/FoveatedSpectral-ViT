# FoveatedSpectral-ViT

Official code for the MedMNIST full-resolution ablation study — a foveated
spectral-spatial Vision Transformer (spiral token ordering + learned-DCT
frequency-domain patch embedding + Gaussian attention bias) evaluated as a
2×2×2 factorial ablation (configs C1–C8) across all 12 MedMNIST v2 datasets
at native 224px resolution.

This repo contains the training/ablation code and the evaluation notebook
generators used to produce the paper's results — no data, no trained
checkpoints, no result tables/figures. Everything here is code: run it
against your own downloaded data and trained checkpoints to reproduce the
experiment end to end.

## Model axes (`src/models/vit.py`)

- **`embed_type`**: `conv` (strided Conv2d baseline) or `learned_dct` (a
  differentiable sigmoid-gated DCT frequency mask — see
  `src/experiments/adaptive_dct/dct_variants.py`).
- **`token_order`**: `spiral` (centre-outward) or `raster` (row-major).
- **`gaussian_bias`**: soft foveal prior amplifying early-sequence
  (= central, under spiral ordering) attention logits.

| Config | token_order | embed_type   | gaussian_bias |
|--------|-------------|--------------|----------------|
| C1 (baseline)   | raster | conv        | False |
| C2              | spiral | conv        | False |
| C3              | raster | learned_dct | False |
| C4              | raster | conv        | True  |
| C5              | spiral | learned_dct | False |
| C6              | spiral | conv        | True  |
| C7              | raster | learned_dct | True  |
| C8 (full model) | spiral | learned_dct | True  |

`src/experiments/ablation_medmnist.py` trains all 8 configs for a given
dataset, selecting on validation macro-F1 (not accuracy — important for
class-imbalanced medical datasets).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires a CUDA-capable GPU for reasonable training time.

## Training

```bash
# Default: organa, organc, organs — all 8 configs each, 100 epochs
./run_ablation.sh

# Explicit subset
./run_ablation.sh --datasets organa,organc,organs --epochs 100

# Data already downloaded from a previous run — skip straight to training
./run_ablation.sh --datasets organa --skip-download
```

Every dataset key from the full MedMNIST v2 benchmark is supported
(`blood`, `breast`, `pneumonia`, `retina`, `derma`, `path`, `chest`, `oct`,
`tissue`, `organa`, `organc`, `organs`) — `chest` is the one multi-label
dataset (14 independent disease flags) and is handled automatically via
`src/utils/training_multilabel.py`.

`run_ablation.sh` downloads every requested dataset's native-224px files
first (`download_medmnist_datasets.py`), then trains. medmnist's own
download check is idempotent (skips files already cached), so re-running
after an interruption only fetches what's missing.

### Resuming after an interruption

Two layers, both automatic, no flags needed:

- **Within a config**: a per-epoch "latest" checkpoint resumes mid-training
  from the last completed epoch.
- **Across configs**: a small JSON state file per dataset
  (`checkpoints/medmnist/ablation_medmnist_state_<dataset>.json`) tracks
  which of the 8 configs are already completed and skips them on re-run.
- `run_ablation.sh` also retries a dataset's whole training call up to 10
  times on a non-zero exit code (transient CUDA/driver hiccups, etc.) —
  each retry resumes via the two mechanisms above rather than restarting
  from scratch.

### Output layout

```
data/MedMnist_fullres/       # downloaded MedMNIST npz files (gitignored)
checkpoints/medmnist/        # per-config best + latest checkpoints, state JSON (gitignored)
results/ablation_medmnist/   # ablation_results_<dataset>_<ts>.csv, loss_curves_<dataset>_<ts>.csv (gitignored)
```

Checkpoint filenames: `vit_<dataset>_best_<embed_type>_<token_order>_<gauss_tag>.pth`.

## Evaluation

Two notebook generators build the analysis notebooks programmatically
(instead of hand-maintaining `.ipynb` JSON) — run either from the repo
root once you have checkpoints in `checkpoints/medmnist/`:

```bash
# Per-dataset EDA + test-set evaluation (one notebook per dataset)
python src/evals/generate_medmnist_notebooks.py

# The consolidated paper benchmark: all 12 datasets, all 8 configs,
# bootstrap CIs, McNemar's test + Cohen's Kappa pairwise significance,
# comparison against named SOTA baselines, Grad-CAM
python src/evals/generate_paper_benchmark_notebook.py
```

Both write plain, unexecuted `.ipynb` files into `src/evals/` — open them
in Jupyter and run all cells. They cache their own intermediate results
under `results/MedMnist_fullres/benchmark_comparison/` (gitignored) so
re-running after an interruption skips already-evaluated (dataset, config)
pairs.

`src/evals/medmnist_eval_utils.py` holds the shared evaluation
infrastructure both generators depend on: checkpoint loading
(`load_spiral_vit`, `load_all_configs`), bootstrap confidence intervals
(`evaluate_model_with_ci`), and the McNemar's-test/Cohen's-Kappa pairwise
comparison protocol (`compare_models`).

**Note (Windows):** the bootstrap comparison loop in `compare_models`
calls native sklearn/numpy code (`confusion_matrix` via
`cohen_kappa_score`) thousands of times per dataset. On some Windows/CUDA
driver combinations this has been observed to trigger intermittent access
violations. If you hit crashes here, replacing `cohen_kappa_score` with a
pure-numpy equivalent and reseeding the bootstrap RNG per call (instead of
a fixed seed) resolves it.

## Runtime configuration reference

Standard hyperparameters: `dim=256`, `depth=6`, `heads=8`, `mlp_dim=512`,
dropout `0.1`, `freq_keep=16` (unused by `learned_dct`, which learns its
own frequency mask). `num_workers` defaults to 4.

See `python src/experiments/ablation_medmnist.py --help` for the full CLI
(`--dataset`, `--epochs`, `--data_root`, `--out_dir`, `--ckpt_dir`,
`--device`, `--seed`, `--num_workers`, `--native_size`).

## License

MIT — see [LICENSE](LICENSE).
