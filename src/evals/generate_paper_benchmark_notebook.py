"""
generate_paper_benchmark_notebook.py
------------------------------------
THE notebook for the paper's final benchmark comparison table. Forked from
generate_full_benchmark_notebook.py after the LearnedDCTEmbedding einsum fix
(the 2-D DCT-II computation was collapsing to an identity transform;
corrected to the standard C @ X @ C^T form -- see
src/experiments/adaptive_dct/dct_variants.py). C3/C5/C7/C8 checkpoints for
all 12 datasets were retrained under the fix (results validated dataset-by-
dataset against the pre-fix baselines: 40/48 runs improve, OCT regresses in
all 4 configs, chest shows a large F1 gain by breaking a trivial
all-negative collapse). Reads from the same CKPT_DIR as before -- the fixed
checkpoints were merged into that directory in place (old ones backed up to
checkpoints/MedMnist_fullres/ablation_prefix_dct_bug_backup/), so C1/C2/C4/C6
(untouched conv configs) and the fixed C3/C5/C7/C8 both live there together.

Generates ONE consolidated notebook covering all 12 MedMNIST datasets
(8-config ablation, native-224px, learned-DCT round) and comparing all
eight of our configs -- C1 (baseline), C2/C3/C4 (spiral/DCT/Gaussian in
isolation), C5/C6/C7 (the remaining pairwise combinations), C8 (full
model) -- against ten named, verifiable SOTA methods:

  - ResNet-18 (224), ResNet-50 (224), auto-sklearn, AutoKeras,
    Google AutoML Vision -- the official MedMNIST v2 benchmark leaderboard
    (224px-resolution rows specifically, to match our own resolution).
  - MedViT-Tiny, CoMViT -- from the CoMViT paper's Table 2 (accuracy only,
    no AUC reported for either).
  - SwinMedNet -- from the SwinMedNet paper's Tables 5/6 (their own result
    only; the anonymous bracketed citations "[6],[39],[40],[13],[41],[42],
    [33],[43]" in that same source were deliberately dropped since we
    cannot verify their evaluation methodology).
  - FPRViT -- the "Ours" row from that paper's Table II.
  - LC2R-ViT -- the "Ours" row from that paper's benchmark table (no
    Blood/Tissue reported).

IMPORTANT: these SOTA numbers were transcribed from screenshots, not the
original papers/CSVs -- double check every value in SOTA_METHODS below
against the source papers before using this in the actual manuscript.
Kept as one clearly-editable block at the top of the generated notebook
specifically so mistakes are easy to spot and fix.

Run from the repository root:
    python src/evals/generate_paper_benchmark_notebook.py
"""

import json
import os
import textwrap

OUT_DIR = os.path.dirname(__file__)
OUT_PATH = os.path.join(OUT_DIR, "paper_benchmark_comparison.ipynb")

# ── Dataset registry (single-label datasets; chest handled separately) ─────
SINGLE_LABEL_DATASETS = {
    "blood":     {"n_classes": 8,  "medmnist_cls": "BloodMNIST"},
    "breast":    {"n_classes": 2,  "medmnist_cls": "BreastMNIST"},
    "pneumonia": {"n_classes": 2,  "medmnist_cls": "PneumoniaMNIST"},
    "retina":    {"n_classes": 5,  "medmnist_cls": "RetinaMNIST"},
    "derma":     {"n_classes": 7,  "medmnist_cls": "DermaMNIST"},
    "oct":       {"n_classes": 4,  "medmnist_cls": "OCTMNIST"},
    "tissue":    {"n_classes": 8,  "medmnist_cls": "TissueMNIST"},
    "organa":    {"n_classes": 11, "medmnist_cls": "OrganAMNIST"},
    "organc":    {"n_classes": 11, "medmnist_cls": "OrganCMNIST"},
    "organs":    {"n_classes": 11, "medmnist_cls": "OrganSMNIST"},
    "path":      {"n_classes": 9,  "medmnist_cls": "PathMNIST"},
}

# Column labels used in both SOTA tables, mapped to our internal dataset keys.
TABLE_COLUMNS = [
    ("Path", "path"), ("Chest", "chest"), ("Derma", "derma"), ("OCT", "oct"),
    ("Pneumonia", "pneumonia"), ("Retina", "retina"), ("Breast", "breast"),
    ("Blood", "blood"), ("Tissue", "tissue"), ("OrganA", "organa"),
    ("OrganC", "organc"), ("OrganS", "organs"),
]

# ── Notebook cell helpers (same pattern as generate_medmnist_notebooks.py) ──

def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip().splitlines(keepends=True)}

def code(text):
    lines = textwrap.dedent(text).strip().splitlines(keepends=True)
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": lines}

def nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }


cells = []

# ── 0. Title ─────────────────────────────────────────────────────────────
cells.append(md("""
# Full 12-Dataset Benchmark Comparison — SPIRAL-ViT vs. Published SOTA

Consolidates the complete 8-config ablation (C1–C8, learned-DCT round) across
all 12 MedMNIST v2 datasets at native 224×224 resolution, and compares all
eight configs -- **C1** (baseline), **C2/C3/C4** (spiral / DCT / Gaussian
in isolation), **C5/C6/C7** (the remaining pairwise combinations), **C8**
(full model) -- against published SOTA methods, with equal billing for
every cell of the 2x2x2 factorial design. C2/C3/C4 isolate each design
axis's own contribution; C6 vs. C8 shows whether adding DCT on top of
spiral+Gaussian helps or hurts. The full 8-config breakdown is also
available per-dataset in the Section 7 appendix, alongside its own
component-contribution and interaction analysis.

**Sections**
1. Setup
2. SOTA reference tables (⚠️ transcribed from screenshots — verify against source papers)
3. Evaluate all 12 datasets (all 8 configs, test set, bootstrap CIs)
4. Chest — multi-label evaluation (separate protocol)
5. Comparison Table — Accuracy + AUC, fused (vs. all ten SOTA methods)
5.1 Ours-only — Macro-F1 + Macro-Recall (not reported by any SOTA source)
6. Paper-ready comparison figures (bar-chart grids)
6.1 Paper-ready LaTeX table (Table-II style: Methods / Para(M) / ACC+AUC per dataset)
7. Full 8-config ablation appendix — table + component-contribution & interaction figures
8. Save tables to CSV
9. Grad-CAM visualisation -- per-dataset C1 vs C6 vs C8 comparison grids
"""))

cells.append(code("%load_ext autoreload\n%autoreload 2"))

# ── 1. Setup ─────────────────────────────────────────────────────────────
cells.append(md("---\n## 1. Setup"))
cells.append(code("""
import os, sys

# Windows-specific native-crash defenses, must run before torch/numpy/
# matplotlib are imported -- both confirmed root causes (via faulthandler)
# in the General ViT evaluation notebook's own debugging:
#  1. PyTorch's bundled MKL and numpy/scipy's own OpenMP runtime ship
#     separate copies of libiomp5md.dll; loading both risks a Windows
#     access violation once both get exercised together.
#  2. faulthandler gives a real C-level stack trace on any future segfault
#     instead of a bare "Kernel died" with no diagnostic at all.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import faulthandler
faulthandler.enable()

from pathlib import Path

ROOT = Path().resolve()
while not (ROOT / "src").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
SRC = ROOT / "src"
for _p in [str(ROOT), str(SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gc
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm.auto import tqdm
from IPython.display import display, Image

# tqdm's background monitor thread has a known race with sklearn's internal
# warnings/inspect-based parameter validation under tight bootstrap loops --
# confirmed root cause of crashes in this exact evaluation pattern elsewhere
# in this project. Must be set before any tqdm instance is created.
tqdm.monitor_interval = 0


def _safe_call(fn, *args, max_retries=5, **kwargs):
    \"\"\"Retry a pure sklearn-metric call on AttributeError/KeyError -- belt
    and suspenders alongside monitor_interval=0 (see comment above).\"\"\"
    last_exc = None
    for _ in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except (AttributeError, KeyError) as exc:
            last_exc = exc
            continue
    raise last_exc

from evals.medmnist_eval_utils import (
    ABLATION_CONFIGS_LEARNED as ABLATION_CONFIGS,
    CONFIG_LABELS_LEARNED as CONFIG_LABELS,
    load_all_configs, evaluate_model_with_ci,
)
from data.medmnist_dataset import (
    MULTI_LABEL_DATASETS, MEDMNIST_INFO, GrayToGET, _MedMNISTWrapper,
)
from torch.utils.data import DataLoader
from torchvision import transforms as T

def get_test_loader_only(dataset_name, batch_size, resize_to, data_root, native_size, use_gray_to_get=False):
    # Builds ONLY the test-split loader. get_medmnist_loaders() also
    # constructs train+val in memory even when we only need test -- for
    # large datasets (tissue/chest/oct/path) that can spike RAM by tens of
    # GB before it even returns, since train is often far bigger than test
    # (e.g. path: ~90K train vs ~7K test images). We never use train/val
    # here, so skip constructing them entirely.
    MedDS, is_rgb = MEDMNIST_INFO[dataset_name]
    multi_label = dataset_name in MULTI_LABEL_DATASETS

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    pil_ops, tensor_ops = [], []
    if not is_rgb:
        if use_gray_to_get:
            tensor_ops = [GrayToGET()]
        else:
            pil_ops = [T.Grayscale(num_output_channels=3)]

    clean_tf = T.Compose([
        T.Resize((resize_to, resize_to), interpolation=T.InterpolationMode.BILINEAR),
        *pil_ops,
        T.ToTensor(),
        *tensor_ops,
        T.Normalize(mean, std),
    ])

    kw = {"root": data_root, "download": True, "size": native_size}
    raw_test = MedDS(split="test", transform=clean_tf, **kw)
    test_ds  = _MedMNISTWrapper(raw_test, multi_label=multi_label)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                       num_workers=0, pin_memory=True)

NATIVE_SIZE = 224
DATA_ROOT   = str(ROOT / "data" / "MedMnist_fullres")
CKPT_DIR    = str(ROOT / "checkpoints" / "medmnist")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SINGLE_LABEL_DATASETS = {
""" + "\n".join(
    f'    "{k}": {{"n_classes": {v["n_classes"]}}},' for k, v in SINGLE_LABEL_DATASETS.items()
) + """
}

TABLE_COLUMNS = [
""" + "\n".join(
    f'    ("{label}", "{ds}"),' for label, ds in TABLE_COLUMNS
) + """
]

print(f"Device   : {DEVICE}")
print(f"CKPT dir : {CKPT_DIR}")
print(f"Data root: {DATA_ROOT}")
"""))

# ── 2. SOTA reference tables ─────────────────────────────────────────────
cells.append(md("""---
## 2. SOTA Reference Tables

⚠️ **Transcribed from pasted screenshots — verify every value against the
original source papers before using this in the manuscript.** Kept as one
editable block below specifically so mistakes are easy to spot-check and fix.
"""))

cells.append(code('''
# All values are (AUC, ACC) tuples, both as fractions in [0, 1]. AUC is None
# where the source paper did not report it (MedViT-Tiny, CoMViT: accuracy
# only). A dataset key absent entirely means that paper did not evaluate on
# that dataset (FPRViT, LC2R-ViT: no Blood/Tissue; SwinMedNet: no Tissue).
SOTA_METHODS = {
    # Official MedMNIST v2 benchmark leaderboard, 224px-resolution rows
    # (matching our own native-224 resolution, not the 28px rows).
    "ResNet-18 (224)": {
        "Path": (0.989, 0.909), "Chest": (0.773, 0.947), "Derma": (0.920, 0.754),
        "OCT": (0.958, 0.763), "Pneumonia": (0.956, 0.864), "Retina": (0.710, 0.493),
        "Breast": (0.891, 0.833), "Blood": (0.998, 0.963), "Tissue": (0.933, 0.681),
        "OrganA": (0.998, 0.951), "OrganC": (0.994, 0.920), "OrganS": (0.974, 0.778),
    },
    "ResNet-50 (224)": {
        "Path": (0.989, 0.892), "Chest": (0.773, 0.948), "Derma": (0.912, 0.731),
        "OCT": (0.958, 0.776), "Pneumonia": (0.962, 0.884), "Retina": (0.716, 0.511),
        "Breast": (0.866, 0.842), "Blood": (0.997, 0.950), "Tissue": (0.932, 0.680),
        "OrganA": (0.998, 0.947), "OrganC": (0.993, 0.911), "OrganS": (0.975, 0.785),
    },
    "auto-sklearn": {
        "Path": (0.934, 0.716), "Chest": (0.649, 0.779), "Derma": (0.902, 0.719),
        "OCT": (0.887, 0.601), "Pneumonia": (0.942, 0.855), "Retina": (0.690, 0.515),
        "Breast": (0.836, 0.803), "Blood": (0.984, 0.878), "Tissue": (0.828, 0.532),
        "OrganA": (0.963, 0.762), "OrganC": (0.976, 0.829), "OrganS": (0.945, 0.672),
    },
    "AutoKeras": {
        "Path": (0.959, 0.834), "Chest": (0.742, 0.937), "Derma": (0.915, 0.749),
        "OCT": (0.955, 0.763), "Pneumonia": (0.947, 0.878), "Retina": (0.719, 0.503),
        "Breast": (0.871, 0.831), "Blood": (0.998, 0.961), "Tissue": (0.941, 0.703),
        "OrganA": (0.994, 0.905), "OrganC": (0.990, 0.879), "OrganS": (0.974, 0.813),
    },
    "Google AutoML Vision": {
        "Path": (0.944, 0.728), "Chest": (0.778, 0.948), "Derma": (0.914, 0.768),
        "OCT": (0.963, 0.771), "Pneumonia": (0.991, 0.946), "Retina": (0.750, 0.531),
        "Breast": (0.919, 0.861), "Blood": (0.998, 0.966), "Tissue": (0.924, 0.673),
        "OrganA": (0.990, 0.886), "OrganC": (0.988, 0.877), "OrganS": (0.964, 0.749),
    },
    # CoMViT paper's Table 2 -- accuracy only, no AUC reported for either.
    "MedViT-Tiny": {
        "Path": (None, 0.956), "Chest": (None, 0.956), "Derma": (None, 0.768),
        "OCT": (None, 0.767), "Pneumonia": (None, 0.949), "Retina": (None, 0.534),
        "Breast": (None, 0.896), "Blood": (None, 0.950), "Tissue": (None, 0.703),
        "OrganA": (None, 0.931), "OrganC": (None, 0.901), "OrganS": (None, 0.789),
    },
    "CoMViT": {
        "Path": (None, 0.9108), "Chest": (None, 0.9512), "Derma": (None, 0.770),
        "OCT": (None, 0.846), "Pneumonia": (None, 0.9214), "Retina": (None, 0.539),
        "Breast": (None, 0.8397), "Blood": (None, 0.9804), "Tissue": (None, 0.6979),
        "OrganA": (None, 0.9515), "OrganC": (None, 0.9284), "OrganS": (None, 0.8044),
    },
    # SwinMedNet paper -- only their own result kept (the anonymous bracketed
    # citations in that same source table were dropped, methodology unverified).
    "SwinMedNet": {
        "Path": (0.970, 0.847), "Chest": (0.916, 0.846), "Derma": (0.959, 0.757),
        "OCT": (0.8709, 0.7233), "Pneumonia": (0.9211, 0.8478), "Blood": (0.995, 0.927),
        "Retina": (0.834, 0.568), "Breast": (0.905, 0.808), "OrganA": (0.9934, 0.9143),
        "OrganC": (0.977, 0.824), "OrganS": (0.9303, 0.6451),
        # Tissue not reported.
    },
    # "Ours" row from the FPRViT paper's Table II (28.4M params).
    "FPRViT": {
        "Path": (0.9861, 0.8825), "Chest": (0.7360, 0.9475), "Derma": (0.9193, 0.7651),
        "OCT": (0.9549, 0.7990), "Pneumonia": (0.9634, 0.9119), "Retina": (0.7409, 0.5325),
        "Breast": (0.8782, 0.8462), "OrganA": (0.9938, 0.9085), "OrganC": (0.9921, 0.9009),
        "OrganS": (0.9861, 0.7525),
        # Blood, Tissue not reported.
    },
    # "Ours" row from the LC2R-ViT paper's benchmark table.
    "LC2R-ViT": {
        "Path": (0.994, 0.918), "Chest": (0.725, 0.948), "Derma": (0.923, 0.766),
        "OCT": (0.968, 0.813), "Pneumonia": (0.973, 0.896), "Retina": (0.753, 0.568),
        "Breast": (0.938, 0.891), "OrganA": (0.997, 0.935), "OrganC": (0.993, 0.903),
        "OrganS": (0.976, 0.785),
        # Blood, Tissue not reported.
    },
}

print("SOTA methods loaded:", len(SOTA_METHODS))
'''))

# ── 3. Evaluate all single-label datasets ────────────────────────────────
cells.append(md("""---
## 3. Evaluate All 12 Datasets (all 8 configs, test set)

Loads all 8 ablation checkpoints per dataset and runs test-set evaluation
with 1000-bootstrap 95% CIs (seed=42). This is the slow cell — a few
minutes per dataset.

**Resumable**: each config's scalar results (accuracy/F1/AUC + their 95% CIs
— the only fields Tables A/B/the appendix actually use) are cached to
`results/MedMnist_fullres/benchmark_comparison/eval_results_<dataset>.json`
immediately after that config finishes. If this cell is interrupted (kernel
restart, computer sleep, etc.), re-running it skips every already-cached
config/dataset and only computes what's still missing.
"""))

cells.append(code("""
import json as _json

EVAL_CACHE_DIR = ROOT / "results" / "MedMnist_fullres" / "benchmark_comparison"
EVAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _cache_path(dataset):
    return EVAL_CACHE_DIR / f"eval_results_{dataset}.json"

def _load_cache(dataset):
    path = _cache_path(dataset)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return _json.load(f)
    return {}

def _save_cache(dataset, cache):
    with open(_cache_path(dataset), "w", encoding="utf-8") as f:
        _json.dump(cache, f, indent=2)

def _scalarize(res):
    # Only the fields Tables A/B and the appendix actually use downstream --
    # deliberately dropping confusion_matrix/True_Class/etc, which are large
    # and not needed by anything in this notebook.
    return {k: res[k] for k in
            ("accuracy", "accuracy_ci", "recall", "recall_ci", "f1", "f1_ci", "roc_auc", "roc_auc_ci")}

print(f"Cache dir: {EVAL_CACHE_DIR}")
"""))

cells.append(code("""
all_results = {}   # dataset -> {config_id: metrics_dict}

_dataset_bar = tqdm(SINGLE_LABEL_DATASETS.items(), desc="Datasets", total=len(SINGLE_LABEL_DATASETS))

# gc.disable() for the whole loop: a manual gc.collect() at the end of each
# dataset iteration (freeing 8 checkpoints' worth of CUDA tensors + a test
# set) is itself a confirmed Windows access-violation trigger elsewhere in
# this project (cyclic GC firing mid-teardown) -- calling collect() at
# exactly that point made it worse, not better. Disable GC for the entire
# evaluation loop and collect exactly once, after every dataset is done.
gc.disable()
try:
    for ds, dcfg in _dataset_bar:
        _dataset_bar.set_postfix(current=ds)
        print(f"=== {ds} ===")

        cache = _load_cache(ds)
        missing = [c for c in ABLATION_CONFIGS if c["config_id"] not in cache]
        if not missing:
            print(f"  [CACHED] all 8 configs already evaluated -- skipping data/checkpoint load")
            all_results[ds] = cache
            continue
        if cache:
            print(f"  [RESUME] {len(cache)}/8 configs already cached: {sorted(cache.keys())}")

        test_loader = get_test_loader_only(
            dataset_name=ds, batch_size=64, resize_to=224,
            data_root=DATA_ROOT, native_size=NATIVE_SIZE, use_gray_to_get=False,
        )
        print(f"  test set size: {len(test_loader.dataset)}")
        models = load_all_configs(ds, dcfg["n_classes"], DEVICE, ckpt_dir=CKPT_DIR,
                                   ablation_configs=ABLATION_CONFIGS)
        print(f"  loaded {len(models)} / 8 checkpoints")

        for cfg in tqdm(ABLATION_CONFIGS, desc=f"  {ds} configs", leave=False):
            cid = cfg["config_id"]
            if cid in cache:
                continue
            if cid not in models:
                print(f"  [{cid}] SKIP -- checkpoint not found")
                continue
            model, _ = models[cid]
            # A fixed seed here makes every retry draw the IDENTICAL bootstrap
            # resample sequence for the same (dataset, config, weights) --
            # confirmed to reproduce a native access-violation deterministically
            # on this machine (6/6 retries died at the exact same config).
            # Seeding from wall-clock time breaks that reproducibility just
            # enough to let retries actually make progress; bootstrap CI
            # width/center is insensitive to the exact seed at n=1000.
            _seed = int(time.time() * 1000) % (2**31)
            res = evaluate_model_with_ci(model, test_loader, DEVICE, n_bootstraps=1000,
                                          random_seed=_seed, show_progress=True)
            cache[cid] = _scalarize(res)
            _save_cache(ds, cache)   # persist immediately -- a crash loses at most this one config
            print(f"  [{cid}] Acc={res['accuracy']:.4f}  F1={res['f1']:.4f}  AUC={res['roc_auc']:.4f}")
            del res

        all_results[ds] = cache

        # Drop this dataset's test-set tensors and 8 loaded models before
        # moving to the next dataset -- no gc.collect() here (see above);
        # empty_cache() alone is safe since it doesn't walk the object graph.
        del test_loader, models
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
finally:
    gc.enable()
    gc.collect()

print("\\nDone -- evaluated", len(all_results), "datasets.")
"""))

# ── 4. Chest multi-label evaluation ──────────────────────────────────────
cells.append(md("""---
## 4. Chest — Multi-Label Evaluation

ChestMNIST's 14 independent disease flags don't fit the single-label
argmax protocol above. Follows the official MedMNIST multi-label
protocol: per-label ROC-AUC and per-label accuracy (threshold 0.5), each
averaged across the 14 labels.

Same resumable caching as Section 3 — cached to
`eval_results_chest.json`, one config at a time.
"""))

cells.append(code("""
def evaluate_chest_multilabel(model, test_loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="  chest batches", leave=False):
            imgs = imgs.to(device)
            probs = torch.sigmoid(model(imgs)).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    y_score = np.concatenate(all_probs, axis=0)
    y_true  = np.concatenate(all_labels, axis=0)
    y_pred  = (y_score > 0.5).astype(int)

    n_labels = y_true.shape[1]
    aucs, accs = [], []
    for i in range(n_labels):
        if len(np.unique(y_true[:, i])) > 1:
            aucs.append(float(roc_auc_score(y_true[:, i], y_score[:, i])))
        accs.append(float(accuracy_score(y_true[:, i], y_pred[:, i])))

    return {"roc_auc": float(np.mean(aucs)), "accuracy": float(np.mean(accs)),
            "per_label_auc": aucs, "per_label_acc": accs}


chest_cache = _load_cache("chest")
chest_missing = [c for c in ABLATION_CONFIGS if c["config_id"] not in chest_cache]

if not chest_missing:
    print("[CACHED] all 8 chest configs already evaluated -- skipping data/checkpoint load")
    chest_results = chest_cache
else:
    if chest_cache:
        print(f"[RESUME] {len(chest_cache)}/8 chest configs already cached: {sorted(chest_cache.keys())}")

    # gc.disable() around this block -- see the main evaluation loop above
    # for why (manual gc.collect() right after freeing 8 checkpoints' worth
    # of CUDA tensors is a confirmed Windows access-violation trigger here).
    gc.disable()
    try:
        chest_test_loader = get_test_loader_only(
            dataset_name="chest", batch_size=64, resize_to=224,
            data_root=DATA_ROOT, native_size=NATIVE_SIZE, use_gray_to_get=False,
        )
        chest_models = load_all_configs("chest", 14, DEVICE, ckpt_dir=CKPT_DIR,
                                         ablation_configs=ABLATION_CONFIGS)

        for cfg in tqdm(ABLATION_CONFIGS, desc="chest configs"):
            cid = cfg["config_id"]
            if cid in chest_cache:
                continue
            if cid not in chest_models:
                print(f"  [{cid}] SKIP -- checkpoint not found")
                continue
            model, _ = chest_models[cid]
            res = evaluate_chest_multilabel(model, chest_test_loader, DEVICE)
            chest_cache[cid] = res   # already JSON-safe -- floats and lists of floats only
            _save_cache("chest", chest_cache)
            print(f"  [{cid}] Acc={res['accuracy']:.4f}  AUC={res['roc_auc']:.4f}")
            del res

        chest_results = chest_cache
        del chest_test_loader, chest_models
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    finally:
        gc.enable()
        gc.collect()

all_results["chest"] = chest_results
"""))

# ── 5. Comparison Table — Accuracy + AUC (fused) ─────────────────────────
cells.append(md("""---
## 5. Comparison Table — Accuracy + AUC (fused)

Six of our configs against all ten SOTA methods, one **ACC | AUC** column
pair per dataset. This replaces what used to be two separate tables (an
accuracy-only ranking and a combined ACC+AUC table) — AUC and ACC together
already carry everything the accuracy-only table would have shown, so
there's no reason to show accuracy twice. Best per column in **bold**,
second-best *italic*. Missing values (not reported by that source) are left
blank.

- **C1** baseline (raster+conv, no Gaussian)
- **C2** spiral only — isolates the spiral-ordering contribution
- **C3** DCT only — isolates the learned-DCT contribution
- **C4** Gaussian only — isolates the Gaussian-bias contribution
- **C5** spiral+DCT, no Gaussian
- **C6** spiral+Gaussian, no DCT — the two foveal-prior components together
- **C7** DCT+Gaussian, no spiral
- **C8** full model — all three combined

All eight cells of the 2x2x2 factorial design get equal billing here; the
same eight are also broken down per-dataset with component-contribution and
interaction analysis in the Section 7 appendix.

**95% CI**: our eight rows show `value [CI-low, CI-high]` for both AUC and
ACC (1000-bootstrap percentile CI from Section 3); SOTA cells and Chest
(no bootstrap in the multi-label protocol) show a plain value.
"""))

cells.append(code('''
OUR_CONFIGS = {
    "C1": "SPIRAL-ViT C1 (Baseline)",
    "C2": "SPIRAL-ViT C2 (Spiral only)",
    "C3": "SPIRAL-ViT C3 (DCT only)",
    "C4": "SPIRAL-ViT C4 (Gaussian only)",
    "C5": "SPIRAL-ViT C5 (Spiral+DCT, no Gaussian)",
    "C6": "SPIRAL-ViT C6 (Spiral+Gaussian, no DCT)",
    "C7": "SPIRAL-ViT C7 (DCT+Gaussian, no Spiral)",
    "C8": "SPIRAL-ViT C8 (Full Model)",
}

table_b_data = dict(SOTA_METHODS)
for cid, method_name in OUR_CONFIGS.items():
    row = {}
    for label, ds in TABLE_COLUMNS:
        r = all_results[ds][cid]
        row[label] = (r["roc_auc"], r["accuracy"])
    table_b_data[method_name] = row

table_b_rows = []
methods_b = list(SOTA_METHODS.keys()) + list(OUR_CONFIGS.values())

for method in methods_b:
    row = {"Method": method}
    for label, _ in TABLE_COLUMNS:
        val = table_b_data[method].get(label)
        if val is None:
            row[f"{label} AUC"] = np.nan
            row[f"{label} ACC"] = np.nan
        else:
            row[f"{label} AUC"] = val[0] if val[0] is not None else np.nan
            row[f"{label} ACC"] = val[1]
    table_b_rows.append(row)

df_b = pd.DataFrame(table_b_rows).set_index("Method")

# CI lookup (fraction scale, matches df_b directly) -- only our eight configs.
OUR_CI_B = {}
for cid, method_name in OUR_CONFIGS.items():
    OUR_CI_B[method_name] = {}
    for label, ds in TABLE_COLUMNS:
        r = all_results[ds][cid]
        OUR_CI_B[method_name][label] = {
            "auc_ci": r.get("roc_auc_ci"),
            "acc_ci": r.get("accuracy_ci"),
            # Chest reports per-label AUC/Acc under its multi-label protocol,
            # not a single macro-F1 -- .get() leaves f1/f1_ci unset there.
            "f1_ci": r.get("f1_ci"),
        }

# F1 point estimates for the paper table -- SOTA methods report none of the
# ten sources include F1/Recall (see Section 5.1), and Chest has no
# macro-F1 in this protocol, so both stay unset (renders as "---").
table_f1_data = {method: {} for method in SOTA_METHODS}
for cid, method_name in OUR_CONFIGS.items():
    table_f1_data[method_name] = {
        label: all_results[ds][cid]["f1"]
        for label, ds in TABLE_COLUMNS if ds != "chest"
    }

def _fmt_frac_ci(val, ci):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    if not ci or ci[0] is None or np.isnan(ci[0]) or np.isnan(ci[1]):
        return f"{val:.4f}"
    return f"{val:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"

df_b_display = df_b.copy().astype(object)
for method in methods_b:
    ci_row = OUR_CI_B.get(method, {})
    for label, _ in TABLE_COLUMNS:
        auc_col, acc_col = f"{label} AUC", f"{label} ACC"
        df_b_display.loc[method, auc_col] = _fmt_frac_ci(df_b.loc[method, auc_col], ci_row.get(label, {}).get("auc_ci"))
        df_b_display.loc[method, acc_col] = _fmt_frac_ci(df_b.loc[method, acc_col], ci_row.get(label, {}).get("acc_ci"))

def _best_second(vals):
    """Index of best and second-best among non-NaN values (NaN-safe)."""
    order = [i for i in np.argsort(vals)[::-1] if not np.isnan(vals[i])]
    best   = order[0] if len(order) > 0 else None
    second = order[1] if len(order) > 1 else None
    return best, second

def highlight_bold_italic(col, numeric_df):
    # Ranking always uses the numeric df, not the CI-formatted text in col.
    vals = numeric_df[col.name].to_numpy(dtype=float)
    best, second = _best_second(vals)
    styles = [""] * len(col)
    if best is not None:
        styles[best] = "font-weight: bold"
    if second is not None:
        styles[second] = "font-style: italic"
    return styles

display(df_b_display.style.apply(lambda col: highlight_bold_italic(col, df_b), axis=0))
'''))

# ── 5.1 Ours-only — Macro-F1 + Macro-Recall ──────────────────────────────
cells.append(md("""---
## 5.1 Ours-Only — Macro-F1 + Macro-Recall

**None of the ten SOTA sources report F1 or Recall** — there is nothing to
compare these against, so rather than padding the fused table above with
empty columns for every SOTA row, they get their own compact table here,
covering just our eight configs. Chest is excluded (its multi-label protocol
reports per-label AUC/accuracy, not a single macro-F1/recall).

Two summary columns — mean across all 11 datasets, and its delta against
the **C1** baseline — answer directly "how much did isolating this axis
change F1/Recall on average," which is the whole point of including
C2/C3/C4 here.

**95% CI**: `value [CI-low, CI-high]`, same 1000-bootstrap percentile CI as
the fused table above.
"""))

cells.append(code('''
F1_TABLE_COLUMNS = [(label, ds) for label, ds in TABLE_COLUMNS if ds != "chest"]

f1r_rows = []
for cid, method_name in OUR_CONFIGS.items():
    row = {"Method": method_name}
    for label, ds in F1_TABLE_COLUMNS:
        r = all_results[ds][cid]
        row[f"{label} F1"] = r["f1"]
        row[f"{label} Recall"] = r["recall"]
    f1r_rows.append(row)

df_f1r = pd.DataFrame(f1r_rows).set_index("Method")

OUR_CI_F1R = {}
for cid, method_name in OUR_CONFIGS.items():
    OUR_CI_F1R[method_name] = {}
    for label, ds in F1_TABLE_COLUMNS:
        r = all_results[ds][cid]
        OUR_CI_F1R[method_name][label] = {"f1_ci": r.get("f1_ci"), "recall_ci": r.get("recall_ci")}

df_f1r_display = df_f1r.copy().astype(object)
for method in df_f1r.index:
    ci_row = OUR_CI_F1R.get(method, {})
    for label, _ in F1_TABLE_COLUMNS:
        f1_col, rec_col = f"{label} F1", f"{label} Recall"
        df_f1r_display.loc[method, f1_col]  = _fmt_frac_ci(df_f1r.loc[method, f1_col],  ci_row.get(label, {}).get("f1_ci"))
        df_f1r_display.loc[method, rec_col] = _fmt_frac_ci(df_f1r.loc[method, rec_col], ci_row.get(label, {}).get("recall_ci"))

# Summary columns: mean across all 11 datasets + delta vs the C1 baseline.
baseline_name = OUR_CONFIGS["C1"]
f1_cols  = [f"{l} F1"     for l, _ in F1_TABLE_COLUMNS]
rec_cols = [f"{l} Recall" for l, _ in F1_TABLE_COLUMNS]
mean_f1  = df_f1r[f1_cols].mean(axis=1)
mean_rec = df_f1r[rec_cols].mean(axis=1)

df_f1r_display["Mean F1 (Δ vs C1)"] = [
    f"{mean_f1[m]:.4f} ({mean_f1[m] - mean_f1[baseline_name]:+.4f})" for m in df_f1r_display.index
]
df_f1r_display["Mean Recall (Δ vs C1)"] = [
    f"{mean_rec[m]:.4f} ({mean_rec[m] - mean_rec[baseline_name]:+.4f})" for m in df_f1r_display.index
]

display(df_f1r_display.style.apply(
    lambda col: highlight_bold_italic(col, df_f1r), axis=0,
    subset=[c for c in df_f1r_display.columns if c in df_f1r.columns],
))

print("\\nMean F1 by config:", mean_f1.round(4).to_dict())
print("Mean Recall by config:", mean_rec.round(4).to_dict())
'''))

# ── 6. Paper-ready comparison figures ────────────────────────────────────
cells.append(md("""---
## 6. Paper-Ready Comparison Figures

Small-multiples bar charts, one per dataset, all ten SOTA methods plus our
eight configs. Our bars are highlighted (a distinct color family from the
SOTA methods) so the comparison is immediately legible. Saved as
high-resolution PNGs suitable for direct inclusion in the manuscript.
"""))

cells.append(code('''
FIG_DIR = ROOT / "results" / "MedMnist_fullres" / "benchmark_comparison" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def plot_metric_grid(metric_index, metric_name, filename):
    """metric_index: 0 for AUC, 1 for ACC, within the (AUC, ACC) tuples."""
    all_methods = list(SOTA_METHODS.keys()) + list(OUR_CONFIGS.values())
    sota_color = "#7f8c8d"
    our_colors = {"SPIRAL-ViT C1 (Baseline)": "#f4a261",
                  "SPIRAL-ViT C2 (Spiral only)": "#e9c46a",
                  "SPIRAL-ViT C3 (DCT only)": "#2a9d8f",
                  "SPIRAL-ViT C4 (Gaussian only)": "#457b9d",
                  "SPIRAL-ViT C5 (Spiral+DCT, no Gaussian)": "#8ab17d",
                  "SPIRAL-ViT C6 (Spiral+Gaussian, no DCT)": "#e76f51",
                  "SPIRAL-ViT C7 (DCT+Gaussian, no Spiral)": "#6d597a",
                  "SPIRAL-ViT C8 (Full Model)": "#c1121f"}

    # gc.disable(): CPython's cyclic GC firing mid-render inside matplotlib's
    # tick/title-position machinery is a confirmed Windows access-violation
    # source (via faulthandler) in the General ViT evaluation notebook's own
    # debugging of an equivalent per-axes-titled grid. Collect once, manually,
    # after the figure is fully saved instead.
    gc.disable()
    try:
        fig, axes = plt.subplots(3, 4, figsize=(20, 13))
        axes = axes.flatten()

        for i, (label, ds) in enumerate(TABLE_COLUMNS):
            ax = axes[i]
            names, values, colors = [], [], []
            for m in all_methods:
                val = table_b_data[m].get(label)
                if val is None or val[metric_index] is None:
                    continue
                names.append(m)
                values.append(val[metric_index])
                colors.append(our_colors.get(m, sota_color))

            order = np.argsort(values)[::-1]
            names  = [names[j] for j in order]
            values = [values[j] for j in order]
            colors = [colors[j] for j in order]

            y = np.arange(len(names))
            ax.barh(y, values, color=colors, edgecolor="black", linewidth=0.4)
            ax.set_yticks(y)
            ax.set_yticklabels(names, fontsize=7)
            ax.invert_yaxis()
            ax.set_xlim(0, 1.0 if metric_index == 0 else max(values) * 1.15)
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.grid(axis="x", alpha=0.3)

        for j in range(len(TABLE_COLUMNS), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"{metric_name} Comparison — SPIRAL-ViT vs. Published SOTA",
                     fontsize=15, fontweight="bold", y=1.0)
        # Fixed margins instead of tight_layout()/bbox_inches="tight" -- see
        # the 9.1 Grad-CAM cells below for why (confirmed Windows access
        # violation in get_tightbbox() with this axes-grid + per-axes-title
        # pattern).
        fig.subplots_adjust(left=0.16, right=0.98, top=0.92, bottom=0.03, wspace=0.7, hspace=0.35)
        fig.savefig(FIG_DIR / filename, dpi=200)
        plt.close(fig)
        # plt.show() also crashes here (Windows access violation computing
        # tick-label bboxes at draw time) -- display the already-saved PNG
        # bytes instead, which sidesteps matplotlib's render pipeline.
        display(Image(filename=str(FIG_DIR / filename)))
        print(f"Saved: {FIG_DIR / filename}")
    finally:
        gc.enable()
        gc.collect()

plot_metric_grid(1, "Accuracy", "benchmark_accuracy_grid.png")
'''))

cells.append(code('''
plot_metric_grid(0, "ROC-AUC", "benchmark_auc_grid.png")
'''))

# ── 6.1 Paper-ready LaTeX table (Table-II style) ──────────────────────────
cells.append(md("""---
## 6.1 Paper-Ready LaTeX Table (Table-II style)

Same grid layout as the "Local ViT" paper's Table II, extended with a third
metric: **Methods** / **Para(M)** columns, then one **ACC | AUC | F1**
column-triple per dataset, best per column in **bold**, second-best in
*italic*. Split into two blocks of 6 datasets each (18 columns of numbers is
too wide for one block) — mirrors the source table's own two-row-block
layout.

**F1 is populated for our eight configs only** — none of the ten SOTA sources
report F1 (see Section 5.1), and Chest has no macro-F1 in this protocol
(its multi-label evaluation reports per-label AUC/Accuracy instead), so both
render as "---".

**Para(M) is only filled in where we have a verified value** — our own eight
configs (computed exactly from the instantiated architecture — spiral
ordering and the Gaussian bias are both zero-cost, so Para(M) only depends
on `embed_type`, giving two distinct values across all eight), plus
ResNet-18/ResNet-50 (11.7M / 25.6M, standard well-known values for these
architectures) and FPRViT (28.4M, stated in its own Table II). All other
SOTA methods leave Para(M) blank rather than guess.

**95% CI**: our eight rows report `value [CI-low, CI-high]` (1000-bootstrap
percentile CI from Section 3, explicit bounds rather than a symmetric
$\\pm$ margin — a percentile bootstrap CI isn't generally symmetric around
the point estimate) for ACC, AUC, **and now F1**, in both the LaTeX output
and the preview table below; SOTA cells and Chest show a plain value (no
bootstrap available). Bold/italic ranking is always based on the point
estimate.
"""))

cells.append(code('''
from models.vit import ViT as _ViTForParamCount

def _count_params(embed_type):
    m = _ViTForParamCount(
        image_size=224, patch_size=14, num_classes=2,
        dim=256, depth=6, heads=8, mlp_dim=512, channels=3,
        dropout=0.0, emb_dropout=0.0,
        embed_type=embed_type, freq_keep=16,
        token_order="spiral", gaussian_bias=True,
    )
    return sum(p.numel() for p in m.parameters()) / 1e6

_OUR_EMBED = {"C1": "conv", "C2": "conv", "C3": "learned_dct",
              "C4": "conv", "C5": "learned_dct", "C6": "conv",
              "C7": "learned_dct", "C8": "learned_dct"}
PARA_M = {method_name: round(_count_params(_OUR_EMBED[cid]), 2)
          for cid, method_name in OUR_CONFIGS.items()}
PARA_M["ResNet-18 (224)"] = 11.7
PARA_M["ResNet-50 (224)"] = 25.6
PARA_M["FPRViT"] = 28.4
# All other SOTA methods: no verified Para(M) -- left unset (renders as "---").

print("Para(M):")
for k, v in PARA_M.items():
    print(f"  {k:35s} {v}")
'''))

cells.append(code('''
def _fmt_val_latex(v, is_best, is_second, ci=None):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "---"
    s = f"{v:.4f}"
    if ci and ci[0] is not None and not np.isnan(ci[0]) and not np.isnan(ci[1]):
        # Explicit [lo, hi] rather than value$\\pm$half-width: a percentile
        # bootstrap CI is not generally symmetric around the point estimate
        # (especially for metrics bounded near 0/1), so collapsing it to a
        # single half-width silently discards that asymmetry. Matches the
        # convention already used everywhere else in this project (CSVs,
        # _fmt_frac_ci in medmnist_eval_utils.py) -- the LaTeX table was the
        # one place still using the symmetric shorthand.
        s = rf"{s} [{ci[0]:.4f}, {ci[1]:.4f}]"
    if is_best:
        return rf"\\textbf{{{s}}}"
    if is_second:
        return rf"\\textit{{{s}}}"
    return s

def generate_paper_table_latex(methods_order, column_block, caption):
    n = len(column_block)
    col_spec = "|c|c|" + "ccc|" * n
    lines = [
        r"\\begin{table}[htbp]", r"\\centering",
        rf"\\caption{{{caption}}}",
        r"\\resizebox{\\textwidth}{!}{%",
        rf"\\begin{{tabular}}{{{col_spec}}}",
        r"\\hline",
    ]
    row1 = r"\\multirow{2}{*}{\\textbf{Methods}} & \\multirow{2}{*}{\\textbf{Para(M)}}"
    for label, _ in column_block:
        row1 += rf" & \\multicolumn{{3}}{{c|}}{{\\textbf{{{label}MNIST}}}}"
    lines.append(row1 + r" \\\\")
    row2 = " & "
    for _ in column_block:
        row2 += r" & \\textit{ACC} & \\textit{AUC} & \\textit{F1}"
    lines.append(row2 + r" \\\\")
    lines.append(r"\\hline")

    acc_vals = {label: [] for label, _ in column_block}
    auc_vals = {label: [] for label, _ in column_block}
    f1_vals = {label: [] for label, _ in column_block}
    for m in methods_order:
        for label, _ in column_block:
            val = table_b_data[m].get(label)
            auc_v, acc_v = val if val is not None else (None, None)
            acc_vals[label].append(acc_v if acc_v is not None else np.nan)
            auc_vals[label].append(auc_v if auc_v is not None else np.nan)
            f1_v = table_f1_data.get(m, {}).get(label)
            f1_vals[label].append(f1_v if f1_v is not None else np.nan)

    best2_acc = {label: _best_second(np.array(v)) for label, v in acc_vals.items()}
    best2_auc = {label: _best_second(np.array(v)) for label, v in auc_vals.items()}
    best2_f1 = {label: _best_second(np.array(v)) for label, v in f1_vals.items()}

    for mi, m in enumerate(methods_order):
        para = PARA_M.get(m)
        para_s = f"{para:.2f}" if para is not None else "---"
        row = f"{m} & {para_s}"
        for label, _ in column_block:
            ba, sa = best2_acc[label]
            bu, su = best2_auc[label]
            bf, sf = best2_f1[label]
            acc_ci = OUR_CI_B.get(m, {}).get(label, {}).get("acc_ci")
            auc_ci = OUR_CI_B.get(m, {}).get(label, {}).get("auc_ci")
            f1_ci = OUR_CI_B.get(m, {}).get(label, {}).get("f1_ci")
            row += " & " + _fmt_val_latex(acc_vals[label][mi], mi == ba, mi == sa, acc_ci)
            row += " & " + _fmt_val_latex(auc_vals[label][mi], mi == bu, mi == su, auc_ci)
            row += " & " + _fmt_val_latex(f1_vals[label][mi], mi == bf, mi == sf, f1_ci)
        lines.append(row + r" \\\\")
        lines.append(r"\\hline")

    lines += [r"\\end{tabular}}", r"\\end{table}"]
    return "\\n".join(lines)

methods_paper_order = list(SOTA_METHODS.keys()) + list(OUR_CONFIGS.values())
block1 = TABLE_COLUMNS[:6]
block2 = TABLE_COLUMNS[6:]

latex_table1 = generate_paper_table_latex(
    methods_paper_order, block1,
    "Quantitative evaluation on MedMNIST datasets (1/2). For each dataset, "
    "the best result is highlighted in \\\\textbf{bold}, the second-best in \\\\textit{italic}.",
)
latex_table2 = generate_paper_table_latex(
    methods_paper_order, block2,
    "Quantitative evaluation on MedMNIST datasets (2/2, continued).",
)

paper_table_tex = latex_table1 + "\\n\\n" + latex_table2
print(paper_table_tex[:1500])
'''))

cells.append(code('''
def build_preview_df(column_block):
    cols = pd.MultiIndex.from_tuples(
        [(label, metric) for label, _ in column_block for metric in ("ACC", "AUC", "F1")]
    )
    rows = {}
    for m in methods_paper_order:
        row = []
        for label, _ in column_block:
            val = table_b_data[m].get(label)
            auc_v, acc_v = val if val is not None else (None, None)
            f1_v = table_f1_data.get(m, {}).get(label)
            row.append(acc_v if acc_v is not None else np.nan)
            row.append(auc_v if auc_v is not None else np.nan)
            row.append(f1_v if f1_v is not None else np.nan)
        rows[m] = row
    df = pd.DataFrame.from_dict(rows, orient="index", columns=cols)
    df.insert(0, "Para(M)", [PARA_M.get(m, np.nan) for m in df.index])
    return df

df_paper1 = build_preview_df(block1)
df_paper2 = build_preview_df(block2)

def build_preview_df_display(column_block, numeric_df):
    # CI-formatted text twin of numeric_df -- same shape/index/columns, so
    # position-aligned styling computed from numeric_df still applies.
    display_df = numeric_df.copy().astype(object)
    for m in numeric_df.index:
        ci_row = OUR_CI_B.get(m, {})
        for label, _ in column_block:
            acc_ci = ci_row.get(label, {}).get("acc_ci")
            auc_ci = ci_row.get(label, {}).get("auc_ci")
            f1_ci = ci_row.get(label, {}).get("f1_ci")
            display_df.loc[m, (label, "ACC")] = _fmt_frac_ci(numeric_df.loc[m, (label, "ACC")], acc_ci)
            display_df.loc[m, (label, "AUC")] = _fmt_frac_ci(numeric_df.loc[m, (label, "AUC")], auc_ci)
            display_df.loc[m, (label, "F1")] = _fmt_frac_ci(numeric_df.loc[m, (label, "F1")], f1_ci)
        # pandas silently pads a plain string column into ("Para(M)", "")
        # to match the surrounding MultiIndex depth when it was inserted --
        # indexing back with the bare string does a partial match on level 0
        # and returns a length-1 Series instead of a scalar, which then
        # blows up on np.isnan(). Index with the exact tuple pandas created.
        para = numeric_df.loc[m, ("Para(M)", "")]
        display_df.loc[m, ("Para(M)", "")] = f"{para:.2f}" if not np.isnan(para) else "---"
    return display_df

def highlight_bold_italic_ci(col, numeric_df):
    # Ranking always uses numeric_df, not the CI-formatted text in col.
    vals = numeric_df[col.name].to_numpy(dtype=float)
    best, second = _best_second(vals)
    styles = [""] * len(col)
    if best is not None:
        styles[best] = "font-weight: bold"
    if second is not None:
        styles[second] = "font-style: italic"
    return styles

df_paper1_display = build_preview_df_display(block1, df_paper1)
df_paper2_display = build_preview_df_display(block2, df_paper2)

display(df_paper1_display.style.apply(lambda col: highlight_bold_italic_ci(col, df_paper1), axis=0,
        subset=[c for c in df_paper1.columns if c != ("Para(M)", "")]))
display(df_paper2_display.style.apply(lambda col: highlight_bold_italic_ci(col, df_paper2), axis=0,
        subset=[c for c in df_paper2.columns if c != ("Para(M)", "")]))
'''))

cells.append(code('''
tex_out_path = EVAL_CACHE_DIR / "paper_table_medmnist.tex"
with open(tex_out_path, "w", encoding="utf-8") as f:
    f.write(paper_table_tex)
print(f"Saved: {tex_out_path}")
'''))

# ── 7. Full ablation appendix table ──────────────────────────────────────
cells.append(md("""---
## 7. Full 8-Config Ablation Table (Appendix)

Complete Accuracy / Macro-F1 / ROC-AUC for all 8 configs across all 12
datasets — the underlying ablation data behind the headline C8 numbers above.

**95% CI**: `*-CI Low` / `*-CI High` columns give the 1000-bootstrap
percentile CI for each metric (Section 3). Chest has no CI columns populated
(`NaN`) — its multi-label protocol evaluates once per config, no bootstrap.
"""))

cells.append(code('''
def _ci_lo_hi(ci):
    if not ci or ci[0] is None:
        return np.nan, np.nan
    return ci[0], ci[1]

appendix_rows = []
for ds in list(SINGLE_LABEL_DATASETS.keys()) + ["chest"]:
    for cfg in ABLATION_CONFIGS:
        cid = cfg["config_id"]
        if ds == "chest":
            if cid not in all_results["chest"]:
                continue
            r = all_results["chest"][cid]
            appendix_rows.append({
                "Dataset": ds, "Config": cid,
                "Embed": cfg["embed_type"], "Order": cfg["token_order"], "Gaussian": cfg["gaussian_bias"],
                "Accuracy": r["accuracy"], "Macro-F1": np.nan, "ROC-AUC": r["roc_auc"],
                "Accuracy-CI Low": np.nan, "Accuracy-CI High": np.nan,
                "Macro-F1-CI Low": np.nan, "Macro-F1-CI High": np.nan,
                "ROC-AUC-CI Low": np.nan, "ROC-AUC-CI High": np.nan,
            })
        else:
            if cid not in all_results[ds]:
                continue
            r = all_results[ds][cid]
            acc_lo, acc_hi = _ci_lo_hi(r.get("accuracy_ci"))
            f1_lo, f1_hi   = _ci_lo_hi(r.get("f1_ci"))
            auc_lo, auc_hi = _ci_lo_hi(r.get("roc_auc_ci"))
            appendix_rows.append({
                "Dataset": ds, "Config": cid,
                "Embed": cfg["embed_type"], "Order": cfg["token_order"], "Gaussian": cfg["gaussian_bias"],
                "Accuracy": r["accuracy"], "Macro-F1": r["f1"], "ROC-AUC": r["roc_auc"],
                "Accuracy-CI Low": acc_lo, "Accuracy-CI High": acc_hi,
                "Macro-F1-CI Low": f1_lo, "Macro-F1-CI High": f1_hi,
                "ROC-AUC-CI Low": auc_lo, "ROC-AUC-CI High": auc_hi,
            })

df_appendix = pd.DataFrame(appendix_rows)
display(df_appendix)
'''))

# ── 7.1 Full 8-config comparison figure ──────────────────────────────────
cells.append(md("### 7.1 Full 8-Config Comparison — All Datasets"))
cells.append(code('''
cfg_order      = [c["config_id"] for c in ABLATION_CONFIGS]
cfg_palette    = dict(zip(cfg_order, sns.color_palette("tab10", 8)))
datasets_order = df_appendix["Dataset"].unique().tolist()

gc.disable()
try:
    fig, ax = plt.subplots(figsize=(18, 6))
    x = np.arange(len(datasets_order))
    width = 0.1
    for i, cid in enumerate(cfg_order):
        vals = []
        for ds in datasets_order:
            row = df_appendix[(df_appendix["Dataset"] == ds) & (df_appendix["Config"] == cid)]
            vals.append(row["Accuracy"].values[0] if len(row) else np.nan)
        ax.bar(x + (i - 3.5) * width, vals, width, label=cid,
               color=cfg_palette[cid], edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets_order, rotation=30, ha="right")
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Full 8-Config Ablation -- Accuracy across all 12 Datasets", fontsize=13, fontweight="bold")
    ax.legend(ncol=8, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(axis="y", alpha=0.3)
    # Fixed margins instead of tight_layout()/bbox_inches="tight" -- see the
    # 9.1 Grad-CAM cells below for why (confirmed Windows access violation).
    fig.subplots_adjust(left=0.05, right=0.99, top=0.90, bottom=0.32)
    fig.savefig(FIG_DIR / "ablation_full_8config_grid.png", dpi=200)
    plt.close(fig)
    display(Image(filename=str(FIG_DIR / "ablation_full_8config_grid.png")))
    print("Saved:", FIG_DIR / "ablation_full_8config_grid.png")
finally:
    gc.enable()
    gc.collect()
'''))

# ── 7.2 Component contribution (per dataset + aggregate) ────────────────
cells.append(md("""### 7.2 Component Contribution — How Much Does Each Axis Help?

Marginal effect of each design axis, isolated by averaging Accuracy across
all configs where that axis is "on" minus all configs where it is "off"
(same method as the PCA ablation notebooks' component-contribution
analysis, generalised to Accuracy so Chest -- which has no Macro-F1 --
is included). Positive = that axis helps, on average, for that dataset.
"""))
cells.append(code('''
CONTRIB_AXES = [
    ("Spiral",   "Order",    "spiral",      "raster"),
    ("DCT",      "Embed",    "learned_dct", "conv"),
    ("Gaussian", "Gaussian", True,          False),
]

def axis_delta(sub, col, on_val, off_val, metric="Accuracy"):
    on  = sub[sub[col] == on_val][metric]
    off = sub[sub[col] == off_val][metric]
    if len(on) == 0 or len(off) == 0:
        return np.nan
    return float(on.mean() - off.mean())

contrib_rows = []
for ds in datasets_order:
    sub = df_appendix[df_appendix["Dataset"] == ds]
    row = {"Dataset": ds}
    for name, col, on_val, off_val in CONTRIB_AXES:
        row[name] = axis_delta(sub, col, on_val, off_val)
    contrib_rows.append(row)

df_contrib = pd.DataFrame(contrib_rows).set_index("Dataset")
display(df_contrib.round(4))

axis_colors = {"Spiral": "#3498db", "DCT": "#9b59b6", "Gaussian": "#e67e22"}

gc.disable()
try:
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()
    for i, ds in enumerate(df_contrib.index):
        ax = axes[i]
        vals = df_contrib.loc[ds]
        colors = [axis_colors[name] for name in vals.index]
        bars = ax.bar(vals.index, vals.values, color=colors, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(ds, fontsize=11, fontweight="bold")
        ax.set_ylabel("$\\\\Delta$Accuracy", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals.values):
            if np.isnan(v):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, v + (0.002 if v >= 0 else -0.006),
                    f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    for j in range(len(df_contrib.index), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Marginal Contribution of Each Design Axis, per Dataset ($\\\\Delta$Accuracy = on - off)",
                 fontsize=14, fontweight="bold", y=1.01)
    # Fixed margins instead of tight_layout()/bbox_inches="tight" -- see the
    # 9.1 Grad-CAM cells below for why (confirmed Windows access violation).
    fig.subplots_adjust(left=0.05, right=0.99, top=0.90, bottom=0.05, wspace=0.35, hspace=0.4)
    fig.savefig(FIG_DIR / "ablation_component_contribution_per_dataset.png", dpi=200)
    plt.close(fig)
    display(Image(filename=str(FIG_DIR / "ablation_component_contribution_per_dataset.png")))
finally:
    gc.enable()
    gc.collect()
'''))

cells.append(md("### 7.3 Aggregate Contribution — Overall Headline Numbers"))
cells.append(code('''
agg_mean = df_contrib.mean()
agg_std  = df_contrib.std()

gc.disable()
try:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [axis_colors[name] for name in agg_mean.index]
    bars = ax.bar(agg_mean.index, agg_mean.values, yerr=agg_std.values, capsize=6,
                  color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Mean $\\\\Delta$Accuracy across all 12 datasets", fontsize=11)
    ax.set_title("Overall Marginal Contribution of Each Design Axis", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, agg_mean.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + (0.002 if v >= 0 else -0.008),
                f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=11, fontweight="bold")
    # Fixed margins instead of tight_layout()/bbox_inches="tight" -- see the
    # 9.1 Grad-CAM cells below for why (confirmed Windows access violation).
    fig.subplots_adjust(left=0.15, right=0.97, top=0.90, bottom=0.10)
    fig.savefig(FIG_DIR / "ablation_component_contribution_aggregate.png", dpi=200)
    plt.close(fig)
    display(Image(filename=str(FIG_DIR / "ablation_component_contribution_aggregate.png")))
finally:
    gc.enable()
    gc.collect()

print("Mean ΔAccuracy across all 12 datasets:")
print(agg_mean.round(4).to_string())
print("\\nStd across datasets (consistency of the effect):")
print(agg_std.round(4).to_string())
'''))

# ── 7.4 Interaction heatmap ───────────────────────────────────────────────
cells.append(md("""### 7.4 Interaction Heatmap — Token Order x Embedding

Does Spiral ordering and the DCT embedding help *more together* than the
sum of their individual effects would predict? Mean Accuracy across all
12 datasets, faceted by Gaussian bias on/off.
"""))
cells.append(code('''
gc.disable()
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, gauss_val, title in zip(axes, [False, True], ["Gaussian Bias = Off", "Gaussian Bias = On"]):
        sub = df_appendix[df_appendix["Gaussian"] == gauss_val]
        pivot = sub.pivot_table(index="Order", columns="Embed", values="Accuracy", aggfunc="mean")
        pivot = pivot.reindex(index=["raster", "spiral"], columns=["conv", "learned_dct"])
        sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlGn", ax=ax,
                    cbar_kws={"label": "Mean Accuracy"}, linewidths=0.5, linecolor="white")
        ax.set_title(title, fontsize=12, fontweight="bold")

    fig.suptitle("Token-Order x Embedding Interaction (mean Accuracy across all 12 datasets)",
                 fontsize=13, fontweight="bold", y=1.03)
    # Fixed margins instead of tight_layout()/bbox_inches="tight" -- see the
    # 9.1 Grad-CAM cells below for why (confirmed Windows access violation).
    fig.subplots_adjust(left=0.08, right=0.95, top=0.82, bottom=0.10, wspace=0.35)
    fig.savefig(FIG_DIR / "ablation_interaction_heatmap.png", dpi=200)
    plt.close(fig)
    display(Image(filename=str(FIG_DIR / "ablation_interaction_heatmap.png")))
finally:
    gc.enable()
    gc.collect()
'''))

# ── 8. Save ───────────────────────────────────────────────────────────────
cells.append(md("""---
## 8. Save Tables to CSV

Both the plain numeric CSVs (used for further analysis/plotting) and the
CI-annotated display versions (`*_with_ci.csv`, ready to paste as text) are
saved — the appendix table already carries CI as separate numeric columns,
so it only needs the one file.
"""))
cells.append(code('''
out_dir = ROOT / "results" / "MedMnist_fullres" / "benchmark_comparison"
out_dir.mkdir(parents=True, exist_ok=True)

df_b.round(4).to_csv(out_dir / "table_b_auc_acc.csv")
df_b_display.to_csv(out_dir / "table_b_auc_acc_with_ci.csv")
df_f1r.round(4).to_csv(out_dir / "table_f1_recall_ours_only.csv")
df_f1r_display.to_csv(out_dir / "table_f1_recall_ours_only_with_ci.csv")
df_appendix.to_csv(out_dir / "full_ablation_appendix.csv", index=False)

print("Saved to", out_dir)
'''))

# ── 9. Grad-CAM visualisation ────────────────────────────────────────────
cells.append(md("""---
## 9. Grad-CAM Visualisation — C1 vs C6 vs C8, Every Dataset

"Grad-CAM" adapted to a token sequence rather than a conv feature map: for a
chosen block's output tokens ``A``, relevance per token is
``ReLU(sum_d dY/dA[n,d] * A[n,d])`` — the embedding-dimension analogue of
Grad-CAM's channel-pooling step, remapped back to the 2-D patch grid
(spiral-aware, via the same inversion used for attention maps in
`attention_viz.py`).

**Layer choice matters**: hooking the *last* block is degenerate here —
`forward_features` only reads the CLS token (position 0) out of the final
block's output, so patch positions carry exactly zero gradient there
(verified: 0/256 non-zero patches at the last block, vs. ~227/256 at the
middle block on this architecture). `extract_gradcam_map` defaults to the
middle block for this reason — do not override `layer_idx` to a late block.

**Scope**: runs for all 11 single-label datasets (**Chest is excluded** — its
labels are multi-hot vectors, not a single ground-truth class index, so it
doesn't fit the same target-class comparison; it already gets separate
treatment everywhere else in this notebook, see Section 4).

**Design**: for each dataset, `N_SAMPLES_PER_DATASET` test images are run
through all three configs (**C1** baseline, **C6**, **C8** full model).
Grad-CAM is computed **w.r.t. the ground-truth class logit for all three
configs** (not each model's own top-1 prediction) — this keeps the
comparison apples-to-apples: any visual difference between columns reflects
where each config's attention actually goes for the *same* target, not a
side effect of the three configs disagreeing on the answer. Each panel's
title still reports that config's own predicted label (green border/text if
correct, red if wrong).

Change `GRADCAM_DATASETS` / `GRADCAM_CONFIGS` / `N_SAMPLES_PER_DATASET` /
`GRADCAM_SEED` below to adjust scope or draw a fresh batch. One grid PNG is
saved per dataset (rows = samples, columns = Original/C1/C6/C8); every row is
also saved as its own standalone 4-panel PNG so a single sample can be
dropped straight into the manuscript without re-running anything.
"""))

cells.append(code('''
GRADCAM_DATASETS = list(SINGLE_LABEL_DATASETS.keys())   # all 11 single-label datasets
GRADCAM_CONFIGS  = ["C1", "C6", "C8"]
N_SAMPLES_PER_DATASET = 3
GRADCAM_SEED = 0

from utils.attention_viz import extract_gradcam_map, gradcam_to_spatial
import medmnist as _medmnist

_MEDMNIST_INFO_KEY = {
    "blood": "bloodmnist", "breast": "breastmnist", "pneumonia": "pneumoniamnist",
    "retina": "retinamnist", "derma": "dermamnist", "oct": "octmnist",
    "tissue": "tissuemnist", "organa": "organamnist", "organc": "organcmnist",
    "organs": "organsmnist", "path": "pathmnist",
}

_GC_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_GC_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def _gc_denorm(img_t):
    t = img_t.squeeze(0).cpu()
    t = t * _GC_STD + _GC_MEAN
    return t.permute(1, 2, 0).clamp(0, 1).numpy()

def _gc_upsample(cam_map, h, w):
    from PIL import Image as _PILImage
    u8 = (cam_map * 255).clip(0, 255).astype(np.uint8)
    return np.array(_PILImage.fromarray(u8, "L").resize((w, h), _PILImage.BICUBIC)).astype(np.float32) / 255.0

GRADCAM_DIR = ROOT / "results" / "MedMnist_fullres" / "benchmark_comparison" / "figures" / "gradcam"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

print(f"Datasets : {GRADCAM_DATASETS}")
print(f"Configs  : {GRADCAM_CONFIGS}")
print(f"Samples/dataset: {N_SAMPLES_PER_DATASET}")
'''))

cells.append(code('''
gradcam_all = {}   # dataset -> {"samples": [...], "class_names": {...}}

# gc.disable() around the whole loop -- see the main evaluation loop in
# Section 5 for why (manual gc.collect() right after freeing a dataset's
# models/loader is a confirmed Windows access-violation trigger here).
gc.disable()
try:
  for ds in tqdm(GRADCAM_DATASETS, desc="Grad-CAM datasets"):
    class_names = _medmnist.INFO[_MEDMNIST_INFO_KEY[ds]]["label"]
    n_classes   = SINGLE_LABEL_DATASETS[ds]["n_classes"]

    models = load_all_configs(
        ds, n_classes, DEVICE, ckpt_dir=CKPT_DIR,
        ablation_configs=[c for c in ABLATION_CONFIGS if c["config_id"] in GRADCAM_CONFIGS],
    )
    missing = set(GRADCAM_CONFIGS) - set(models.keys())
    if missing:
        print(f"  [{ds}] SKIP -- missing checkpoints for {sorted(missing)}")
        continue

    test_loader = get_test_loader_only(
        dataset_name=ds, batch_size=1, resize_to=224,
        data_root=DATA_ROOT, native_size=NATIVE_SIZE, use_gray_to_get=False,
    )
    dataset_obj = test_loader.dataset

    rng = np.random.default_rng(GRADCAM_SEED)
    n_pick = min(N_SAMPLES_PER_DATASET, len(dataset_obj))
    sample_indices = rng.choice(len(dataset_obj), size=n_pick, replace=False)

    samples = []
    for idx in sample_indices:
        img_t, label = dataset_obj[int(idx)]
        img_t = img_t.unsqueeze(0).to(DEVICE)
        true_label = int(label)
        image_np = _gc_denorm(img_t)

        per_config = {}
        for cid in GRADCAM_CONFIGS:
            model, _ = models[cid]
            order_idx = model.token_order_idx.tolist() if model.token_order == "spiral" else None
            grid_size = model.grid_h
            # Ground-truth-targeted CAM for all three configs -- keeps the
            # comparison apples-to-apples (see Section 9 markdown above).
            cam, _ = extract_gradcam_map(model, img_t, layer_idx=None, target_class=true_label)
            with torch.no_grad():
                pred_label = int(model(img_t).argmax(dim=-1).item())
            spatial = gradcam_to_spatial(cam, grid_size, order_idx)
            per_config[cid] = {"cam": spatial, "pred": pred_label}

        samples.append({"idx": int(idx), "image": image_np, "true": true_label, "configs": per_config})

    gradcam_all[ds] = {"samples": samples, "class_names": class_names}
    print(f"  [{ds}] computed {len(samples)} samples x {len(GRADCAM_CONFIGS)} configs")

    del models, test_loader, dataset_obj
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
finally:
  gc.enable()
  gc.collect()

print(f"\\nDone -- Grad-CAM comparisons computed for {len(gradcam_all)}/{len(GRADCAM_DATASETS)} datasets.")
'''))

cells.append(md("### 9.1 Per-Dataset Comparison Grids (Original | C1 | C6 | C8)"))

cells.append(code('''
# Confirmed via faulthandler in the General ViT evaluation notebook's own
# debugging of this exact pattern (axis("off") grid + per-axes titles):
# plt.tight_layout()'s internal bbox computation (get_tightbbox() ->
# _update_title_position() / Text.get_window_extent()) causes a Windows
# access violation on this matplotlib install. gc.disable() alone did NOT
# fully prevent it there -- avoiding tight_layout()/bbox_inches="tight"
# entirely (fixed subplot margins instead) is what actually worked.
gc.disable()
try:
    for ds, payload in gradcam_all.items():
        samples     = payload["samples"]
        class_names = payload["class_names"]
        n           = len(samples)
        ncols       = 1 + len(GRADCAM_CONFIGS)

        fig, axes = plt.subplots(n, ncols, figsize=(ncols * 3.6, n * 3.6))
        axes = np.array(axes).reshape(n, ncols)   # normalise to 2-D regardless of n

        for row, s in enumerate(samples):
            h, w = s["image"].shape[:2]
            true_name = class_names.get(str(s["true"]), str(s["true"]))

            ax0 = axes[row, 0]
            ax0.imshow(np.clip(s["image"], 0, 1))
            ax0.set_title(f"Original\\ntrue={true_name}" if row == 0 else f"true={true_name}", fontsize=9)
            ax0.axis("off")

            for col, cid in enumerate(GRADCAM_CONFIGS, start=1):
                info = s["configs"][cid]
                cam_up = _gc_upsample(info["cam"], h, w)
                pred_name = class_names.get(str(info["pred"]), str(info["pred"]))
                correct = info["pred"] == s["true"]
                ax = axes[row, col]
                ax.imshow(np.clip(s["image"], 0, 1))
                ax.imshow(cam_up, cmap="jet", alpha=0.45, vmin=0, vmax=1)
                title = f"{cid}\\npred={pred_name}" if row == 0 else f"pred={pred_name}"
                ax.set_title(title, fontsize=9, color="#2ecc71" if correct else "#e74c3c")
                ax.axis("off")

        fig.suptitle(f"Grad-CAM Comparison -- {ds.upper()}  (target = ground truth, mid-block)",
                     fontsize=13, fontweight="bold", y=0.995)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.01, wspace=0.04, hspace=0.20)

        out_path = GRADCAM_DIR / f"gradcam_compare_{ds}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        # plt.show() also crashes here (Windows access violation in
        # _update_title_position() -> get_tightbbox(), which fires for any
        # axes with a title regardless of axis("off")) -- display the
        # already-saved PNG instead.
        display(Image(filename=str(out_path)))
        print("Saved:", out_path)
finally:
    gc.enable()
    gc.collect()
'''))

cells.append(md("""### 9.2 Individual Per-Sample Figures (Original | C1 | C6 | C8)

One standalone 4-panel PNG per sample — saved only (not displayed inline, to
avoid dozens of duplicate figures after the grids above) — so a single row
can be dropped straight into the manuscript without re-running anything.
Browse `results/MedMnist_fullres/benchmark_comparison/figures/gradcam/` and
pick the best `gradcam_compare_<dataset>_img<idx>.png` per dataset.
"""))

cells.append(code('''
# Same tight_layout()/axis("off")-grid crash pattern as 9.1 above -- fixed
# margins instead, gc.disable() around the whole loop.
_n_saved = 0
gc.disable()
try:
    for ds, payload in gradcam_all.items():
        class_names = payload["class_names"]
        for s in payload["samples"]:
            h, w = s["image"].shape[:2]
            true_name = class_names.get(str(s["true"]), str(s["true"]))
            ncols = 1 + len(GRADCAM_CONFIGS)

            fig, axes = plt.subplots(1, ncols, figsize=(ncols * 3.6, 3.6))
            axes[0].imshow(np.clip(s["image"], 0, 1))
            axes[0].set_title(f"Original\\ntrue={true_name}", fontsize=10)
            axes[0].axis("off")

            for col, cid in enumerate(GRADCAM_CONFIGS, start=1):
                info = s["configs"][cid]
                cam_up = _gc_upsample(info["cam"], h, w)
                pred_name = class_names.get(str(info["pred"]), str(info["pred"]))
                correct = info["pred"] == s["true"]
                axes[col].imshow(np.clip(s["image"], 0, 1))
                axes[col].imshow(cam_up, cmap="jet", alpha=0.45, vmin=0, vmax=1)
                axes[col].set_title(f"{cid}\\npred={pred_name}", fontsize=10,
                                     color="#2ecc71" if correct else "#e74c3c")
                axes[col].axis("off")

            fig.suptitle(f"{ds.upper()}  |  sample #{s[\'idx\']}", fontsize=11)
            fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.02, wspace=0.05)

            out_path = GRADCAM_DIR / f"gradcam_compare_{ds}_img{s[\'idx\']}.png"
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
            _n_saved += 1
finally:
    gc.enable()
    gc.collect()

print(f"Saved {_n_saved} individual per-sample figures to {GRADCAM_DIR}")
'''))

notebook = nb(cells)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Written: {OUT_PATH}")
print(f"Total cells: {len(cells)}")
