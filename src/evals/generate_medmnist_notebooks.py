"""
generate_medmnist_notebooks.py
-------------------------------
Generates one evaluation notebook per MedMNIST dataset.
Run from the repository root:
    python src/evals/generate_medmnist_notebooks.py
"""

import json, os, textwrap

OUT_DIR = os.path.join(os.path.dirname(__file__))

# ── Dataset configurations ────────────────────────────────────────────────────
DATASETS = {
    "blood": {
        "title":       "BloodMNIST — Blood Cell Type Classification",
        "medmnist_cls":"BloodMNIST",
        "n_classes":   8,
        "is_rgb":      True,
        "class_names": ["Basophil","Eosinophil","Erythroblast","Myeloblast",
                         "Promyelocyte","Metamyelocyte","Monocyte","Lymphocyte"],
        "task":        "8-class blood cell type classification (RGB)",
    },
    "breast": {
        "title":       "BreastMNIST — Ultrasound Malignancy Detection",
        "medmnist_cls":"BreastMNIST",
        "n_classes":   2,
        "is_rgb":      False,
        "class_names": ["Benign","Malignant"],
        "task":        "Binary ultrasound malignancy detection (grayscale → 3-ch)",
    },
    "pneumonia": {
        "title":       "PneumoniaMNIST — Chest X-Ray Pneumonia Detection",
        "medmnist_cls":"PneumoniaMNIST",
        "n_classes":   2,
        "is_rgb":      False,
        "class_names": ["Normal","Pneumonia"],
        "task":        "Binary chest X-ray pneumonia detection (grayscale → 3-ch)",
    },
    "retina": {
        "title":       "RetinaMNIST — Retinal Disease Grading",
        "medmnist_cls":"RetinaMNIST",
        "n_classes":   5,
        "is_rgb":      True,
        "class_names": ["Grade 0","Grade 1","Grade 2","Grade 3","Grade 4"],
        "task":        "5-class retinal vascular grading (RGB)",
    },
    "derma": {
        "title":       "DermaMNIST — Skin Lesion Classification",
        "medmnist_cls":"DermaMNIST",
        "n_classes":   7,
        "is_rgb":      True,
        "class_names": ["Melanocytic nevi","Melanoma","Benign keratosis",
                         "Basal cell carcinoma","Actinic keratosis",
                         "Vascular lesions","Dermatofibroma"],
        "task":        "7-class skin lesion classification (RGB)",
    },
    "oct": {
        "title":       "OCTMNIST — Retinal OCT Disease Classification",
        "medmnist_cls":"OCTMNIST",
        "n_classes":   4,
        "is_rgb":      False,
        "class_names": ["Choroidal neovascularization","Diabetic macular edema",
                         "Drusen","Normal"],
        "task":        "4-class retinal OCT classification (grayscale → 3-ch)",
    },
    "tissue": {
        "title":       "TissueMNIST — Kidney Cortex Cell Type Classification",
        "medmnist_cls":"TissueMNIST",
        "n_classes":   8,
        "is_rgb":      False,
        "class_names": ["Collecting Duct/Connecting Tubule","Distal Convoluted Tubule",
                         "Glomerular endothelial cells","Interstitial endothelial cells",
                         "Leukocytes","Podocytes","Proximal Tubule Segments",
                         "Thick Ascending Limb"],
        "task":        "8-class kidney cortex cell type classification (grayscale → 3-ch)",
    },
    "organa": {
        "title":       "OrganAMNIST — Axial Organ Classification",
        "medmnist_cls":"OrganAMNIST",
        "n_classes":   11,
        "is_rgb":      False,
        "class_names": ["Bladder","Femur-left","Femur-right","Heart","Kidney-left",
                         "Kidney-right","Liver","Lung-left","Lung-right","Pancreas","Spleen"],
        "task":        "11-class axial-plane organ classification (grayscale → 3-ch)",
    },
    "organc": {
        "title":       "OrganCMNIST — Coronal Organ Classification",
        "medmnist_cls":"OrganCMNIST",
        "n_classes":   11,
        "is_rgb":      False,
        "class_names": ["Bladder","Femur-left","Femur-right","Heart","Kidney-left",
                         "Kidney-right","Liver","Lung-left","Lung-right","Pancreas","Spleen"],
        "task":        "11-class coronal-plane organ classification (grayscale → 3-ch)",
    },
    "organs": {
        "title":       "OrganSMNIST — Sagittal Organ Classification",
        "medmnist_cls":"OrganSMNIST",
        "n_classes":   11,
        "is_rgb":      False,
        "class_names": ["Bladder","Femur-left","Femur-right","Heart","Kidney-left",
                         "Kidney-right","Liver","Lung-left","Lung-right","Pancreas","Spleen"],
        "task":        "11-class sagittal-plane organ classification (grayscale → 3-ch)",
    },
    "path": {
        "title":       "PathMNIST — Colorectal Histology Classification",
        "medmnist_cls":"PathMNIST",
        "n_classes":   9,
        "is_rgb":      True,
        "class_names": ["Adipose","Background","Debris","Lymphocytes","Mucus",
                         "Smooth muscle","Normal colon mucosa",
                         "Cancer-associated stroma","Colorectal adenocarcinoma epithelium"],
        "task":        "9-class colorectal histology tissue classification (RGB)",
    },
}

# ── Notebook helpers ──────────────────────────────────────────────────────────

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

# ── Cell factories ────────────────────────────────────────────────────────────

def cell_imports(ds, cfg):
    return code(f"""
import os, sys
from pathlib import Path

# Locate repo root (contains src/) regardless of notebook cwd
ROOT = Path().resolve()
while not (ROOT / "src").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
SRC = ROOT / "src"
for _p in [str(ROOT), str(SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from IPython.display import display

from evals.medmnist_eval_utils import (
    ABLATION_CONFIGS_LEARNED as ABLATION_CONFIGS,
    CONFIG_LABELS_LEARNED as CONFIG_LABELS,
    load_all_configs, evaluate_model_with_ci,
    compare_models, compare_models_probabilities,
    plot_roc_curves, plot_confusion_matrix,
    plot_all_confusion_matrices, plot_metrics_comparison,
    plot_component_contribution, print_results_table,
)
from data.medmnist_dataset import get_medmnist_loaders

# ── Global settings
DATASET     = "{ds}"
N_CLASSES   = {cfg['n_classes']}
CLASS_NAMES = {cfg['class_names']}
TASK        = "{cfg['task']}"
NATIVE_SIZE = 224   # true 224px MedMNIST release, not the 28px-upsampled baseline
DATA_ROOT   = str(ROOT / "data" / "MedMnist_fullres")
CKPT_DIR    = str(ROOT / "checkpoints" / "medmnist")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Dataset : {{DATASET}}")
print(f"Classes : {{N_CLASSES}}")
print(f"Device  : {{DEVICE}}")
print(f"CKPT dir: {{CKPT_DIR}}")
""")


def cell_eda_header():
    return md("""---
## 1. Dataset Characteristics (EDA)
Explore the raw MedMNIST dataset: split sizes, class distribution, sample images, and the preprocessing pipeline the model actually sees.
""")


def cell_eda_load(ds, cfg):
    return code(f"""
from medmnist import {cfg['medmnist_cls']}, INFO

# Load all three splits (no preprocessing — raw PIL images for EDA)
from torchvision import transforms as T
raw_tf = T.Compose([T.ToTensor()])

_kw = {{"root": DATA_ROOT, "download": True, "size": NATIVE_SIZE}}
raw_train = {cfg['medmnist_cls']}(split="train", transform=raw_tf, **_kw)
raw_val   = {cfg['medmnist_cls']}(split="val",   transform=raw_tf, **_kw)
raw_test  = {cfg['medmnist_cls']}(split="test",  transform=raw_tf, **_kw)

info_key  = "{cfg['medmnist_cls'].lower()}"
info      = INFO[info_key]

print(f"Task   : {{info['task']}}")
print(f"Train  : {{len(raw_train)}} images")
print(f"Val    : {{len(raw_val)}} images")
print(f"Test   : {{len(raw_test)}} images")
print(f"Classes: {{len(info['label'])}}")
print(f"Labels : {{info['label']}}")
""")


def cell_eda_distribution(cfg):
    return code("""
import numpy as np

def class_counts(dataset):
    labels = dataset.labels.squeeze().astype(int)
    counts = np.bincount(labels, minlength=N_CLASSES)
    return counts

train_counts = class_counts(raw_train)
val_counts   = class_counts(raw_val)
test_counts  = class_counts(raw_test)

df_dist = pd.DataFrame({
    "Class": CLASS_NAMES,
    "Train": train_counts,
    "Val":   val_counts,
    "Test":  test_counts,
    "Train %": (100 * train_counts / train_counts.sum()).round(1),
    "Test %":  (100 * test_counts  / test_counts.sum()).round(1),
})
display(df_dist)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
x = np.arange(N_CLASSES)
axes[0].bar(x, train_counts, color="#1f77b4", alpha=0.85, edgecolor="black", lw=0.5)
axes[0].set_xticks(x)
axes[0].set_xticklabels(CLASS_NAMES, rotation=40, ha="right", fontsize=8)
axes[0].set_title("Training Set — Class Distribution", fontsize=11)
axes[0].set_ylabel("Count")
axes[0].grid(axis="y", alpha=0.3)

axes[1].bar(x, test_counts, color="#ff7f0e", alpha=0.85, edgecolor="black", lw=0.5)
axes[1].set_xticks(x)
axes[1].set_xticklabels(CLASS_NAMES, rotation=40, ha="right", fontsize=8)
axes[1].set_title("Test Set — Class Distribution", fontsize=11)
axes[1].set_ylabel("Count")
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()
""")


def cell_eda_montage(cfg):
    return code("""
# Show one sample per class from the training set
from collections import defaultdict

# Collect one example per class
class_samples = defaultdict(list)
for img, label in raw_train:
    lbl = int(np.asarray(label).squeeze())
    if len(class_samples[lbl]) < 3:
        class_samples[lbl].append(img)
    if all(len(v) >= 3 for v in class_samples.values()) and len(class_samples) == N_CLASSES:
        break

fig, axes = plt.subplots(3, N_CLASSES, figsize=(N_CLASSES * 1.8, 6))
if N_CLASSES == 1:
    axes = np.array([[axes[0]], [axes[1]], [axes[2]]])
elif axes.ndim == 1:
    axes = axes.reshape(3, 1)

for cls_idx in range(N_CLASSES):
    for row, img in enumerate(class_samples[cls_idx]):
        ax = axes[row][cls_idx]
        img_np = img.permute(1, 2, 0).numpy()
        if img_np.shape[2] == 1:
            ax.imshow(img_np[:, :, 0], cmap="gray")
        else:
            img_np = np.clip(img_np, 0, 1)
            ax.imshow(img_np)
        ax.axis("off")
        if row == 0:
            ax.set_title(CLASS_NAMES[cls_idx], fontsize=7, fontweight="bold")

fig.suptitle(f"Sample Images — {DATASET.upper()} (3 per class)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()
""")


def cell_eda_preprocessing(cfg):
    rgb_note = "RGB images are kept as-is (no channel conversion)." if cfg["is_rgb"] else \
               "Grayscale images are expanded to 3 channels via `T.Grayscale(3)` (simple replication)."
    return code(f"""
# Show the preprocessing pipeline: raw → resized → normalised (what the model sees)
from torchvision import transforms as T

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

# {rgb_note}
{"pil_ops = []" if cfg["is_rgb"] else "pil_ops = [T.Grayscale(num_output_channels=3)]"}
preproc_tf = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
    *pil_ops,
    T.ToTensor(),
    T.Normalize(mean, std),
])

# Reverse-normalise for display
def denorm(t):
    m = torch.tensor(mean).view(3,1,1)
    s = torch.tensor(std).view(3,1,1)
    return torch.clamp(t * s + m, 0, 1)

# Grab one example per class (raw PIL)
from medmnist import {cfg['medmnist_cls']}
pil_train = {cfg['medmnist_cls']}(split="train", transform=None, root=DATA_ROOT, download=True, size=NATIVE_SIZE)

class_pil = {{}}
for img_pil, label in pil_train:
    lbl = int(np.asarray(label).squeeze())
    if lbl not in class_pil:
        class_pil[lbl] = img_pil
    if len(class_pil) == N_CLASSES:
        break

fig, axes = plt.subplots(2, N_CLASSES, figsize=(N_CLASSES * 2, 5))
if N_CLASSES == 1:
    axes = np.array([[axes[0]], [axes[1]]])
elif axes.ndim == 1:
    axes = axes.reshape(2, 1)

for cls_idx in range(N_CLASSES):
    pil = class_pil.get(cls_idx)
    if pil is None:
        continue

    # Row 0: raw
    ax_raw = axes[0][cls_idx]
    raw_arr = np.array(pil)
    if raw_arr.ndim == 2:
        ax_raw.imshow(raw_arr, cmap="gray")
    else:
        ax_raw.imshow(raw_arr)
    ax_raw.axis("off")
    ax_raw.set_title(CLASS_NAMES[cls_idx], fontsize=7, fontweight="bold")

    # Row 1: after preprocessing (denorm for display)
    ax_pp = axes[1][cls_idx]
    tensor = preproc_tf(pil)
    disp   = denorm(tensor).permute(1,2,0).numpy()
    ax_pp.imshow(disp)
    ax_pp.axis("off")

axes[0][0].set_ylabel("Raw", fontsize=9)
axes[1][0].set_ylabel("Preprocessed\\n(224×224, norm.)", fontsize=9)
fig.suptitle("Preprocessing Pipeline", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()
""")


def cell_eval_header():
    return md("""---
## 2. Load Models & Run Test Evaluation
Load all 8 ablation checkpoints and evaluate on the held-out test set.
Each config is evaluated with 1000-bootstrap 95% confidence intervals (seed=42), matching the advisor's ViTDCT protocol.
""")


def cell_eval_loaders():
    return code("""
# Build test loader (clean, deterministic — no augmentation)
train_loader, val_loader, test_loader, _class_names = get_medmnist_loaders(
    dataset_name=DATASET,
    batch_size=64,
    resize_to=224,
    download=True,
    data_root=DATA_ROOT,
    num_workers=0,
    use_gray_to_get=False,   # baseline: T.Grayscale(3), not GET
    native_size=NATIVE_SIZE,
)

# Class prevalence from training split (used for ROC decision points)
from collections import Counter
_all_labels = []
for _, lbl in train_loader.dataset:
    _all_labels.append(int(lbl))
prevalence = {k: v / len(_all_labels) for k, v in Counter(_all_labels).items()}
print("Class prevalence (train):")
for k, v in sorted(prevalence.items()):
    print(f"  {CLASS_NAMES[k]}: {v*100:.1f}%")
""")


def cell_eval_run():
    return code("""
# Evaluate all 8 configs (takes a few minutes — 1000 bootstrap samples each)
models = load_all_configs(DATASET, N_CLASSES, DEVICE, ckpt_dir=CKPT_DIR, ablation_configs=ABLATION_CONFIGS)
print(f"Loaded {len(models)} / 8 checkpoints\\n")

results = {}
for cfg in ABLATION_CONFIGS:
    cid = cfg["config_id"]
    if cid not in models:
        print(f"  [{cid}] SKIP — checkpoint not found")
        continue
    model, _ = models[cid]
    print(f"  [{cid}] Evaluating ...", end=" ", flush=True)
    res = evaluate_model_with_ci(model, test_loader, DEVICE,
                                  n_bootstraps=1000, random_seed=42)
    results[cid] = res
    print(f"Acc={res['accuracy']:.4f}  F1={res['f1']:.4f}  AUC={res['roc_auc']:.4f}")
""")


def cell_results_table_header():
    return md("""---
## 3. Results Summary Table
Point estimates and 95% bootstrap confidence intervals for all 8 configurations.
""")


def cell_results_table():
    return code("""
print_results_table(results, DATASET)

# Pandas version for richer display
rows = []
for cfg in ABLATION_CONFIGS:
    cid = cfg["config_id"]
    if cid not in results:
        continue
    r = results[cid]
    rows.append({
        "Config": cid,
        "Embed": cfg["embed_type"],
        "Order": cfg["token_order"],
        "Gaussian": cfg["gaussian_bias"],
        "Accuracy": f"{r['accuracy']:.4f} [{r['accuracy_ci'][0]:.4f}–{r['accuracy_ci'][1]:.4f}]",
        "Recall":   f"{r['recall']:.4f} [{r['recall_ci'][0]:.4f}–{r['recall_ci'][1]:.4f}]",
        "Macro-F1": f"{r['f1']:.4f} [{r['f1_ci'][0]:.4f}–{r['f1_ci'][1]:.4f}]",
        "ROC-AUC":  f"{r['roc_auc']:.4f} [{r['roc_auc_ci'][0]:.4f}–{r['roc_auc_ci'][1]:.4f}]",
    })

df_res = pd.DataFrame(rows).set_index("Config")
display(df_res)

# Best config by Macro-F1
best_cid = max(results, key=lambda c: results[c]["f1"])
print(f"\\nBest config by Macro-F1: {best_cid} — F1={results[best_cid]['f1']:.4f}")
""")


def cell_confusion_matrices_header():
    return md("""---
## 4. Confusion Matrices
All 8 configurations side-by-side.
""")


def cell_confusion_matrices():
    return code("""
plot_all_confusion_matrices(results, CLASS_NAMES, DATASET)
""")


def cell_roc_header():
    return md("""---
## 5. ROC Curves
Per-class one-vs-rest ROC curves.
The filled dot on each curve marks the threshold closest to class training prevalence — the clinically calibrated operating point used by the advisor.
""")


def cell_roc():
    return code("""
n_cols = 4
n_rows = 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 10))
fig.suptitle(f"ROC Curves — {DATASET.upper()}", fontsize=14, fontweight="bold")

for i, cfg in enumerate(ABLATION_CONFIGS):
    cid = cfg["config_id"]
    ax  = axes[i // n_cols][i % n_cols]
    if cid not in models:
        ax.set_visible(False)
        continue
    model, _ = models[cid]
    plot_roc_curves(
        model, test_loader, DEVICE,
        prevalence=prevalence,
        class_names=CLASS_NAMES,
        title=CONFIG_LABELS[cid].replace("\\n", " "),
        ax=ax,
    )

plt.tight_layout()
plt.show()
""")


def cell_metrics_bar_header():
    return md("""---
## 6. Ablation Metrics Bar Chart
Visual comparison of Accuracy, Macro-F1, and ROC-AUC across all 8 configurations.
Error bars show the 95% bootstrap CI.  The full Spiral-ViT (C8) is highlighted in red.
""")


def cell_metrics_bar():
    return code("""
plot_metrics_comparison(results, DATASET)
""")


def cell_component_header():
    return md("""---
## 7. Component Contribution Analysis
Marginal contribution of each Spiral-ViT component (spiral ordering, DCT embedding, Gaussian bias)
on Macro-F1, computed by switching each axis on/off and averaging across the other two axes.
""")


def cell_component():
    return code("""
deltas = plot_component_contribution(results, DATASET)
print(f"ΔF1 Spiral:   {deltas['delta_spiral']:+.4f}")
print(f"ΔF1 DCT:      {deltas['delta_dct']:+.4f}")
print(f"ΔF1 Gaussian: {deltas['delta_gauss']:+.4f}")
""")


def cell_stat_header():
    return md("""---
## 8. Statistical Comparison: C1 (ViT Baseline) vs C8 (Full Spiral-ViT)
Mirrors the advisor's statistical comparison protocol:
- **`compare_models_probabilities`**: paired t-test + Wilcoxon signed-rank on true-class softmax probability
- **`compare_models`**: McNemar's test (corrected) + Cohen's Kappa with 2000-bootstrap CI
""")


def cell_stat():
    return code("""
if "C1" in models and "C8" in models:
    m_c1, _ = models["C1"]
    m_c8, _ = models["C8"]

    # Probability comparison
    cmp_prob = compare_models_probabilities(
        model1=m_c8, model2=m_c1,
        test_loader=test_loader, device=DEVICE,
        label1="C8 (Spiral-ViT)", label2="C1 (ViT baseline)",
    )

    # Prediction-level comparison
    cmp_pred = compare_models(
        y_true  = cmp_prob["True_Class"],
        y_pred1 = cmp_prob["Predicted_Class_1"],
        y_pred2 = cmp_prob["Predicted_Class_2"],
        label1  = "C8 (Spiral-ViT)",
        label2  = "C1 (ViT baseline)",
    )
else:
    print("C1 or C8 checkpoint not available — skipping comparison.")
""")


def cell_summary_header():
    return md("""---
## 9. Summary
""")


def cell_summary(ds, cfg):
    task_escaped = cfg['task'].replace("'", "\\'")
    ds_upper = ds.upper()
    return code(f"""
print("=" * 70)
print("  SUMMARY — {ds_upper}")
print("=" * 70)

if results:
    best_cid  = max(results, key=lambda c: results[c]["f1"])
    worst_cid = min(results, key=lambda c: results[c]["f1"])
    r_best  = results[best_cid]
    r_worst = results[worst_cid]
    r_c1    = results.get("C1", {{}})
    r_c8    = results.get("C8", {{}})

    print(f"  Dataset        : {ds_upper}")
    print(f"  Task           : {task_escaped}")
    print(f"  Test samples   : {{len(test_loader.dataset)}}")
    print()
    lo = r_best["f1_ci"][0]
    hi = r_best["f1_ci"][1]
    print(f"  Best config    : {{best_cid}}  F1={{r_best['f1']:.4f}} [{{lo:.4f}}, {{hi:.4f}}]")
    print(f"  Worst config   : {{worst_cid}}  F1={{r_worst['f1']:.4f}}")
    if r_c1 and r_c8:
        delta = r_c8["f1"] - r_c1["f1"]
        print(f"  C8 vs C1 ΔF1   : {{delta:+.4f}}")
    print()
    print("  Full table:")
    print_results_table(results, DATASET)
""")


# ── Notebook builder ──────────────────────────────────────────────────────────

def build_notebook(ds, cfg):
    title = cfg["title"]
    cells = [
        md(f"# {title}\n\nSPIRAL-VIT ablation evaluation: {cfg['task']}\n\n"
           f"This notebook evaluates all **8 ablation configurations** (C1–C8) on the "
           f"held-out test set, reproducing the advisor's ViTDCT evaluation protocol with "
           f"1000-bootstrap 95% confidence intervals.\n"),
        code("%load_ext autoreload\n%autoreload 2"),
        cell_imports(ds, cfg),

        cell_eda_header(),
        cell_eda_load(ds, cfg),
        cell_eda_distribution(cfg),
        cell_eda_montage(cfg),
        cell_eda_preprocessing(cfg),

        cell_eval_header(),
        cell_eval_loaders(),
        cell_eval_run(),

        cell_results_table_header(),
        cell_results_table(),

        cell_confusion_matrices_header(),
        cell_confusion_matrices(),

        cell_roc_header(),
        cell_roc(),

        cell_metrics_bar_header(),
        cell_metrics_bar(),

        cell_component_header(),
        cell_component(),

        cell_stat_header(),
        cell_stat(),

        cell_summary_header(),
        cell_summary(ds, cfg),
    ]
    return nb(cells)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for ds, cfg in DATASETS.items():
        path = os.path.join(OUT_DIR, f"medmnist_{ds}_evaluation.ipynb")
        notebook = build_notebook(ds, cfg)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print(f"Written: {path}")

    print(f"\nAll {len(DATASETS)} notebooks generated.")
