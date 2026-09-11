"""
src/evals/medmnist_eval_utils.py
=================================
Shared evaluation utilities for the SPIRAL-VIT MedMNIST ablation notebooks.

Mirrors the advisor's ViTDCT evaluation protocol so results are directly
comparable:
  - evaluate_model_with_ci : point estimates + 1000-bootstrap 95% CIs
  - compare_models          : McNemar's test + Cohen's Kappa (2000-bootstrap CI)
  - compare_models_probabilities : paired t-test + Wilcoxon signed-rank
  - plot_roc_curves          : per-class OvR ROC with prevalence decision points
  - plot_confusion_matrix    : annotated seaborn heatmap
  - load_spiral_vit          : load any of the 8 ablation checkpoints
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    auc,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils import resample
import scipy.stats as stats

# statsmodels for McNemar's test
try:
    from statsmodels.stats.contingency_tables import mcnemar
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False

# tqdm's background monitor thread has a known race with sklearn's internal
# warnings/inspect-based parameter validation under tight bootstrap loops --
# manifests as anything from a soft AttributeError/KeyError deep inside
# roc_auc_score/f1_score to a hard Windows access violation (confirmed via
# faulthandler in the General ViT evaluation notebook's own debugging of
# this exact pattern). Setting this to 0 stops the monitor thread from ever
# spawning; must happen before any tqdm instance is created anywhere in the
# process, which this module-level assignment guarantees for any caller that
# imports this file before creating its own tqdm bars.
from tqdm.auto import tqdm as _tqdm_class
_tqdm_class.monitor_interval = 0


def _safe_call(fn, *args, max_retries=5, **kwargs):
    """Retry a pure sklearn-metric call on AttributeError/KeyError -- belt
    and suspenders alongside monitor_interval=0 above. Since the call is a
    pure function of its (already-resampled) arguments, retrying recomputes
    the exact same thing with no side effects and no re-drawing of the
    sample."""
    last_exc = None
    for _ in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except (AttributeError, KeyError) as exc:
            last_exc = exc
            continue
    raise last_exc

# Project root (contains src/) and src/ itself both need to be on the path.
# vit.py uses `from utils.token_order import ...` which requires src/ in sys.path.
_ROOT = Path(__file__).resolve().parents[2]   # repo root
_SRC  = _ROOT / "src"
for _p in [str(_ROOT), str(_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.vit import ViT  # noqa: E402  (resolved via _SRC on sys.path)


# ---------------------------------------------------------------------------
# Ablation config registry
# ---------------------------------------------------------------------------

ABLATION_CONFIGS: List[Dict] = [
    {"config_id": "C1", "embed_type": "conv",     "token_order": "raster",  "gaussian_bias": False},
    {"config_id": "C2", "embed_type": "conv",     "token_order": "spiral",  "gaussian_bias": False},
    {"config_id": "C3", "embed_type": "dct_fast", "token_order": "raster",  "gaussian_bias": False},
    {"config_id": "C4", "embed_type": "conv",     "token_order": "raster",  "gaussian_bias": True},
    {"config_id": "C5", "embed_type": "dct_fast", "token_order": "spiral",  "gaussian_bias": False},
    {"config_id": "C6", "embed_type": "conv",     "token_order": "spiral",  "gaussian_bias": True},
    {"config_id": "C7", "embed_type": "dct_fast", "token_order": "raster",  "gaussian_bias": True},
    {"config_id": "C8", "embed_type": "dct_fast", "token_order": "spiral",  "gaussian_bias": True},
]

CONFIG_LABELS: Dict[str, str] = {
    "C1": "C1\nViT baseline\n(raster+conv)",
    "C2": "C2\nSpiral\n(conv)",
    "C3": "C3\nDCT\n(raster)",
    "C4": "C4\nGaussian\n(raster+conv)",
    "C5": "C5\nSpiral+DCT",
    "C6": "C6\nSpiral+Gaussian\n(conv)",
    "C7": "C7\nDCT+Gaussian\n(raster)",
    "C8": "C8\nFull Spiral-ViT\n(spiral+DCT+Gauss)",
}

CKPT_DIR = str(_ROOT / "checkpoints" / "medmnist")

# Learned-DCT paper round: dct_fast replaced by learned_dct in the same 4
# slots (C3/C5/C7/C8). Kept separate from ABLATION_CONFIGS/CONFIG_LABELS
# above (rather than mutating them) so any existing notebook built on the
# original dct_fast checkpoints keeps working unmodified.
ABLATION_CONFIGS_LEARNED: List[Dict] = [
    {"config_id": "C1", "embed_type": "conv",        "token_order": "raster", "gaussian_bias": False},
    {"config_id": "C2", "embed_type": "conv",        "token_order": "spiral", "gaussian_bias": False},
    {"config_id": "C3", "embed_type": "learned_dct", "token_order": "raster", "gaussian_bias": False},
    {"config_id": "C4", "embed_type": "conv",        "token_order": "raster", "gaussian_bias": True},
    {"config_id": "C5", "embed_type": "learned_dct", "token_order": "spiral", "gaussian_bias": False},
    {"config_id": "C6", "embed_type": "conv",        "token_order": "spiral", "gaussian_bias": True},
    {"config_id": "C7", "embed_type": "learned_dct", "token_order": "raster", "gaussian_bias": True},
    {"config_id": "C8", "embed_type": "learned_dct", "token_order": "spiral", "gaussian_bias": True},
]

CONFIG_LABELS_LEARNED: Dict[str, str] = {
    "C1": "C1\nViT baseline\n(raster+conv)",
    "C2": "C2\nSpiral\n(conv)",
    "C3": "C3\nLearned-DCT\n(raster)",
    "C4": "C4\nGaussian\n(raster+conv)",
    "C5": "C5\nSpiral+Learned-DCT",
    "C6": "C6\nSpiral+Gaussian\n(conv)",
    "C7": "C7\nLearned-DCT+Gaussian\n(raster)",
    "C8": "C8\nFull Spiral-ViT\n(spiral+Learned-DCT+Gauss)",
}


def ckpt_path(dataset: str, embed_type: str, token_order: str, gaussian_bias: bool,
              ckpt_dir: str = CKPT_DIR) -> str:
    gauss = "gaussian" if gaussian_bias else "nogaussian"
    return os.path.join(ckpt_dir, f"vit_{dataset}_best_{embed_type}_{token_order}_{gauss}.pth")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_spiral_vit(path: str, num_classes: int, device: torch.device) -> Tuple[ViT, Dict]:
    """Load a Spiral-ViT checkpoint and return (model, cfg)."""
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Corrupted checkpoint — {path}\n  ({exc})") from exc
    cfg  = ckpt["cfg"]
    model = ViT(
        image_size    = int(cfg.get("image_size",    224)),
        patch_size    = int(cfg.get("patch_size",     14)),
        num_classes   = num_classes,
        dim           = int(cfg.get("dim",           256)),
        depth         = int(cfg.get("depth",           6)),
        heads         = int(cfg.get("heads",           8)),
        mlp_dim       = int(cfg.get("mlp_dim",       512)),
        dropout       = float(cfg.get("dropout",     0.1)),
        embed_type    = str(cfg.get("embed_type",  "conv")),
        token_order   = str(cfg.get("token_order", "raster")),
        gaussian_bias = bool(cfg.get("gaussian_bias", False)),
        freq_keep     = int(cfg.get("freq_keep",      16)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, cfg


def load_all_configs(
    dataset: str, num_classes: int, device: torch.device,
    ckpt_dir: str = CKPT_DIR, ablation_configs: Optional[List[Dict]] = None,
) -> Dict[str, Tuple[ViT, Dict]]:
    """Load all 8 ablation checkpoints for a dataset. Returns {config_id: (model, cfg)}."""
    if ablation_configs is None:
        ablation_configs = ABLATION_CONFIGS
    models = {}
    for c in ablation_configs:
        path = ckpt_path(dataset, c["embed_type"], c["token_order"], c["gaussian_bias"], ckpt_dir=ckpt_dir)
        if not os.path.exists(path):
            print(f"[WARN] Checkpoint not found: {path}")
            continue
        try:
            model, cfg = load_spiral_vit(path, num_classes, device)
            models[c["config_id"]] = (model, cfg)
        except RuntimeError as exc:
            print(f"[WARN] {exc}")
    return models


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(arr: np.ndarray, statistic=np.mean, n_boot: int = 2000,
                 random_seed: int = 42) -> Tuple[float, float]:
    """Percentile 95% CI via bootstrap resampling."""
    rng = np.random.default_rng(random_seed)
    boot = []
    for _ in range(n_boot):
        idx  = rng.integers(0, len(arr), len(arr))
        boot.append(statistic(arr[idx]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_kappa_ci(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray,
                        n_boot: int = 2000, random_seed: int = 42) -> Tuple[float, float]:
    """Bootstrap 95% CI for Cohen's Kappa between y_pred1 and y_pred2."""
    rng   = np.random.default_rng(random_seed)
    kappas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        try:
            k = _safe_call(cohen_kappa_score, y_pred1[idx], y_pred2[idx])
            kappas.append(k)
        except Exception:
            pass
    if not kappas:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(kappas, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_model_with_ci(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    n_bootstraps: int = 1000,
    random_seed: int = 42,
    model_base: Optional[torch.nn.Module] = None,
    show_progress: bool = False,
) -> Dict:
    """
    Run test-set inference and compute point-estimate metrics with 95% CIs.

    Mirrors the advisor's ViTDCT ``evaluate_model_with_ci()`` exactly:
      - Accuracy, Macro-Recall, Macro-F1, ROC-AUC
      - 95% CI via percentile bootstrap (n_bootstraps=1000, seed=42)
      - Optionally runs a second baseline model in the same forward pass

    Returns
    -------
    dict with keys:
        accuracy, accuracy_ci, recall, recall_ci, f1, f1_ci,
        roc_auc, roc_auc_ci, confusion_matrix,
        True_Class, Predicted_Class, Predicted_Score,
        Predicted_Base_Class, Predicted_Base_Score  (if model_base given)
    """
    model.eval()
    if model_base is not None:
        model_base.eval()

    y_true_list, y_pred_list, y_score_list = [], [], []
    y_pred_ba_list, y_score_ba_list         = [], []

    with torch.no_grad():
        for batch in test_loader:
            imgs, labels = batch[0].to(device), batch[1]
            labels = labels.squeeze().long()

            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)
            preds  = logits.argmax(dim=1)

            y_true_list.append(labels.cpu().numpy())
            y_pred_list.append(preds.cpu().numpy())
            y_score_list.append(probs.cpu().numpy())

            if model_base is not None:
                logits_ba = model_base(imgs)
                probs_ba  = torch.softmax(logits_ba, dim=1)
                preds_ba  = logits_ba.argmax(dim=1)
                y_pred_ba_list.append(preds_ba.cpu().numpy())
                y_score_ba_list.append(probs_ba.cpu().numpy())

    y_true  = np.concatenate(y_true_list)
    y_pred  = np.concatenate(y_pred_list)
    y_score = np.concatenate(y_score_list)

    # Point estimates
    accuracy = float((y_pred == y_true).mean())
    recall   = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1       = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm       = confusion_matrix(y_true, y_pred)

    n_classes = y_score.shape[1]
    try:
        if n_classes == 2:
            roc_auc = float(roc_auc_score(y_true, y_score[:, 1]))
        else:
            roc_auc = float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))
    except Exception:
        roc_auc = float("nan")

    # Bootstrap CIs
    rng = np.random.default_rng(random_seed)
    acc_scores, rec_scores, f1_scores, auc_scores = [], [], [], []

    _boot_iter = range(n_bootstraps)
    if show_progress:
        from tqdm.auto import tqdm as _tqdm
        _boot_iter = _tqdm(_boot_iter, desc="  bootstrap CI", leave=False)

    for _ in _boot_iter:
        idx       = rng.integers(0, len(y_true), len(y_true))
        yt, yp, ys = y_true[idx], y_pred[idx], y_score[idx]

        acc_scores.append((yp == yt).mean())
        rec_scores.append(_safe_call(recall_score, yt, yp, average="macro", zero_division=0))
        f1_scores.append(_safe_call(f1_score, yt, yp, average="macro", zero_division=0))

        try:
            if n_classes == 2:
                if len(np.unique(yt)) > 1:
                    auc_scores.append(_safe_call(roc_auc_score, yt, ys[:, 1]))
            else:
                auc_scores.append(_safe_call(roc_auc_score, yt, ys, multi_class="ovr", average="macro"))
        except Exception:
            pass

    def ci(vals):
        return np.percentile(vals, [2.5, 97.5]).tolist() if vals else [float("nan"), float("nan")]

    result = {
        "accuracy":     accuracy,
        "accuracy_ci":  ci(acc_scores),
        "recall":       recall,
        "recall_ci":    ci(rec_scores),
        "f1":           f1,
        "f1_ci":        ci(f1_scores),
        "roc_auc":      roc_auc,
        "roc_auc_ci":   ci(auc_scores),
        "confusion_matrix": cm,
        "True_Class":   y_true,
        "Predicted_Class": y_pred,
        "Predicted_Score": y_score,
    }

    if model_base is not None:
        result["Predicted_Base_Class"] = np.concatenate(y_pred_ba_list)
        result["Predicted_Base_Score"] = np.concatenate(y_score_ba_list)

    return result


# ---------------------------------------------------------------------------
# Statistical model comparison
# ---------------------------------------------------------------------------

def compare_models(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray,
                   label1: str = "Model 1", label2: str = "Model 2") -> Dict:
    """
    McNemar's test + Cohen's Kappa comparison between two prediction arrays.

    Matches the advisor's compare_models() protocol.
    """
    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)

    # Concordance table: [both wrong, 1 right 2 wrong, 1 wrong 2 right, both right]
    b = int(( correct1 & ~correct2).sum())   # model1 right, model2 wrong
    c = int((~correct1 &  correct2).sum())   # model1 wrong, model2 right

    # McNemar's test
    if _HAS_STATSMODELS:
        table = np.array([[int((~correct1 & ~correct2).sum()), b],
                          [c,                                   int(( correct1 &  correct2).sum())]])
        result_mn = mcnemar(table, exact=False, correction=True)
        mcnemar_p  = float(result_mn.pvalue)
        mcnemar_st = float(result_mn.statistic)
    else:
        # Fallback: hand-compute chi-squared
        if (b + c) > 0:
            mcnemar_st = (abs(b - c) - 1.0) ** 2 / (b + c)
        else:
            mcnemar_st = 0.0
        import scipy.stats as st
        mcnemar_p = float(st.chi2.sf(mcnemar_st, df=1))

    kappa = float(cohen_kappa_score(y_pred1, y_pred2))
    kappa_ci = bootstrap_kappa_ci(y_true, y_pred1, y_pred2)

    acc1 = float(correct1.mean())
    acc2 = float(correct2.mean())
    ci1  = bootstrap_ci(correct1.astype(float))
    ci2  = bootstrap_ci(correct2.astype(float))

    significant = mcnemar_p < 0.05

    print(f"\n{'='*60}")
    print(f"  Statistical Comparison: {label1} vs {label2}")
    print(f"{'='*60}")
    print(f"  {label1:20s}  Acc={acc1:.4f}  95% CI [{ci1[0]:.4f}, {ci1[1]:.4f}]")
    print(f"  {label2:20s}  Acc={acc2:.4f}  95% CI [{ci2[0]:.4f}, {ci2[1]:.4f}]")
    print(f"  Cohen's Kappa         = {kappa:.4f}  95% CI [{kappa_ci[0]:.4f}, {kappa_ci[1]:.4f}]")
    print(f"  McNemar statistic     = {mcnemar_st:.4f}")
    print(f"  McNemar p-value       = {mcnemar_p:.6f}  {'** SIGNIFICANT **' if significant else '(not significant)'}")
    print(f"{'='*60}\n")

    return {
        f"acc_{label1}": acc1, f"acc_ci_{label1}": ci1,
        f"acc_{label2}": acc2, f"acc_ci_{label2}": ci2,
        "kappa": kappa, "kappa_ci": kappa_ci,
        "mcnemar_stat": mcnemar_st, "mcnemar_p": mcnemar_p,
        "significant": significant,
    }


def compare_models_probabilities(
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    test_loader,
    device: torch.device,
    label1: str = "Model 1",
    label2: str = "Model 2",
    alpha: float = 0.05,
) -> Dict:
    """
    Compare two models on their true-class softmax probability.

    Paired t-test + Wilcoxon signed-rank (both must agree for significance —
    conservative criterion matching the advisor's protocol).
    """
    model1.eval(); model2.eval()

    probs1_list, probs2_list, y_true_list = [], [], []
    preds1_list, preds2_list = [], []

    with torch.no_grad():
        for batch in test_loader:
            imgs, labels = batch[0].to(device), batch[1]
            labels = labels.squeeze().long()
            n = len(labels)

            p1 = torch.softmax(model1(imgs), dim=1).cpu().numpy()
            p2 = torch.softmax(model2(imgs), dim=1).cpu().numpy()
            yt = labels.cpu().numpy()

            probs1_list.append(p1[np.arange(n), yt])
            probs2_list.append(p2[np.arange(n), yt])
            preds1_list.append(p1.argmax(axis=1))
            preds2_list.append(p2.argmax(axis=1))
            y_true_list.append(yt)

    probs1  = np.concatenate(probs1_list)
    probs2  = np.concatenate(probs2_list)
    y_true  = np.concatenate(y_true_list)
    y_pred1 = np.concatenate(preds1_list)
    y_pred2 = np.concatenate(preds2_list)

    t_stat, t_p    = stats.ttest_rel(probs1, probs2)
    w_stat, w_p    = stats.wilcoxon(probs1 - probs2)

    # Both must agree for significance (conservative)
    significant = (max(t_p, w_p) < alpha)
    winner = label1 if probs1.mean() > probs2.mean() else label2

    print(f"\n{'='*60}")
    print(f"  Probability Comparison: {label1} vs {label2}")
    print(f"{'='*60}")
    print(f"  {label1:20s}  mean_prob={probs1.mean():.4f} ± {probs1.std():.4f}")
    print(f"  {label2:20s}  mean_prob={probs2.mean():.4f} ± {probs2.std():.4f}")
    print(f"  Paired t-test:    t={t_stat:.4f}, p={t_p:.6f}")
    print(f"  Wilcoxon:         W={w_stat:.4f}, p={w_p:.6f}")
    print(f"  Significant:      {'YES' if significant else 'NO'}  (both tests must agree, α={alpha})")
    if significant:
        print(f"  Winner:           {winner}")
    print(f"{'='*60}\n")

    return {
        f"mean_prob_{label1}": float(probs1.mean()),
        f"std_prob_{label1}": float(probs1.std()),
        f"mean_prob_{label2}": float(probs2.mean()),
        f"std_prob_{label2}": float(probs2.std()),
        "t_stat": float(t_stat), "t_p": float(t_p),
        "w_stat": float(w_stat), "w_p": float(w_p),
        "significant": significant, "winner": winner if significant else "none",
        "True_Class": y_true,
        "Predicted_Class_1": y_pred1,
        "Predicted_Class_2": y_pred2,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_roc_curves(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    prevalence: Optional[Dict[int, float]] = None,
    class_names: Optional[List[str]] = None,
    title: str = "ROC Curves",
    figsize: Tuple = (12, 9),
    ax: Optional[plt.Axes] = None,
) -> None:
    """Per-class OvR ROC curves with prevalence-based decision points."""
    model.eval()
    y_true_list, y_score_list = [], []

    with torch.no_grad():
        for batch in test_loader:
            imgs, labels = batch[0].to(device), batch[1]
            labels = labels.squeeze().long()
            probs  = torch.softmax(model(imgs), dim=1)
            y_true_list.append(labels.cpu().numpy())
            y_score_list.append(probs.cpu().numpy())

    y_true  = np.concatenate(y_true_list)
    y_score = np.concatenate(y_score_list)
    n_cls   = y_score.shape[1]

    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_cls)]
    if prevalence is None:
        prevalence = {i: 1.0 / n_cls for i in range(n_cls)}

    y_onehot = np.eye(n_cls)[y_true]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.get_cmap("tab10")
    for i in range(n_cls):
        fpr, tpr, thresholds = roc_curve(y_onehot[:, i], y_score[:, i])
        roc_auc_val = auc(fpr, tpr)
        color = cmap(i % 10)

        ax.plot(fpr, tpr, lw=2, color=color,
                label=f"{class_names[i]} (AUC={roc_auc_val:.3f})")

        prev = prevalence.get(i, 0.5)
        if len(thresholds) > 0:
            idx = np.argmin(np.abs(thresholds - prev))
            ax.scatter(fpr[idx], tpr[idx], s=100, color=color,
                       edgecolors="black", linewidths=1.5, zorder=5)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    if own_fig:
        plt.tight_layout()
        plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "",
    ax: Optional[plt.Axes] = None,
    cmap: str = "Blues",
    fmt: str = "d",
) -> None:
    """Annotated confusion matrix heatmap."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(max(4, len(class_names)), max(4, len(class_names))))

    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor="gray", cbar=False)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)

    if own_fig:
        plt.tight_layout()
        plt.show()


def plot_all_confusion_matrices(
    results_dict: Dict[str, Dict],
    class_names: List[str],
    dataset: str = "",
    ablation_configs: Optional[List[Dict]] = None,
    config_labels: Optional[Dict[str, str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """2×4 grid of confusion matrices for all 8 configs."""
    if ablation_configs is None:
        ablation_configs = ABLATION_CONFIGS
    if config_labels is None:
        config_labels = CONFIG_LABELS

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"Confusion Matrices — {dataset.upper()}", fontsize=14, fontweight="bold")

    for i, cfg in enumerate(ablation_configs):
        cid = cfg["config_id"]
        ax  = axes[i // 4][i % 4]
        if cid in results_dict:
            cm = results_dict[cid]["confusion_matrix"]
            plot_confusion_matrix(cm, class_names, title=config_labels[cid].replace("\n", " "), ax=ax)
        else:
            ax.set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_metrics_comparison(
    results_dict: Dict[str, Dict], dataset: str = "",
    ablation_configs: Optional[List[Dict]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Bar chart comparing all 8 configs across Accuracy, Macro-F1, and ROC-AUC."""
    if ablation_configs is None:
        ablation_configs = ABLATION_CONFIGS
    cids    = [c["config_id"] for c in ablation_configs if c["config_id"] in results_dict]
    labels  = [c["config_id"] for c in ablation_configs if c["config_id"] in results_dict]
    metrics = {"Accuracy": "accuracy", "Macro F1": "f1", "ROC AUC": "roc_auc"}
    ci_keys = {"Accuracy": "accuracy_ci", "Macro F1": "f1_ci", "ROC AUC": "roc_auc_ci"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Ablation Metrics — {dataset.upper()}", fontsize=14, fontweight="bold")

    colors = ["#d62728" if c == "C8" else "#1f77b4" for c in cids]

    for ax, (metric_name, key) in zip(axes, metrics.items()):
        vals  = [results_dict[c][key]        for c in cids]
        cis   = [results_dict[c][ci_keys[metric_name]] for c in cids]
        errs  = np.array([[v - lo, hi - v] for v, (lo, hi) in zip(vals, cis)]).T

        bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.errorbar(labels, vals, yerr=errs, fmt="none", color="black", capsize=4, linewidth=1.2)
        ax.set_title(metric_name, fontsize=12)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.set_ylim(max(0, min(vals) - 0.05), min(1, max(vals) + 0.05))
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_component_contribution(
    results_dict: Dict[str, Dict], dataset: str = "", dct_axis_label: str = "DCT\nEmbedding",
    save_path: Optional[str] = None,
) -> None:
    """
    Show the marginal contribution of each component (spiral, DCT, Gaussian)
    by computing delta-F1 when each axis is switched on, averaged across the
    other two axes (same method as the PCA ablation notebooks).

    The config_id -> axis-on/off grouping (C2/C5/C6/C8 = "DCT-axis on", etc.)
    is invariant to whether that axis is dct_fast or learned_dct — only
    ``dct_axis_label`` needs to change for the Learned-DCT round (pass
    "Learned-DCT\\nEmbedding").
    """
    f1 = {c["config_id"]: results_dict[c["config_id"]]["f1"]
          for c in ABLATION_CONFIGS if c["config_id"] in results_dict}

    # Spiral: mean(C2,C5,C6,C8) - mean(C1,C3,C4,C7)
    spiral_on  = np.mean([f1.get("C2", np.nan), f1.get("C5", np.nan),
                           f1.get("C6", np.nan), f1.get("C8", np.nan)])
    spiral_off = np.mean([f1.get("C1", np.nan), f1.get("C3", np.nan),
                           f1.get("C4", np.nan), f1.get("C7", np.nan)])
    delta_spiral = float(spiral_on - spiral_off)

    # DCT: mean(C3,C5,C7,C8) - mean(C1,C2,C4,C6)
    dct_on  = np.mean([f1.get("C3", np.nan), f1.get("C5", np.nan),
                        f1.get("C7", np.nan), f1.get("C8", np.nan)])
    dct_off = np.mean([f1.get("C1", np.nan), f1.get("C2", np.nan),
                        f1.get("C4", np.nan), f1.get("C6", np.nan)])
    delta_dct = float(dct_on - dct_off)

    # Gaussian: mean(C4,C6,C7,C8) - mean(C1,C2,C3,C5)
    gauss_on  = np.mean([f1.get("C4", np.nan), f1.get("C6", np.nan),
                          f1.get("C7", np.nan), f1.get("C8", np.nan)])
    gauss_off = np.mean([f1.get("C1", np.nan), f1.get("C2", np.nan),
                          f1.get("C3", np.nan), f1.get("C5", np.nan)])
    delta_gauss = float(gauss_on - gauss_off)

    components = ["Spiral\nOrdering", dct_axis_label, "Gaussian\nBias"]
    deltas     = [delta_spiral, delta_dct, delta_gauss]
    colors     = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(components, deltas, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("ΔMacro-F1", fontsize=11)
    ax.set_title(f"Marginal Component Contribution — {dataset.upper()}", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, deltas):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                y + (0.001 if y >= 0 else -0.003),
                f"{val:+.4f}", ha="center",
                va="bottom" if y >= 0 else "top", fontsize=10, fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    return {"delta_spiral": delta_spiral, "delta_dct": delta_dct, "delta_gauss": delta_gauss}


def print_results_table(
    results_dict: Dict[str, Dict], dataset: str = "",
    ablation_configs: Optional[List[Dict]] = None,
) -> None:
    """Print a formatted results table matching the advisor's output style."""
    if ablation_configs is None:
        ablation_configs = ABLATION_CONFIGS
    print(f"\n{'='*90}")
    print(f"  SPIRAL-VIT Ablation Results — {dataset.upper()}")
    print(f"{'='*90}")
    header = f"  {'Config':<6} {'Acc':>7}  {'Acc CI':>20}  {'F1':>7}  {'F1 CI':>20}  {'AUC':>7}  {'AUC CI':>20}"
    print(header)
    print(f"  {'-'*84}")
    for cfg in ablation_configs:
        cid = cfg["config_id"]
        if cid not in results_dict:
            continue
        r = results_dict[cid]
        print(
            f"  {cid:<6} "
            f"{r['accuracy']:>7.4f}  "
            f"[{r['accuracy_ci'][0]:.4f}, {r['accuracy_ci'][1]:.4f}]  "
            f"{r['f1']:>7.4f}  "
            f"[{r['f1_ci'][0]:.4f}, {r['f1_ci'][1]:.4f}]  "
            f"{r['roc_auc']:>7.4f}  "
            f"[{r['roc_auc_ci'][0]:.4f}, {r['roc_auc_ci'][1]:.4f}]"
        )
    print(f"{'='*90}\n")
