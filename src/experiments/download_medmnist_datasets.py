"""
src/experiments/download_medmnist_datasets.py
================================================
Pre-downloads MedMNIST dataset files (train/val/test splits, at a given
native resolution) with no training involved.

Run this once, ahead of a long overnight ablation sweep, so every dataset
the sweep needs is already cached locally -- a training crash or watchdog
restart mid-sweep then never blocks on a network download again, it just
resumes training against data that's already on disk.

medmnist's own download mechanism is idempotent (it skips re-downloading a
file that's already present and valid), so re-running this script after a
partial previous run only fetches what's still missing -- safe to run
again if it was interrupted.

Usage
-----
    python src/experiments/download_medmnist_datasets.py --datasets chest,oct,tissue,organa,organc,organs,path
    python src/experiments/download_medmnist_datasets.py --datasets path --native_size 224
"""

from __future__ import annotations

import argparse
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.medmnist_dataset import MEDMNIST_INFO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--datasets", type=str, required=True,
        help="Comma-separated dataset keys, e.g. chest,oct,tissue,organa,organc,organs,path",
    )
    p.add_argument("--data_root", type=str, default="data/MedMnist_fullres")
    p.add_argument("--native_size", type=int, default=224, choices=[28, 64, 128, 224])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_keys = [d.strip() for d in args.datasets.split(",") if d.strip()]

    unknown = [d for d in dataset_keys if d not in MEDMNIST_INFO]
    if unknown:
        raise ValueError(f"Unknown dataset(s): {unknown}. Choose from: {sorted(MEDMNIST_INFO)}")

    print(f"Downloading {len(dataset_keys)} dataset(s) at native_size={args.native_size} into {args.data_root}")
    print(f"Datasets: {', '.join(dataset_keys)}\n")

    os.makedirs(args.data_root, exist_ok=True)

    for i, key in enumerate(dataset_keys, 1):
        MedDS, _ = MEDMNIST_INFO[key]
        print(f"[{i}/{len(dataset_keys)}] {key} ...", flush=True)
        for split in ("train", "val", "test"):
            MedDS(split=split, root=args.data_root, download=True, size=args.native_size)
        print(f"[{i}/{len(dataset_keys)}] {key} -- OK (cached in {args.data_root})")

    print("\nAll requested datasets are downloaded and cached.")


if __name__ == "__main__":
    main()
