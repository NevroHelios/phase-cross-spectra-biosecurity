"""
Sensitivity analysis: effect of including back-translated sequences.

Runs the main phase cross-spectral experiment twice:
  (A) Real CDS only       — excludes back-translated sequences
  (B) All sequences       — includes back-translated synthetic seqs

Prints the AUC difference and saves comparison CSV.

Usage:
    python experiments/run_backtranslation.py \\
        --input data/sequences.csv \\
        --backtranslated data/backtranslated.txt \\
        --w 90 --pca 64
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.fingerprint import phase_cross_spectral_fingerprint
from src.evaluate import cluster_aware_eval, bootstrap_ci
from src.utils import load_dataset


def _worker(args):
    acc, seq, W = args
    return acc, phase_cross_spectral_fingerprint(seq, W)


def run_subset(
    accs, seqs, labels, cluster_ids,
    W: int, pca_components: int, n_seeds: int, C: float, label: str,
    n_workers: int,
) -> list[float]:
    tasks = [(a, s, W) for a, s in zip(accs, seqs)]
    fps_map: dict[str, np.ndarray] = {}
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        for acc, fp in exe.map(_worker, tasks):
            if fp is not None:
                fps_map[acc] = fp

    valid    = [a for a in accs if a in fps_map]
    idx_map  = {a: i for i, a in enumerate(accs)}
    vidx     = np.array([idx_map[a] for a in valid])
    fps      = np.vstack([fps_map[a] for a in valid])
    labs     = labels[vidx]
    cids     = cluster_ids[vidx]
    n_clust  = len(np.unique(cids))

    print(f"\n[{label}] n={len(valid)} | clusters={n_clust} | "
          f"danger={(labs==1).sum()} | benign={(labs==0).sum()}")

    aucs = cluster_aware_eval(fps, labs, cids, n_seeds=n_seeds,
                               pca_components=pca_components, C=C)
    mean, std = np.mean(aucs), np.std(aucs)
    lo, hi    = bootstrap_ci(aucs)
    print(f"  Mean ± Std: {mean:.4f} ± {std:.4f}  CI=[{lo:.4f},{hi:.4f}]")
    return aucs


def run(
    csv_path: str,
    backtranslated_path: str,
    W: int,
    pca_components: int,
    n_seeds: int = 20,
    C: float = 0.1,
) -> None:
    import multiprocessing
    n_workers = max(1, multiprocessing.cpu_count() - 1)

    accs, seqs, labels, cluster_ids, backtranslated = load_dataset(
        csv_path, backtranslated_path
    )
    print(f"Total: {len(accs)} sequences, {len(backtranslated)} back-translated")

    # (A) real CDS only
    mask_real = np.array([a not in backtranslated for a in accs])
    aucs_real = run_subset(
        [a for a, m in zip(accs, mask_real) if m],
        [s for s, m in zip(seqs, mask_real) if m],
        labels[mask_real], cluster_ids[mask_real],
        W, pca_components, n_seeds, C,
        label="REAL CDS only", n_workers=n_workers,
    )

    # (B) all sequences
    aucs_all = run_subset(
        accs, seqs, labels, cluster_ids,
        W, pca_components, n_seeds, C,
        label="ALL (incl back-translated)", n_workers=n_workers,
    )

    delta = np.mean(aucs_real) - np.mean(aucs_all)
    print(f"\nΔ (real − all): {delta:+.4f}  "
          f"(back-translation depresses AUC by {abs(delta):.3f})")

    n = min(len(aucs_real), len(aucs_all))
    df = pd.DataFrame({
        "seed":     range(n),
        "real_cds": aucs_real[:n],
        "all_seqs": aucs_all[:n],
        "delta":    [r - a for r, a in zip(aucs_real[:n], aucs_all[:n])],
    })
    out = f"results/backtranslation_W{W}.csv"
    pathlib.Path("results").mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Back-translation sensitivity analysis."
    )
    parser.add_argument("--input",          required=True)
    parser.add_argument("--backtranslated", required=True)
    parser.add_argument("--w",              type=int,   default=90)
    parser.add_argument("--pca",            type=int,   default=64)
    parser.add_argument("--seeds",          type=int,   default=20)
    parser.add_argument("--C",              type=float, default=0.1)
    args = parser.parse_args()
    run(args.input, args.backtranslated, args.w, args.pca, args.seeds, args.C)


if __name__ == "__main__":
    main()
