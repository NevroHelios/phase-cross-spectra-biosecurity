"""
Reproduce Table 1: Phase cross-spectral fingerprint vs Voss baseline.

Usage:
    python experiments/run_main.py --input data/sequences.csv --w 90 --pca 64
    python experiments/run_main.py --input data/sequences.csv --w 420 --pca 128

Input CSV columns: accession, sequence, label, cluster_id
Optional: --backtranslated data/backtranslated.txt  (to exclude synthetic seqs)
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
from src.baseline import voss_power_fingerprint
from src.evaluate import cluster_aware_eval, bootstrap_ci, wilcoxon_greater, wilcoxon_full
from src.utils import load_dataset


def _compute_fp(args: tuple) -> tuple[str, np.ndarray | None, np.ndarray | None]:
    acc, seq, W = args
    return (
        acc,
        phase_cross_spectral_fingerprint(seq, W),
        voss_power_fingerprint(seq, W),
    )


def run(
    csv_path: str,
    W: int,
    pca_components: int,
    n_seeds: int = 20,
    C: float = 0.1,
    backtranslated_path: str | None = None,
    output_csv: str | None = None,
    n_workers: int | None = None,
) -> None:
    import multiprocessing
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)

    print(f"Loading data from {csv_path} ...")
    accs, seqs, labels, cluster_ids, backtranslated = load_dataset(
        csv_path, backtranslated_path
    )

    # Filter back-translated if flag provided
    if backtranslated_path and backtranslated:
        mask = np.array([a not in backtranslated for a in accs])
        accs        = [a for a, m in zip(accs, mask) if m]
        seqs        = [s for s, m in zip(seqs, mask) if m]
        labels      = labels[mask]
        cluster_ids = cluster_ids[mask]
        print(f"  Excluded {(~mask).sum()} back-translated sequences.")

    print(f"Computing fingerprints: W={W}, n={len(accs)}, workers={n_workers} ...")
    tasks = [(acc, seq, W) for acc, seq in zip(accs, seqs)]

    phase_fps: dict[str, np.ndarray] = {}
    voss_fps:  dict[str, np.ndarray] = {}

    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        for acc, fp_ph, fp_vo in exe.map(_compute_fp, tasks):
            if fp_ph is not None:
                phase_fps[acc] = fp_ph
            if fp_vo is not None:
                voss_fps[acc] = fp_vo

    # Intersect: keep only sequences that have both fingerprints
    valid = [a for a in accs if a in phase_fps and a in voss_fps]
    idx   = {a: i for i, a in enumerate(accs)}
    valid_idx    = np.array([idx[a] for a in valid])
    fps_phase    = np.vstack([phase_fps[a] for a in valid])
    fps_voss     = np.vstack([voss_fps[a]  for a in valid])
    labs_valid   = labels[valid_idx]
    cids_valid   = cluster_ids[valid_idx]
    n_clusters   = len(np.unique(cids_valid))

    print(f"\nW={W} | n={len(valid)} | clusters={n_clusters} | "
          f"danger={(labs_valid==1).sum()} | benign={(labs_valid==0).sum()}")

    # ── Voss baseline ────────────────────────────────────────────────────────
    print("\n--- Voss power spectrum baseline ---")
    voss_aucs = cluster_aware_eval(fps_voss, labs_valid, cids_valid,
                                   n_seeds=n_seeds, pca_components=pca_components,
                                   C=C)
    v_mean, v_std = np.mean(voss_aucs), np.std(voss_aucs)
    v_lo, v_hi   = bootstrap_ci(voss_aucs)
    print(f"  Mean ± Std : {v_mean:.4f} ± {v_std:.4f}")
    print(f"  Bootstrap 95% CI: [{v_lo:.4f}, {v_hi:.4f}]")

    # ── Phase cross-spectral ─────────────────────────────────────────────────
    print("\n--- Phase cross-spectral + PCA + LR ---")
    phase_aucs = cluster_aware_eval(fps_phase, labs_valid, cids_valid,
                                    n_seeds=n_seeds, pca_components=pca_components,
                                    C=C)
    p_mean, p_std = np.mean(phase_aucs), np.std(phase_aucs)
    p_lo, p_hi    = bootstrap_ci(phase_aucs)
    print(f"  Mean ± Std : {p_mean:.4f} ± {p_std:.4f}")
    print(f"  Bootstrap 95% CI: [{p_lo:.4f}, {p_hi:.4f}]")

    # ── Wilcoxon test ────────────────────────────────────────────────────────
    n_min = min(len(phase_aucs), len(voss_aucs))
    wx = wilcoxon_full(phase_aucs[:n_min], voss_aucs[:n_min],
                       label=f"phase > Voss W={W}")
    if wx["p"] < 0.01:
        print(f"  → p < 0.01: phase cross-spectra SIGNIFICANTLY outperforms Voss")

    # ── Save results ─────────────────────────────────────────────────────────
    if output_csv is None:
        output_csv = f"results/main_W{W}_pca{pca_components}.csv"
    pathlib.Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed, (pa, va) in enumerate(zip(phase_aucs, voss_aucs)):
        rows.append({"seed": seed, "phase_auc": round(pa, 6),
                     "voss_auc": round(va, 6)})
    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_csv, index=False)
    print(f"\nPer-seed results saved to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce Table 1: phase cross-spectra vs Voss baseline."
    )
    parser.add_argument("--input",          required=True,
                        help="Path to input CSV (accession,sequence,label,cluster_id)")
    parser.add_argument("--w",              type=int, default=90,
                        help="Window size in bp (default: 90)")
    parser.add_argument("--pca",            type=int, default=64,
                        help="PCA components (default: 64; use 128 for W=420)")
    parser.add_argument("--seeds",          type=int, default=20)
    parser.add_argument("--C",              type=float, default=0.1)
    parser.add_argument("--backtranslated", default=None,
                        help="Path to text file listing back-translated accessions")
    parser.add_argument("--output",         default=None,
                        help="Output CSV path for per-seed results")
    parser.add_argument("--workers",        type=int, default=None)
    args = parser.parse_args()

    run(
        csv_path=args.input,
        W=args.w,
        pca_components=args.pca,
        n_seeds=args.seeds,
        C=args.C,
        backtranslated_path=args.backtranslated,
        output_csv=args.output,
        n_workers=args.workers,
    )


if __name__ == "__main__":
    main()
