"""
Reproduce Table 2 ablation experiments.

  Exp A: Phase coherence (Welch squared) vs raw phase angle — W=90, W=420
  Exp B: Multi-scale concatenation (W=90 PCA-32 + W=420 PCA-32 → LR)

Usage:
    python experiments/run_ablation.py --input data/sequences.csv
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.fft import rfft
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.encoding import voss_encode
from src.fingerprint import phase_cross_spectral_fingerprint
from src.evaluate import _get_split, bootstrap_ci, wilcoxon_full
from src.utils import load_dataset

PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]


def _coherence_fingerprint(sequence: str, W: int) -> np.ndarray | None:
    """
    Welch squared coherence fingerprint (ablation — discards phase sign).

    C_ij(k) = |Σ F_i · conj(F_j)|² / (Σ|F_i|² · Σ|F_j|²)

    Returns a real-valued vector of dimension 6 × (W//2 + 1).
    """
    dna = sequence.upper()
    if len(dna) < W:
        return None
    step = W // 2
    K = W // 2 + 1
    cross_sum = np.zeros((6, K), dtype=np.complex128)
    power_sum = np.zeros((4, K), dtype=np.float64)
    n = 0
    for s in range(0, len(dna) - W + 1, step):
        sig = voss_encode(dna[s:s + W])
        F = rfft(sig, axis=1)
        power_sum += np.abs(F) ** 2
        for idx, (i, j) in enumerate(PAIRS):
            cross_sum[idx] += F[i] * np.conj(F[j])
        n += 1
    if n == 0:
        return None
    coh = np.zeros((6, K), dtype=np.float32)
    for idx, (i, j) in enumerate(PAIRS):
        num = np.abs(cross_sum[idx]) ** 2
        den = power_sum[i] * power_sum[j]
        coh[idx] = np.where(den > 0, num / den, 0.0)
    return coh.flatten()


def _lr_multiscale(
    X90_tr, X90_te, X420_tr, X420_te,
    y_tr, y_te,
    pca_dims: int = 32,
    C: float = 0.1,
) -> float:
    """Fit independent PCA on each scale, concatenate, train LR."""
    parts_tr, parts_te = [], []
    for Xtr, Xte in [(X90_tr, X90_te), (X420_tr, X420_te)]:
        n_comp = min(pca_dims, Xtr.shape[0] - 1, Xtr.shape[1])
        sc  = StandardScaler().fit(Xtr)
        pca = PCA(n_components=n_comp).fit(sc.transform(Xtr))
        parts_tr.append(pca.transform(sc.transform(Xtr)))
        parts_te.append(pca.transform(sc.transform(Xte)))
    X_tr_cat = np.hstack(parts_tr)
    X_te_cat = np.hstack(parts_te)
    lr = LogisticRegression(C=C, max_iter=1000, solver="lbfgs",
                             class_weight="balanced")
    lr.fit(X_tr_cat, y_tr)
    return roc_auc_score(y_te, lr.predict_proba(X_te_cat)[:, 1])


def _worker(args):
    acc, seq, W = args
    return acc, W, phase_cross_spectral_fingerprint(seq, W), _coherence_fingerprint(seq, W)


def run(csv_path: str, n_seeds: int = 20, C: float = 0.1) -> None:
    print(f"Loading data from {csv_path} ...")
    accs, seqs, labels, cluster_ids, _ = load_dataset(csv_path)

    tasks = [(a, s, W) for a, s in zip(accs, seqs) for W in [90, 420]]
    print(f"Computing fingerprints — {len(tasks)} tasks ...")

    import multiprocessing
    n_workers = max(1, multiprocessing.cpu_count() - 1)
    res: dict[tuple[str, int], tuple] = {}
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        for acc, W, fp_ph, fp_coh in exe.map(_worker, tasks):
            res[(acc, W)] = (fp_ph, fp_coh)

    # ── Exp A: coherence vs raw phase ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXP A — Phase coherence vs raw phase angle")
    print("=" * 70)

    for W in [90, 420]:
        valid = [a for a in accs
                 if res.get((a, W), (None, None))[0] is not None
                 and res[(a, W)][1] is not None]
        idx_map = {a: i for i, a in enumerate(accs)}
        vidx    = np.array([idx_map[a] for a in valid])
        fps_ph  = np.vstack([res[(a, W)][0] for a in valid])
        fps_coh = np.vstack([res[(a, W)][1] for a in valid])
        labs    = labels[vidx]
        cids    = cluster_ids[vidx]

        ph_aucs, coh_aucs = [], []
        for seed in range(n_seeds):
            tr, te = _get_split(cids, seed)
            if len(np.unique(labs[te])) < 2:
                continue
            from scipy.spatial.distance import cdist
            def dist_auc(fps):
                D = cdist(fps[te], fps[tr], metric="cosine")
                sc = np.array([D[i, labs[tr]==0].mean() - D[i, labs[tr]==1].mean()
                               for i in range(len(te))])
                return roc_auc_score(labs[te], sc)
            ph  = dist_auc(fps_ph)
            coh = dist_auc(fps_coh)
            ph_aucs.append(ph)
            coh_aucs.append(coh)
            print(f"  [W={W}] seed={seed:>2} phase={ph:.4f} coherence={coh:.4f}")

        pm, ps = np.mean(ph_aucs), np.std(ph_aucs)
        cm, cs = np.mean(coh_aucs), np.std(coh_aucs)
        lo, hi = bootstrap_ci(coh_aucs)
        print(f"\n  W={W} raw phase    : {pm:.4f} ± {ps:.4f}")
        print(f"  W={W} coherence    : {cm:.4f} ± {cs:.4f}  "
              f"Δ = {cm-pm:+.4f}  CI=[{lo:.4f},{hi:.4f}]")
        if len(ph_aucs) >= 5:
            wilcoxon_full(coh_aucs, ph_aucs,
                          label=f"coherence > raw phase W={W}")
        print()

    # ── Exp B: multi-scale ───────────────────────────────────────────────────
    print("=" * 70)
    print("EXP B — Multi-scale (W=90 PCA-32 + W=420 PCA-32 → LR)")
    print("=" * 70)

    both = [a for a in accs
            if res.get((a, 90), (None,))[0] is not None
            and res.get((a, 420), (None,))[0] is not None]
    idx_map = {a: i for i, a in enumerate(accs)}
    bidx    = np.array([idx_map[a] for a in both])
    fps_90  = np.vstack([res[(a, 90)][0]  for a in both])
    fps_420 = np.vstack([res[(a, 420)][0] for a in both])
    labs_b  = labels[bidx]
    cids_b  = cluster_ids[bidx]

    ms_aucs = []
    for seed in range(n_seeds):
        tr, te = _get_split(cids_b, seed)
        if len(np.unique(labs_b[te])) < 2:
            continue
        auc = _lr_multiscale(fps_90[tr], fps_90[te],
                             fps_420[tr], fps_420[te],
                             labs_b[tr], labs_b[te], pca_dims=32, C=C)
        ms_aucs.append(auc)
        print(f"  [multiscale] seed={seed:>2} auc={auc:.4f}")

    mm, ms = np.mean(ms_aucs), np.std(ms_aucs)
    lo, hi = bootstrap_ci(ms_aucs)
    print(f"\n  Multi-scale: {mm:.4f} ± {ms:.4f}  Bootstrap CI=[{lo:.4f},{hi:.4f}]")

    pathlib.Path("results").mkdir(exist_ok=True)
    pd.DataFrame({"seed": range(len(ms_aucs)), "multiscale_auc": ms_aucs}).to_csv(
        "results/ablation_multiscale.csv", index=False
    )
    print("Saved results/ablation_multiscale.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation experiments.")
    parser.add_argument("--input",  required=True)
    parser.add_argument("--seeds",  type=int,   default=20)
    parser.add_argument("--C",      type=float, default=0.1)
    args = parser.parse_args()
    run(args.input, args.seeds, args.C)


if __name__ == "__main__":
    main()
