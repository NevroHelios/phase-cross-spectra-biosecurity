"""
Statistical analysis for journal submission.

Runs 20 seeds for each config, saves raw per-seed arrays, then:
  1. Wilcoxon signed-rank test (TestA_i vs TestB_i, paired)
  2. 95% CI on generalisation gap via t-distribution
  3. Full tables with mean ± std for TestA and TestB

Configs covered:
  Codon-level (LR + PCA-128):
    codon_unigram 64d           ← primary
    delta_RSCU 64d
    positional_unigram 192d
    codon_bigram 4096d → PCA-128
    unigram + delta_RSCU 128d
  Nonlinear (XGBoost CUDA, depth=8):
    codon_unigram 64d           ← ceiling test

Saves:
  results/stats_raw_arrays.npz   (all per-seed AUROC arrays)
  results/stats_tables.csv       (mean ± std for all configs)

Usage: uv run python3 exp_stats_analysis.py
"""
from __future__ import annotations
import json, pathlib, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.fft import rfft
from scipy.stats import wilcoxon, t as t_dist
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from pipeline import DataBundle, load_dataset, NEG_CACHE

# ── constants ──────────────────────────────────────────────────────────────────
N_SEEDS     = 20
PCA_K       = 128
C_LR        = 0.1
N_FP_WORK   = 8
GPU_WORKERS = 4
RESULTS     = pathlib.Path("results")

_XGB_D8 = dict(
    device="cuda", tree_method="hist",
    n_estimators=600, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", verbosity=0,
)

# ── codon alphabet ─────────────────────────────────────────────────────────────
_BASES = {'A': 0, 'T': 1, 'G': 2, 'C': 3}

_CODON_TABLE: dict[str, str] = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S','AGT':'S','AGC':'S','CCT':'P','CCC':'P',
    'CCA':'P','CCG':'P','ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A',
    'GCA':'A','GCG':'A','TGT':'C','TGC':'C','TGG':'W','TGA':'*','CGT':'R','CGC':'R',
    'CGA':'R','CGG':'R','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}
_SYN: dict[str, list[str]] = defaultdict(list)
for _c, _a in _CODON_TABLE.items():
    if _a != '*': _SYN[_a].append(_c)

def _cint(c: str) -> int:
    b = _BASES; return b[c[0]]*16 + b[c[1]]*4 + b[c[2]]

def _parse(seq: str) -> list[str]:
    dna = seq.upper()
    return [dna[i:i+3] for i in range(0, (len(dna)//3)*3, 3)
            if all(dna[i+k] in _BASES for k in range(3))]

# ── feature functions ──────────────────────────────────────────────────────────

def codon_unigram_fp(seq: str) -> np.ndarray:
    codons = _parse(seq); v = np.zeros(64, dtype=np.float32)
    for c in codons: v[_cint(c)] += 1
    s = v.sum(); return v / s if s > 0 else v

def _rscu(codons):
    cc: dict[str, int] = defaultdict(int); ac: dict[str, int] = defaultdict(int)
    for c in codons:
        aa = _CODON_TABLE.get(c)
        if aa and aa != '*': cc[c] += 1; ac[aa] += 1
    v = np.ones(64, dtype=np.float32)
    for c, aa in _CODON_TABLE.items():
        if aa == '*': continue
        n = len(_SYN[aa]); fc = cc.get(c,0); faa = ac.get(aa,0)
        if faa > 0: v[_cint(c)] = n * fc / faa
    return v

def compute_genome_rscu(neg_map):
    all_c = []
    for seqs in neg_map.values():
        for s in seqs: all_c.extend(_parse(s))
    return _rscu(all_c)

def delta_rscu_fp(seq: str, genome_rscu: np.ndarray) -> np.ndarray:
    codons = _parse(seq)
    if not codons: return np.zeros(64, dtype=np.float32)
    return (_rscu(codons) - genome_rscu).astype(np.float32)

def positional_unigram_fp(seq: str) -> np.ndarray | None:
    codons = _parse(seq); n = len(codons)
    if n < 9: return None
    t = n // 3; parts = []
    for seg in [codons[:t], codons[t:2*t], codons[2*t:]]:
        v = np.zeros(64, dtype=np.float32)
        for c in seg: v[_cint(c)] += 1
        s = v.sum(); parts.append(v/s if s > 0 else v)
    return np.concatenate(parts)

def codon_bigram_fp(seq: str) -> np.ndarray | None:
    codons = _parse(seq)
    if len(codons) < 2: return None
    v = np.zeros(4096, dtype=np.float32)
    for a, b in zip(codons, codons[1:]): v[_cint(a)*64 + _cint(b)] += 1
    s = v.sum(); return v / s if s > 0 else v

# ── seed runners — return raw per-seed arrays ──────────────────────────────────

def _run_seeds_lr_raw(X_tr, y_tr, cl_tr, X_tb, y_tb, pca_k=PCA_K):
    """Returns (aucs_A, aucs_B) as ordered lists indexed by seed."""
    per_seed: dict[int, tuple] = {}
    lock = threading.Lock()

    def _one(seed):
        rng = np.random.default_rng(seed)
        uc  = np.unique(cl_tr); rng.shuffle(uc)
        msk = np.array([c in set(uc[:max(1, int(len(uc)*.8))]) for c in cl_tr])
        Xtr, ytr = X_tr[msk],  y_tr[msk]
        Xta, yta = X_tr[~msk], y_tr[~msk]
        if len(np.unique(ytr)) < 2 or len(np.unique(yta)) < 2: return
        k  = min(pca_k, Xtr.shape[0]-1, Xtr.shape[1])
        sc = StandardScaler().fit(Xtr)
        Xtr_s = sc.transform(Xtr)
        pca = PCA(n_components=k, random_state=0).fit(Xtr_s)
        def _proj(X): return pca.transform(sc.transform(X))
        lr = LogisticRegression(C=C_LR, max_iter=2000, solver='lbfgs',
                                class_weight='balanced', random_state=0)
        lr.fit(_proj(Xtr), ytr)
        def _metrics(X, y):
            if len(np.unique(y)) < 2: return float('nan'), float('nan')
            prob = lr.predict_proba(_proj(X))[:, 1]
            return roc_auc_score(y, prob), tpr_at_fpr(y, prob)
        aA, tA = _metrics(Xta, yta); aB, tB = _metrics(X_tb, y_tb)
        with lock: per_seed[seed] = (aA, aB, tA, tB)

    with ThreadPoolExecutor(max_workers=N_SEEDS) as ex:
        list(ex.map(_one, range(N_SEEDS)))

    paired_A, paired_B, paired_tA, paired_tB = [], [], [], []
    for s in range(N_SEEDS):
        if s in per_seed:
            a, b, tA, tB = per_seed[s]
            if not any(np.isnan(v) for v in (a, b, tA, tB)):
                paired_A.append(a); paired_B.append(b)
                paired_tA.append(tA); paired_tB.append(tB)
    return (np.array(paired_A), np.array(paired_B),
            np.array(paired_tA), np.array(paired_tB))


def _run_seeds_xgb_raw(X_tr, y_tr, cl_tr, X_tb, y_tb):
    """XGBoost CUDA depth=8, returns paired (aucs_A, aucs_B)."""
    per_seed: dict[int, tuple] = {}
    lock = threading.Lock()

    def _one(seed):
        rng = np.random.default_rng(seed)
        uc  = np.unique(cl_tr); rng.shuffle(uc)
        msk = np.array([c in set(uc[:max(1, int(len(uc)*.8))]) for c in cl_tr])
        Xtr, ytr = X_tr[msk],  y_tr[msk]
        Xta, yta = X_tr[~msk], y_tr[~msk]
        if len(np.unique(ytr)) < 2 or len(np.unique(yta)) < 2: return
        sc  = StandardScaler().fit(Xtr)
        clf = XGBClassifier(**_XGB_D8, random_state=seed)
        clf.fit(sc.transform(Xtr), ytr)
        def _metrics(X, y):
            if len(np.unique(y)) < 2: return float('nan'), float('nan')
            prob = clf.predict_proba(sc.transform(X))[:, 1]
            return roc_auc_score(y, prob), tpr_at_fpr(y, prob)
        aA, tA = _metrics(Xta, yta); aB, tB = _metrics(X_tb, y_tb)
        with lock: per_seed[seed] = (aA, aB, tA, tB)

    with ThreadPoolExecutor(max_workers=GPU_WORKERS) as ex:
        list(ex.map(_one, range(N_SEEDS)))

    paired_A, paired_B, paired_tA, paired_tB = [], [], [], []
    for s in range(N_SEEDS):
        if s in per_seed:
            a, b, tA, tB = per_seed[s]
            if not any(np.isnan(v) for v in (a, b, tA, tB)):
                paired_A.append(a); paired_B.append(b)
                paired_tA.append(tA); paired_tB.append(tB)
    return (np.array(paired_A), np.array(paired_B),
            np.array(paired_tA), np.array(paired_tB))

# ── stats helpers ──────────────────────────────────────────────────────────────

def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr_target: float = 0.01) -> float:
    """TPR at the operating point where FPR ≤ fpr_target (interpolated)."""
    if len(np.unique(y_true)) < 2: return float('nan')
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # find largest fpr index still ≤ fpr_target
    idx = np.searchsorted(fpr, fpr_target, side='right') - 1
    idx = max(0, min(idx, len(tpr) - 1))
    return float(tpr[idx])


def _stats_block(label: str,
                 aA: np.ndarray, aB: np.ndarray,
                 tA: np.ndarray, tB: np.ndarray) -> dict:
    """Compute mean±std, gap CI, Wilcoxon for one config."""
    gap = aA - aB
    n   = len(gap)
    mean_gap = gap.mean(); sem_gap = gap.std(ddof=1) / np.sqrt(n)
    ci_lo, ci_hi = t_dist.interval(0.95, df=n-1, loc=mean_gap, scale=sem_gap)
    W, p = wilcoxon(aA, aB, alternative='two-sided')
    return {
        "config":           label,
        "n_seeds":          n,
        "testA_auroc":      aA.mean(),  "testA_auroc_std":  aA.std(ddof=1),
        "testB_auroc":      aB.mean(),  "testB_auroc_std":  aB.std(ddof=1),
        "testA_tpr1fpr":    tA.mean(),  "testA_tpr1fpr_std": tA.std(ddof=1),
        "testB_tpr1fpr":    tB.mean(),  "testB_tpr1fpr_std": tB.std(ddof=1),
        "gap_mean":         mean_gap,   "gap_std":    gap.std(ddof=1),
        "gap_ci_lo":        ci_lo,      "gap_ci_hi":  ci_hi,
        "wilcoxon_W":       W,          "wilcoxon_p": p,
    }

def _fmt(mean, std): return f"{mean:.4f} ± {std:.4f}"

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS.mkdir(exist_ok=True)
    bundle: DataBundle = load_dataset()
    n_tr = bundle.n_train

    print("\nComputing genome RSCU background ...")
    neg_map     = json.loads(NEG_CACHE.read_text())
    genome_rscu = compute_genome_rscu(neg_map)

    all_seqs = bundle.train_seqs + bundle.tb_seqs
    n_all    = len(all_seqs)

    def _par(fn, desc):
        out = [None] * n_all
        def _w(i): out[i] = fn(i)
        print(f"  {desc} ...")
        with ThreadPoolExecutor(max_workers=N_FP_WORK) as ex:
            list(ex.map(_w, range(n_all)))
        return out

    print("\nPrecomputing features ...")
    UNI   = _par(lambda i: codon_unigram_fp(all_seqs[i]),            "codon_unigram 64d")
    DRSCU = _par(lambda i: delta_rscu_fp(all_seqs[i], genome_rscu),  "delta_RSCU 64d")
    POS   = _par(lambda i: positional_unigram_fp(all_seqs[i]),       "positional_unigram 192d")
    BIG   = _par(lambda i: codon_bigram_fp(all_seqs[i]),             "codon_bigram 4096d")

    def _build(*feat_lists):
        tr, tb, ytr, ytb, cl = [], [], [], [], []
        for i in range(n_tr):
            parts = [fl[i] for fl in feat_lists]
            if all(p is not None for p in parts):
                tr.append(np.concatenate(parts)); ytr.append(bundle.train_labels[i])
                cl.append(bundle.cl_ids[i])
        for j in range(len(bundle.tb_seqs)):
            parts = [fl[n_tr+j] for fl in feat_lists]
            if all(p is not None for p in parts):
                tb.append(np.concatenate(parts)); ytb.append(bundle.tb_labels[j])
        return (np.vstack(tr), np.array(ytr), np.array(cl),
                np.vstack(tb), np.array(ytb))

    CONFIGS_LR = [
        ("codon_unigram 64d [LR]",          (UNI,),        PCA_K),
        ("delta_RSCU 64d [LR]",             (DRSCU,),      PCA_K),
        ("positional_unigram 192d [LR]",    (POS,),        PCA_K),
        ("codon_bigram 4096d→PCA-128 [LR]", (BIG,),        128),
        ("unigram+delta_RSCU 128d [LR]",    (UNI, DRSCU),  PCA_K),
    ]
    CONFIGS_XGB = [
        ("codon_unigram 64d [XGB d=8]",     (UNI,)),
    ]

    raw_arrays: dict[str, np.ndarray] = {}
    stat_rows:  list[dict]            = []

    cw = 38
    print(f"\n{'='*116}")
    print(f"  {'Config':<{cw}}  {'A_AUROC':>15}  {'B_AUROC':>15}"
          f"  {'A_TPR@1%FPR':>13}  {'B_TPR@1%FPR':>13}"
          f"  {'Gap(AUROC)':>12}  {'W':>6}  {'p':>8}")
    print(f"  {'─'*110}")

    def _print_and_record(label, aA, aB, tA, tB, runner_fn=None):
        raw_arrays[f"{label}_A"]  = aA;  raw_arrays[f"{label}_B"]  = aB
        raw_arrays[f"{label}_tA"] = tA;  raw_arrays[f"{label}_tB"] = tB
        r = _stats_block(label, aA, aB, tA, tB)
        stat_rows.append(r)
        print(f"  {label:<{cw}}"
              f"  {_fmt(r['testA_auroc'], r['testA_auroc_std']):>15}"
              f"  {_fmt(r['testB_auroc'], r['testB_auroc_std']):>15}"
              f"  {_fmt(r['testA_tpr1fpr'], r['testA_tpr1fpr_std']):>13}"
              f"  {_fmt(r['testB_tpr1fpr'], r['testB_tpr1fpr_std']):>13}"
              f"  {r['gap_mean']:>+.4f}±{r['gap_std']:.4f}"
              f"  {r['wilcoxon_W']:>6.1f}  {r['wilcoxon_p']:>8.4f}")

    for label, feat_lists, k in CONFIGS_LR:
        X_tr, y_tr, cl_tr, X_tb, y_tb = _build(*feat_lists)
        print(f"  {label:<{cw}}", end="", flush=True)
        aA, aB, tA, tB = _run_seeds_lr_raw(X_tr, y_tr, cl_tr, X_tb, y_tb, pca_k=k)
        _print_and_record(label, aA, aB, tA, tB)

    for label, feat_lists in CONFIGS_XGB:
        X_tr, y_tr, cl_tr, X_tb, y_tb = _build(*feat_lists)
        print(f"  {label:<{cw}}", end="", flush=True)
        aA, aB, tA, tB = _run_seeds_xgb_raw(X_tr, y_tr, cl_tr, X_tb, y_tb)
        _print_and_record(label, aA, aB, tA, tB)

    print(f"{'='*116}")

    # ── wilcoxon detail for primary config ────────────────────────────────────
    key = "codon_unigram 64d [LR]"
    aA  = raw_arrays[f"{key}_A"];  aB  = raw_arrays[f"{key}_B"]
    tA  = raw_arrays[f"{key}_tA"]; tB  = raw_arrays[f"{key}_tB"]
    gap = aA - aB
    n   = len(gap)
    sem = gap.std(ddof=1) / np.sqrt(n)
    ci  = t_dist.interval(0.95, df=n-1, loc=gap.mean(), scale=sem)
    W,  p  = wilcoxon(aA, aB, alternative='two-sided')
    W1, p1 = wilcoxon(aA, aB, alternative='greater')

    print(f"\n{'─'*62}")
    print(f"  Primary config: {key}")
    print(f"  n_seeds          : {n}")
    print(f"  TestA AUROC      : {aA.mean():.4f} ± {aA.std(ddof=1):.4f}  "
          f"[{aA.min():.4f}, {aA.max():.4f}]")
    print(f"  TestB AUROC      : {aB.mean():.4f} ± {aB.std(ddof=1):.4f}  "
          f"[{aB.min():.4f}, {aB.max():.4f}]")
    print(f"  TestA TPR@1%FPR  : {tA.mean():.4f} ± {tA.std(ddof=1):.4f}  "
          f"[{tA.min():.4f}, {tA.max():.4f}]")
    print(f"  TestB TPR@1%FPR  : {tB.mean():.4f} ± {tB.std(ddof=1):.4f}  "
          f"[{tB.min():.4f}, {tB.max():.4f}]")
    print(f"  AUROC gap        : {gap.mean():+.4f} ± {gap.std(ddof=1):.4f}")
    print(f"  95% CI (gap)     : [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    print(f"  Wilcoxon 2-sided : W={W:.1f}  p={p:.4f}"
          + ("  *" if p < 0.05 else "  ns"))
    print(f"  Wilcoxon A>B     : W={W1:.1f}  p={p1:.4f}")
    print(f"{'─'*62}")

    # ── save ──────────────────────────────────────────────────────────────────
    np.savez(RESULTS / "stats_raw_arrays.npz", **raw_arrays)
    pd.DataFrame(stat_rows).to_csv(RESULTS / "stats_tables.csv", index=False)
    print(f"\n  Saved raw arrays : results/stats_raw_arrays.npz")
    print(f"  Saved stats table: results/stats_tables.csv")

    # ── also save individual .npy for quick one-liners ────────────────────────
    uni_key = "codon_unigram 64d [LR]"
    xgb_key = "codon_unigram 64d [XGB d=8]"
    np.save(RESULTS / "testa_aucs_codon_unigram_lr.npy",  raw_arrays[f"{uni_key}_A"])
    np.save(RESULTS / "testb_aucs_codon_unigram_lr.npy",  raw_arrays[f"{uni_key}_B"])
    np.save(RESULTS / "testb_aucs_xgb_best.npy",          raw_arrays[f"{xgb_key}_B"])
    print(f"  Saved .npy files for downstream one-liners")


if __name__ == "__main__":
    main()
