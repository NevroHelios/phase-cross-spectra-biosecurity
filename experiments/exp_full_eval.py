"""
Two-way evaluation framework: Phase+Voss W=110 PCA-128.

┌─────────────────────────────────────────────────────────────┐
│  TRAINING SET  (~80% of non-holdout sequences)              │
│  • VFDB setB, 75 organisms (5 genera held out)              │
│  • Same-strain non-VF CDS negatives                         │
│  • Gene-family-disjoint split via 40% AA linclust           │
├─────────────────────────────────────────────────────────────┤
│  TEST SET A — same-strain holdout (20% gene-family split)   │
│  • Organism-controlled; primary AUC for the paper           │
├─────────────────────────────────────────────────────────────┤
│  TEST SET B — novel pathogen generalization                 │
│  • 5 held-out genera: Chlamydia, Coxiella, Helicobacter,   │
│    Campylobacter, Bordetella                                │
│  • Positive = VF CDS from those genera (VFDB setB)         │
│  • Negative = non-VF CDS from exact same genomes           │
└─────────────────────────────────────────────────────────────┘

Parallelism
-----------
  • Vectorized Voss encoding (numpy broadcasting, no Python loops)
  • ThreadPoolExecutor: fingerprints computed in parallel across seqs
  • All 20 seeds run concurrently (independent train/test splits)

Usage: uv run python3 exp_full_eval.py
"""
from __future__ import annotations
import gzip, re, json, pathlib, subprocess, tempfile, itertools, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from Bio.Seq import Seq
from scipy.fft import rfft
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# ── config ────────────────────────────────────────────────────────────────────

VFDB_NT   = pathlib.Path("data/VFDB_setB_nt.fas.gz")
NEG_CACHE = pathlib.Path("data/same_strain_neg.json")

def _find_mmseqs() -> str:
    import shutil, os
    if p := os.environ.get("MMSEQS_PATH"): return p
    if p := shutil.which("mmseqs"): return p
    for candidate in ["mmseqs/bin/mmseqs", "../mmseqs/bin/mmseqs",
                      "/tmp/mmseqs/bin/mmseqs"]:
        if pathlib.Path(candidate).exists(): return candidate
    raise FileNotFoundError(
        "mmseqs2 not found. Install it or set MMSEQS_PATH=/path/to/mmseqs")

MMSEQS = _find_mmseqs()

HOLDOUT_GENERA = ["Chlamydia", "Coxiella", "Helicobacter", "Campylobacter", "Bordetella"]
MAX_ORGS   = 80
W          = 110     # best from Phase+Voss window sweep
PCA_K      = 128
N_SEEDS    = 20
C_LR       = 0.1
RNG_SEED   = 42
N_FP_WORK  = 8       # parallel threads for fingerprint computation

CHANNELS   = "ATGC"
CH_IDX     = {c: i for i, c in enumerate(CHANNELS)}
PAIRS      = [(i, j) for i in range(4) for j in range(i + 1, 4)]
CH_ARR     = np.array([ord(c) for c in CHANNELS], dtype=np.uint8)

# ── vectorized Voss encoder ───────────────────────────────────────────────────

def _voss_batch(seq: str, W: int) -> np.ndarray | None:
    """
    Returns (n_windows, 4, W) Voss one-hot signals — fully vectorized.
    No Python-level loop over bases.
    """
    step = max(1, W // 2)
    raw  = np.frombuffer(seq.upper().encode("ascii", errors="replace"), dtype=np.uint8)
    n    = (len(raw) - W) // step + 1
    if n <= 0:
        return None
    lut = np.full(256, -1, dtype=np.int8)
    for i, c in enumerate(CHANNELS):
        lut[ord(c)] = i
    arr = lut[raw]                                            # (L,)
    idx = step * np.arange(n)[:, None] + np.arange(W)[None, :]  # (n, W)
    wins = arr[idx]                                            # (n, W)
    # one-hot: (n, 4, W)  — single broadcast, no loop
    sig  = (wins[:, None, :] == np.arange(4)[None, :, None]).astype(np.float32)
    return sig                                                 # (n, 4, W)


# ── fingerprints ──────────────────────────────────────────────────────────────

def phase_fp(sig: np.ndarray) -> np.ndarray:
    """Phase cross-spectral fingerprint from (n_win, 4, W) Voss signal."""
    K     = sig.shape[2] // 2 + 1
    F     = rfft(sig, axis=2)                         # (n, 4, K) complex
    Fn    = F / (np.abs(F) + 1e-10)
    cross = np.zeros((6, K), dtype=np.complex128)
    for idx, (i, j) in enumerate(PAIRS):
        cross[idx] = (Fn[:, i, :] * np.conj(Fn[:, j, :])).mean(axis=0)
    def l1(x):
        s = np.abs(x).sum(); return x / s if s > 0 else x
    return np.concatenate(
        [v for c in cross for v in (l1(np.cos(np.angle(c))), l1(np.sin(np.angle(c))))]
    ).astype(np.float32)


def voss_psd_fp(sig: np.ndarray) -> np.ndarray:
    """Per-channel PSD: 4 × (W//2+1) features."""
    F   = rfft(sig, axis=2)                           # (n, 4, K)
    psd = (np.abs(F) ** 2).mean(axis=0)               # (4, K)
    out = []
    for c in range(4):
        s = psd[c].sum()
        out.append(psd[c] / s if s > 0 else psd[c])
    return np.concatenate(out).astype(np.float32)


def phase_voss_fp(seq: str, W: int = W) -> np.ndarray | None:
    sig = _voss_batch(seq, W)
    if sig is None:
        return None
    return np.concatenate([phase_fp(sig), voss_psd_fp(sig)])


def compute_fps_parallel(seqs: list[str], W: int = W) -> tuple[np.ndarray, list[int]]:
    """Compute fingerprints in parallel; return (X, valid_indices)."""
    results: list[np.ndarray | None] = [None] * len(seqs)

    def _worker(i: int) -> None:
        results[i] = phase_voss_fp(seqs[i], W)

    with ThreadPoolExecutor(max_workers=N_FP_WORK) as ex:
        futs = {ex.submit(_worker, i): i for i in range(len(seqs))}
        for fut in as_completed(futs):
            fut.result()   # propagate exceptions

    valid_idx = [i for i, r in enumerate(results) if r is not None]
    X = np.vstack([results[i] for i in valid_idx])
    return X, valid_idx


# ── NR90 ─────────────────────────────────────────────────────────────────────

def _translate(seq: str) -> str | None:
    s = seq.upper().replace("-", "")
    trim = (len(s) // 3) * 3
    if trim < 90: return None
    try:
        p = str(Seq(s[:trim]).translate(to_stop=True))
        return p if len(p) >= 30 else None
    except Exception: return None


def nr90(seqs: dict[str, str], tmp: pathlib.Path, tag: str) -> dict[str, str]:
    prots = {k: p for k, s in seqs.items() if (p := _translate(s))}
    if not prots: return seqs
    fa = tmp / f"{tag}.faa"
    with open(fa, "w") as f:
        for k, p in prots.items(): f.write(f">{k}\n{p}\n")
    pfx = tmp / f"{tag}_nr90"
    subprocess.run(
        [MMSEQS, "easy-linclust", str(fa), str(pfx), str(tmp / f"{tag}_tmp"),
         "--min-seq-id", "0.90", "-c", "0.80", "--cov-mode", "0",
         "-v", "0", "--threads", "4"],
        capture_output=True,
    )
    tsv = tmp / f"{tag}_nr90_cluster.tsv"
    if not tsv.exists(): return seqs
    df  = pd.read_csv(tsv, sep="\t", header=None, names=["rep", "mem"])
    reps = set(df["rep"].unique())
    return {k: seqs[k] for k in reps if k in seqs}


# ── 40% AA cluster split ──────────────────────────────────────────────────────

def cluster_40(seqs: dict[str, str], tmp: pathlib.Path, tag: str) -> dict[str, str]:
    """Cluster at 40% AA identity; return {seq_key: cluster_rep}."""
    prots = {k: p for k, s in seqs.items() if (p := _translate(s))}
    if not prots: return {k: k for k in seqs}
    fa = tmp / f"{tag}.faa"
    with open(fa, "w") as f:
        for k, p in prots.items(): f.write(f">{k}\n{p}\n")
    pfx = tmp / f"{tag}_cl40"
    subprocess.run(
        [MMSEQS, "easy-linclust", str(fa), str(pfx), str(tmp / f"{tag}_tmp40"),
         "--min-seq-id", "0.40", "-c", "0.80", "--cov-mode", "0",
         "-v", "0", "--threads", "4"],
        capture_output=True,
    )
    tsv = tmp / f"{tag}_cl40_cluster.tsv"
    if not tsv.exists(): return {k: k for k in seqs}
    df = pd.read_csv(tsv, sep="\t", header=None, names=["rep", "mem"])
    m  = dict(zip(df["mem"], df["rep"]))
    return {k: m.get(k, k) for k in seqs}


# ── VFDB parse ────────────────────────────────────────────────────────────────

def parse_vfdb_by_org(fasta_gz: pathlib.Path, max_orgs: int) -> dict[str, list[str]]:
    hdr_re = re.compile(r'^>(\S+)\(gb\|([^)]+)\).*\[([^\[\]]+)\]\s*$')
    org_seqs: dict[str, list[str]] = defaultdict(list)
    cur_hdr: str | None = None; cur_seq: list[str] = []

    def _emit():
        nonlocal cur_hdr, cur_seq
        if cur_hdr:
            m = hdr_re.match(cur_hdr)
            if m:
                _, _, org = m.groups()
                s = "".join(cur_seq).upper()
                if len(s) >= W: org_seqs[org].append(s)
        cur_hdr = None; cur_seq.clear()

    with gzip.open(fasta_gz, "rt") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"): _emit(); cur_hdr = line
            else: cur_seq.append(line)
    _emit()
    top = sorted(org_seqs.items(), key=lambda x: len(x[1]), reverse=True)[:max_orgs]
    return dict(top)


# ── seed evaluation ───────────────────────────────────────────────────────────

def _one_seed(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    cluster_ids_tr: np.ndarray,
    seed: int,
) -> dict[str, float]:
    """
    Gene-family split on training pool, train model, return AUROCs.
    X_te / y_te here is Test Set A (gene-family holdout within train orgs).
    cluster_ids_tr : cluster ID per training-pool sample.
    """
    rng = np.random.default_rng(seed)
    unique_cl = np.unique(cluster_ids_tr)
    rng.shuffle(unique_cl)
    n_tr = max(1, int(len(unique_cl) * 0.8))
    tr_cl = set(unique_cl[:n_tr])

    tr_mask = np.array([c in tr_cl for c in cluster_ids_tr])
    te_mask  = ~tr_mask

    Xtr, ytr = X_tr[tr_mask], y_tr[tr_mask]
    Xta, yta = X_tr[te_mask], y_tr[te_mask]

    if len(np.unique(ytr)) < 2 or len(np.unique(yta)) < 2:
        return {}

    k   = min(PCA_K, Xtr.shape[0] - 1, Xtr.shape[1])
    sc  = StandardScaler().fit(Xtr)
    pca = PCA(n_components=k, random_state=0).fit(sc.transform(Xtr))
    lr  = LogisticRegression(C=C_LR, max_iter=2000, solver="lbfgs",
                             class_weight="balanced", random_state=0)
    lr.fit(pca.transform(sc.transform(Xtr)), ytr)

    def _scores(X, y):
        if len(np.unique(y)) < 2: return float("nan"), float("nan")
        prob = lr.predict_proba(pca.transform(sc.transform(X)))[:, 1]
        pred = (prob >= 0.5).astype(int)
        return roc_auc_score(y, prob), f1_score(y, pred, zero_division=0)

    aA, fA = _scores(Xta, yta)
    aB, fB = _scores(X_te, y_te)
    return {"A": aA, "f1A": fA, "B": aB, "f1B": fB}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print(f"Full 3-way evaluation  |  Phase+Voss W={W}  PCA-{PCA_K}")
    print(f"Holdout genera (Test B): {', '.join(HOLDOUT_GENERA)}")
    print("=" * 70)

    # 1. Parse VFDB by organism
    print("\nParsing VFDB_setB_nt.fas.gz ...")
    org_pos = parse_vfdb_by_org(VFDB_NT, MAX_ORGS)
    holdout_orgs = {o for o in org_pos if any(o.startswith(g) for g in HOLDOUT_GENERA)}
    train_orgs   = set(org_pos.keys()) - holdout_orgs
    print(f"  Train organisms: {len(train_orgs)}  "
          f"Holdout organisms: {len(holdout_orgs)} ({', '.join(sorted(holdout_orgs))})")

    # 2. Load negatives
    print("Loading negative cache ...")
    neg_map: dict[str, list[str]] = json.loads(NEG_CACHE.read_text())

    # 3. Build pools (raw, before NR90)
    def _pool(orgs, pos_src, neg_src, prefix):
        p, n = {}, {}
        pi = ni = 0
        for o in orgs:
            for s in pos_src.get(o, []):
                p[f"{prefix}p{pi}"] = s; pi += 1
            for s in neg_src.get(o, []):
                n[f"{prefix}n{ni}"] = s; ni += 1
        return p, n

    tr_pos_raw, tr_neg_raw = _pool(train_orgs,   org_pos, neg_map, "tr_")
    tb_pos_raw, tb_neg_raw = _pool(holdout_orgs, org_pos, neg_map, "tb_")
    print(f"  Train pool raw:  pos={len(tr_pos_raw)}  neg={len(tr_neg_raw)}")
    print(f"  TestB pool raw:  pos={len(tb_pos_raw)}  neg={len(tb_neg_raw)}")

    # 4. NR90 each pool
    print("\nNR90 (4 pools in parallel) ...")
    nr90_results: dict[str, dict[str, str]] = {}
    nr90_lock = threading.Lock()

    def _nr90_worker(seqs, tag):
        with tempfile.TemporaryDirectory() as tmp:
            result = nr90(seqs, pathlib.Path(tmp), tag)
        with nr90_lock:
            nr90_results[tag] = result

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [
            ex.submit(_nr90_worker, tr_pos_raw, "tr_pos"),
            ex.submit(_nr90_worker, tr_neg_raw, "tr_neg"),
            ex.submit(_nr90_worker, tb_pos_raw, "tb_pos"),
            ex.submit(_nr90_worker, tb_neg_raw, "tb_neg"),
        ]
        for f in as_completed(futs): f.result()

    tr_pos = nr90_results["tr_pos"]; tr_neg = nr90_results["tr_neg"]
    tb_pos = nr90_results["tb_pos"]; tb_neg = nr90_results["tb_neg"]

    # Balance negatives in training pool
    n_pos_tr = len(tr_pos)
    if len(tr_neg) > n_pos_tr:
        keys = resample(list(tr_neg.keys()), n_samples=n_pos_tr,
                        replace=False, random_state=RNG_SEED)
        tr_neg = {k: tr_neg[k] for k in keys}

    # Balance TestB
    n_pos_tb = len(tb_pos)
    if len(tb_neg) > n_pos_tb:
        keys = resample(list(tb_neg.keys()), n_samples=n_pos_tb,
                        replace=False, random_state=RNG_SEED)
        tb_neg = {k: tb_neg[k] for k in keys}

    print(f"  After NR90+balance:  Train pos={len(tr_pos)}  neg={len(tr_neg)}")
    print(f"                       TestB pos={len(tb_pos)}  neg={len(tb_neg)}")

    # 5. Gene-family cluster at 40% AA on training pool
    print("\nClustering training pool at 40% AA for gene-family split ...")
    all_train = {**tr_pos, **tr_neg}
    train_labels_dict = {k: 1 for k in tr_pos} | {k: 0 for k in tr_neg}
    with tempfile.TemporaryDirectory() as tmp:
        cl_map = cluster_40(all_train, pathlib.Path(tmp), "train")
    # Map cluster rep strings → integer IDs
    cl_strs  = [cl_map[k] for k in all_train]
    uniq_cls = {s: i for i, s in enumerate(sorted(set(cl_strs)))}
    cl_ids   = np.array([uniq_cls[s] for s in cl_strs])
    print(f"  Unique gene families: {len(uniq_cls)}")

    train_seqs_list  = list(all_train.values())
    train_labels_arr = np.array([train_labels_dict[k] for k in all_train])

    # 6. Compute fingerprints (parallel)
    print(f"\nComputing Phase+Voss fingerprints (W={W}, {N_FP_WORK} threads) ...")

    tb_seqs = list(tb_pos.values()) + list(tb_neg.values())
    tb_labs = np.array([1] * len(tb_pos) + [0] * len(tb_neg))

    # Compute train + TestB fingerprints in parallel across all sequences
    all_seqs_combined = train_seqs_list + tb_seqs
    X_all, valid_idx = compute_fps_parallel(all_seqs_combined, W)

    n_train = len(train_seqs_list)
    tr_valid = [i for i in valid_idx if i < n_train]
    tb_valid = [i - n_train for i in valid_idx if i >= n_train]

    X_tr_full = X_all[[i for i, vi in enumerate(valid_idx) if vi < n_train]]
    y_tr_full = train_labels_arr[[vi for vi in valid_idx if vi < n_train]]
    cl_tr_full = cl_ids[[vi for vi in valid_idx if vi < n_train]]

    X_tb = X_all[[i for i, vi in enumerate(valid_idx) if vi >= n_train]]
    y_tb = tb_labs[[vi - n_train for vi in valid_idx if vi >= n_train]]

    print(f"  Train pool: {X_tr_full.shape}  TestB: {X_tb.shape}  feat_dim={X_all.shape[1]}")

    # 7. Run 20 seeds in parallel
    print(f"\nRunning {N_SEEDS} seeds (concurrent) ...")

    seed_results: list[dict[str, float]] = [{}] * N_SEEDS
    res_lock = threading.Lock()

    def _seed_worker(seed: int) -> None:
        r = _one_seed(X_tr_full, y_tr_full, X_tb, y_tb, cl_tr_full, seed)
        with res_lock:
            seed_results[seed] = r

    with ThreadPoolExecutor(max_workers=N_SEEDS) as ex:
        futs = {ex.submit(_seed_worker, s): s for s in range(N_SEEDS)}
        for fut in as_completed(futs):
            s = futs[fut]
            fut.result()
            print(f"  seed {s:>2} done", flush=True)

    # 9. Aggregate results
    def _agg(key: str) -> tuple[float, float]:
        vals = [r[key] for r in seed_results if key in r and not np.isnan(r.get(key, float("nan")))]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))

    auc_A, std_A = _agg("A")
    auc_B, std_B = _agg("B")
    f1_A,  sf1_A = _agg("f1A")
    f1_B,  sf1_B = _agg("f1B")

    print(f"\n{'='*70}")
    print(f"  Phase+Voss W={W}  PCA-{PCA_K}  |  {N_SEEDS} seeds  |  same-strain control")
    print(f"  {'-'*65}")
    print(f"  Test A  same-strain holdout  (20% gene-families)  : AUROC {auc_A:.4f} ± {std_A:.4f}  F1 {f1_A:.4f} ± {sf1_A:.4f}")
    print(f"  Test B  novel pathogen genera ({len(holdout_orgs)} genera held out) : AUROC {auc_B:.4f} ± {std_B:.4f}  F1 {f1_B:.4f} ± {sf1_B:.4f}")
    print(f"{'='*70}")

    # Save
    pathlib.Path("results").mkdir(exist_ok=True)
    rows = [
        {"test_set": "A_same_strain_holdout",
         "auroc_mean": auc_A, "auroc_std": std_A, "f1_mean": f1_A, "f1_std": sf1_A,
         "description": "Gene-family holdout, same organisms"},
        {"test_set": "B_novel_genera",
         "auroc_mean": auc_B, "auroc_std": std_B, "f1_mean": f1_B, "f1_std": sf1_B,
         "description": f"Organism holdout: {', '.join(HOLDOUT_GENERA)}"},
    ]
    seed_rows = [{"seed": s, **r} for s, r in enumerate(seed_results)]
    pd.DataFrame(rows).to_csv("results/full_eval_summary.csv", index=False)
    pd.DataFrame(seed_rows).to_csv("results/full_eval_seeds.csv", index=False)
    print(f"\n  Saved: results/full_eval_summary.csv  full_eval_seeds.csv")


if __name__ == "__main__":
    main()
