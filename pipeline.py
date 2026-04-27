"""
Data pipeline — load once, reuse forever.

Exports:
    load_dataset()  →  DataBundle
    build_features(bundle, feature_cfg) → (X_tr, y_tr, cl_tr, X_tb, y_tb)

DataBundle holds:
  - raw NR90-deduplicated sequences (train / testB)
  - cluster IDs (40% AA gene-family split)
  - precomputed feature arrays: cp, ac, cpc_nmf, fg, ph
  - genome background dicts (for foreignness)

feature_cfg is a list of strings from:
  {"cp", "ac", "cpc_nmf", "fg", "ph"}

Usage:
    from pipeline import load_dataset, build_features
    bundle = load_dataset()                    # ~2–3 min, run once
    X_tr, y_tr, cl_tr, X_tb, y_tb = build_features(bundle, ["cp","ac","cpc_nmf","fg"])
"""
from __future__ import annotations
import gzip, re, json, pathlib, pickle, subprocess, tempfile, threading, itertools
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from Bio.Seq import Seq
from scipy.fft import rfft
from sklearn.decomposition import NMF
from sklearn.utils import resample

# ── paths / constants ─────────────────────────────────────────────────────────

VFDB_NT      = pathlib.Path("data/VFDB_setB_nt.fas.gz")
NEG_CACHE    = pathlib.Path("data/same_strain_neg.json")
BUNDLE_CACHE = pathlib.Path("data/bundle_cache.pkl")
def _find_mmseqs() -> str:
    import shutil, os
    if p := os.environ.get("MMSEQS_PATH"): return p
    if p := shutil.which("mmseqs"): return p
    for candidate in ["mmseqs/bin/mmseqs", "../mmseqs/bin/mmseqs",
                      "/tmp/mmseqs/bin/mmseqs"]:
        if pathlib.Path(candidate).exists(): return candidate
    raise FileNotFoundError(
        "mmseqs2 not found. Install it (https://github.com/soedinglab/MMseqs2) "
        "or set MMSEQS_PATH=/path/to/mmseqs")

MMSEQS = _find_mmseqs()

HOLDOUT_GENERA = ["Chlamydia", "Coxiella", "Helicobacter", "Campylobacter", "Bordetella"]
MAX_ORGS       = 80
W_PHASE        = 110
AUTOCORR_LAGS  = 10
NMF_COMPONENTS = 64
RNG_SEED       = 42
N_FP_WORK      = 8

CHANNELS = "ATGC"
CH_IDX   = {c: i for i, c in enumerate(CHANNELS)}
PAIRS    = [(i, j) for i in range(4) for j in range(i + 1, 4)]

CODONS    = ["".join(k) for k in itertools.product("ACGT", repeat=3)]
CODON_IDX = {k: i for i, k in enumerate(CODONS)}

_LUT = np.full(256, -1, dtype=np.int8)
for _i, _c in enumerate(CHANNELS): _LUT[ord(_c)] = _i

_ECOLI_FREQ = {
    "TTT":0.58,"TTC":0.42,"TTA":0.14,"TTG":0.13,"CTT":0.12,"CTC":0.10,
    "CTA":0.04,"CTG":0.47,"ATT":0.49,"ATC":0.39,"ATA":0.11,"ATG":1.00,
    "GTT":0.28,"GTC":0.20,"GTA":0.17,"GTG":0.35,"TAT":0.57,"TAC":0.43,
    "TAA":0.61,"TAG":0.09,"TGA":0.30,"CAT":0.57,"CAC":0.43,"CAA":0.34,
    "CAG":0.66,"AAT":0.45,"AAC":0.55,"AAA":0.74,"AAG":0.26,"GAT":0.63,
    "GAC":0.37,"GAA":0.68,"GAG":0.32,"TCT":0.17,"TCC":0.15,"TCA":0.14,
    "TCG":0.14,"AGT":0.16,"AGC":0.25,"CCT":0.18,"CCC":0.13,"CCA":0.20,
    "CCG":0.49,"ACT":0.19,"ACC":0.40,"ACA":0.17,"ACG":0.25,"GCT":0.18,
    "GCC":0.26,"GCA":0.23,"GCG":0.33,"TGT":0.46,"TGC":0.54,"TGG":1.00,
    "CGT":0.36,"CGC":0.36,"CGA":0.07,"CGG":0.11,"AGA":0.07,"AGG":0.04,
    "GGT":0.35,"GGC":0.37,"GGA":0.13,"GGG":0.15,
}

# ── NR90 / cluster-40 ─────────────────────────────────────────────────────────

def _translate(seq: str) -> str | None:
    s = seq.upper().replace("-", "")
    trim = (len(s) // 3) * 3
    if trim < 90: return None
    try:
        p = str(Seq(s[:trim]).translate(to_stop=True))
        return p if len(p) >= 30 else None
    except Exception: return None


def _mmseqs_clust(seqs: dict[str, str], tmp: pathlib.Path,
                  tag: str, identity: float) -> dict[str, str]:
    prots = {k: p for k, s in seqs.items() if (p := _translate(s))}
    if not prots: return {k: k for k in seqs}
    fa = tmp / f"{tag}.faa"
    with open(fa, "w") as f:
        for k, p in prots.items(): f.write(f">{k}\n{p}\n")
    pfx = tmp / f"{tag}_cl"
    subprocess.run(
        [MMSEQS, "easy-linclust", str(fa), str(pfx), str(tmp / f"{tag}_tmp"),
         "--min-seq-id", str(identity), "-c", "0.80", "--cov-mode", "0",
         "-v", "0", "--threads", "4"],
        capture_output=True,
    )
    tsv = tmp / f"{tag}_cl_cluster.tsv"
    if not tsv.exists(): return {k: k for k in seqs}
    df = pd.read_csv(tsv, sep="\t", header=None, names=["rep", "mem"])
    m  = dict(zip(df["mem"], df["rep"]))
    return {k: m.get(k, k) for k in seqs}


def nr90(seqs: dict[str, str], tmp: pathlib.Path, tag: str) -> dict[str, str]:
    cl = _mmseqs_clust(seqs, tmp, tag, 0.90)
    reps = set(cl[k] for k in seqs)
    return {k: seqs[k] for k in reps if k in seqs}


def cluster_40(seqs: dict[str, str], tmp: pathlib.Path, tag: str) -> dict[str, str]:
    return _mmseqs_clust(seqs, tmp, tag, 0.40)

# ── parsers ───────────────────────────────────────────────────────────────────

def _parse_vfdb_by_org(fasta_gz, max_orgs) -> dict[str, list[str]]:
    hdr_re = re.compile(r'^>(\S+)\(gb\|([^)]+)\).*\[([^\[\]]+)\]\s*$')
    org_seqs: dict[str, list[str]] = defaultdict(list)
    cur_hdr = None; cur_seq = []

    def _emit():
        nonlocal cur_hdr, cur_seq
        if cur_hdr:
            m = hdr_re.match(cur_hdr)
            if m:
                _, _, org = m.groups()
                s = "".join(cur_seq).upper()
                if len(s) >= 90: org_seqs[org].append(s)
        cur_hdr = None; cur_seq.clear()

    with gzip.open(fasta_gz, "rt") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"): _emit(); cur_hdr = line
            else: cur_seq.append(line)
    _emit()
    top = sorted(org_seqs.items(), key=lambda x: len(x[1]), reverse=True)[:max_orgs]
    return dict(top)


def _build_pool(orgs, pos_src, neg_src, prefix):
    p, n, p_org, n_org = {}, {}, {}, {}
    pi = ni = 0
    for o in orgs:
        for s in pos_src.get(o, []):
            k = f"{prefix}p{pi}"; p[k] = s; p_org[k] = o; pi += 1
        for s in neg_src.get(o, []):
            k = f"{prefix}n{ni}"; n[k] = s; n_org[k] = o; ni += 1
    return p, n, {**p_org, **n_org}

# ── background stats ──────────────────────────────────────────────────────────

def _gc(dna: str) -> float:
    return sum(1 for b in dna if b in "GC") / max(len(dna), 1)


def _cai(dna: str) -> float:
    trim = (len(dna) // 3) * 3
    if trim < 3: return 0.0
    ws = [_ECOLI_FREQ[c] for c in [dna[i:i+3] for i in range(0, trim, 3)]
          if c in _ECOLI_FREQ]
    return float(np.exp(np.mean(np.log(np.clip(ws, 1e-6, None))))) if ws else 0.0


def _dinuc_rho(dna: str) -> np.ndarray:
    raw = _LUT[np.frombuffer(dna.encode("ascii", errors="replace"), dtype=np.uint8)]
    mono = np.zeros(4, dtype=np.float64)
    dinu = np.zeros((4, 4), dtype=np.float64)
    for b in raw:
        if b >= 0: mono[b] += 1
    for i in range(len(raw) - 1):
        a, b = int(raw[i]), int(raw[i+1])
        if a >= 0 and b >= 0: dinu[a, b] += 1
    N = mono.sum()
    if N < 2: return np.zeros(16, dtype=np.float32)
    f  = mono / N
    fd = dinu / max(dinu.sum(), 1)
    rho = np.zeros((4, 4), dtype=np.float64)
    for a in range(4):
        for b in range(4):
            denom = f[a] * f[b]
            rho[a, b] = fd[a, b] / denom if denom > 1e-10 else 1.0
    return rho.flatten().astype(np.float32)


def _compute_genome_backgrounds(neg_map: dict[str, list[str]]) -> dict[str, dict]:
    bg = {}
    for org, seqs in neg_map.items():
        if not seqs:
            bg[org] = {"gc": 0.5, "cai": 0.5, "rho": np.ones(16, np.float32) / 16}
            continue
        bg[org] = {
            "gc":  float(np.mean([_gc(s)  for s in seqs])),
            "cai": float(np.mean([_cai(s) for s in seqs])),
            "rho": np.mean([_dinuc_rho(s) for s in seqs], axis=0).astype(np.float32),
        }
    return bg

# ── fingerprint functions ─────────────────────────────────────────────────────

def _codon_pos_fp(seq: str) -> np.ndarray:
    dna = seq.upper()
    raw = _LUT[np.frombuffer(dna.encode("ascii", errors="replace"), dtype=np.uint8)]
    mats = [np.zeros((4, 4), dtype=np.float32) for _ in range(3)]
    for i in range(len(raw) - 1):
        a, b = int(raw[i]), int(raw[i+1])
        if 0 <= a <= 3 and 0 <= b <= 3:
            mats[i % 3][a, b] += 1
    out = []
    for m in mats:
        s = m.sum(); out.append((m / s if s > 0 else m).flatten())
    return np.concatenate(out)


def _codon_pos_autocorr_fp(seq: str, max_lag: int = AUTOCORR_LAGS) -> np.ndarray:
    dna = seq.upper()
    raw = _LUT[np.frombuffer(dna.encode("ascii", errors="replace"), dtype=np.uint8)]
    out = []
    for p in range(3):
        subseq = raw[p::3]
        L = len(subseq)
        oh = (subseq[:, None] == np.arange(4)[None, :]).astype(np.float32)
        autocorrs = np.zeros(max_lag, dtype=np.float32)
        for lag in range(1, min(max_lag + 1, L)):
            x = oh[:L - lag]; y = oh[lag:]
            mx = x.mean(0); my = y.mean(0)
            sx = x.std(0);  sy = y.std(0)
            valid = (sx > 1e-8) & (sy > 1e-8)
            corr = np.where(valid,
                            ((x - mx) * (y - my)).mean(0) / (sx * sy + 1e-10),
                            0.0)
            autocorrs[lag - 1] = corr[valid].mean() if valid.any() else 0.0
        out.append(autocorrs)
    return np.concatenate(out)


def _codon_pair_counts_fp(seq: str) -> np.ndarray | None:
    dna = seq.upper()
    trim = (len(dna) // 3) * 3
    if trim < 6: return None
    codons = [dna[i:i+3] for i in range(0, trim, 3)]
    counts = np.zeros(64 * 64, dtype=np.float32)
    for i in range(len(codons) - 1):
        ci = CODON_IDX.get(codons[i], -1)
        cj = CODON_IDX.get(codons[i+1], -1)
        if ci >= 0 and cj >= 0:
            counts[ci * 64 + cj] += 1
    s = counts.sum()
    return counts / s if s > 0 else counts


def _foreignness_fp(seq: str, bg: dict) -> np.ndarray:
    dna = seq.upper()
    return np.concatenate([
        [_gc(dna) - bg["gc"], _cai(dna) - bg["cai"]],
        _dinuc_rho(dna) - bg["rho"],
    ]).astype(np.float32)


def _voss_batch(seq: str, W: int) -> np.ndarray | None:
    step = max(1, W // 2)
    raw  = np.frombuffer(seq.upper().encode("ascii", errors="replace"), dtype=np.uint8)
    n    = (len(raw) - W) // step + 1
    if n <= 0: return None
    arr  = _LUT[raw]
    idx  = step * np.arange(n)[:, None] + np.arange(W)[None, :]
    wins = arr[idx]
    return (wins[:, None, :] == np.arange(4)[None, :, None]).astype(np.float32)


def _phase_fp(seq: str, W: int = W_PHASE) -> np.ndarray | None:
    sig = _voss_batch(seq, W)
    if sig is None: return None
    K = W // 2 + 1
    F = rfft(sig, axis=2); Fn = F / (np.abs(F) + 1e-10)
    cross = np.zeros((6, K), dtype=np.complex128)
    for idx, (i, j) in enumerate(PAIRS):
        cross[idx] = (Fn[:, i, :] * np.conj(Fn[:, j, :])).mean(axis=0)
    def l1(x):
        s = np.abs(x).sum(); return x / s if s > 0 else x
    return np.concatenate(
        [v for c in cross for v in (l1(np.cos(np.angle(c))), l1(np.sin(np.angle(c))))]
    ).astype(np.float32)

# ── DataBundle ────────────────────────────────────────────────────────────────

@dataclass
class DataBundle:
    # sequences
    train_seqs:  list[str]
    train_keys:  list[str]
    train_labels: np.ndarray    # 1=VF, 0=non-VF
    cl_ids:      np.ndarray     # gene-family cluster id (int) per train seq

    tb_seqs:     list[str]
    tb_keys:     list[str]
    tb_labels:   np.ndarray

    # precomputed raw feature arrays (None where computation failed)
    cp:      list               # codon_pos 48d
    ac:      list               # autocorr  30d
    cpc_nmf: list               # codon_pair NMF-64d
    fg:      list               # foreignness 18d
    ph:      list               # phase 12×(W//2+1)d

    n_train: int = field(init=False)

    def __post_init__(self):
        self.n_train = len(self.train_seqs)

    @property
    def feat_dims(self) -> dict[str, int]:
        dims = {}
        for name in ("cp", "ac", "cpc_nmf", "fg", "ph"):
            arr = getattr(self, name)
            v = next((x for x in arr if x is not None), None)
            dims[name] = int(v.shape[0]) if v is not None else 0
        return dims

# ── public API ────────────────────────────────────────────────────────────────

def load_dataset(verbose: bool = True, use_cache: bool = True) -> DataBundle:
    """
    Parse VFDB + same-strain negatives, deduplicate (NR90), cluster (40% AA),
    compute all feature arrays.  ~2–3 min on first run; instant on subsequent
    runs via pickle cache at data/bundle_cache.pkl.

    Pass use_cache=False to force a full recompute and overwrite the cache.
    """
    def log(msg):
        if verbose: print(msg, flush=True)

    if use_cache and BUNDLE_CACHE.exists():
        log(f"Loading cached DataBundle from {BUNDLE_CACHE} ...")
        with BUNDLE_CACHE.open("rb") as f:
            bundle = pickle.load(f)
        log(f"  train={bundle.n_train}  testB={len(bundle.tb_seqs)}")
        log("=" * 70)
        return bundle

    log("=" * 70)
    log("Loading dataset ...")
    log("=" * 70)

    log("\nParsing VFDB setB ...")
    org_pos = _parse_vfdb_by_org(VFDB_NT, MAX_ORGS)
    holdout  = {o for o in org_pos if any(o.startswith(g) for g in HOLDOUT_GENERA)}
    train_or = set(org_pos.keys()) - holdout
    neg_map  = json.loads(NEG_CACHE.read_text())

    tr_p_raw, tr_n_raw, tr_org_raw = _build_pool(train_or, org_pos, neg_map, "tr_")
    tb_p_raw, tb_n_raw, tb_org_raw = _build_pool(holdout,  org_pos, neg_map, "tb_")
    log(f"  Train raw: pos={len(tr_p_raw)}  neg={len(tr_n_raw)}")
    log(f"  TestB raw: pos={len(tb_p_raw)}  neg={len(tb_n_raw)}")

    log("NR90 (4 pools in parallel) ...")
    nr_res: dict = {}; nr_lock = threading.Lock()

    def _do_nr90(seqs, tag):
        with tempfile.TemporaryDirectory() as tmp:
            r = nr90(seqs, pathlib.Path(tmp), tag)
        with nr_lock: nr_res[tag] = r

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(_do_nr90, d, t) for d, t in
                [(tr_p_raw,"tr_pos"),(tr_n_raw,"tr_neg"),
                 (tb_p_raw,"tb_pos"),(tb_n_raw,"tb_neg")]]
        for f in as_completed(futs): f.result()

    tr_pos = nr_res["tr_pos"]; tr_neg_all = nr_res["tr_neg"]
    tb_pos = nr_res["tb_pos"]; tb_neg_all = nr_res["tb_neg"]

    tr_neg_keys = resample(list(tr_neg_all.keys()),
                           n_samples=min(len(tr_pos), len(tr_neg_all)),
                           replace=False, random_state=RNG_SEED)
    tb_neg_keys = resample(list(tb_neg_all.keys()),
                           n_samples=min(len(tb_pos), len(tb_neg_all)),
                           replace=False, random_state=RNG_SEED)
    tr_neg = {k: tr_neg_all[k] for k in tr_neg_keys}
    tb_neg = {k: tb_neg_all[k] for k in tb_neg_keys}

    log(f"  After NR90+balance: train pos={len(tr_pos)} neg={len(tr_neg)}"
        f"  testB pos={len(tb_pos)} neg={len(tb_neg)}")

    all_train = {**tr_pos, **tr_neg}
    tr_labels = np.array([1]*len(tr_pos) + [0]*len(tr_neg))
    tb_seqs_d = {**tb_pos, **tb_neg}
    tb_labels = np.array([1]*len(tb_pos) + [0]*len(tb_neg))

    org_map = {**tr_org_raw, **tb_org_raw}

    log("Clustering at 40% AA ...")
    with tempfile.TemporaryDirectory() as tmp:
        cl_map = cluster_40(all_train, pathlib.Path(tmp), "train")
    cl_strs  = [cl_map[k] for k in all_train]
    uniq_cl  = {s: i for i, s in enumerate(sorted(set(cl_strs)))}
    cl_ids   = np.array([uniq_cl[s] for s in cl_strs])
    log(f"  Gene families: {len(uniq_cl)}")

    train_seqs = list(all_train.values())
    train_keys = list(all_train.keys())
    tb_seqs    = list(tb_seqs_d.values())
    tb_keys    = list(tb_seqs_d.keys())
    all_seqs   = train_seqs + tb_seqs
    all_keys   = train_keys + tb_keys
    n_tr       = len(train_seqs)

    log("Computing genome backgrounds ...")
    genome_bg = _compute_genome_backgrounds(neg_map)
    global_bg = {
        "gc":  float(np.mean([bg["gc"]  for bg in genome_bg.values()])),
        "cai": float(np.mean([bg["cai"] for bg in genome_bg.values()])),
        "rho": np.mean([bg["rho"] for bg in genome_bg.values()], axis=0),
    }

    def get_bg(key):
        org = org_map.get(key)
        return genome_bg.get(org, global_bg)

    log("\nPrecomputing feature arrays ...")

    def _par(fn):
        out = [None] * len(all_keys)
        def _w(i): out[i] = fn(i)
        with ThreadPoolExecutor(max_workers=N_FP_WORK) as ex:
            list(ex.map(_w, range(len(all_keys))))
        return out

    log("  codon_pos ...")
    cp_raw  = _par(lambda i: _codon_pos_fp(all_seqs[i]))
    log("  codon_pos_autocorr ...")
    ac_raw  = _par(lambda i: _codon_pos_autocorr_fp(all_seqs[i]))
    log("  codon_pair_counts (for NMF) ...")
    cpc_raw = _par(lambda i: _codon_pair_counts_fp(all_seqs[i]))
    log("  foreignness ...")
    fg_raw  = _par(lambda i: _foreignness_fp(all_seqs[i], get_bg(all_keys[i])))
    log("  phase ...")
    ph_raw  = _par(lambda i: _phase_fp(all_seqs[i]))

    log(f"  Fitting NMF-{NMF_COMPONENTS} on codon-pair counts ...")
    cp_train_valid = [(i, cpc_raw[i]) for i in range(n_tr) if cpc_raw[i] is not None]
    if cp_train_valid:
        X_cpc_train = np.vstack([v for _, v in cp_train_valid])
        nmf = NMF(n_components=NMF_COMPONENTS, random_state=42, max_iter=500)
        nmf.fit(X_cpc_train)
        cpc_nmf_raw = [nmf.transform(v[None, :])[0] if v is not None else None
                       for v in cpc_raw]
    else:
        cpc_nmf_raw = [None] * len(all_seqs)
        log("  [NMF skipped]")

    log(f"\nDataset ready.  train={n_tr}  testB={len(tb_seqs)}")
    log("=" * 70)

    bundle = DataBundle(
        train_seqs=train_seqs, train_keys=train_keys, train_labels=tr_labels,
        cl_ids=cl_ids,
        tb_seqs=tb_seqs, tb_keys=tb_keys, tb_labels=tb_labels,
        cp=cp_raw, ac=ac_raw, cpc_nmf=cpc_nmf_raw, fg=fg_raw, ph=ph_raw,
    )
    log(f"Saving DataBundle cache to {BUNDLE_CACHE} ...")
    BUNDLE_CACHE.parent.mkdir(exist_ok=True)
    with BUNDLE_CACHE.open("wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    log("  Done.")
    return bundle


def build_features(
    bundle: DataBundle,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Assemble (X_tr, y_tr, cl_tr, X_tb, y_tb) from selected feature arrays.

    feature_names: subset of {"cp", "ac", "cpc_nmf", "fg", "ph"}

    Only rows where ALL selected features are non-None are kept.
    """
    arrays = [getattr(bundle, name) for name in feature_names]
    n_tr   = bundle.n_train

    fps_tr, fps_tb = [], []
    y_tr_v, y_tb_v, cl_tr_v = [], [], []

    for i in range(n_tr):
        parts = [arrays[c][i] for c in range(len(arrays))]
        if all(p is not None for p in parts):
            fps_tr.append(np.concatenate(parts))
            y_tr_v.append(bundle.train_labels[i])
            cl_tr_v.append(bundle.cl_ids[i])

    for j in range(len(bundle.tb_seqs)):
        i = n_tr + j
        parts = [arrays[c][i] for c in range(len(arrays))]
        if all(p is not None for p in parts):
            fps_tb.append(np.concatenate(parts))
            y_tb_v.append(bundle.tb_labels[j])

    X_tr = np.vstack(fps_tr)
    X_tb = np.vstack(fps_tb)
    return (X_tr, np.array(y_tr_v), np.array(cl_tr_v),
            X_tb, np.array(y_tb_v))
