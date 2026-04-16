# Inter-Nucleotide Phase Cross-Spectra Enable Alignment-Free Hazard Sequence Screening at Sub-150 bp Resolution

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

## Abstract

Phase cross-spectral fingerprinting of DNA sequences using Voss 4-channel FFT encoding achieves AUROC 0.913 [0.897, 0.929] at 90 bp and 0.914 [0.907, 0.922] at 420 bp on native CDS sequences from SafeProtein-Bench, with cluster-aware evaluation (MMseqs2 40% identity, 20 seeds). This is the first published classifier performance below 150 bp for biosecurity hazard sequence screening. Method operates entirely on nucleotide sequences in O(n log n) without amino acid translation.

---

## Key Claims

- Phase cross-spectra between Voss channels (not the power spectrum) carry discriminative signal for hazard sequence screening
- AUROC 0.913 ± 0.035 at W=90 bp (n=788 real CDS sequences, 605 clusters)
- AUROC 0.914 ± 0.018 at W=420 bp (n=695 real CDS sequences, 552 clusters)
- Back-translated sequences depress AUC by 0.026–0.039 — report real-CDS results separately
- Phase coherence is inferior to raw phase angle (drops AUC by 0.06 at W=90, 0.23 at W=420)
- Voss power spectrum baseline: AUROC 0.785 ± 0.068 (W=90), 0.792 ± 0.054 (W=420) under identical cluster-aware evaluation

> **Note on baseline:** An earlier 3-seed pre-cluster-aware estimate of 0.877 appears in internal notes. The correct 20-seed cluster-aware value is 0.785–0.792.

---

## Method Summary

### Algorithm

Given a DNA sequence, the fingerprint is computed as follows:

```python
import numpy as np
from scipy.fft import rfft

CHANNELS = ['A', 'T', 'G', 'C']
PAIRS    = [(i, j) for i in range(4) for j in range(i + 1, 4)]  # 6 pairs

def phase_cross_spectral_fingerprint(sequence: str, W: int) -> np.ndarray:
    """
    Args:
        sequence: DNA string (any case)
        W:        window size in bp (e.g. 90 or 420)

    Returns:
        1-D float32 array of dimension 2 × 6 × (W//2 + 1)
        e.g. W=90  → 552-dim, W=420 → 2532-dim
    """
    dna  = sequence.upper()
    step = W // 2
    K    = W // 2 + 1
    ch   = {b: i for i, b in enumerate(CHANNELS)}

    cross = np.zeros((6, K), dtype=np.complex128)
    n = 0
    for s in range(0, len(dna) - W + 1, step):
        # Step 1 — Voss 4-channel binary encoding
        sig = np.zeros((4, W))
        for pos, base in enumerate(dna[s:s + W]):
            idx = ch.get(base, -1)
            if idx >= 0:
                sig[idx, pos] = 1.0

        # Step 2 — rfft on each channel (complex, phase preserved)
        F  = rfft(sig, axis=1)                  # shape (4, K)
        Fn = F / (np.abs(F) + 1e-10)            # unit magnitude, phase intact

        # Step 3 — 6 cross-spectra: (A,T),(A,G),(A,C),(T,G),(T,C),(G,C)
        for idx, (i, j) in enumerate(PAIRS):
            cross[idx] += Fn[i] * np.conj(Fn[j])
        n += 1

    if n == 0:
        return None
    cm = cross / n                               # coherent average

    # Step 4 — cos(arg) and sin(arg), Step 5 — L1 normalize each part
    def l1(x):
        s = np.abs(x).sum()
        return x / s if s > 0 else x

    return np.concatenate(
        [v for c in cm
         for v in (l1(np.cos(np.angle(c))),
                   l1(np.sin(np.angle(c))))]
    ).astype(np.float32)
```

### Full Pipeline

1. **Voss encoding** — each position mapped to a one-hot vector over {A, T, G, C}
2. **rfft per channel** — complex spectrum, phase information retained (not |F|²)
3. **6 cross-spectra** — all channel pairs: F̂ᵢ · conj(F̂ⱼ), averaged coherently over windows
4. **cos + sin of phase angle** — extract arg(cross) → cos, sin; concatenate
5. **L1 normalization** — each of the 12 sub-vectors independently normalized
6. **StandardScaler → PCA → LogisticRegression (C=0.1)** — fitted on train fold only per seed
   - W=90:  PCA(64),  feature dim=552
   - W=420: PCA(128), feature dim=2532

---

## Dataset

**SafeProtein-Bench** (Fan et al., NeurIPS 2025): 858 sequences (429 dangerous toxin/viral proteins, 429 benign SwissProt controls).

| Split | Total | With native CDS | Back-translated | Pass W=90 | Pass W=420 |
|-------|-------|-----------------|-----------------|-----------|------------|
| Dangerous | 429 | 362 (84%) | 67 | 362 | 302 |
| Benign | 429 | 426 (99%) | 2 | 426 | 393 |
| **Total** | **858** | **788** | **69** | **788** | **695** |

The 67 dangerous sequences with no native nucleotide records (synthetic peptides with no NCBI/EMBL links) were back-translated using human codon optimization. **These are excluded from main experiments** — back-translation inflates dangerous-class regularity and depresses AUC by 0.026–0.039. Results on all 857 sequences are reported separately as a sensitivity analysis.

---

## Evaluation Protocol

- **Clustering**: MMseqs2 `easy-cluster` at 40% amino acid sequence identity, coverage 0.8 (`--cov-mode 0`)
- **Split**: cluster-aware 80/20 split — no sequence from the same cluster appears in both train and test
- **Seeds**: 20 independent shuffles of cluster assignments
- **Metric**: AUROC (best of cosine-kNN and distance-to-centroid, or PCA+LR predict\_proba)
- **No leakage**: conservation scores and labels are never passed to fingerprint functions; verified by code audit

---

## Results

### Table 1 — Main Results (real CDS sequences only)

| Method | W (bp) | n | Mean AUC | Std | 95% CI (bootstrap) |
|--------|--------|---|----------|-----|---------------------|
| Voss power + dist-to-centroid | 90 | 788 | 0.785 | 0.068 | [0.755, 0.815] |
| Phase cross-spectra + dist | 90 | 788 | 0.912 | 0.023 | [0.902, 0.922] |
| **Phase cross-spectra + PCA(64) + LR** | **90** | **788** | **0.913** | **0.035** | **[0.897, 0.929]** |
| Voss power + dist-to-centroid | 420 | 695 | 0.792 | 0.054 | [0.767, 0.815] |
| Phase cross-spectra + dist | 420 | 695 | 0.900 | 0.030 | [0.887, 0.913] |
| **Phase cross-spectra + PCA(128) + LR** | **420** | **695** | **0.914** | **0.018** | **[0.907, 0.922]** |

### Table 2 — Ablation

| Method | W (bp) | Mean AUC | Std | Δ vs phase |
|--------|--------|----------|-----|------------|
| Multi-scale (W=90 + W=420 PCA-32 → LR) | — | 0.938 | 0.040 | +0.025 |
| Phase coherence (Welch) instead of raw phase | 90 | 0.852 | 0.034 | −0.061 |
| Phase coherence (Welch) instead of raw phase | 420 | 0.675 | 0.052 | −0.226 |
| All sequences incl. back-translated | 90 | 0.887 | 0.032 | −0.026 |
| All sequences incl. back-translated | 420 | 0.875 | 0.038 | −0.039 |

---

## Repository Structure

```
phase-cross-spectra-biosecurity/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── encoding.py       # Voss 4-channel encoding
│   ├── fingerprint.py    # Phase cross-spectral feature extraction
│   ├── evaluate.py       # Cluster-aware evaluation loop (20 seeds)
│   ├── baseline.py       # Voss power spectrum baseline
│   └── utils.py          # Data loading, NCBI fetch helpers
├── experiments/
│   ├── run_main.py            # Reproduces Table 1 (W=90 and W=420)
│   ├── run_ablation.py        # Coherence ablation, multi-scale
│   └── run_backtranslation.py # Table 2 sensitivity analysis
├── results/
│   ├── README.md         # CSV format documentation
│   └── .gitkeep
└── notebooks/
    └── exploration.ipynb # Interactive walkthrough
```

---

## Quickstart

```bash
git clone https://github.com/<your-org>/phase-cross-spectra-biosecurity
cd phase-cross-spectra-biosecurity
pip install -r requirements.txt

# Reproduce Table 1
python experiments/run_main.py --input data/sequences.csv --w 90 --pca 64
python experiments/run_main.py --input data/sequences.csv --w 420 --pca 128
```

Input CSV format: `accession,sequence,label,cluster_id` where `label ∈ {0, 1}` and `cluster_id` is the MMseqs2 representative accession.

---

## Dependencies

| Package | Version |
|---------|---------|
| numpy | ≥ 2.4.4 |
| scipy | ≥ 1.17.1 |
| scikit-learn | ≥ 1.8.0 |
| biopython | ≥ 1.84 |
| pandas | ≥ 2.0 |

---

## Citation

```bibtex
@misc{phase-cross-spectra-biosecurity-2026,
  title  = {Inter-Nucleotide Phase Cross-Spectra Enable Alignment-Free Hazard
             Sequence Screening at Sub-150 bp Resolution},
  author = {},
  year   = {2026},
  note   = {Preprint forthcoming on bioRxiv. GitHub timestamp: first commit date.}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
