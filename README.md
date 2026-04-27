# HGT Leaves a Linear Fingerprint in Codon Space

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

> Arka Dash — Independent, With Apart Research

## Abstract

Alignment-free nucleotide classifiers for virulence factor (VF) prediction routinely report AUROC above 0.90. We show these figures are inflated by approximately 0.30 AUROC from two compounded evaluation artefacts: organism-level codon usage bias (CUB) confounds from mismatched negative classes (~0.09) and gene-family identity leakage from random splits (~0.21). Under a same-strain negative class design (non-VF CDS from identical genome assemblies as positives; 14,894 sequences) and gene-family-disjoint GroupShuffleSplit (9,100+ families, 40% amino acid identity, 20 seeds), we benchmark six alignment-free feature classes. The best-generalising configuration is 64-dimensional codon unigram frequency with logistic regression: TestA = 0.741 ± 0.014, TestB = 0.734 ± 0.002 on five entirely withheld pathogen genera (n = 1,312). The generalisation gap (0.006) is not statistically distinguishable from zero (Wilcoxon p = 0.097, two-sided). Every feature beyond 64 dimensions produces a significant gap (all p ≤ 0.024), confirming monotonic overfitting with complexity. The underlying signal is horizontal gene transfer (HGT)-derived CUB deviation, linear in codon frequency space and genus-invariant.

---

## Key Results

| Feature | TestA AUROC | TestB AUROC | Gap | Wilcoxon p |
|---------|-------------|-------------|-----|------------|
| **Codon unigram 64d [LR]** | **0.741 ± 0.014** | **0.734 ± 0.002** | **+0.006** | **0.097 NS** |
| Unigram + delta RSCU 128d [LR] | 0.740 ± 0.015 | 0.732 ± 0.002 | +0.009 | 0.024 * |
| Positional unigram 192d [LR] | 0.726 ± 0.015 | 0.706 ± 0.004 | +0.020 | 0.0001 *** |
| Codon bigram 4096d → PCA-128 [LR] | 0.719 ± 0.014 | 0.689 ± 0.005 | +0.030 | < 0.0001 *** |
| Codon unigram 64d [XGB d=8] | 0.773 ± 0.012 | 0.714 ± 0.006 | +0.059 | < 0.0001 *** |
| Delta RSCU 64d [LR] | 0.596 ± 0.011 | 0.588 ± 0.005 | +0.008 | 0.011 * |

TestA = gene-family holdout (20% of clusters, 20 seeds). TestB = 5 entirely withheld pathogen genera: *Chlamydia*, *Coxiella*, *Helicobacter*, *Campylobacter*, *Bordetella* (n = 1,312).

---

## Inflation Sources

Prior nucleotide-level VF classifiers overestimate AUROC by ~0.30 from two compounded sources, illustrated here with Phase + Voss PSD as example feature:

| Stage | AUROC | Drop |
|-------|-------|------|
| Random split, organism-mismatched | 0.979 ± 0.007 | — |
| Cluster-aware, same-organism distribution | 0.890 ± 0.010 | −0.089 |
| **Same-strain, gene-family-disjoint** | **0.682 ± 0.012** | **−0.208** |

---

## Method

**Dataset.** VFDB setB, top 80 organisms, NR90-clustered (CD-HIT 90% AA identity): 7,447 VF sequences. Negative class: non-VF CDS from the identical 344 genome assemblies, with full VFDB exclusion list (30,177 unique protein IDs, BLASTP > 50%). Final: 14,894 sequences (7,447 per class).

**TestB.** Five genera withheld before any modelling: *Bordetella*, *Campylobacter*, *Helicobacter* (moderate HGT amelioration), *Chlamydia*, *Coxiella* (obligate intracellular, fully ameliorated VFs). 1,312 sequences (656 per class).

**Evaluation.** Gene-family-disjoint GroupShuffleSplit: all training sequences clustered at 40% AA identity (MMseqs2), yielding 9,143 families. Families split 80/20 across 20 random seeds. Two-sided Wilcoxon signed-rank test on 20 paired TestA−TestB gaps.

**Signal.** VFs disproportionately acquired via HGT carry the donor organism's codon preferences. In the same-strain design, the detectable CUB difference is between HGT-derived genes and the chromosomal background — a linear signal in 64-dimensional codon frequency space.

---

## Repository Structure

```
phase-cross-spectra-biosecurity/
├── pipeline.py               # Same-strain DataBundle loader (cache at data/bundle_cache.pkl)
├── stats_utils.py            # Wilcoxon / Benjamini-Hochberg helpers
├── requirements.txt
├── experiments/
│   ├── exp_stats_analysis.py     # Reproduces Table 2 — all 6 feature configs + stats
│   ├── exp_full_eval.py          # Phase+Voss Stage 3 baseline (Table 1, 0.682)
│   ├── run_main.py               # Phase cross-spectral Stage 1 baseline (Table 1, 0.979)
│   └── run_ablation.py           # Phase cross-spectral Stage 2 (Table 1, 0.890)
├── results/
│   ├── stats_tables.csv          # Table 2 data (mean±std, Wilcoxon, 95% CI)
│   ├── stats_raw_arrays.npz      # Per-seed AUROC arrays for all 6 configs
│   ├── full_eval_summary.csv     # Phase+Voss Stage 3 reference
│   ├── testa_aucs_codon_unigram_lr.npy
│   ├── testb_aucs_codon_unigram_lr.npy
│   └── testb_aucs_xgb_best.npy
└── src/                          # Phase cross-spectral source (Stage 1/2 context)
    ├── encoding.py
    ├── fingerprint.py
    ├── evaluate.py
    └── baseline.py
```

---

## Quickstart

```bash
git clone https://github.com/NevroHelios/phase-cross-spectra-biosecurity
cd phase-cross-spectra-biosecurity
pip install -r requirements.txt

# Requires: data/VFDB_setB_nt.fas.gz, data/same_strain_neg.json
# First run builds data/bundle_cache.pkl (~2-3 min); subsequent runs load from cache.

# Reproduce Table 2 (all feature configs + Wilcoxon stats)
python experiments/exp_stats_analysis.py

# Reproduce Phase+Voss Stage 3 baseline (Table 1, third row)
python experiments/exp_full_eval.py
```

**MMseqs2** must be installed and on `PATH`, or set `MMSEQS_PATH=/path/to/mmseqs`.  
Install: `conda install -c conda-forge -c bioconda mmseqs2` or see [MMseqs2 releases](https://github.com/soedinglab/MMseqs2/releases).

---

## Data

Raw data files (not tracked by git — too large):

| File | Description |
|------|-------------|
| `data/VFDB_setB_nt.fas.gz` | VFDB setB nucleotide CDS sequences |
| `data/same_strain_neg.json` | Non-VF CDS from the same 344 genome assemblies |
| `data/bundle_cache.pkl` | Pre-built DataBundle (auto-generated on first run) |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥ 2.4.4 | Arrays |
| scipy | ≥ 1.17.1 | FFT, stats |
| scikit-learn | ≥ 1.8.0 | LR, PCA, StandardScaler |
| biopython | ≥ 1.84 | Sequence translation |
| pandas | ≥ 2.0 | Result tables |
| xgboost | ≥ 2.0 | XGB ceiling test (GPU optional) |

---

## Citation

```bibtex
@article{dash2026hgt,
  title   = {{HGT} Leaves a Linear Fingerprint in Codon Space},
  author  = {Dash, Arka},
  year    = {2026},
  note    = {Preprint. Code: https://github.com/NevroHelios/phase-cross-spectra-biosecurity}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
