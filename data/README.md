# Data

## Files

### `VFDB_setB_nt.fas.gz` (11 MB)
VFDB setB nucleotide CDS sequences (experimentally verified virulence factors).

**Source:** Virulence Factor Database (VFDB), downloaded April 2025.  
**URL:** http://www.mgc.ac.cn/VFs/download.htm  
**License:** Freely accessible for academic research. Cite:

> Liu B, Zheng D, Zhou S, Chen L, Yang J. VFDB 2022: a general classification scheme for bacterial virulence factors. *Nucleic Acids Research*, 2022, 50(D1):D912–D917. https://doi.org/10.1093/nar/gkab1107

The file contains raw nucleotide sequences. `pipeline.py` filters to the top 80 organisms by sequence count, applies NR90 deduplication, and holds out five genera for TestB.

---

### `same_strain_neg.json` (319 MB)
Non-VF CDS sequences downloaded from the same 80 genome assemblies that contribute to the positive class.

**Source:** NCBI RefSeq complete genome assemblies, fetched via NCBI Datasets CLI (`datasets download genome accession ...`), April 2025.  
**License:** Public domain — NCBI/GenBank sequences are produced by the US government and are not subject to copyright.

**Construction:** `exp_same_strain_control.py` identifies one representative complete genome per VFDB organism (via NCBI taxonomy), downloads all CDS, removes any sequence matching the VFDB exclusion list (BLASTP identity > 50%, 30,177 unique protein IDs), and caches the result here.

**Note:** This file is large (319 MB). It is tracked directly in this repository. If you need to regenerate it (e.g., after updating the exclusion list), run:
```bash
python experiments/exp_same_strain_control.py --rebuild-neg
```

---

## Not Included

`bundle_cache.pkl` — pre-computed DataBundle (features + cluster IDs). Not tracked; auto-generated on first run of any experiment script via `pipeline.py`. Stored at `data/bundle_cache.pkl` locally.
