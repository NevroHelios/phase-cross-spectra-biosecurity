"""
Data loading, NCBI fetch helpers, and back-translation utilities.
"""

from __future__ import annotations

import pathlib
import time
from typing import Iterator

import numpy as np
import pandas as pd
import requests

# ── Human codon optimisation table (most frequent codon per AA, H. sapiens) ─
HUMAN_CODONS: dict[str, str] = {
    "A": "GCC", "R": "AGG", "N": "AAC", "D": "GAC", "C": "TGC",
    "Q": "CAG", "E": "GAG", "G": "GGC", "H": "CAC", "I": "ATC",
    "L": "CTG", "K": "AAG", "M": "ATG", "F": "TTC", "P": "CCC",
    "S": "AGC", "T": "ACC", "W": "TGG", "Y": "TAC", "V": "GTG",
    "*": "TGA",
}


def back_translate(aa_sequence: str) -> str:
    """
    Back-translate an amino acid sequence using human codon optimisation.

    Args:
        aa_sequence: Single-letter amino acid string.

    Returns:
        Nucleotide string (3 × len(aa_sequence) characters).
        Unknown amino acids map to 'NNN'.
    """
    return "".join(HUMAN_CODONS.get(aa.upper(), "NNN") for aa in aa_sequence)


def load_dataset(
    csv_path: str | pathlib.Path,
    backtranslated_path: str | pathlib.Path | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray, set[str]]:
    """
    Load sequences, labels, and cluster IDs from a CSV file.

    Expected CSV columns:
        accession  : unique identifier
        sequence   : nucleotide CDS string
        label      : 0 (benign) or 1 (dangerous)
        cluster_id : MMseqs2 representative accession (cluster membership)

    Args:
        csv_path:            Path to the input CSV.
        backtranslated_path: Optional path to a text file listing one
                             accession per line that were back-translated.
                             Used to filter if desired.

    Returns:
        (accessions, sequences_array, labels_array, cluster_ids_array,
         backtranslated_set)
        where sequences_array is a list of str, labels is int64 ndarray,
        cluster_ids is an int64 ndarray of cluster indices.
    """
    df = pd.read_csv(csv_path)
    required = {"accession", "sequence", "label", "cluster_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df = df.dropna(subset=["sequence"]).reset_index(drop=True)

    accessions  = df["accession"].tolist()
    sequences   = df["sequence"].tolist()
    labels      = df["label"].to_numpy(dtype=np.int64)

    # Map string cluster IDs to integer indices
    unique_clusters = list(dict.fromkeys(df["cluster_id"]))
    cluster_map     = {r: i for i, r in enumerate(unique_clusters)}
    cluster_ids     = np.array([cluster_map[c] for c in df["cluster_id"]],
                                dtype=np.int64)

    backtranslated: set[str] = set()
    if backtranslated_path is not None:
        p = pathlib.Path(backtranslated_path)
        if p.exists():
            backtranslated = set(p.read_text().strip().splitlines())

    return accessions, sequences, labels, cluster_ids, backtranslated


def fetch_ncbi_cds(
    ncbi_id: str,
    db: str,
    cache_dir: pathlib.Path,
    email: str,
    delay: float = 0.35,
) -> str | None:
    """
    Fetch coding sequence FASTA from NCBI Entrez (fasta_cds_na rettype).

    Args:
        ncbi_id:   NCBI accession (e.g. NM_000207.3 or NP_000198.1).
        db:        Entrez database — 'nucleotide' or 'protein'.
        cache_dir: Directory to cache downloaded FASTA files.
        email:     E-mail address for NCBI Entrez (required by NCBI ToS).
        delay:     Seconds to sleep after each API call (default: 0.35).

    Returns:
        Nucleotide CDS string, or None on failure.
    """
    from Bio import Entrez
    Entrez.email = email
    Entrez.tool  = "phase-cross-spectra-biosecurity"

    cache = cache_dir / f"{ncbi_id}.fasta"
    if cache.exists():
        raw = cache.read_text().strip()
        seq = "".join(l for l in raw.splitlines() if not l.startswith(">"))
        return seq or None

    try:
        handle = Entrez.efetch(db=db, id=ncbi_id, rettype="fasta_cds_na",
                               retmode="text")
        raw = handle.read()
        handle.close()
        if raw.startswith(">"):
            cache.write_text(raw)
            seq = "".join(l for l in raw.strip().splitlines()
                          if not l.startswith(">"))
            return seq or None
    except Exception as e:  # noqa: BLE001
        print(f"  NCBI fetch failed for {ncbi_id} ({db}): {e}")
    finally:
        time.sleep(delay)

    return None


def is_back_translated(sequence: str, threshold: float = 0.95) -> bool:
    """
    Heuristic: detect back-translated sequences by codon uniformity.

    Back-translated sequences using human codon optimisation always use
    the same codon per amino acid, producing very regular triplet patterns.
    This function checks whether >threshold fraction of codons match the
    human optimisation table.

    Args:
        sequence:  Nucleotide string (must be multiple of 3).
        threshold: Fraction of codons that must match to flag as
                   back-translated (default: 0.95).

    Returns:
        True if likely back-translated.
    """
    seq = sequence.upper()
    if len(seq) % 3 != 0 or len(seq) < 30:
        return False
    human_codon_set = set(HUMAN_CODONS.values())
    codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
    frac_match = sum(1 for c in codons if c in human_codon_set) / len(codons)
    return frac_match >= threshold
