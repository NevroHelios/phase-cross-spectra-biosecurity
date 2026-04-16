"""Voss 4-channel binary encoding of DNA sequences."""

import numpy as np

CHANNELS: list[str] = ["A", "T", "G", "C"]
_CH_IDX: dict[str, int] = {c: i for i, c in enumerate(CHANNELS)}


def voss_encode(sequence: str) -> np.ndarray:
    """
    Encode a DNA sequence as a 4-channel binary (Voss) matrix.

    Each row corresponds to one nucleotide channel (A, T, G, C).
    Position (i, j) is 1 if sequence[j] == CHANNELS[i], else 0.
    Ambiguous bases (N, R, Y, …) are silently mapped to all-zero columns.

    Args:
        sequence: DNA string, any case.

    Returns:
        np.ndarray of shape (4, len(sequence)), dtype float32.
    """
    seq = sequence.upper()
    N = len(seq)
    sig = np.zeros((4, N), dtype=np.float32)
    for pos, base in enumerate(seq):
        idx = _CH_IDX.get(base, -1)
        if idx >= 0:
            sig[idx, pos] = 1.0
    return sig
