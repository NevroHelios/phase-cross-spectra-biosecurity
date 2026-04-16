"""
Phase cross-spectral fingerprint for DNA sequences.

The fingerprint encodes inter-nucleotide phase relationships across all
6 Voss channel pairs (A-T, A-G, A-C, T-G, T-C, G-C), averaged coherently
over sliding windows of width W.

Feature dimension: 2 × 6 × (W//2 + 1)
  W=90  → 552 dims
  W=420 → 2532 dims
"""

import numpy as np
from scipy.fft import rfft

from .encoding import voss_encode, CHANNELS

_PAIRS: list[tuple[int, int]] = [
    (i, j) for i in range(4) for j in range(i + 1, 4)
]  # (A,T),(A,G),(A,C),(T,G),(T,C),(G,C)


def phase_cross_spectral_fingerprint(
    sequence: str,
    W: int,
    step: int | None = None,
) -> np.ndarray | None:
    """
    Compute the phase cross-spectral fingerprint for a DNA sequence.

    Algorithm:
      1. Slide a window of width W with stride W//2.
      2. Voss-encode each window → 4×W binary matrix.
      3. rfft per channel → complex spectrum F of shape (4, W//2+1).
      4. Unit-normalise: F̂ = F / (|F| + ε).
      5. Accumulate cross-spectrum: Σ F̂ᵢ · conj(F̂ⱼ) for all 6 pairs.
      6. Divide by window count → coherent average C of shape (6, W//2+1).
      7. For each pair: extract cos(arg(C)) and sin(arg(C)), L1-normalise.
      8. Concatenate all 12 sub-vectors → 2×6×(W//2+1)-dim float32 vector.

    Args:
        sequence: DNA string, any case.
        W:        Window size in base pairs.
        step:     Stride between windows. Defaults to W//2 (50% overlap).

    Returns:
        1-D float32 array of shape (2 * 6 * (W//2 + 1),), or None if the
        sequence is shorter than W.
    """
    dna = sequence.upper()
    if len(dna) < W:
        return None

    if step is None:
        step = W // 2

    K = W // 2 + 1
    cross = np.zeros((6, K), dtype=np.complex128)
    n_windows = 0

    for start in range(0, len(dna) - W + 1, step):
        sig = voss_encode(dna[start : start + W])          # (4, W)
        F   = rfft(sig, axis=1)                             # (4, K) complex
        Fn  = F / (np.abs(F) + 1e-10)                      # unit magnitude

        for idx, (i, j) in enumerate(_PAIRS):
            cross[idx] += Fn[i] * np.conj(Fn[j])
        n_windows += 1

    if n_windows == 0:
        return None

    cm = cross / n_windows  # coherent average, shape (6, K)

    def _l1(x: np.ndarray) -> np.ndarray:
        s = np.abs(x).sum()
        return x / s if s > 0 else x

    parts: list[np.ndarray] = []
    for c in cm:
        parts.append(_l1(np.cos(np.angle(c))))
        parts.append(_l1(np.sin(np.angle(c))))

    return np.concatenate(parts).astype(np.float32)


def fingerprint_dim(W: int) -> int:
    """Return the expected feature dimension for window size W."""
    return 2 * 6 * (W // 2 + 1)
