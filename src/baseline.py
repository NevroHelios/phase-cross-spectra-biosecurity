"""
Voss power spectrum baseline fingerprint.

Computes the average power spectral density |F_i(k)|² for each of the
4 Voss channels, L1-normalised, then concatenated.

Feature dimension: 4 × (W//2 + 1)
  W=90  → 184 dims
  W=420 → 844 dims
"""

import numpy as np
from scipy.fft import rfft

from .encoding import voss_encode


def voss_power_fingerprint(
    sequence: str,
    W: int,
    step: int | None = None,
) -> np.ndarray | None:
    """
    Compute the Voss power spectrum (magnitude-only) fingerprint.

    For each sliding window, computes |rfft(channel)|² for all 4 channels,
    averages across windows, L1-normalises each channel, and concatenates.

    Args:
        sequence: DNA string, any case.
        W:        Window size in base pairs.
        step:     Stride between windows. Defaults to W//2 (50% overlap).

    Returns:
        1-D float32 array of shape (4 * (W//2 + 1),), or None if sequence
        is shorter than W.
    """
    dna = sequence.upper()
    if len(dna) < W:
        return None

    if step is None:
        step = W // 2

    K = W // 2 + 1
    power = np.zeros((4, K), dtype=np.float64)
    n_windows = 0

    for start in range(0, len(dna) - W + 1, step):
        sig = voss_encode(dna[start : start + W])   # (4, W)
        F   = rfft(sig, axis=1)                      # (4, K) complex
        power += np.abs(F) ** 2
        n_windows += 1

    if n_windows == 0:
        return None

    avg_power = power / n_windows  # shape (4, K)

    def _l1(x: np.ndarray) -> np.ndarray:
        s = np.abs(x).sum()
        return x / s if s > 0 else x

    return np.concatenate(
        [_l1(avg_power[i]) for i in range(4)]
    ).astype(np.float32)
