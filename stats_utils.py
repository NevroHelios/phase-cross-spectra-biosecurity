"""
Shared statistical utilities: Wilcoxon W + effect size, BH FDR correction.

Usage in experiment scripts:
    from stats_utils import wilcoxon_report, bh_correct

wilcoxon_report(a, b, label, n=None)
    → prints  W=210, p<0.0001, Z=4.32, r=0.97 (large effect)
    → returns dict with keys: label, W, p, p_raw, Z, r, n

bh_correct(results)
    → results: list of dicts from wilcoxon_report
    → prints FDR-corrected table, returns list of dicts with p_adj added
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def wilcoxon_report(
    a: list[float] | np.ndarray,
    b: list[float] | np.ndarray,
    label: str,
    n: int | None = None,
    alternative: str = "greater",
) -> dict:
    """
    Run Wilcoxon signed-rank test and print full report.

    Uses asymptotic Z from scipy (method='asymptotic') when n >= 10,
    exact otherwise. Rank-biserial r = Z / sqrt(n).

    Returns dict: label, W, p, Z, r, n, effect_size
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n_pairs = min(len(a), len(b))
    a, b = a[:n_pairs], b[:n_pairs]

    if n is None:
        n = n_pairs

    method = "asymptotic" if n >= 10 else "exact"
    result = stats.wilcoxon(a, b, alternative=alternative, method=method)
    W = float(result.statistic)
    p = float(result.pvalue)

    # Z score
    if hasattr(result, "zstatistic") and result.zstatistic is not None:
        Z = float(result.zstatistic)
    else:
        # Fall back to normal approximation from p-value
        # For one-sided 'greater', p = P(Z >= z), so z = norm.ppf(1-p)
        Z = float(stats.norm.ppf(1.0 - p)) if p < 1.0 else 0.0

    r = Z / np.sqrt(n)

    # Effect size label (Cohen's conventions for r)
    if abs(r) >= 0.5:
        eff = "large"
    elif abs(r) >= 0.3:
        eff = "medium"
    elif abs(r) >= 0.1:
        eff = "small"
    else:
        eff = "negligible"

    p_str = f"p<0.0001" if p < 0.0001 else f"p={p:.4f}"

    print(f"  Wilcoxon [{label}]: "
          f"W={W:.0f}, {p_str}, Z={Z:.2f}, r={r:.2f} ({eff} effect), n={n}")

    return dict(label=label, W=W, p=p, Z=Z, r=r, n=n, effect_size=eff,
                alternative=alternative)


def bh_correct(results: list[dict], alpha: float = 0.05) -> list[dict]:
    """
    Apply Benjamini-Hochberg FDR correction to a list of wilcoxon_report dicts.

    Prints a formatted table with raw p, adjusted p, and significance.
    Returns list of dicts with p_adj added.
    """
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        print("  [bh_correct] statsmodels not available — install with: pip install statsmodels")
        for r in results:
            r["p_adj"] = r["p"]
        return results

    pvals = np.array([r["p"] for r in results])
    reject, p_adj, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")

    out = []
    print(f"\n{'─'*80}")
    print(f"  Benjamini-Hochberg FDR correction (α={alpha}, m={len(pvals)} tests)")
    print(f"  {'Test':45s}  {'p_raw':>9}  {'p_adj':>9}  sig")
    print(f"{'─'*80}")
    for r, pa, rej in zip(results, p_adj, reject):
        p_raw_s = "<0.0001" if r["p"] < 0.0001 else f"{r['p']:.4f}"
        pa_s    = "<0.0001" if pa      < 0.0001 else f"{pa:.4f}"
        sig     = "***" if pa < 0.001 else ("**" if pa < 0.01 else ("*" if pa < 0.05 else "ns"))
        print(f"  {r['label']:45s}  {p_raw_s:>9}  {pa_s:>9}  {sig}")
        rr = dict(r)
        rr["p_adj"] = float(pa)
        rr["reject"] = bool(rej)
        out.append(rr)
    print(f"{'─'*80}\n")
    return out


def tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, threshold: float) -> float:
    """TPR at the highest FPR point that is <= threshold."""
    mask = fpr <= threshold
    return float(tpr[mask][-1]) if mask.any() else 0.0
