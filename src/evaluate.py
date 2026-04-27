"""
Cluster-aware evaluation loop for biosecurity sequence classifiers.

Splits sequences by cluster membership (80% train / 20% test), ensuring
no sequence from the same sequence-similarity cluster appears in both
partitions. Evaluates AUROC over 20 random seeds.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon


def _get_split(
    cluster_ids: np.ndarray,
    seed: int,
    train_frac: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (train_indices, test_indices) split by cluster.

    Args:
        cluster_ids:  Integer cluster ID for each sample.
        seed:         Random seed for cluster shuffling.
        train_frac:   Fraction of clusters in train set.

    Returns:
        Tuple of (train_idx, test_idx) index arrays.
    """
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(cluster_ids)
    rng.shuffle(unique_clusters)
    cut = int(len(unique_clusters) * train_frac)
    train_clusters = set(unique_clusters[:cut].tolist())
    test_clusters  = set(unique_clusters[cut:].tolist())
    train_idx = np.where([c in train_clusters for c in cluster_ids])[0]
    test_idx  = np.where([c in test_clusters  for c in cluster_ids])[0]
    return train_idx, test_idx


def _dist_score(
    fps: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> float:
    """AUROC via distance-to-class-centroid (cosine metric)."""
    X_tr, X_te = fps[train_idx], fps[test_idx]
    y_tr, y_te = labels[train_idx], labels[test_idx]
    D = cdist(X_te, X_tr, metric="cosine")
    scores = np.array([
        D[i, y_tr == 0].mean() - D[i, y_tr == 1].mean()
        for i in range(len(test_idx))
    ])
    return roc_auc_score(y_te, scores)


def _lr_score(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    pca_components: int,
    C: float = 0.1,
) -> float:
    """AUROC via StandardScaler + PCA + LogisticRegression pipeline."""
    n_comp = min(pca_components, X_tr.shape[0] - 1, X_tr.shape[1])
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n_comp)),
        ("lr",     LogisticRegression(C=C, max_iter=1000, solver="lbfgs",
                                      class_weight="balanced")),
    ])
    pipe.fit(X_tr, y_tr)
    return roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])


def cluster_aware_eval(
    fingerprints: np.ndarray,
    labels: np.ndarray,
    cluster_ids: np.ndarray,
    n_seeds: int = 20,
    pca_components: int = 64,
    C: float = 0.1,
    train_frac: float = 0.8,
) -> list[float]:
    """
    Cluster-aware evaluation: AUROC across n_seeds random train/test splits.

    Args:
        fingerprints:   Feature matrix, shape (n_samples, n_features).
        labels:         Binary labels (0/1), shape (n_samples,).
        cluster_ids:    Integer cluster ID per sample, shape (n_samples,).
        n_seeds:        Number of random seeds (default: 20).
        pca_components: PCA dimensionality for LR pipeline (default: 64).
        C:              LR regularisation strength (default: 0.1).
        train_frac:     Fraction of clusters used for training (default: 0.8).

    Returns:
        List of per-seed AUROCs (may be shorter than n_seeds if a seed
        produces a single-class test set).
    """
    aucs: list[float] = []
    for seed in range(n_seeds):
        tr_idx, te_idx = _get_split(cluster_ids, seed, train_frac)
        if len(np.unique(labels[te_idx])) < 2:
            continue
        auc = _lr_score(
            fingerprints[tr_idx], labels[tr_idx],
            fingerprints[te_idx], labels[te_idx],
            pca_components, C,
        )
        aucs.append(auc)
    return aucs


def bootstrap_ci(
    aucs: list[float],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Bootstrap percentile confidence interval on the mean AUROC.

    Args:
        aucs:         Per-seed AUROC values.
        n_resamples:  Number of bootstrap resamples.
        alpha:        Significance level (default: 0.05 → 95% CI).
        seed:         RNG seed for reproducibility.

    Returns:
        (ci_lower, ci_upper)
    """
    a = np.array(aucs)
    rng = np.random.default_rng(seed)
    means = [rng.choice(a, len(a), replace=True).mean() for _ in range(n_resamples)]
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def wilcoxon_greater(
    aucs_a: list[float],
    aucs_b: list[float],
) -> float:
    """Paired Wilcoxon H₁: aucs_a > aucs_b. Returns p-value."""
    a, b = np.array(aucs_a), np.array(aucs_b)
    n = min(len(a), len(b))
    _, p = wilcoxon(a[:n], b[:n], alternative="greater")
    return float(p)


def wilcoxon_full(
    aucs_a: list[float],
    aucs_b: list[float],
    label: str = "",
    alternative: str = "greater",
) -> dict:
    """
    Paired Wilcoxon signed-rank test with W statistic and rank-biserial effect size.

    Returns dict: W, p, Z, r, effect_size, n, label
    """
    from scipy import stats as _stats
    a, b = np.array(aucs_a, dtype=float), np.array(aucs_b, dtype=float)
    n = min(len(a), len(b))
    method = "asymptotic" if n >= 10 else "exact"
    result = wilcoxon(a[:n], b[:n], alternative=alternative, method=method)
    W = float(result.statistic)
    p = float(result.pvalue)
    if hasattr(result, "zstatistic") and result.zstatistic is not None:
        Z = float(result.zstatistic)
    else:
        Z = float(_stats.norm.ppf(1.0 - p)) if p < 1.0 else 0.0
    r = Z / np.sqrt(n)
    eff = ("large" if abs(r) >= 0.5 else
           "medium" if abs(r) >= 0.3 else
           "small" if abs(r) >= 0.1 else "negligible")
    p_str = "p<0.0001" if p < 0.0001 else f"p={p:.4f}"
    print(f"  Wilcoxon [{label}]: W={W:.0f}, {p_str}, Z={Z:.2f}, r={r:.2f} ({eff} effect)")
    return dict(label=label, W=W, p=p, Z=Z, r=r, n=n, effect_size=eff)
