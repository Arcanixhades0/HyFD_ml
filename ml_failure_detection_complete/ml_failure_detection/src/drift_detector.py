"""
Drift Detection Module
Implements statistical tests for data distribution shift detection.
Supports: KS-test, PSI (Population Stability Index), Chi-squared test.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional


def kolmogorov_smirnov_test(reference: np.ndarray, production: np.ndarray,
                             alpha: float = 0.05) -> Dict:
    """
    KS test for continuous feature distribution shift.
    Returns statistic, p-value, and drift flag.
    """
    stat, p_value = stats.ks_2samp(reference, production)
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "drift_detected": bool(p_value < alpha),
        "method": "KS-Test"
    }


def population_stability_index(reference: np.ndarray, production: np.ndarray,
                                bins: int = 10) -> Dict:
    """
    PSI measures how much a distribution has shifted.
    PSI < 0.1 : No significant change
    PSI 0.1-0.25 : Moderate change
    PSI > 0.25 : Significant change
    """
    # Create bins from reference distribution
    min_val = min(reference.min(), production.min())
    max_val = max(reference.max(), production.max())
    bin_edges = np.linspace(min_val, max_val, bins + 1)

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    prod_counts, _ = np.histogram(production, bins=bin_edges)

    # Avoid zero divisions
    ref_pct = (ref_counts + 1e-6) / (len(reference) + 1e-6 * bins)
    prod_pct = (prod_counts + 1e-6) / (len(production) + 1e-6 * bins)

    psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))

    return {
        "psi": float(psi),
        "drift_detected": bool(psi > 0.1),
        "severity": "none" if psi < 0.1 else ("moderate" if psi < 0.25 else "severe"),
        "method": "PSI"
    }


def chi_squared_test(reference: np.ndarray, production: np.ndarray,
                      bins: int = 10, alpha: float = 0.05) -> Dict:
    """
    Chi-squared test for categorical or binned continuous features.
    """
    min_val = min(reference.min(), production.min())
    max_val = max(reference.max(), production.max())
    bin_edges = np.linspace(min_val, max_val, bins + 1)

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    prod_counts, _ = np.histogram(production, bins=bin_edges)

    # Normalize prod counts to same total as reference
    scale = ref_counts.sum() / (prod_counts.sum() + 1e-9)
    expected = prod_counts * scale + 1e-6

    stat, p_value = stats.chisquare(ref_counts + 1e-6, f_exp=expected)

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "drift_detected": bool(p_value < alpha),
        "method": "Chi-Squared"
    }


class DataDriftDetector:
    """
    Multi-test drift detector for all features in a dataset.
    Combines KS-test and PSI for robust drift detection.
    """

    def __init__(self, alpha: float = 0.05, psi_threshold: float = 0.1):
        self.alpha = alpha
        self.psi_threshold = psi_threshold
        self.reference_data: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None

    def fit(self, reference_data: np.ndarray,
            feature_names: Optional[List[str]] = None):
        """Store reference distribution."""
        self.reference_data = reference_data
        n_features = reference_data.shape[1] if reference_data.ndim > 1 else 1
        self.feature_names = feature_names or [f"feature_{i}" for i in range(n_features)]
        return self

    def detect(self, production_data: np.ndarray) -> Dict:
        """
        Run drift detection on production data.
        Returns per-feature results and overall drift flag.
        """
        if self.reference_data is None:
            raise ValueError("Call fit() before detect()")

        ref = self.reference_data
        prod = production_data

        if ref.ndim == 1:
            ref = ref.reshape(-1, 1)
        if prod.ndim == 1:
            prod = prod.reshape(-1, 1)

        feature_results = {}
        drifted_features = []

        for i, fname in enumerate(self.feature_names):
            ref_col = ref[:, i]
            prod_col = prod[:, i]

            ks_result = kolmogorov_smirnov_test(ref_col, prod_col, self.alpha)
            psi_result = population_stability_index(ref_col, prod_col)

            # Feature drifted if BOTH signals agree
            drifted = ks_result["drift_detected"] or psi_result["drift_detected"]
            if drifted:
                drifted_features.append(fname)

            feature_results[fname] = {
                "ks_test": ks_result,
                "psi": psi_result,
                "drift_detected": drifted
            }

        drift_ratio = len(drifted_features) / len(self.feature_names)

        return {
            "overall_drift_detected": drift_ratio > 0.2,  # >20% features drifted
            "drift_ratio": drift_ratio,
            "drifted_features": drifted_features,
            "n_drifted": len(drifted_features),
            "n_total_features": len(self.feature_names),
            "feature_results": feature_results,
            "drift_score": float(np.mean([
                r["psi"]["psi"] for r in feature_results.values()
            ]))
        }
