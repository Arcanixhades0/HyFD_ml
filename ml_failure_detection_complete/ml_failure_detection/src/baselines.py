"""
Baseline Single-Signal Detection Methods
=========================================
Implements individual monitoring approaches for comparison with HyFD.
These are the "ablation" baselines used in the paper's experiments.
"""

import numpy as np
from typing import Dict, Optional, Any
from scipy import stats


class DriftOnlyDetector:
    """Baseline: detect failures using data drift only (KS-test)."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.reference_data: Optional[np.ndarray] = None

    def fit(self, X_ref: np.ndarray, **kwargs):
        self.reference_data = X_ref
        return self

    def detect(self, X_prod: np.ndarray, **kwargs) -> Dict:
        drifted = 0
        total = X_prod.shape[1] if X_prod.ndim > 1 else 1

        ref = self.reference_data
        if ref.ndim == 1:
            ref = ref.reshape(-1, 1)
        if X_prod.ndim == 1:
            X_prod = X_prod.reshape(-1, 1)

        for i in range(total):
            _, p = stats.ks_2samp(ref[:, i], X_prod[:, i])
            if p < self.alpha:
                drifted += 1

        drift_ratio = drifted / total
        return {
            "failure_detected": drift_ratio > 0.2,
            "composite_score": drift_ratio,
            "method": "DriftOnly"
        }


class UncertaintyOnlyDetector:
    """Baseline: detect failures using prediction uncertainty only."""

    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        self.ref_entropy_mean: Optional[float] = None
        self.ref_entropy_std: Optional[float] = None
        self.model: Optional[Any] = None

    def fit(self, X_ref: np.ndarray, model: Any, **kwargs):
        self.model = model
        proba = model.predict_proba(X_ref)
        probs = np.clip(proba, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log2(probs), axis=1)
        self.ref_entropy_mean = float(np.mean(entropy))
        self.ref_entropy_std = float(np.std(entropy)) + 1e-9
        return self

    def detect(self, X_prod: np.ndarray, **kwargs) -> Dict:
        proba = self.model.predict_proba(X_prod)
        probs = np.clip(proba, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log2(probs), axis=1)
        confidence = np.max(proba, axis=1)

        mean_entropy = float(np.mean(entropy))
        low_conf_ratio = float(np.mean(confidence < self.confidence_threshold))

        z_score = abs((mean_entropy - self.ref_entropy_mean) / self.ref_entropy_std)
        uncertainty_score = (low_conf_ratio + z_score / 10.0)

        return {
            "failure_detected": low_conf_ratio > 0.3 or z_score > 2.0,
            "composite_score": float(np.clip(uncertainty_score, 0, 1)),
            "method": "UncertaintyOnly"
        }


class QualityOnlyDetector:
    """Baseline: detect failures using data quality checks only."""

    def __init__(self, missing_threshold: float = 0.05,
                 outlier_threshold: float = 0.1):
        self.missing_threshold = missing_threshold
        self.outlier_threshold = outlier_threshold
        self.ref_means: Optional[np.ndarray] = None
        self.ref_stds: Optional[np.ndarray] = None

    def fit(self, X_ref: np.ndarray, **kwargs):
        self.ref_means = np.nanmean(X_ref, axis=0)
        self.ref_stds = np.nanstd(X_ref, axis=0) + 1e-9
        return self

    def detect(self, X_prod: np.ndarray, **kwargs) -> Dict:
        # Missing rate
        missing_rate = float(np.isnan(X_prod).mean())

        # Outlier rate (z-score > 3.5)
        z_scores = np.abs((X_prod - self.ref_means) / self.ref_stds)
        outlier_rate = float(np.nanmean(z_scores > 3.5))

        quality_score = missing_rate * 0.5 + outlier_rate * 0.5

        return {
            "failure_detected": (missing_rate > self.missing_threshold or
                                  outlier_rate > self.outlier_threshold),
            "composite_score": float(np.clip(quality_score, 0, 1)),
            "method": "QualityOnly"
        }


class OODOnlyDetector:
    """Baseline: OOD detection using Mahalanobis distance only."""

    def __init__(self, threshold_percentile: float = 97.5):
        self.threshold_percentile = threshold_percentile
        self.mean: Optional[np.ndarray] = None
        self.inv_cov: Optional[np.ndarray] = None
        self.threshold: Optional[float] = None

    def fit(self, X_ref: np.ndarray, **kwargs):
        self.mean = np.mean(X_ref, axis=0)
        cov = np.cov(X_ref.T) + np.eye(X_ref.shape[1]) * 1e-6
        self.inv_cov = np.linalg.inv(cov)
        diff = X_ref - self.mean
        distances = np.sqrt(np.einsum('ij,jk,ik->i', diff, self.inv_cov, diff))
        self.threshold = float(np.percentile(distances, self.threshold_percentile))
        return self

    def detect(self, X_prod: np.ndarray, **kwargs) -> Dict:
        # Fill NaN with mean for distance computation
        X_filled = np.where(np.isnan(X_prod), self.mean, X_prod)
        diff = X_filled - self.mean
        distances = np.sqrt(np.clip(
            np.einsum('ij,jk,ik->i', diff, self.inv_cov, diff), 0, None
        ))
        ood_rate = float(np.mean(distances > self.threshold))

        return {
            "failure_detected": ood_rate > 0.05,
            "composite_score": float(np.clip(ood_rate * 2, 0, 1)),
            "method": "OODOnly"
        }


def get_all_baselines(X_ref: np.ndarray, model: Any) -> Dict:
    """
    Initialize and fit all baseline detectors.
    Returns dict: method_name -> detector instance.
    """
    baselines = {
        "DriftOnly": DriftOnlyDetector().fit(X_ref),
        "UncertaintyOnly": UncertaintyOnlyDetector().fit(X_ref, model=model),
        "QualityOnly": QualityOnlyDetector().fit(X_ref),
        "OODOnly": OODOnlyDetector().fit(X_ref),
    }
    return baselines
