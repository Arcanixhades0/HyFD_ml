"""
Data Quality Monitor Module
Detects data quality issues in production:
- Missing values
- Out-of-range values
- Corrupted / noisy features
- Out-of-distribution samples
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.stats import zscore


class DataQualityMonitor:
    """
    Monitors data quality in production batches.
    Detects: missing values, outliers, feature range violations, noise.
    """

    def __init__(self, missing_threshold: float = 0.05,
                 outlier_threshold: float = 0.1,
                 z_score_cutoff: float = 3.5):
        self.missing_threshold = missing_threshold
        self.outlier_threshold = outlier_threshold
        self.z_score_cutoff = z_score_cutoff

        # Reference statistics (set during fit)
        self.ref_means: Optional[np.ndarray] = None
        self.ref_stds: Optional[np.ndarray] = None
        self.ref_mins: Optional[np.ndarray] = None
        self.ref_maxs: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None

    def fit(self, reference_data: np.ndarray,
            feature_names: Optional[List[str]] = None):
        """Compute reference statistics from clean training data."""
        self.ref_means = np.nanmean(reference_data, axis=0)
        self.ref_stds = np.nanstd(reference_data, axis=0) + 1e-9
        self.ref_mins = np.nanmin(reference_data, axis=0)
        self.ref_maxs = np.nanmax(reference_data, axis=0)
        n_feat = reference_data.shape[1]
        self.feature_names = feature_names or [f"feature_{i}" for i in range(n_feat)]
        return self

    def check_missing(self, data: np.ndarray) -> Dict:
        """Detect missing value rate per feature and overall."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        missing_per_feature = np.isnan(data).mean(axis=0)
        overall_missing = float(np.isnan(data).mean())

        high_missing_features = [
            self.feature_names[i]
            for i in range(len(self.feature_names))
            if missing_per_feature[i] > self.missing_threshold
        ]

        return {
            "overall_missing_rate": overall_missing,
            "per_feature_missing": {
                self.feature_names[i]: float(missing_per_feature[i])
                for i in range(len(self.feature_names))
            },
            "high_missing_features": high_missing_features,
            "issue_detected": overall_missing > self.missing_threshold or
                              len(high_missing_features) > 0
        }

    def check_outliers(self, data: np.ndarray) -> Dict:
        """Detect feature values far outside the reference distribution."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Z-score relative to reference distribution
        z_scores = np.abs((data - self.ref_means) / self.ref_stds)
        outlier_mask = z_scores > self.z_score_cutoff

        outlier_rate_per_feature = np.nanmean(outlier_mask, axis=0)
        overall_outlier_rate = float(np.nanmean(outlier_mask))

        high_outlier_features = [
            self.feature_names[i]
            for i in range(len(self.feature_names))
            if outlier_rate_per_feature[i] > self.outlier_threshold
        ]

        return {
            "overall_outlier_rate": overall_outlier_rate,
            "per_feature_outlier_rate": {
                self.feature_names[i]: float(outlier_rate_per_feature[i])
                for i in range(len(self.feature_names))
            },
            "high_outlier_features": high_outlier_features,
            "issue_detected": overall_outlier_rate > self.outlier_threshold or
                              len(high_outlier_features) > 0
        }

    def check_range_violations(self, data: np.ndarray) -> Dict:
        """Check if feature values exceed training data range."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Allow 10% beyond training range
        range_span = self.ref_maxs - self.ref_mins
        lower_bound = self.ref_mins - 0.1 * range_span
        upper_bound = self.ref_maxs + 0.1 * range_span

        below_min = (data < lower_bound).mean(axis=0)
        above_max = (data > upper_bound).mean(axis=0)
        violation_rate = (below_min + above_max)

        violated_features = [
            self.feature_names[i]
            for i in range(len(self.feature_names))
            if violation_rate[i] > 0.05
        ]

        return {
            "violated_features": violated_features,
            "per_feature_violation_rate": {
                self.feature_names[i]: float(violation_rate[i])
                for i in range(len(self.feature_names))
            },
            "issue_detected": len(violated_features) > 0
        }

    def check(self, production_data: np.ndarray) -> Dict:
        """Run all quality checks and return combined report."""
        missing_result = self.check_missing(production_data)
        outlier_result = self.check_outliers(production_data)
        range_result = self.check_range_violations(production_data)

        any_issue = (
            missing_result["issue_detected"] or
            outlier_result["issue_detected"] or
            range_result["issue_detected"]
        )

        # Composite quality score (0 = perfect, 1 = very poor)
        quality_score = (
            missing_result["overall_missing_rate"] * 0.4 +
            outlier_result["overall_outlier_rate"] * 0.4 +
            (0.2 if range_result["issue_detected"] else 0.0)
        )

        return {
            "quality_issue_detected": any_issue,
            "quality_score": float(quality_score),
            "missing_check": missing_result,
            "outlier_check": outlier_result,
            "range_check": range_result
        }


class OODDetector:
    """
    Out-of-Distribution (OOD) detector using Mahalanobis distance.
    Samples with large Mahalanobis distance from training distribution
    are flagged as OOD.
    """

    def __init__(self, threshold_percentile: float = 97.5):
        self.threshold_percentile = threshold_percentile
        self.mean: Optional[np.ndarray] = None
        self.inv_cov: Optional[np.ndarray] = None
        self.threshold: Optional[float] = None

    def fit(self, reference_data: np.ndarray):
        """Fit OOD detector on reference (training) data."""
        self.mean = np.mean(reference_data, axis=0)
        cov = np.cov(reference_data.T)
        # Regularize covariance matrix for stability
        cov += np.eye(cov.shape[0]) * 1e-6
        self.inv_cov = np.linalg.inv(cov)

        # Compute threshold from reference distances
        ref_distances = self._mahalanobis(reference_data)
        self.threshold = float(np.percentile(ref_distances, self.threshold_percentile))
        return self

    def _mahalanobis(self, X: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distance for each sample."""
        diff = X - self.mean
        # Vectorized Mahalanobis
        left = diff @ self.inv_cov
        distances = np.sqrt(np.einsum('ij,ij->i', left, diff))
        return distances

    def detect(self, production_data: np.ndarray) -> Dict:
        """Detect OOD samples in production batch."""
        distances = self._mahalanobis(production_data)
        ood_mask = distances > self.threshold
        ood_rate = float(np.mean(ood_mask))

        return {
            "ood_rate": ood_rate,
            "mean_distance": float(np.mean(distances)),
            "max_distance": float(np.max(distances)),
            "threshold": self.threshold,
            "ood_detected": ood_rate > 0.05,  # >5% OOD samples is alarming
            "n_ood": int(np.sum(ood_mask)),
            "n_samples": len(production_data),
            "ood_score": float(ood_rate + np.mean(distances) / (self.threshold + 1e-9))
        }
