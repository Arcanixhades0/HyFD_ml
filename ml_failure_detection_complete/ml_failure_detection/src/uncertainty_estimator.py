"""
Prediction Uncertainty Module
Implements uncertainty estimation for ML model predictions.
Uses Monte Carlo Dropout simulation and prediction confidence analysis.
"""

import numpy as np
from typing import Dict, Optional
from sklearn.calibration import calibration_curve
from sklearn.base import BaseEstimator


class UncertaintyEstimator:
    """
    Estimates prediction uncertainty using:
    1. Prediction entropy (for classifiers with predict_proba)
    2. Confidence score distribution
    3. Low-confidence sample ratio
    """

    def __init__(self, confidence_threshold: float = 0.7,
                 entropy_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        self.reference_entropy_mean: Optional[float] = None
        self.reference_entropy_std: Optional[float] = None

    def _compute_entropy(self, probabilities: np.ndarray) -> np.ndarray:
        """Shannon entropy of prediction probability distributions."""
        # Clip to avoid log(0)
        probs = np.clip(probabilities, 1e-10, 1.0)
        if probs.ndim == 1:
            probs = np.column_stack([1 - probs, probs])
        entropy = -np.sum(probs * np.log2(probs), axis=1)
        return entropy

    def _compute_confidence(self, probabilities: np.ndarray) -> np.ndarray:
        """Max class probability as confidence score."""
        if probabilities.ndim == 1:
            return np.maximum(probabilities, 1 - probabilities)
        return np.max(probabilities, axis=1)

    def fit_reference(self, reference_probabilities: np.ndarray):
        """
        Compute reference entropy statistics from training/validation set.
        Used to detect if production uncertainty deviates significantly.
        """
        entropy = self._compute_entropy(reference_probabilities)
        self.reference_entropy_mean = float(np.mean(entropy))
        self.reference_entropy_std = float(np.std(entropy))
        return self

    def analyze(self, probabilities: np.ndarray) -> Dict:
        """
        Analyze uncertainty of a batch of predictions.
        Returns uncertainty metrics and failure flag.
        """
        entropy = self._compute_entropy(probabilities)
        confidence = self._compute_confidence(probabilities)

        low_confidence_mask = confidence < self.confidence_threshold
        high_entropy_mask = entropy > self.entropy_threshold

        mean_entropy = float(np.mean(entropy))
        mean_confidence = float(np.mean(confidence))
        low_conf_ratio = float(np.mean(low_confidence_mask))
        high_entropy_ratio = float(np.mean(high_entropy_mask))

        # Detect if uncertainty has increased vs reference
        uncertainty_shift = False
        z_score = None
        if self.reference_entropy_mean is not None:
            z_score = (mean_entropy - self.reference_entropy_mean) / (
                self.reference_entropy_std + 1e-9
            )
            uncertainty_shift = abs(z_score) > 2.0  # 2-sigma rule

        failure_detected = (
            low_conf_ratio > 0.3 or      # >30% low-confidence predictions
            high_entropy_ratio > 0.3 or   # >30% high-entropy predictions
            uncertainty_shift
        )

        return {
            "mean_entropy": mean_entropy,
            "mean_confidence": mean_confidence,
            "low_confidence_ratio": low_conf_ratio,
            "high_entropy_ratio": high_entropy_ratio,
            "uncertainty_shift_detected": uncertainty_shift,
            "z_score": float(z_score) if z_score is not None else None,
            "failure_detected": failure_detected,
            "uncertainty_score": float(mean_entropy + low_conf_ratio),
            "n_samples": len(probabilities),
            "n_low_confidence": int(np.sum(low_confidence_mask))
        }


class ModelCalibrationChecker:
    """
    Checks if model confidence is well-calibrated.
    A miscalibrated model (overconfident or underconfident) is a failure signal.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.reference_ece: Optional[float] = None

    def expected_calibration_error(self, y_true: np.ndarray,
                                   y_prob: np.ndarray) -> float:
        """
        ECE: average gap between predicted confidence and actual accuracy.
        Lower is better. ECE > 0.1 indicates miscalibration.
        """
        if y_prob.ndim > 1:
            y_prob = np.max(y_prob, axis=1)

        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        n = len(y_true)

        for i in range(self.n_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (y_prob >= lo) & (y_prob < hi)
            if mask.sum() == 0:
                continue
            bin_acc = np.mean(y_true[mask] == (y_prob[mask] > 0.5).astype(int))
            bin_conf = np.mean(y_prob[mask])
            bin_weight = mask.sum() / n
            ece += bin_weight * abs(bin_acc - bin_conf)

        return float(ece)

    def fit_reference(self, y_true: np.ndarray, y_prob: np.ndarray):
        self.reference_ece = self.expected_calibration_error(y_true, y_prob)
        return self

    def check(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
        ece = self.expected_calibration_error(y_true, y_prob)
        ece_increase = None
        if self.reference_ece is not None:
            ece_increase = ece - self.reference_ece

        return {
            "ece": ece,
            "reference_ece": self.reference_ece,
            "ece_increase": ece_increase,
            "miscalibration_detected": ece > 0.1 or (
                ece_increase is not None and ece_increase > 0.05
            )
        }
