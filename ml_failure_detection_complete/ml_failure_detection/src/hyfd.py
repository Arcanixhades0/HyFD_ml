"""
HyFD: Hybrid Failure Detection System
=====================================
The proposed multi-signal failure detection framework that combines:
  1. Data drift detection (KS + PSI)
  2. Prediction uncertainty analysis
  3. Slice-based performance analysis
  4. Data quality monitoring (outliers, missing, OOD)

Implements weighted signal fusion with adaptive thresholding.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import warnings

from drift_detector import DataDriftDetector
from uncertainty_estimator import UncertaintyEstimator, ModelCalibrationChecker
from slice_analyzer import SliceBasedAnalyzer
from data_quality import DataQualityMonitor, OODDetector


# ============================================================
# PSEUDOCODE for HyFD (reference for paper)
# ============================================================
"""
ALGORITHM: HyFD - Hybrid Failure Detection

INPUT:
  M       : trained ML model
  D_ref   : reference dataset (training/validation)
  D_prod  : production data batch
  W       : signal weights [w_drift, w_uncertainty, w_slice, w_quality]
  tau     : detection threshold (default 0.5)

OUTPUT:
  failure_detected : Boolean
  failure_report   : Dict with per-signal analysis

PROCEDURE:
  1. INITIALIZE detectors with D_ref:
       DriftDetector.fit(D_ref.X)
       UncertaintyEstimator.fit_reference(M.predict_proba(D_ref.X))
       SliceAnalyzer.fit_reference(D_ref.y, M.predict(D_ref.X))
       QualityMonitor.fit(D_ref.X)
       OODDetector.fit(D_ref.X)

  2. FOR each production batch D_prod:
       a. COMPUTE per-signal scores:
            s_drift       <- DriftDetector.detect(D_prod.X).drift_score
            s_uncertainty <- UncertaintyEstimator.analyze(proba_prod).uncertainty_score
            s_slice       <- SliceAnalyzer.analyze(D_prod).max_accuracy_drop
            s_quality     <- QualityMonitor.check(D_prod.X).quality_score
            s_ood         <- OODDetector.detect(D_prod.X).ood_score

       b. NORMALIZE scores to [0, 1]

       c. COMPUTE composite score:
            S_composite = w_drift * s_drift
                        + w_uncertainty * s_uncertainty
                        + w_slice * s_slice
                        + w_quality * s_quality
                        + w_ood * s_ood

       d. DETECT failure:
            IF S_composite > tau:
                failure_detected = True
                failure_type = argmax(signal_scores)
            ELSE:
                failure_detected = False

  3. RETURN failure_detected, detailed report

COMPLEXITY: O(n * d) per batch, n = batch size, d = n_features
"""


class HyFD:
    """
    Hybrid Failure Detection System for Production ML.

    Integrates multiple monitoring signals into a unified failure score
    using configurable weighted fusion.
    """

    DEFAULT_WEIGHTS = {
        "drift": 0.30,
        "uncertainty": 0.25,
        "slice": 0.25,
        "quality": 0.10,
        "ood": 0.10
    }

    def __init__(self,
                 signal_weights: Optional[Dict[str, float]] = None,
                 detection_threshold: float = 0.35,
                 confidence_threshold: float = 0.7,
                 drift_alpha: float = 0.05,
                 min_slice_size: int = 30):
        """
        Parameters
        ----------
        signal_weights : dict, optional
            Weights for each detection signal. Must sum to 1.0.
        detection_threshold : float
            Composite score above which failure is declared (0-1).
        confidence_threshold : float
            Minimum acceptable prediction confidence.
        drift_alpha : float
            Significance level for drift statistical tests.
        min_slice_size : int
            Minimum samples per data slice for analysis.
        """
        self.signal_weights = signal_weights or self.DEFAULT_WEIGHTS
        self.detection_threshold = detection_threshold
        self.is_fitted = False

        # Initialize individual detectors
        self.drift_detector = DataDriftDetector(alpha=drift_alpha)
        self.uncertainty_estimator = UncertaintyEstimator(
            confidence_threshold=confidence_threshold
        )
        self.calibration_checker = ModelCalibrationChecker()
        self.slice_analyzer = SliceBasedAnalyzer(min_slice_size=min_slice_size)
        self.quality_monitor = DataQualityMonitor()
        self.ood_detector = OODDetector()

        # History for trend analysis
        self.detection_history: List[Dict] = []

    def fit(self, X_ref: np.ndarray, y_ref: np.ndarray,
            model: Any,
            feature_names: Optional[List[str]] = None):
        """
        Fit all detectors using reference (training/validation) data.

        Parameters
        ----------
        X_ref : array, shape (n_samples, n_features)
        y_ref : array, shape (n_samples,)
        model : sklearn-compatible model with predict() and predict_proba()
        feature_names : list of str, optional
        """
        n_feat = X_ref.shape[1]
        self.feature_names = feature_names or [f"feature_{i}" for i in range(n_feat)]
        self.model = model

        # Get reference predictions
        y_pred_ref = model.predict(X_ref)
        y_proba_ref = model.predict_proba(X_ref)

        # Fit all detectors
        self.drift_detector.fit(X_ref, self.feature_names)
        self.uncertainty_estimator.fit_reference(y_proba_ref)
        self.calibration_checker.fit_reference(y_ref, y_proba_ref[:, 1])
        self.slice_analyzer.fit_reference(y_ref, y_pred_ref)
        self.quality_monitor.fit(X_ref, self.feature_names)
        self.ood_detector.fit(X_ref)

        self.is_fitted = True
        return self

    def _normalize_score(self, score: float, min_val: float = 0.0,
                          max_val: float = 1.0) -> float:
        """Clip score to [0, 1] range."""
        return float(np.clip((score - min_val) / (max_val - min_val + 1e-9), 0, 1))

    def detect(self, X_prod: np.ndarray, y_prod: Optional[np.ndarray] = None,
               batch_id: Optional[str] = None) -> Dict:
        """
        Run full failure detection on a production batch.

        Parameters
        ----------
        X_prod : array, shape (n_samples, n_features)
        y_prod : array, optional — if available, enables slice and calibration analysis
        batch_id : str, optional

        Returns
        -------
        dict with failure_detected, composite_score, per_signal_scores, details
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before detect()")

        results = {}
        signal_scores = {}

        # ── Signal 1: Data Drift ──────────────────────────────────
        drift_result = self.drift_detector.detect(X_prod)
        results["drift"] = drift_result
        # Normalize PSI-based drift score
        signal_scores["drift"] = self._normalize_score(
            drift_result["drift_score"], 0, 0.5
        )

        # ── Signal 2: Prediction Uncertainty ─────────────────────
        y_proba_prod = self.model.predict_proba(X_prod)
        uncertainty_result = self.uncertainty_estimator.analyze(y_proba_prod)
        results["uncertainty"] = uncertainty_result
        signal_scores["uncertainty"] = self._normalize_score(
            uncertainty_result["uncertainty_score"], 0, 1.5
        )

        # ── Signal 3: Slice Performance ───────────────────────────
        y_pred_prod = self.model.predict(X_prod)
        if y_prod is not None:
            slice_result = self.slice_analyzer.analyze_manual_slices(
                X_prod, y_prod, y_pred_prod, self.feature_names
            )
            results["slice"] = slice_result
            max_drop = max(
                [s["accuracy_drop"] for s in slice_result["all_slices"]], default=0.0
            )
            signal_scores["slice"] = self._normalize_score(max_drop, 0, 0.5)
        else:
            # Without ground truth, use proxy: entropy spread across slices
            results["slice"] = {"failure_detected": False, "note": "No labels available"}
            signal_scores["slice"] = 0.0

        # ── Signal 4: Data Quality ────────────────────────────────
        quality_result = self.quality_monitor.check(X_prod)
        results["quality"] = quality_result
        signal_scores["quality"] = self._normalize_score(
            quality_result["quality_score"], 0, 1.0
        )

        # ── Signal 5: OOD Detection ───────────────────────────────
        ood_result = self.ood_detector.detect(X_prod)
        results["ood"] = ood_result
        signal_scores["ood"] = self._normalize_score(
            ood_result["ood_score"], 0, 2.0
        )

        # ── Weighted Composite Score ──────────────────────────────
        composite_score = sum(
            self.signal_weights.get(sig, 0) * score
            for sig, score in signal_scores.items()
        )

        failure_detected = composite_score > self.detection_threshold

        # Identify dominant failure signal
        primary_signal = max(signal_scores, key=signal_scores.get)

        detection_record = {
            "batch_id": batch_id,
            "failure_detected": failure_detected,
            "composite_score": float(composite_score),
            "detection_threshold": self.detection_threshold,
            "signal_scores": signal_scores,
            "primary_failure_signal": primary_signal if failure_detected else None,
            "per_signal_results": results,
            "n_samples": len(X_prod)
        }

        self.detection_history.append(detection_record)
        return detection_record

    def get_summary_report(self) -> pd.DataFrame:
        """Return a DataFrame summary of all detection runs."""
        if not self.detection_history:
            return pd.DataFrame()

        rows = []
        for rec in self.detection_history:
            row = {
                "batch_id": rec["batch_id"],
                "failure_detected": rec["failure_detected"],
                "composite_score": rec["composite_score"],
                "score_drift": rec["signal_scores"].get("drift", 0),
                "score_uncertainty": rec["signal_scores"].get("uncertainty", 0),
                "score_slice": rec["signal_scores"].get("slice", 0),
                "score_quality": rec["signal_scores"].get("quality", 0),
                "score_ood": rec["signal_scores"].get("ood", 0),
                "primary_signal": rec["primary_failure_signal"]
            }
            rows.append(row)
        return pd.DataFrame(rows)
