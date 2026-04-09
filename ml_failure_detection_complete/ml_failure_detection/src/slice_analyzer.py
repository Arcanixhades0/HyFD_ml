"""
Slice-Based Performance Analysis Module
Detects subgroup (slice) level failures in ML model predictions.
Implements automatic slice discovery and performance monitoring.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_slice_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           mask: np.ndarray) -> Dict:
    """Compute performance metrics for a data slice (subset)."""
    if mask.sum() == 0:
        return {"n_samples": 0, "accuracy": None, "f1": None}

    yt = y_true[mask]
    yp = y_pred[mask]

    try:
        acc = accuracy_score(yt, yp)
        f1 = f1_score(yt, yp, average="weighted", zero_division=0)
        prec = precision_score(yt, yp, average="weighted", zero_division=0)
        rec = recall_score(yt, yp, average="weighted", zero_division=0)
    except Exception:
        acc = f1 = prec = rec = 0.0

    return {
        "n_samples": int(mask.sum()),
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec)
    }


class SliceBasedAnalyzer:
    """
    Automatically discovers underperforming data slices.

    Strategy:
    1. Manual slices: user-defined feature buckets (e.g., age groups)
    2. Automatic slices: decision tree finds worst-performing subgroups
    3. Compares slice performance vs overall baseline
    """

    def __init__(self, performance_drop_threshold: float = 0.1,
                 min_slice_size: int = 30):
        self.performance_drop_threshold = performance_drop_threshold
        self.min_slice_size = min_slice_size
        self.baseline_accuracy: Optional[float] = None
        self.baseline_f1: Optional[float] = None

    def fit_reference(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Establish baseline performance from reference/validation data."""
        self.baseline_accuracy = float(accuracy_score(y_true, y_pred))
        self.baseline_f1 = float(f1_score(y_true, y_pred, average="weighted",
                                           zero_division=0))
        return self

    def analyze_manual_slices(self, X: np.ndarray, y_true: np.ndarray,
                               y_pred: np.ndarray,
                               feature_names: Optional[List[str]] = None,
                               n_bins: int = 3) -> Dict:
        """
        Analyze performance across quantile bins of each feature.
        Flags slices where performance drops significantly vs baseline.
        """
        if feature_names is None:
            n_feat = X.shape[1] if X.ndim > 1 else 1
            feature_names = [f"feature_{i}" for i in range(n_feat)]

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        failing_slices = []
        all_slices = []

        for i, fname in enumerate(feature_names):
            col = X[:, i]
            try:
                bin_edges = np.quantile(col, np.linspace(0, 1, n_bins + 1))
                bin_edges = np.unique(bin_edges)  # Remove duplicates
                if len(bin_edges) < 3:
                    continue
            except Exception:
                continue

            for j in range(len(bin_edges) - 1):
                lo, hi = bin_edges[j], bin_edges[j + 1]
                mask = (col >= lo) & (col <= hi)

                if mask.sum() < self.min_slice_size:
                    continue

                metrics = compute_slice_metrics(y_true, y_pred, mask)
                if metrics["accuracy"] is None:
                    continue

                acc_drop = (self.baseline_accuracy or 0) - metrics["accuracy"]
                f1_drop = (self.baseline_f1 or 0) - metrics["f1"]

                slice_info = {
                    "feature": fname,
                    "range": (float(lo), float(hi)),
                    "n_samples": metrics["n_samples"],
                    "accuracy": metrics["accuracy"],
                    "f1": metrics["f1"],
                    "accuracy_drop": float(acc_drop),
                    "f1_drop": float(f1_drop),
                    "failing": acc_drop > self.performance_drop_threshold
                }
                all_slices.append(slice_info)
                if slice_info["failing"]:
                    failing_slices.append(slice_info)

        return {
            "baseline_accuracy": self.baseline_accuracy,
            "baseline_f1": self.baseline_f1,
            "all_slices": all_slices,
            "failing_slices": failing_slices,
            "n_failing_slices": len(failing_slices),
            "failure_detected": len(failing_slices) > 0,
            "worst_slice": max(all_slices, key=lambda s: s["accuracy_drop"],
                               default=None)
        }

    def auto_discover_failing_slices(self, X: np.ndarray, y_true: np.ndarray,
                                      y_pred: np.ndarray,
                                      feature_names: Optional[List[str]] = None,
                                      max_depth: int = 3) -> Dict:
        """
        Use a shallow decision tree to automatically discover failing subgroups.
        The tree is trained to predict ERRORS (where model is wrong).
        Leaves with high error rate = failing slices.
        """
        if feature_names is None:
            n_feat = X.shape[1] if X.ndim > 1 else 1
            feature_names = [f"feature_{i}" for i in range(n_feat)]

        # Binary error label: 1 = model is wrong, 0 = correct
        error_label = (y_true != y_pred).astype(int)

        # Train tree to find patterns in errors
        tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=20,
                                       random_state=42)
        tree.fit(X, error_label)

        # Get leaf assignments
        leaf_ids = tree.apply(X)
        unique_leaves = np.unique(leaf_ids)

        discovered_slices = []
        for leaf in unique_leaves:
            mask = leaf_ids == leaf
            if mask.sum() < self.min_slice_size:
                continue

            leaf_error_rate = float(np.mean(error_label[mask]))
            overall_error_rate = float(np.mean(error_label))
            slice_acc = 1.0 - leaf_error_rate

            if (overall_error_rate - leaf_error_rate) < -self.performance_drop_threshold:
                discovered_slices.append({
                    "leaf_id": int(leaf),
                    "n_samples": int(mask.sum()),
                    "error_rate": leaf_error_rate,
                    "overall_error_rate": overall_error_rate,
                    "accuracy": slice_acc,
                    "excess_error_rate": float(leaf_error_rate - overall_error_rate),
                    "failing": True
                })

        # Sort by severity
        discovered_slices.sort(key=lambda s: s["excess_error_rate"], reverse=True)

        return {
            "discovered_failing_slices": discovered_slices,
            "n_discovered": len(discovered_slices),
            "failure_detected": len(discovered_slices) > 0,
            "max_excess_error": max(
                [s["excess_error_rate"] for s in discovered_slices], default=0.0
            )
        }
