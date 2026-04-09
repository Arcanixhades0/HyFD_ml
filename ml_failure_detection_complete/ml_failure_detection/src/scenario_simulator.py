"""
Production Failure Scenario Simulator
======================================
Simulates common production failure modes for evaluation:
  1. Data drift (covariate shift)
  2. Noisy features (sensor/data pipeline noise)
  3. Missing / corrupted data
  4. Out-of-distribution (OOD) samples
  5. Concept drift (label distribution shift)
"""

import numpy as np
from typing import Tuple, Dict
from sklearn.datasets import make_classification


def generate_reference_dataset(n_samples: int = 2000, n_features: int = 10,
                                n_informative: int = 5, random_state: int = 42
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a clean reference (training) dataset.
    Uses sklearn's make_classification for reproducibility.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=2,
        n_clusters_per_class=2,
        flip_y=0.02,  # Slight label noise (realistic)
        random_state=random_state
    )
    return X, y


# ────────────────────────────────────────────────────────────────────────────
# Failure Scenario Generators
# ────────────────────────────────────────────────────────────────────────────

def scenario_no_failure(X_ref: np.ndarray, n_samples: int = 500,
                         random_state: int = 0) -> Tuple[np.ndarray, str]:
    """Healthy production: same distribution as reference."""
    rng = np.random.RandomState(random_state)
    indices = rng.choice(len(X_ref), size=n_samples, replace=True)
    X_prod = X_ref[indices].copy()
    # Add tiny noise to avoid identical distributions
    X_prod += rng.normal(0, 0.01, X_prod.shape)
    return X_prod, "no_failure"


def scenario_data_drift(X_ref: np.ndarray, n_samples: int = 500,
                         shift_magnitude: float = 2.0,
                         random_state: int = 1) -> Tuple[np.ndarray, str]:
    """
    Covariate shift: features shift to a different mean.
    Simulates: seasonal change, new user population, region change.
    """
    rng = np.random.RandomState(random_state)
    indices = rng.choice(len(X_ref), size=n_samples, replace=True)
    X_prod = X_ref[indices].copy()

    # Shift first 60% of features
    n_shifted = max(1, int(X_prod.shape[1] * 0.6))
    shift_vector = rng.uniform(shift_magnitude * 0.5, shift_magnitude,
                                size=n_shifted)
    X_prod[:, :n_shifted] += shift_vector
    return X_prod, "data_drift"


def scenario_noisy_features(X_ref: np.ndarray, n_samples: int = 500,
                              noise_std: float = 3.0,
                              random_state: int = 2) -> Tuple[np.ndarray, str]:
    """
    Feature noise: significant random noise added to observations.
    Simulates: sensor degradation, data pipeline bugs, measurement errors.
    """
    rng = np.random.RandomState(random_state)
    indices = rng.choice(len(X_ref), size=n_samples, replace=True)
    X_prod = X_ref[indices].copy()

    # Add heavy Gaussian noise to all features
    noise = rng.normal(0, noise_std, X_prod.shape)
    X_prod += noise
    return X_prod, "noisy_features"


def scenario_missing_data(X_ref: np.ndarray, n_samples: int = 500,
                           missing_rate: float = 0.25,
                           random_state: int = 3) -> Tuple[np.ndarray, str]:
    """
    Missing values: NaN injected into random feature positions.
    Simulates: data pipeline failures, sensor outages, API errors.
    """
    rng = np.random.RandomState(random_state)
    indices = rng.choice(len(X_ref), size=n_samples, replace=True)
    X_prod = X_ref[indices].copy()

    # Randomly set values to NaN
    mask = rng.rand(*X_prod.shape) < missing_rate
    X_prod[mask] = np.nan
    return X_prod, "missing_data"


def scenario_ood_samples(X_ref: np.ndarray, n_samples: int = 500,
                          ood_fraction: float = 0.4,
                          ood_distance: float = 6.0,
                          random_state: int = 4) -> Tuple[np.ndarray, str]:
    """
    Out-of-distribution samples: fraction of batch is from different domain.
    Simulates: wrong input format, model serving unintended use cases.
    """
    rng = np.random.RandomState(random_state)
    n_normal = int(n_samples * (1 - ood_fraction))
    n_ood = n_samples - n_normal

    # Normal samples
    indices = rng.choice(len(X_ref), size=n_normal, replace=True)
    X_normal = X_ref[indices].copy()

    # OOD samples: far from reference distribution
    ref_std = np.std(X_ref, axis=0)
    ood_center = np.mean(X_ref, axis=0) + ood_distance * ref_std
    X_ood = rng.normal(loc=ood_center, scale=ref_std * 0.5,
                        size=(n_ood, X_ref.shape[1]))

    X_prod = np.vstack([X_normal, X_ood])
    rng.shuffle(X_prod)
    return X_prod, "ood_samples"


def scenario_corrupted_data(X_ref: np.ndarray, n_samples: int = 500,
                              corruption_rate: float = 0.15,
                              random_state: int = 5) -> Tuple[np.ndarray, str]:
    """
    Data corruption: extreme outlier values injected.
    Simulates: database corruption, type casting errors, encoding issues.
    """
    rng = np.random.RandomState(random_state)
    indices = rng.choice(len(X_ref), size=n_samples, replace=True)
    X_prod = X_ref[indices].copy()

    # Inject corrupted extreme values
    corruption_mask = rng.rand(*X_prod.shape) < corruption_rate
    extreme_values = rng.choice([-999, 999, -9999, 9999],
                                 size=corruption_mask.sum())
    X_prod[corruption_mask] = extreme_values
    return X_prod, "corrupted_data"


def get_all_scenarios(X_ref: np.ndarray, n_samples: int = 500) -> Dict:
    """
    Generate all failure scenarios and the no-failure baseline.
    Returns dict: scenario_name -> (X_prod, scenario_label).
    """
    scenarios = {
        "no_failure": scenario_no_failure(X_ref, n_samples),
        "data_drift": scenario_data_drift(X_ref, n_samples),
        "noisy_features": scenario_noisy_features(X_ref, n_samples),
        "missing_data": scenario_missing_data(X_ref, n_samples),
        "ood_samples": scenario_ood_samples(X_ref, n_samples),
        "corrupted_data": scenario_corrupted_data(X_ref, n_samples),
    }
    return scenarios
