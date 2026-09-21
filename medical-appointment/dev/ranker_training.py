from __future__ import annotations

import hashlib
import numpy as np


def fold_for(transcript_id: str, folds: int) -> int:
    digest = hashlib.sha1(str(transcript_id).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float):
    means = X.mean(axis=0)
    scales = X.std(axis=0)
    scales = np.where(scales < 1e-9, 1.0, scales)
    Z = (X - means) / scales
    intercept = float(y.mean())
    centered = y - intercept
    eye = np.eye(Z.shape[1], dtype=np.float64)
    weights = np.linalg.solve(Z.T @ Z + float(alpha) * eye, Z.T @ centered)
    return means, scales, weights, intercept


def predict(X, means, scales, weights, intercept):
    scales = np.where(np.asarray(scales) < 1e-9, 1.0, scales)
    Z = (X - means) / scales
    return intercept + Z @ weights
