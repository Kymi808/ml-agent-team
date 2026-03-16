"""Metric computation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn import metrics as sklearn_metrics

from ..core.types import ProblemType


def get_metrics_for_problem_type(problem_type: ProblemType) -> list[str]:
    """Return the list of appropriate metric names for a problem type."""
    mapping: dict[ProblemType, list[str]] = {
        ProblemType.BINARY_CLASSIFICATION: [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc_roc",
            "log_loss",
        ],
        ProblemType.MULTICLASS_CLASSIFICATION: [
            "accuracy",
            "f1_macro",
            "f1_weighted",
            "precision_weighted",
            "recall_weighted",
        ],
        ProblemType.REGRESSION: ["r2", "rmse", "mae", "mape"],
        ProblemType.CLUSTERING: ["silhouette_score", "calinski_harabasz"],
        ProblemType.ANOMALY_DETECTION: ["precision", "recall", "f1"],
    }
    return mapping.get(problem_type, ["accuracy"])


def compute_all_classification_metrics(
    y_true: Any, y_pred: Any, y_proba: Any | None = None
) -> dict[str, float]:
    """Compute a comprehensive set of classification metrics."""
    result: dict[str, float] = {
        "accuracy": float(sklearn_metrics.accuracy_score(y_true, y_pred)),
        "f1_weighted": float(
            sklearn_metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "precision_weighted": float(
            sklearn_metrics.precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            sklearn_metrics.recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }

    n_classes = len(np.unique(y_true))
    if n_classes == 2:
        result["f1"] = float(sklearn_metrics.f1_score(y_true, y_pred, zero_division=0))
        result["precision"] = float(
            sklearn_metrics.precision_score(y_true, y_pred, zero_division=0)
        )
        result["recall"] = float(sklearn_metrics.recall_score(y_true, y_pred, zero_division=0))

        if y_proba is not None:
            proba = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
            result["auc_roc"] = float(sklearn_metrics.roc_auc_score(y_true, proba))
    else:
        result["f1_macro"] = float(
            sklearn_metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)
        )

    return result


def compute_all_regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute a comprehensive set of regression metrics."""
    y_true_arr = np.array(y_true, dtype=float)
    y_pred_arr = np.array(y_pred, dtype=float)

    result: dict[str, float] = {
        "r2": float(sklearn_metrics.r2_score(y_true_arr, y_pred_arr)),
        "rmse": float(np.sqrt(sklearn_metrics.mean_squared_error(y_true_arr, y_pred_arr))),
        "mae": float(sklearn_metrics.mean_absolute_error(y_true_arr, y_pred_arr)),
    }

    nonzero = y_true_arr != 0
    if nonzero.any():
        result["mape"] = float(
            np.mean(np.abs((y_true_arr[nonzero] - y_pred_arr[nonzero]) / y_true_arr[nonzero]))
        )

    return result
