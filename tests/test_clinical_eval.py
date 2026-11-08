"""Tests for clinical metrics and attention diagnostics."""

import pytest
import numpy as np
from patho_attn.metrics.clinical_eval import ClinicalEvaluator


def test_roc_auc_perfect_separation():
    """Verify AUC calculation on perfectly separable predictions."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    auc = ClinicalEvaluator.compute_binary_auc(y_true, y_score)
    assert np.isclose(auc, 1.0)


def test_roc_auc_random_classifier():
    """Verify AUC on random predictions is ~0.50."""
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.4, 0.6, 0.4, 0.6])
    auc = ClinicalEvaluator.compute_binary_auc(y_true, y_score)
    assert 0.4 <= auc <= 0.6


def test_cohort_evaluation_metrics():
    """Verify confusion matrix and sensitivity/specificity."""
    y_true = np.array([0, 0, 1, 1])
    y_probs = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.3, 0.7],
        [0.4, 0.6],
    ])
    metrics = ClinicalEvaluator.evaluate_cohort(y_true, y_probs, num_classes=2)
    assert metrics.accuracy == 1.0
    assert metrics.balanced_accuracy == 1.0
    assert metrics.roc_auc == 1.0
    assert metrics.confusion_matrix[0, 0] == 2
    assert metrics.confusion_matrix[1, 1] == 2


def test_attention_diagnostics():
    """Verify Shannon entropy and focal lesion detection."""
    # Focal lesion: one patch has 0.96 of the attention
    focal_a = np.array([0.96, 0.01, 0.01, 0.01, 0.01])
    diag_focal = ClinicalEvaluator.analyze_attention(focal_a)
    assert diag_focal.is_focal_lesion is True
    assert diag_focal.normalized_entropy < 0.50

    # Diffuse disease: uniform attention across all patches
    diffuse_a = np.ones(50) / 50.0
    diag_diffuse = ClinicalEvaluator.analyze_attention(diffuse_a)
    assert diag_diffuse.is_focal_lesion is False
    assert np.isclose(diag_diffuse.normalized_entropy, 1.0, atol=1e-5)
