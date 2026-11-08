"""Clinical Evaluation Metrics and Attention Diagnostics for Digital Pathology."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np


@dataclass
class DiagnosticMetrics:
    """Clinical subtyping metrics on pathology test cohort."""
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    roc_auc: float
    confusion_matrix: np.ndarray
    per_class_sensitivity: Dict[int, float]
    per_class_specificity: Dict[int, float]


@dataclass
class AttentionDiagnostics:
    """Interpretability diagnostics of slide attention distribution."""
    attention_entropy: float          # Raw Shannon entropy in nats
    normalized_entropy: float         # Entropy normalized by log(K) in [0.0, 1.0]
    effective_instances_count: float  # exp(Entropy) - number of patches driving decision
    top_5_attention_mass: float       # Fraction of total probability mass in top-5 patches
    is_focal_lesion: bool             # True if normalized entropy < 0.70


class ClinicalEvaluator:
    """Computes diagnostic efficacy and attention interpretability metrics."""

    @staticmethod
    def compute_binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """
        Compute Area Under ROC curve using trapezoidal integration (Mann-Whitney U statistic).
        """
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        pos = y_score[y_true == 1]
        neg = y_score[y_true == 0]

        if len(pos) == 0 or len(neg) == 0:
            return 0.5

        # Count pairs where pos > neg
        # pos[:, None] > neg[None, :]
        matches = np.sum(pos[:, None] > neg[None, :])
        ties = np.sum(pos[:, None] == neg[None, :]) * 0.5
        auc = float((matches + ties) / (len(pos) * len(neg)))
        return auc

    @classmethod
    def evaluate_cohort(
        cls,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        num_classes: int = 2,
    ) -> DiagnosticMetrics:
        """
        Compute comprehensive classification metrics across a cohort of patient slides.
        """
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.argmax(y_probs, axis=1)
        n = len(y_true)

        # Confusion matrix
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1

        accuracy = float(np.mean(y_true == y_pred))

        # Per class sensitivity & specificity
        sens: Dict[int, float] = {}
        spec: Dict[int, float] = {}
        f1_list: List[float] = []

        for c in range(num_classes):
            tp = float(cm[c, c])
            fn = float(np.sum(cm[c, :]) - tp)
            fp = float(np.sum(cm[:, c]) - tp)
            tn = float(n - tp - fn - fp)

            s = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = 2 * (prec * s) / (prec + s) if (prec + s) > 0 else 0.0

            sens[c] = s
            spec[c] = sp
            f1_list.append(f1)

        balanced_acc = float(np.mean(list(sens.values())))
        macro_f1 = float(np.mean(f1_list))

        # Multi-class One-vs-Rest AUC
        if num_classes == 2:
            auc = cls.compute_binary_auc(y_true, y_probs[:, 1])
        else:
            aucs = []
            for c in range(num_classes):
                bin_true = (y_true == c).astype(int)
                aucs.append(cls.compute_binary_auc(bin_true, y_probs[:, c]))
            auc = float(np.mean(aucs))

        return DiagnosticMetrics(
            accuracy=accuracy,
            balanced_accuracy=balanced_acc,
            macro_f1=macro_f1,
            roc_auc=auc,
            confusion_matrix=cm,
            per_class_sensitivity=sens,
            per_class_specificity=spec,
        )

    @staticmethod
    def analyze_attention(attention_weights: np.ndarray) -> AttentionDiagnostics:
        """
        Analyze Shannon entropy and sparsity of attention weights.
        """
        a = np.asarray(attention_weights, dtype=np.float64)
        K = len(a)
        if K <= 1:
            return AttentionDiagnostics(0.0, 0.0, 1.0, 1.0, True)

        # Shannon entropy
        valid = a > 1e-15
        p = a[valid]
        entropy = -float(np.sum(p * np.log(p)))
        norm_entropy = float(entropy / np.log(K))
        eff_instances = float(np.exp(entropy))

        top_k = min(5, K)
        sorted_a = np.sort(a)[::-1]
        top_mass = float(np.sum(sorted_a[:top_k]))

        return AttentionDiagnostics(
            attention_entropy=entropy,
            normalized_entropy=norm_entropy,
            effective_instances_count=eff_instances,
            top_5_attention_mass=top_mass,
            is_focal_lesion=(norm_entropy < 0.70),
        )
