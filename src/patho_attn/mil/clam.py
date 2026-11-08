"""Clustering-Constrained Attention Multiple Instance Learning (CLAM)."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
import numpy as np


@dataclass
class CLAMPrediction:
    """Detailed output from CLAM multi-branch inference."""
    predicted_class: int
    probabilities: np.ndarray              # (num_classes,)
    branch_attentions: np.ndarray          # (num_classes, K)
    slide_embeddings: np.ndarray           # (num_classes, embedding_dim)
    top_positive_instances: List[int]      # Indices of top-k highest-attention patches for predicted class
    top_negative_instances: List[int]      # Indices of lowest-attention patches (stroma/normal)
    instance_clustering_loss: float        # Auxiliary instance loss


class CLAMModel:
    """
    CLAM: Clustering-constrained Attention Multiple Instance Learning (Lu et al., Nat Biomed Eng 2021).
    Multi-branch attention with instance-level pseudo-clustering supervision.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        num_classes: int = 2,
        top_k_instances: int = 5,
        inst_loss_weight: float = 0.3,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.top_k = top_k_instances
        self.inst_loss_weight = inst_loss_weight

        rng = np.random.default_rng(seed)
        # Shared projection
        scale_proj = np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_proj = rng.normal(0, scale_proj, (hidden_dim, input_dim)).astype(np.float32)

        # Multi-branch attention heads: one (w, V, U) set per class
        self.V_branches = rng.normal(0, scale_proj, (num_classes, hidden_dim, hidden_dim)).astype(np.float32)
        self.U_branches = rng.normal(0, scale_proj, (num_classes, hidden_dim, hidden_dim)).astype(np.float32)
        self.w_branches = rng.normal(0, scale_proj, (num_classes, 1, hidden_dim)).astype(np.float32)

        # Slide-level linear classifiers per branch
        self.cls_weights = rng.normal(0, scale_proj, (num_classes, hidden_dim)).astype(np.float32)
        self.cls_bias = np.zeros(num_classes, dtype=np.float32)

        # Instance-level clustering classifier: maps hidden_dim to (num_classes + 1) [classes + background]
        self.inst_W = rng.normal(0, scale_proj, (num_classes + 1, hidden_dim)).astype(np.float32)
        self.inst_b = np.zeros(num_classes + 1, dtype=np.float32)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))

    @staticmethod
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def forward(self, features: np.ndarray, ground_truth_label: Optional[int] = None) -> CLAMPrediction:
        """
        Forward pass with multi-branch attention and instance clustering evaluation.
        """
        K, D = features.shape
        # Common projection
        h_proj = (self.W_proj @ features.T).T  # (K, hidden_dim)

        branch_attns = np.zeros((self.num_classes, K), dtype=np.float32)
        slide_embs = np.zeros((self.num_classes, self.hidden_dim), dtype=np.float32)
        logits = np.zeros(self.num_classes, dtype=np.float32)

        # Evaluate each branch independently
        for c in range(self.num_classes):
            # Attention scoring
            vc = np.tanh(self.V_branches[c] @ h_proj.T)     # (hidden_dim, K)
            uc = self.sigmoid(self.U_branches[c] @ h_proj.T) # (hidden_dim, K)
            gated = vc * uc

            a_raw = (self.w_branches[c] @ gated).ravel()     # (K,)
            a_norm = self.softmax(a_raw)
            branch_attns[c] = a_norm

            # Branch slide embedding
            z_c = np.sum(a_norm[:, np.newaxis] * h_proj, axis=0)  # (hidden_dim,)
            slide_embs[c] = z_c

            logits[c] = np.dot(self.cls_weights[c], z_c) + self.cls_bias[c]

        probs = self.softmax(logits)
        pred_class = int(np.argmax(probs))

        # Target branch for instance clustering: use ground truth if available, otherwise prediction
        target_c = ground_truth_label if ground_truth_label is not None else pred_class
        attn_c = branch_attns[target_c]

        k_clamped = min(self.top_k, max(1, K // 2))
        sorted_indices = np.argsort(attn_c)
        neg_indices = sorted_indices[:k_clamped].tolist()
        pos_indices = sorted_indices[-k_clamped:].tolist()

        # Compute instance clustering loss on pseudo-labeled patches
        inst_loss = 0.0
        # Positive patches should predict target_c
        pos_feats = h_proj[pos_indices]  # (k, hidden_dim)
        pos_logits = (self.inst_W @ pos_feats.T).T + self.inst_b
        pos_probs = self.softmax(pos_logits, axis=-1)
        inst_loss += -float(np.mean(np.log(np.clip(pos_probs[:, target_c], 1e-12, 1.0))))

        # Negative patches should predict background (class index num_classes)
        neg_feats = h_proj[neg_indices]
        neg_logits = (self.inst_W @ neg_feats.T).T + self.inst_b
        neg_probs = self.softmax(neg_logits, axis=-1)
        inst_loss += -float(np.mean(np.log(np.clip(neg_probs[:, self.num_classes], 1e-12, 1.0))))

        inst_loss *= 0.5

        return CLAMPrediction(
            predicted_class=pred_class,
            probabilities=probs,
            branch_attentions=branch_attns,
            slide_embeddings=slide_embs,
            top_positive_instances=pos_indices,
            top_negative_instances=neg_indices,
            instance_clustering_loss=inst_loss,
        )

    def train_step(
        self,
        features: np.ndarray,
        target_label: int,
        learning_rate: float = 0.005,
    ) -> Dict[str, float]:
        """
        Gradient descent step combining slide cross-entropy and instance clustering loss.
        """
        pred = self.forward(features, ground_truth_label=target_label)
        slide_loss = -float(np.log(max(pred.probabilities[target_label], 1e-12)))
        total_loss = slide_loss + self.inst_loss_weight * pred.instance_clustering_loss

        # Finite difference / analytical optimization updates for parameters
        # Update classifier head
        d_logits = pred.probabilities.copy()
        d_logits[target_label] -= 1.0

        for c in range(self.num_classes):
            self.cls_weights[c] -= learning_rate * (d_logits[c] * pred.slide_embeddings[c])
            self.cls_bias[c] -= learning_rate * d_logits[c]

        # Update instance classifier
        h_proj = (self.W_proj @ features.T).T
        pos_indices = pred.top_positive_instances
        neg_indices = pred.top_negative_instances

        # Instance pos updates
        pos_logits = (self.inst_W @ h_proj[pos_indices].T).T + self.inst_b
        pos_probs = self.softmax(pos_logits, axis=-1)
        d_pos = pos_probs.copy()
        d_pos[:, target_label] -= 1.0
        d_inst_W = (d_pos.T @ h_proj[pos_indices]) / len(pos_indices)
        d_inst_b = np.mean(d_pos, axis=0)

        # Instance neg updates
        neg_logits = (self.inst_W @ h_proj[neg_indices].T).T + self.inst_b
        neg_probs = self.softmax(neg_logits, axis=-1)
        d_neg = neg_probs.copy()
        d_neg[:, self.num_classes] -= 1.0
        d_inst_W += (d_neg.T @ h_proj[neg_indices]) / len(neg_indices)
        d_inst_b += np.mean(d_neg, axis=0)

        self.inst_W -= learning_rate * self.inst_loss_weight * d_inst_W
        self.inst_b -= learning_rate * self.inst_loss_weight * d_inst_b

        # Update projection weights
        self.W_proj -= learning_rate * 0.01 * np.outer(self.cls_weights[target_label], np.mean(features, axis=0))

        return {
            "slide_loss": slide_loss,
            "instance_loss": pred.instance_clustering_loss,
            "total_loss": total_loss,
        }
