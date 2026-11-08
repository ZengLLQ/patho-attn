"""Attention-Based Deep Multiple Instance Learning (ABMIL)."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import numpy as np


@dataclass
class ABMILPrediction:
    """Output from ABMIL slide-level inference."""
    predicted_class: int
    probabilities: np.ndarray          # (num_classes,)
    attention_weights: np.ndarray      # (K,) normalized attention scores summing to 1.0
    slide_embedding: np.ndarray        # (embedding_dim,) aggregated slide feature vector
    raw_logits: np.ndarray             # (num_classes,)


class GatedAttentionMIL:
    """
    Gated Attention Deep Multiple Instance Learning (Ilse et al., ICML 2018).
    
    Architecture:
      h_k in R^D
      V in R^{L x D}, U in R^{L x D}, w in R^{1 x L}
      a_k = softmax( w * (tanh(V h_k) (x) sigmoid(U h_k)) )
      z_slide = sum_k (a_k * h_k)
      logits = W_c * z_slide + b_c
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        num_classes: int = 2,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        rng = np.random.default_rng(seed)
        # Xavier / Glorot initialization
        scale_v = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale_u = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale_w = np.sqrt(2.0 / (hidden_dim + 1))
        scale_c = np.sqrt(2.0 / (input_dim + num_classes))

        self.V = rng.normal(0, scale_v, (hidden_dim, input_dim)).astype(np.float32)
        self.U = rng.normal(0, scale_u, (hidden_dim, input_dim)).astype(np.float32)
        self.w = rng.normal(0, scale_w, (1, hidden_dim)).astype(np.float32)

        self.W_c = rng.normal(0, scale_c, (num_classes, input_dim)).astype(np.float32)
        self.b_c = np.zeros(num_classes, dtype=np.float32)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))

    @staticmethod
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def forward(self, features: np.ndarray) -> ABMILPrediction:
        """
        Forward pass on instance bag H of shape (K, D).
        """
        K, D = features.shape
        if D != self.input_dim:
            raise ValueError(f"Feature dimension {D} does not match model input_dim {self.input_dim}")

        # Gated attention: a_raw = w @ (tanh(V @ H.T) * sigmoid(U @ H.T))
        # V @ H.T -> (L, K)
        vh = np.tanh(self.V @ features.T)           # (L, K)
        uh = self.sigmoid(self.U @ features.T)       # (L, K)
        gated = vh * uh                              # (L, K)

        a_raw = (self.w @ gated).ravel()            # (K,)
        attn_weights = self.softmax(a_raw)           # (K,)

        # Slide-level representation: z = sum_k (a_k * h_k)
        slide_emb = np.sum(attn_weights[:, np.newaxis] * features, axis=0)  # (D,)

        # Slide classifier logits
        logits = (self.W_c @ slide_emb) + self.b_c   # (num_classes,)
        probs = self.softmax(logits)
        pred_class = int(np.argmax(probs))

        return ABMILPrediction(
            predicted_class=pred_class,
            probabilities=probs,
            attention_weights=attn_weights,
            slide_embedding=slide_emb,
            raw_logits=logits,
        )

    def train_step(
        self,
        features: np.ndarray,
        target_label: int,
        learning_rate: float = 0.005,
        weight_decay: float = 1e-4,
    ) -> float:
        """
        Single gradient descent step using cross-entropy loss on slide label.
        Returns scalar cross-entropy loss.
        """
        pred = self.forward(features)
        y_true = target_label

        # Cross-entropy loss
        loss = -float(np.log(max(pred.probabilities[y_true], 1e-12)))

        # Gradient wrt logits: dL/dlogits = p - y_onehot
        d_logits = pred.probabilities.copy()
        d_logits[y_true] -= 1.0  # (num_classes,)

        # Gradients wrt classifier weights
        d_Wc = np.outer(d_logits, pred.slide_embedding) + weight_decay * self.W_c
        d_bc = d_logits.copy()

        # Gradient wrt slide embedding: dL/dz_slide = W_c.T @ d_logits
        d_slide_emb = self.W_c.T @ d_logits  # (D,)

        # dL/da_k = dL/dz_slide dot h_k
        d_attn = np.sum(features * d_slide_emb[np.newaxis, :], axis=1)  # (K,)

        # Softmax gradient: dL/da_raw_k = a_k * (dL/da_k - sum_j a_j dL/da_j)
        sum_a_da = np.sum(pred.attention_weights * d_attn)
        d_a_raw = pred.attention_weights * (d_attn - sum_a_da)  # (K,)

        # Gradients wrt gated attention weights
        vh = np.tanh(self.V @ features.T)      # (L, K)
        uh = self.sigmoid(self.U @ features.T)  # (L, K)
        gated = vh * uh                         # (L, K)

        # dL/dw = sum_k d_a_raw[k] * gated[:, k]
        d_w = np.sum(d_a_raw[np.newaxis, :] * gated, axis=1, keepdims=True).T + weight_decay * self.w  # (1, L)

        # Backprop through gated activations
        d_gated = self.w.T @ d_a_raw[np.newaxis, :]  # (L, K)
        d_vh = d_gated * uh * (1.0 - vh**2)          # (L, K)
        d_uh = d_gated * vh * uh * (1.0 - uh)        # (L, K)

        d_V = (d_vh @ features) + weight_decay * self.V
        d_U = (d_uh @ features) + weight_decay * self.U

        # Parameter updates
        self.W_c -= learning_rate * d_Wc
        self.b_c -= learning_rate * d_bc
        self.w -= learning_rate * d_w
        self.V -= learning_rate * d_V
        self.U -= learning_rate * d_U

        return loss
