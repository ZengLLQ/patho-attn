"""Tests for Attention-Based Deep Multiple Instance Learning (ABMIL)."""

import pytest
import numpy as np
from patho_attn.mil.abmil import GatedAttentionMIL


def test_abmil_forward_properties():
    """Verify forward dimensions and softmax normalization."""
    model = GatedAttentionMIL(input_dim=64, hidden_dim=32, num_classes=3, seed=42)
    K = 25
    features = np.random.randn(K, 64).astype(np.float32)

    pred = model.forward(features)
    assert pred.predicted_class in [0, 1, 2]
    assert pred.probabilities.shape == (3,)
    assert np.isclose(np.sum(pred.probabilities), 1.0, atol=1e-5)
    assert pred.attention_weights.shape == (K,)
    # Attention scores across all bag instances must sum to 1.0
    assert np.isclose(np.sum(pred.attention_weights), 1.0, atol=1e-5)
    assert np.all(pred.attention_weights >= 0.0)


def test_abmil_train_step():
    """Verify gradient step reduces loss on repeated iterations."""
    model = GatedAttentionMIL(input_dim=32, hidden_dim=16, num_classes=2, seed=123)
    K = 15
    # High signal in first dimension for class 1
    features = np.random.randn(K, 32).astype(np.float32)
    features[:, 0] += 3.0

    initial_loss = model.train_step(features, target_label=1, learning_rate=0.05)
    loss_after = initial_loss
    for _ in range(10):
        loss_after = model.train_step(features, target_label=1, learning_rate=0.05)

    assert loss_after < initial_loss
