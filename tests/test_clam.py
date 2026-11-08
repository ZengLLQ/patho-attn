"""Tests for Clustering-Constrained Attention Multiple Instance Learning (CLAM)."""

import pytest
import numpy as np
from patho_attn.mil.clam import CLAMModel


def test_clam_multi_branch_forward():
    """Verify multi-branch attention and instance selection."""
    model = CLAMModel(input_dim=48, hidden_dim=24, num_classes=3, top_k_instances=4, seed=42)
    K = 20
    features = np.random.randn(K, 48).astype(np.float32)

    pred = model.forward(features, ground_truth_label=1)
    assert pred.branch_attentions.shape == (3, K)
    assert pred.slide_embeddings.shape == (3, 24)
    assert len(pred.top_positive_instances) == 4
    assert len(pred.top_negative_instances) == 4
    # Positive and negative sets should be non-overlapping if K >= 2 * top_k
    assert set(pred.top_positive_instances).isdisjoint(set(pred.top_negative_instances))
    assert pred.instance_clustering_loss >= 0.0


def test_clam_train_step():
    """Verify joint loss calculation during training step."""
    model = CLAMModel(input_dim=32, hidden_dim=16, num_classes=2, top_k_instances=3, seed=77)
    K = 12
    features = np.random.randn(K, 32).astype(np.float32)

    loss_dict = model.train_step(features, target_label=0, learning_rate=0.01)
    assert "slide_loss" in loss_dict
    assert "instance_loss" in loss_dict
    assert "total_loss" in loss_dict
    assert loss_dict["total_loss"] >= loss_dict["slide_loss"]
