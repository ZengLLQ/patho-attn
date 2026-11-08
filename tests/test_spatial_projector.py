"""Tests for spatial attention heatmap projection."""

import pytest
import numpy as np
from patho_attn.tiling.wsi_tiler import PatchCoordinate
from patho_attn.heatmap.spatial_projector import SpatialAttentionProjector


def test_spatial_heatmap_projection():
    """Verify projection from patch list to continuous grid."""
    coords = [
        PatchCoordinate(0, 0, 0, 50, 50, 1.0),
        PatchCoordinate(1, 50, 0, 50, 50, 1.0),
        PatchCoordinate(2, 0, 50, 50, 50, 1.0),
        PatchCoordinate(3, 50, 50, 50, 50, 1.0),
    ]
    # Strong attention in bottom-right patch (patch 3)
    weights = np.array([0.05, 0.05, 0.10, 0.80], dtype=np.float32)

    heatmap = SpatialAttentionProjector.generate_heatmap(
        coordinates=coords,
        attention_weights=weights,
        slide_dimensions=(100, 100),
        grid_shape=(20, 20),
        smooth_sigma=1.0,
    )

    assert heatmap.heatmap_matrix.shape == (20, 20)
    assert np.all(heatmap.heatmap_matrix >= 0.0)
    assert np.all(heatmap.heatmap_matrix <= 1.0)
    # Peak intensity should be near bottom right (indices ~15-19)
    assert heatmap.heatmap_matrix[15, 15] > heatmap.heatmap_matrix[2, 2]
    # Top hotspot should be patch 3 (x=50, y=50)
    top_x, top_y, top_val = heatmap.top_hotspot_coordinates[0]
    assert top_x == 50 and top_y == 50
