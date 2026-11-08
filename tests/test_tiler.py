"""Tests for WSI tiling and tissue segmentation."""

import pytest
import numpy as np
from patho_attn.tiling.wsi_tiler import WSITiler, PatchCoordinate


def test_rgb_to_hsv():
    """Verify RGB to HSV conversion for primary colors."""
    # Red: [255, 0, 0] -> H=0, S=1, V=1
    red = np.array([[[255, 0, 0]]], dtype=np.uint8)
    hsv_red = WSITiler.rgb_to_hsv(red)
    assert np.isclose(hsv_red[0, 0, 0], 0.0, atol=1e-3)
    assert np.isclose(hsv_red[0, 0, 1], 1.0, atol=1e-3)
    assert np.isclose(hsv_red[0, 0, 2], 1.0, atol=1e-3)

    # White glass: [255, 255, 255] -> S=0, V=1
    white = np.array([[[255, 255, 255]]], dtype=np.uint8)
    hsv_white = WSITiler.rgb_to_hsv(white)
    assert np.isclose(hsv_white[0, 0, 1], 0.0, atol=1e-3)
    assert np.isclose(hsv_white[0, 0, 2], 1.0, atol=1e-3)


def test_tissue_segmentation():
    """Verify tissue mask distinguishes H&E tissue from white glass background."""
    img = np.ones((50, 50, 3), dtype=np.uint8) * 250  # White glass
    # Add tissue patch in center with eosin pink [200, 100, 150]
    img[15:35, 15:35] = [200, 100, 150]

    mask = WSITiler.segment_tissue(img, saturation_threshold=0.10)
    assert mask.shape == (50, 50)
    # Background corners should be False
    assert not mask[2, 2]
    # Tissue center should be True
    assert mask[25, 25]


def test_slide_tiling():
    """Verify grid tiling filters out empty background patches."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True  # 60x60 square

    patches = WSITiler.tile_slide(
        slide_dimensions=(100, 100),
        tissue_mask=mask,
        patch_size=20,
        stride=20,
        min_tissue_fraction=0.5,
    )

    assert len(patches) > 0
    # Every returned patch must satisfy min_tissue_fraction
    for p in patches:
        assert p.tissue_fraction >= 0.5
        assert p.width == 20
        assert p.height == 20
