"""Spatial Attention Heatmap Projection and Tumor Hotspot Reconstruction."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
from scipy.ndimage import gaussian_filter
from patho_attn.tiling.wsi_tiler import PatchCoordinate


@dataclass
class SpatialHeatmap:
    """Continuous spatial attention map reconstructed from patch coordinates."""
    heatmap_matrix: np.ndarray        # (GridH, GridW) continuous intensity values [0.0, 1.0]
    slide_dimensions: Tuple[int, int]  # (Height, Width) in pixels
    grid_resolution: Tuple[int, int]   # (GridH, GridW)
    patch_size: int
    top_hotspot_coordinates: List[Tuple[int, int, float]]  # (x, y, attention_score)


class SpatialAttentionProjector:
    """Maps discrete patch attention weights onto the 2D continuous gigapixel slide space."""

    @classmethod
    def generate_heatmap(
        cls,
        coordinates: List[PatchCoordinate],
        attention_weights: np.ndarray,
        slide_dimensions: Tuple[int, int],
        grid_shape: Tuple[int, int] = (64, 64),
        smooth_sigma: float = 1.2,
    ) -> SpatialHeatmap:
        """
        Map patch attention scores into a continuous 2D spatial density matrix.
        """
        if len(coordinates) != len(attention_weights):
            raise ValueError(
                f"Mismatch: {len(coordinates)} coordinates vs {len(attention_weights)} attention weights"
            )

        h_slide, w_slide = slide_dimensions
        gh, gw = grid_shape
        heatmap = np.zeros((gh, gw), dtype=np.float32)
        count_grid = np.zeros((gh, gw), dtype=np.float32)

        scale_y = gh / h_slide
        scale_x = gw / w_slide

        for coord, weight in zip(coordinates, attention_weights):
            # Center of patch
            cx = coord.x + coord.width / 2.0
            cy = coord.y + coord.height / 2.0

            gx = int(np.clip(cx * scale_x, 0, gw - 1))
            gy = int(np.clip(cy * scale_y, 0, gh - 1))

            heatmap[gy, gx] += weight
            count_grid[gy, gx] += 1.0

        # Normalize bins by coverage
        valid = count_grid > 0
        heatmap[valid] /= count_grid[valid]

        # Gaussian smoothing
        if smooth_sigma > 0:
            heatmap = gaussian_filter(heatmap, sigma=smooth_sigma)

        # Rescale smoothed map to [0, 1]
        h_min, h_max = float(np.min(heatmap)), float(np.max(heatmap))
        if h_max > h_min:
            heatmap_norm = (heatmap - h_min) / (h_max - h_min)
        else:
            heatmap_norm = np.zeros_like(heatmap)

        # Identify top hotspots
        top_k = min(5, len(coordinates))
        top_indices = np.argsort(attention_weights)[-top_k:][::-1]
        hotspots = [
            (coordinates[idx].x, coordinates[idx].y, float(attention_weights[idx]))
            for idx in top_indices
        ]

        patch_sz = coordinates[0].width if coordinates else 64

        return SpatialHeatmap(
            heatmap_matrix=heatmap_norm,
            slide_dimensions=slide_dimensions,
            grid_resolution=grid_shape,
            patch_size=patch_sz,
            top_hotspot_coordinates=hotspots,
        )
