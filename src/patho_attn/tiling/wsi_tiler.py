"""Whole Slide Image (WSI) Tiling and Tissue Segmentation."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from scipy.ndimage import binary_fill_holes, binary_opening


@dataclass
class PatchCoordinate:
    """Coordinate and metadata of an individual tile patch."""
    patch_id: int
    x: int  # Top-left horizontal coordinate
    y: int  # Top-left vertical coordinate
    width: int
    height: int
    tissue_fraction: float  # Percentage of patch containing tissue [0.0, 1.0]


@dataclass
class SlideBag:
    """Bag of instance feature vectors extracted from a Whole Slide Image."""
    slide_id: str
    features: np.ndarray        # (K, D) instance embeddings
    coordinates: List[PatchCoordinate]
    label: Optional[int] = None # Slide-level ground truth label
    cancer_subtype: Optional[str] = None
    slide_dimensions: Tuple[int, int] = (1000, 1000)  # (Height, Width)


class WSITiler:
    """Segments tissue foreground and decomposes gigapixel slides into patch bags."""

    @staticmethod
    def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
        """Vectorized RGB to HSV conversion in NumPy."""
        # Normalize rgb to [0, 1]
        rgb_norm = rgb.astype(np.float32)
        if np.max(rgb_norm) > 1.0:
            rgb_norm /= 255.0

        r, g, b = rgb_norm[..., 0], rgb_norm[..., 1], rgb_norm[..., 2]
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin

        # Hue
        h = np.zeros_like(delta)
        mask = delta > 1e-6
        # When r is max
        r_mask = mask & (cmax == r)
        h[r_mask] = (60.0 * (((g[r_mask] - b[r_mask]) / delta[r_mask]) % 6))
        # When g is max
        g_mask = mask & (cmax == g)
        h[g_mask] = (60.0 * (((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2))
        # When b is max
        b_mask = mask & (cmax == b)
        h[b_mask] = (60.0 * (((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4))

        # Saturation
        s = np.zeros_like(cmax)
        s[cmax > 1e-6] = delta[cmax > 1e-6] / cmax[cmax > 1e-6]

        # Value
        v = cmax

        return np.stack([h, s, v], axis=-1)

    @classmethod
    def segment_tissue(
        cls,
        wsi_rgb: np.ndarray,
        saturation_threshold: float = 0.12,
        value_max_threshold: float = 0.95,
        min_hole_fill_size: int = 5,
    ) -> np.ndarray:
        """
        Segment tissue foreground from white background glass slide.
        Tissue stained with H&E exhibits high saturation and reduced brightness compared to bright white glass.
        """
        hsv = cls.rgb_to_hsv(wsi_rgb)
        sat = hsv[..., 1]
        val = hsv[..., 2]

        # Glass is typically very high value (bright white) and near-zero saturation
        tissue_mask = (sat >= saturation_threshold) & (val <= value_max_threshold)

        # Morphological opening and hole filling
        structure = np.ones((min_hole_fill_size, min_hole_fill_size), dtype=bool)
        tissue_mask = binary_opening(tissue_mask, structure=structure)
        tissue_mask = binary_fill_holes(tissue_mask)

        return tissue_mask

    @classmethod
    def tile_slide(
        cls,
        slide_dimensions: Tuple[int, int],
        tissue_mask: np.ndarray,
        patch_size: int = 64,
        stride: Optional[int] = None,
        min_tissue_fraction: float = 0.25,
    ) -> List[PatchCoordinate]:
        """
        Divide whole slide into patch coordinates, filtering by tissue area fraction.
        """
        h_slide, w_slide = slide_dimensions
        step = stride or patch_size

        mask_h, mask_w = tissue_mask.shape
        scale_y = mask_h / h_slide
        scale_x = mask_w / w_slide

        patches: List[PatchCoordinate] = []
        patch_id = 0

        for y in range(0, h_slide - patch_size + 1, step):
            for x in range(0, w_slide - patch_size + 1, step):
                # Sample corresponding region in tissue mask
                my_start = int(round(y * scale_y))
                my_end = int(round((y + patch_size) * scale_y))
                mx_start = int(round(x * scale_x))
                mx_end = int(round((x + patch_size) * scale_x))

                patch_mask = tissue_mask[my_start:my_end, mx_start:mx_end]
                if patch_mask.size == 0:
                    continue

                fraction = float(np.mean(patch_mask))
                if fraction >= min_tissue_fraction:
                    patches.append(
                        PatchCoordinate(
                            patch_id=patch_id,
                            x=x,
                            y=y,
                            width=patch_size,
                            height=patch_size,
                            tissue_fraction=fraction,
                        )
                    )
                    patch_id += 1

        return patches
