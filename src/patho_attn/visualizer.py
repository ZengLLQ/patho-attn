"""Terminal ASCII visualizer for Whole Slide Digital Pathology and attention heatmaps."""

from __future__ import annotations
from typing import Sequence, List, Tuple
import numpy as np
from patho_attn.heatmap.spatial_projector import SpatialHeatmap
from patho_attn.metrics.clinical_eval import AttentionDiagnostics, DiagnosticMetrics


class PathoVisualizer:
    """ASCII visualizations of gigapixel digital pathology heatmaps and metrics."""

    CHARS = [" ", ".", ":", "-", "=", "+", "*", "#", "%", "@"]

    @classmethod
    def render_spatial_heatmap(
        cls,
        heatmap: SpatialHeatmap,
        width: int = 44,
        height: int = 22,
    ) -> str:
        """Render 2D continuous attention heatmap as an ASCII diagnostic canvas."""
        mat = heatmap.heatmap_matrix
        gh, gw = mat.shape

        y_indices = np.linspace(0, gh - 1, height).astype(int)
        x_indices = np.linspace(0, gw - 1, width).astype(int)

        sampled = mat[y_indices[:, None], x_indices]
        char_idx = np.clip((sampled * (len(cls.CHARS) - 1)).astype(int), 0, len(cls.CHARS) - 1)

        lines = []
        lines.append(f"  +{' Whole Slide Attention Heatmap ':-^{width}}+")
        for r in range(height):
            line_str = "".join(cls.CHARS[c] for c in char_idx[r])
            lines.append(f"  |{line_str}|")
        lines.append(f"  +{'-' * width}+")
        lines.append(f"  Scale: [Low]   . : - = + * # % @   [High Malignancy Risk]")
        return "\n".join(lines)

    @classmethod
    def render_confusion_matrix(cls, cm: np.ndarray, class_names: Sequence[str]) -> str:
        """Render ASCII confusion matrix table."""
        n_classes = len(class_names)
        max_name_len = max(len(name) for name in class_names)
        cell_width = 8

        lines = []
        # Header
        header = f"{'':>{max_name_len + 2}}" + "".join(f"{name:^{cell_width}}" for name in class_names)
        lines.append("  Predicted ->")
        lines.append("  " + header)
        lines.append("  " + " " * (max_name_len + 2) + "-" * (cell_width * n_classes))

        for i, name in enumerate(class_names):
            row_str = f"  {name:>{max_name_len}} |"
            for j in range(n_classes):
                row_str += f"{cm[i, j]:^{cell_width}d}"
            lines.append(row_str)

        return "\n".join(lines)

    @classmethod
    def render_attention_bar(cls, attention_weights: np.ndarray, top_n: int = 8, bar_width: int = 30) -> str:
        """Render horizontal ASCII bar chart of top-ranked patches."""
        sorted_indices = np.argsort(attention_weights)[::-1][:top_n]
        max_val = float(attention_weights[sorted_indices[0]]) if len(sorted_indices) > 0 else 1.0

        lines = ["  Top Patch Attention Distribution:"]
        for rank, idx in enumerate(sorted_indices, 1):
            val = float(attention_weights[idx])
            bar_len = int(round((val / max_val) * bar_width)) if max_val > 0 else 0
            bar = "#" * bar_len + "-" * (bar_width - bar_len)
            lines.append(f"    Rank {rank:2d} (Tile #{idx:3d}): [{bar}] {val:.4f}")
        return "\n".join(lines)
