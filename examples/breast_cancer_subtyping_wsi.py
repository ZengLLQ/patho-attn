"""Breast Biopsy Cancer Subtyping: Invasive Ductal vs Invasive Lobular Carcinoma."""

from __future__ import annotations
import numpy as np

from patho_attn.tiling.wsi_tiler import WSITiler, PatchCoordinate
from patho_attn.mil.abmil import GatedAttentionMIL
from patho_attn.heatmap.spatial_projector import SpatialAttentionProjector
from patho_attn.metrics.clinical_eval import ClinicalEvaluator
from patho_attn.visualizer import PathoVisualizer


def simulate_breast_biopsy(
    height: int = 350,
    width: int = 350,
    subtype: str = "IDC",  # "IDC" (Invasive Ductal) or "ILC" (Invasive Lobular)
    seed: int = 42,
):
    """
    Simulate gigapixel breast core needle biopsy slide.
    IDC forms dense cohesive tumor nests with high cellular density.
    ILC forms single-file 'Indian-file' infiltrative strands through stroma.
    """
    rng = np.random.default_rng(seed)
    
    # White background glass
    rgb = rng.normal(242, 4, (height, width, 3)).astype(np.float32)

    # Biopsy tissue core
    y, x = np.ogrid[:height, :width]
    cy, cx = height // 2, width // 2
    # Long cylindrical tissue core shape
    tissue_mask = (np.abs(y - cy) <= height // 3.5) & (np.abs(x - cx) <= width // 2.5)

    # Benign fibrous breast stroma: light pink
    rgb[tissue_mask] = rng.normal([205, 130, 175], 10, (np.sum(tissue_mask), 3))

    # Tumor focus
    tumor_mask = np.zeros_like(tissue_mask)
    if subtype == "IDC":
        # Cohesive circumscribed malignant ductal nest
        d = np.sqrt((y - (cy - height // 8))**2 + (x - (cx + width // 8))**2)
        tumor_mask = tissue_mask & (d <= min(height, width) // 5)
        rgb[tumor_mask] = rng.normal([120, 50, 135], 8, (np.sum(tumor_mask), 3))
    elif subtype == "ILC":
        # Dispersed infiltrative strands across central stroma
        strand_pattern = (np.sin(y / 8.0) * np.cos(x / 8.0) > 0.3)
        tumor_mask = tissue_mask & strand_pattern
        rgb[tumor_mask] = rng.normal([135, 60, 145], 9, (np.sum(tumor_mask), 3))

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb, tissue_mask, tumor_mask


def main():
    print("=" * 75)
    print("PathoAttn :: Breast Cancer Digital Pathology Subtyping (IDC vs ILC)")
    print("=" * 75)

    subtype = "IDC"
    print(f"Acquiring 20x Whole Slide Digital Biopsy Image (Biopsy Target: {subtype})...")
    rgb, tissue_mask, tumor_mask = simulate_breast_biopsy(height=300, width=300, subtype=subtype, seed=2026)

    print("Segmenting Tissue Foreground via HSV Saturation Filtering...")
    detected_mask = WSITiler.segment_tissue(rgb, saturation_threshold=0.10)
    patches = WSITiler.tile_slide((300, 300), detected_mask, patch_size=25, min_tissue_fraction=0.35)
    print(f"Segmented {len(patches)} valid histological tissue patches from glass background.")

    # Morphological feature vectors (128-D)
    rng = np.random.default_rng(42)
    dim = 128
    features = np.zeros((len(patches), dim), dtype=np.float32)

    for i, p in enumerate(patches):
        p_tumor = tumor_mask[p.y : p.y + p.height, p.x : p.x + p.width]
        t_frac = float(np.mean(p_tumor)) if p_tumor.size > 0 else 0.0

        if t_frac > 0.25:
            # IDC signature in first 32 channels
            feat = np.zeros(dim, dtype=np.float32)
            feat[:32] = 2.5
            features[i] = feat + rng.normal(0, 0.4, dim)
        else:
            features[i] = rng.normal(0, 0.5, dim)

    print("\nRunning Gated Attention Deep Multiple Instance Learning (ABMIL)...")
    abmil = GatedAttentionMIL(input_dim=128, hidden_dim=64, num_classes=2, seed=123)

    # Train a few steps to calibrate weights
    for epoch in range(12):
        abmil.train_step(features, target_label=1, learning_rate=0.02)

    pred = abmil.forward(features)
    class_labels = ["Benign / Normal", "Invasive Ductal Carcinoma (IDC)"]

    print("-" * 75)
    print(f"Histopathological Call:  {class_labels[pred.predicted_class]}")
    print(f"Diagnostic Probability:  {pred.probabilities[pred.predicted_class]*100:.2f}%")
    print(f"Benign Probability:      {pred.probabilities[0]*100:.2f}%")

    diag = ClinicalEvaluator.analyze_attention(pred.attention_weights)
    print(f"\nAttention Entropy & Spatial Focus:")
    print(f"  Normalized Shannon Entropy: {diag.normalized_entropy:.3f} (Focal tumor nest: {diag.is_focal_lesion})")
    print(f"  Driving Patch Equivalents:  {diag.effective_instances_count:.1f} of {len(patches)} patches")
    print(f"  Top 5 Patches Concentration: {diag.top_5_attention_mass*100:.1f}%")

    print("\nReconstructing Spatial Attention Heatmap Over Biopsy Grid:")
    heatmap = SpatialAttentionProjector.generate_heatmap(
        coordinates=patches,
        attention_weights=pred.attention_weights,
        slide_dimensions=(300, 300),
        grid_shape=(30, 30),
        smooth_sigma=1.2,
    )

    print(PathoVisualizer.render_spatial_heatmap(heatmap, width=42, height=18))
    print("\n" + PathoVisualizer.render_attention_bar(pred.attention_weights, top_n=6))
    print("=" * 75)


if __name__ == "__main__":
    main()
