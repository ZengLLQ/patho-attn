"""CLAM Multi-Branch Attention & Instance Clustering: Lung Adenocarcinoma vs Squamous Cell Carcinoma."""

from __future__ import annotations
import numpy as np

from patho_attn.tiling.wsi_tiler import WSITiler, PatchCoordinate
from patho_attn.mil.clam import CLAMModel
from patho_attn.heatmap.spatial_projector import SpatialAttentionProjector
from patho_attn.metrics.clinical_eval import ClinicalEvaluator
from patho_attn.visualizer import PathoVisualizer


def simulate_lung_wsi(
    height: int = 320,
    width: int = 320,
    subtype: str = "LUAD",  # "LUAD" (Adenocarcinoma) or "LUSC" (Squamous Cell Carcinoma)
    seed: int = 777,
):
    """
    Simulate lung resection specimen.
    LUAD forms glandular/acinar patterns with mucin production.
    LUSC forms keratin pearls and intercellular bridges.
    """
    rng = np.random.default_rng(seed)
    rgb = rng.normal(246, 3, (height, width, 3)).astype(np.float32)

    # Lobular tissue architecture
    y, x = np.ogrid[:height, :width]
    cy, cx = height // 2, width // 2
    r = min(height, width) // 2.5
    tissue_mask = ((x - cx)**2 + (y - cy)**2) <= r**2

    # Normal alveolar septa: very pale pink
    rgb[tissue_mask] = rng.normal([215, 150, 185], 8, (np.sum(tissue_mask), 3))

    # Infiltrative tumor margin
    tumor_mask = np.zeros_like(tissue_mask)
    ty, tx = cy + height // 6, cx - width // 6
    tumor_mask = tissue_mask & (((x - tx)**2 + (y - ty)**2) <= (r * 0.55)**2)

    if subtype == "LUAD":
        rgb[tumor_mask] = rng.normal([115, 45, 125], 9, (np.sum(tumor_mask), 3))
    else:
        rgb[tumor_mask] = rng.normal([135, 70, 150], 9, (np.sum(tumor_mask), 3))

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb, tissue_mask, tumor_mask


def main():
    print("=" * 75)
    print("PathoAttn :: CLAM Multi-Branch Subtyping: Lung Adenocarcinoma vs Squamous Cell")
    print("=" * 75)

    subtype = "LUAD"
    print(f"Simulating Lung Resection Whole Slide (True Histology: {subtype})...")
    rgb, tissue_mask, tumor_mask = simulate_lung_wsi(subtype=subtype, seed=42)

    print("Segmenting Tissue & Tiling Slide...")
    mask = WSITiler.segment_tissue(rgb, saturation_threshold=0.10)
    coords = WSITiler.tile_slide((320, 320), mask, patch_size=28, min_tissue_fraction=0.30)
    print(f"Extracted {len(coords)} tissue patches from lung resection.")

    rng = np.random.default_rng(42)
    dim = 64
    features = np.zeros((len(coords), dim), dtype=np.float32)

    for i, c in enumerate(coords):
        patch_t = tumor_mask[c.y : c.y + c.height, c.x : c.x + c.width]
        if patch_t.size > 0 and np.mean(patch_t) > 0.2:
            feat = np.zeros(dim, dtype=np.float32)
            feat[:16] = 2.0  # LUAD glandular signature
            features[i] = feat + rng.normal(0, 0.4, dim)
        else:
            features[i] = rng.normal(0, 0.5, dim)

    print("\nInitializing CLAM Model (Clustering-Constrained Multi-Branch MIL)...")
    clam = CLAMModel(
        input_dim=dim,
        hidden_dim=32,
        num_classes=2,
        top_k_instances=4,
        inst_loss_weight=0.3,
        seed=100,
    )

    # Train model on synthetic batches
    for step in range(12):
        clam.train_step(features, target_label=0, learning_rate=0.02)

    pred = clam.forward(features, ground_truth_label=0)
    classes = ["Lung Adenocarcinoma (LUAD)", "Lung Squamous Cell (LUSC)"]

    print("-" * 75)
    print(f"CLAM Diagnostic Subtype:   {classes[pred.predicted_class]}")
    print(f"Subtype Probability:       {pred.probabilities[pred.predicted_class]*100:.1f}%")
    print(f"Instance Clustering Loss:  {pred.instance_clustering_loss:.4f}")
    print(f"Top Positive Patch Foci:   Tiles {pred.top_positive_instances}")
    print(f"Top Negative Stroma Foci:  Tiles {pred.top_negative_instances}")

    heatmap = SpatialAttentionProjector.generate_heatmap(
        coordinates=coords,
        attention_weights=pred.branch_attentions[0],
        slide_dimensions=(320, 320),
        grid_shape=(30, 30),
    )

    print("\nSpatial LUAD Tumor Infiltration Heatmap:")
    print(PathoVisualizer.render_spatial_heatmap(heatmap, width=42, height=18))
    print("=" * 75)


if __name__ == "__main__":
    main()
