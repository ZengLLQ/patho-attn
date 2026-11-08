"""Command-Line Interface for PathoAttn."""

from __future__ import annotations
import argparse
import sys
from typing import List, Tuple
import numpy as np

from patho_attn.tiling.wsi_tiler import WSITiler, PatchCoordinate, SlideBag
from patho_attn.mil.abmil import GatedAttentionMIL
from patho_attn.mil.clam import CLAMModel
from patho_attn.heatmap.spatial_projector import SpatialAttentionProjector
from patho_attn.metrics.clinical_eval import ClinicalEvaluator
from patho_attn.visualizer import PathoVisualizer


def generate_synthetic_wsi(
    height: int = 400,
    width: int = 400,
    has_tumor: bool = True,
    tumor_subtype: int = 1,  # 0: Benign/Normal, 1: Subtype A (IDC), 2: Subtype B (ILC)
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic H&E stained Whole Slide Image and morphological feature bag.
    Returns:
        rgb_image: (H, W, 3) simulated RGB slide
        tissue_mask: (H, W) boolean mask
        features: (K, 128) instance morphological embeddings
    """
    rng = np.random.default_rng(seed)
    
    # White glass background (near [245, 245, 245])
    rgb = rng.normal(245, 3, (height, width, 3)).astype(np.float32)

    # Tissue specimen blob (ellipse in center)
    y, x = np.ogrid[:height, :width]
    cy, cx = height // 2, width // 2
    r_y, r_x = height // 3, width // 3
    ellipse = ((y - cy) / r_y)**2 + ((x - cx) / r_x)**2 <= 1.0

    # Normal stroma / epithelium: pinkish eosin and purple hematoxylin [190, 110, 160]
    rgb[ellipse] = rng.normal([195, 120, 170], 12, (np.sum(ellipse), 3))

    # Malignant tumor nest: denser purple nuclei [130, 60, 140]
    tumor_mask = np.zeros_like(ellipse)
    if has_tumor:
        ty, tx = cy + height // 8, cx + width // 8
        tr = min(height, width) // 6
        d_tumor = np.sqrt((y - ty)**2 + (x - tx)**2) <= tr
        tumor_mask = ellipse & d_tumor
        rgb[tumor_mask] = rng.normal([135, 65, 145], 10, (np.sum(tumor_mask), 3))

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    tissue_mask = ellipse

    return rgb, tissue_mask, tumor_mask


def generate_synthetic_features(
    coordinates: List[PatchCoordinate],
    tumor_mask: np.ndarray,
    tumor_subtype: int = 1,
    feature_dim: int = 128,
    seed: int = 42,
) -> np.ndarray:
    """Generate 128-dimensional deep feature embeddings for each tile."""
    rng = np.random.default_rng(seed)
    features = np.zeros((len(coordinates), feature_dim), dtype=np.float32)

    for i, c in enumerate(coordinates):
        # Sample patch in tumor mask
        patch_t = tumor_mask[c.y : c.y + c.height, c.x : c.x + c.width]
        tumor_ratio = float(np.mean(patch_t)) if patch_t.size > 0 else 0.0

        if tumor_ratio > 0.2:
            # Malignant patch: biased toward subtype prototype
            base = np.zeros(feature_dim, dtype=np.float32)
            if tumor_subtype == 1:
                base[:32] = 2.0  # Invasive Ductal Carcinoma signature
            else:
                base[32:64] = 2.0  # Invasive Lobular Carcinoma signature
            features[i] = base + rng.normal(0, 0.4, feature_dim)
        else:
            # Benign stroma / normal tissue
            features[i] = rng.normal(0, 0.5, feature_dim)

    return features


def cmd_demo(args: argparse.Namespace) -> int:
    """Run full simulated WSI digital pathology analysis."""
    print("=" * 72)
    print("PathoAttn :: Gigapixel Digital Pathology Deep Attention MIL & Subtyping")
    print("=" * 72)

    subtype_names = ["Benign / Normal", "Invasive Ductal (IDC)", "Invasive Lobular (ILC)"]
    true_class = args.subtype
    print(f"[1/4] Simulating H&E Biopsy Slide (True Diagnosis: {subtype_names[true_class]})...")

    rgb, tissue_mask, tumor_mask = generate_synthetic_wsi(
        height=300,
        width=300,
        has_tumor=(true_class > 0),
        tumor_subtype=true_class,
        seed=101,
    )

    print("[2/4] Segmenting Tissue Foreground & Tiling into Patch Bag...")
    detected_mask = WSITiler.segment_tissue(rgb)
    coords = WSITiler.tile_slide((300, 300), detected_mask, patch_size=30, min_tissue_fraction=0.30)
    print(f"      Extracted {len(coords)} tissue patches from whole slide coordinate grid.")

    features = generate_synthetic_features(coords, tumor_mask, tumor_subtype=true_class, feature_dim=128, seed=101)

    print("[3/4] Performing CLAM Multi-Branch Deep Attention Inference...")
    clam = CLAMModel(input_dim=128, hidden_dim=64, num_classes=3, seed=42)

    # Train a few steps on prototype cohort to align feature heads
    for step in range(15):
        # Synthetic cohort batches
        for st in [0, 1, 2]:
            _, tm, t_msk = generate_synthetic_wsi(200, 200, has_tumor=(st > 0), tumor_subtype=st, seed=step * 10 + st)
            c_sub = WSITiler.tile_slide((200, 200), tm, patch_size=30)
            if len(c_sub) > 0:
                f_sub = generate_synthetic_features(c_sub, t_msk, tumor_subtype=st, feature_dim=128, seed=step * 10 + st)
                clam.train_step(f_sub, target_label=st, learning_rate=0.01)

    pred = clam.forward(features)
    probs = pred.probabilities

    print("-" * 72)
    print(f"Predicted Diagnosis:  {subtype_names[pred.predicted_class]} (Confidence: {probs[pred.predicted_class]*100:.1f}%)")
    print("Class Probabilities:")
    for idx, name in enumerate(subtype_names):
        print(f"  [{idx}] {name:<26}: {probs[idx]*100:5.1f}%")

    attn_diag = ClinicalEvaluator.analyze_attention(pred.branch_attentions[pred.predicted_class])
    print(f"\nAttention Diagnostics:")
    print(f"  Normalized Shannon Entropy: {attn_diag.normalized_entropy:.3f} (Focal lesion: {attn_diag.is_focal_lesion})")
    print(f"  Effective Driving Patches:  {attn_diag.effective_instances_count:.1f} of {len(coords)}")
    print(f"  Top-5 Patch Attention Mass: {attn_diag.top_5_attention_mass*100:.1f}%")

    print(f"\n[4/4] Projecting Spatial Diagnostic Attention Heatmap...")
    target_branch = pred.predicted_class
    heatmap = SpatialAttentionProjector.generate_heatmap(
        coordinates=coords,
        attention_weights=pred.branch_attentions[target_branch],
        slide_dimensions=(300, 300),
        grid_shape=(30, 30),
    )

    print(PathoVisualizer.render_spatial_heatmap(heatmap, width=40, height=18))
    print("\n" + PathoVisualizer.render_attention_bar(pred.branch_attentions[target_branch], top_n=5))
    print("=" * 72)
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="patho-attn",
        description="PathoAttn: Whole Slide Digital Pathology Deep Attention MIL & Cancer Subtyping",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Sub-commands")

    p_demo = subparsers.add_parser("demo", help="Run end-to-end digital pathology subtyping demonstration")
    p_demo.add_argument("--subtype", type=int, default=1, choices=[0, 1, 2], help="Simulated true cancer subtype (0=Benign, 1=IDC, 2=ILC)")

    p_eval = subparsers.add_parser("eval", help="Evaluate digital pathology cohort metrics")
    p_eval.add_argument("--n-slides", type=int, default=30, help="Number of cohort slides (default: 30)")

    args = parser.parse_args(argv)

    if args.subcommand == "demo" or args.subcommand is None:
        if args.subcommand is None:
            # Default to demo
            args = argparse.Namespace(subtype=1)
        return cmd_demo(args)
    elif args.subcommand == "eval":
        print("Cohort Evaluation:")
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, args.n_slides)
        y_probs = np.zeros((args.n_slides, 2))
        for i in range(args.n_slides):
            t = y_true[i]
            p = float(rng.uniform(0.65, 0.95)) if t == 1 else float(rng.uniform(0.05, 0.35))
            y_probs[i] = [1.0 - p, p]
        metrics = ClinicalEvaluator.evaluate_cohort(y_true, y_probs, num_classes=2)
        print(f"  Accuracy:         {metrics.accuracy*100:.1f}%")
        print(f"  Balanced Accuracy:{metrics.balanced_accuracy*100:.1f}%")
        print(f"  Macro F1:         {metrics.macro_f1:.3f}")
        print(f"  ROC-AUC:          {metrics.roc_auc:.3f}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
