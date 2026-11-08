"""PathoAttn: Gigapixel Whole Slide Digital Pathology Deep Attention Multiple Instance Learning & Cancer Subtyping."""

from patho_attn.tiling.wsi_tiler import WSITiler, PatchCoordinate, SlideBag
from patho_attn.mil.abmil import GatedAttentionMIL, ABMILPrediction
from patho_attn.mil.clam import CLAMModel, CLAMPrediction
from patho_attn.heatmap.spatial_projector import SpatialAttentionProjector, SpatialHeatmap
from patho_attn.metrics.clinical_eval import ClinicalEvaluator, DiagnosticMetrics, AttentionDiagnostics
from patho_attn.visualizer import PathoVisualizer

__version__ = "0.1.0"
__all__ = [
    "WSITiler",
    "PatchCoordinate",
    "SlideBag",
    "GatedAttentionMIL",
    "ABMILPrediction",
    "CLAMModel",
    "CLAMPrediction",
    "SpatialAttentionProjector",
    "SpatialHeatmap",
    "ClinicalEvaluator",
    "DiagnosticMetrics",
    "AttentionDiagnostics",
    "PathoVisualizer",
]
