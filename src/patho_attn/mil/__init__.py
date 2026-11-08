"""Deep Multiple Instance Learning models."""

from patho_attn.mil.abmil import GatedAttentionMIL, ABMILPrediction
from patho_attn.mil.clam import CLAMModel, CLAMPrediction

__all__ = [
    "GatedAttentionMIL",
    "ABMILPrediction",
    "CLAMModel",
    "CLAMPrediction",
]
