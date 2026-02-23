"""
Batch Inference (backward compatibility)
=========================================
Re-exports the unified Inference class.
"""

from src.backend.inference.model_inference import Inference

# Alias for backward compatibility
BatchInference = Inference
