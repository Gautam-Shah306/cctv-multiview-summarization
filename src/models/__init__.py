"""
src/models/__init__.py

Package initialization for neural network models (detectors, ReID backbones, etc.).
"""

from src.models.reid_model import build_reid_model, OSNet

__all__ = ["build_reid_model", "OSNet"]
