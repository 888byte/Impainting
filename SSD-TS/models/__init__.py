"""Canonical model package."""

from .color_encoder import ColorEncoder, ColorEncoderConfig
from .cond_predictor import ColorToSpecPredictor, CondPredictorConfig
from .denoiser import DenoiserConfig, MambaDenoiser
from .spectral_encoder import ConditionerConfig, MultimodalConditioner
from .physics import PhysicsCfg, FadingForwardModelLab, warmup_weight
