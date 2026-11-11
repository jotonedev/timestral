"""Timestral JAX / Flax implementation."""

from timestral.jax.conversion import (
    flax_to_torch_params,
    load_flax_model_from_safetensors,
    save_flax_model_to_safetensors,
    torch_to_flax_params,
)
from timestral.jax.forecaster import JaxForecast, TimestralJaxForecaster
from timestral.jax.layers import Attention, FusedSwiGLU, ResidualBlock, RMSNorm, SinCosRoPE, Transformer, TransformerBlock
from timestral.jax.model import PatchRepresentation, QuantileHead, TimestralModel
from timestral.jax.scaler import (
    advancing_block_frames,
    apply_affine_asinh_masked,
    causal_patch_loc_scale,
    clamp_nonfinite,
    compute_causal_statistics,
    denormalize_quantiles,
    instance_norm_stats,
    invert_affine_asinh,
)

__all__ = [
    "TimestralModel",
    "TimestralJaxForecaster",
    "JaxForecast",
    "torch_to_flax_params",
    "flax_to_torch_params",
    "load_flax_model_from_safetensors",
    "save_flax_model_to_safetensors",
    "Transformer",
    "TransformerBlock",
    "Attention",
    "FusedSwiGLU",
    "ResidualBlock",
    "RMSNorm",
    "SinCosRoPE",
    "PatchRepresentation",
    "QuantileHead",
    "causal_patch_loc_scale",
    "apply_affine_asinh_masked",
    "invert_affine_asinh",
    "denormalize_quantiles",
    "clamp_nonfinite",
    "compute_causal_statistics",
    "advancing_block_frames",
    "instance_norm_stats",
]
