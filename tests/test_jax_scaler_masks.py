import numpy as np
import pytest
import torch
import jax.numpy as jnp

from timestral.masks import (
    build_forecast_positions as pt_build_forecast_positions,
    build_forecast_time_mask as pt_build_forecast_time_mask,
    build_two_stream_positions as pt_build_two_stream_positions,
    build_two_stream_time_mask as pt_build_two_stream_time_mask,
    build_two_stream_variate_mask as pt_build_two_stream_variate_mask,
    compute_enc_key_validity as pt_compute_enc_key_validity,
)
from timestral.scaler import (
    advancing_block_frames as pt_advancing_block_frames,
    apply_affine_asinh_masked as pt_apply_affine_asinh_masked,
    causal_patch_loc_scale as pt_causal_patch_loc_scale,
    compute_causal_statistics as pt_compute_causal_statistics,
    instance_norm_stats as pt_instance_norm_stats,
    invert_affine_asinh as pt_invert_affine_asinh,
)
from timestral.jax.masks import (
    build_forecast_positions as jax_build_forecast_positions,
    build_forecast_time_mask as jax_build_forecast_time_mask,
    build_two_stream_positions as jax_build_two_stream_positions,
    build_two_stream_time_mask as jax_build_two_stream_time_mask,
    build_two_stream_variate_mask as jax_build_two_stream_variate_mask,
    compute_enc_key_validity as jax_compute_enc_key_validity,
)
from timestral.jax.scaler import (
    advancing_block_frames as jax_advancing_block_frames,
    apply_affine_asinh_masked as jax_apply_affine_asinh_masked,
    causal_patch_loc_scale as jax_causal_patch_loc_scale,
    compute_causal_statistics as jax_compute_causal_statistics,
    instance_norm_stats as jax_instance_norm_stats,
    invert_affine_asinh as jax_invert_affine_asinh,
)


def test_scaler_causal_statistics_matches_torch():
    np.random.seed(42)
    data_np = np.random.randn(2, 64).astype(np.float32)
    mask_np = np.random.rand(2, 64) > 0.1

    pt_data = torch.from_numpy(data_np)
    pt_mask = torch.from_numpy(mask_np)

    jax_data = jnp.array(data_np)
    jax_mask = jnp.array(mask_np)

    pt_means, pt_scales = pt_compute_causal_statistics(pt_data, pt_mask, minimum_scale=1e-6, stabilize_with_global=False)
    jax_means, jax_scales = jax_compute_causal_statistics(jax_data, jax_mask, minimum_scale=1e-6, stabilize_with_global=False)

    np.testing.assert_allclose(np.array(jax_means), pt_means.numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.array(jax_scales), pt_scales.numpy(), rtol=1e-5, atol=1e-6)


def test_causal_patch_loc_scale_matches_torch():
    np.random.seed(7)
    data_np = np.random.randn(2, 64).astype(np.float32)
    mask_np = np.random.rand(2, 64) > 0.1

    pt_loc, pt_scale = pt_causal_patch_loc_scale(
        torch.from_numpy(data_np), torch.from_numpy(mask_np), patch_size=8, minimum_scale=1e-6, stabilize_with_global=False
    )
    jax_loc, jax_scale = jax_causal_patch_loc_scale(
        jnp.array(data_np), jnp.array(mask_np), patch_size=8, minimum_scale=1e-6, stabilize_with_global=False
    )

    np.testing.assert_allclose(np.array(jax_loc), pt_loc.numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.array(jax_scale), pt_scale.numpy(), rtol=1e-5, atol=1e-6)


def test_affine_asinh_roundtrip_matches_torch():
    np.random.seed(13)
    data_np = np.random.randn(2, 32).astype(np.float32)
    mask_np = np.ones((2, 32), dtype=bool)
    loc_np = np.zeros((2, 32), dtype=np.float32)
    scale_np = np.ones((2, 32), dtype=np.float32) * 2.0

    pt_scaled = pt_apply_affine_asinh_masked(
        torch.from_numpy(data_np), torch.from_numpy(loc_np), torch.from_numpy(scale_np), torch.from_numpy(mask_np), use_asinh=True
    )
    jax_scaled = jax_apply_affine_asinh_masked(
        jnp.array(data_np), jnp.array(loc_np), jnp.array(scale_np), jnp.array(mask_np), use_asinh=True
    )
    np.testing.assert_allclose(np.array(jax_scaled), pt_scaled.numpy(), rtol=1e-5, atol=1e-6)

    pt_inv = pt_invert_affine_asinh(pt_scaled, torch.from_numpy(loc_np), torch.from_numpy(scale_np), use_asinh=True)
    jax_inv = jax_invert_affine_asinh(jax_scaled, jnp.array(loc_np), jnp.array(scale_np), use_asinh=True)
    np.testing.assert_allclose(np.array(jax_inv), pt_inv.numpy(), rtol=1e-5, atol=1e-6)


def test_masks_match_torch():
    np.random.seed(101)
    obs_np = np.random.rand(3, 8) > 0.2
    pt_obs = torch.from_numpy(obs_np)
    jax_obs = jnp.array(obs_np)

    pt_key = pt_compute_enc_key_validity(pt_obs, attend_missing=True)
    jax_key = jax_compute_enc_key_validity(jax_obs, attend_missing=True)
    np.testing.assert_array_equal(np.array(jax_key), pt_key.numpy())

    # Two-stream time mask
    min_ctx_np = np.array([2, 3, 2], dtype=np.int64)
    bi_np = np.array([False, True, False])
    pt_tmask = pt_build_two_stream_time_mask(
        torch.from_numpy(min_ctx_np), 8, 4, pt_key, bidirectional_rows=torch.from_numpy(bi_np)
    )
    jax_tmask = jax_build_two_stream_time_mask(
        jnp.array(min_ctx_np), 8, 4, jax_key, bidirectional_rows=jnp.array(bi_np)
    )
    np.testing.assert_array_equal(np.array(jax_tmask), pt_tmask.numpy())

    # Two-stream variate mask
    group_ids_np = np.array([0, 0, 1], dtype=np.int64)
    tgt_np = np.array([True, False, True])
    pt_vmask = pt_build_two_stream_variate_mask(
        torch.from_numpy(group_ids_np), pt_key, 8, target_rows=torch.from_numpy(tgt_np), bidirectional_rows=torch.from_numpy(bi_np)
    )
    jax_vmask = jax_build_two_stream_variate_mask(
        jnp.array(group_ids_np), jax_key, 8, target_rows=jnp.array(tgt_np), bidirectional_rows=jnp.array(bi_np)
    )
    np.testing.assert_array_equal(np.array(jax_vmask), pt_vmask.numpy())
