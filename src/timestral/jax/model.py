"""The Timestral network and its training & inference primitives in Flax / JAX."""

from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp

from timestral.config import TimestralConfig
from timestral.jax.layers import ResidualBlock, Transformer, TwoStreamMask
from timestral.jax.masks import (
    build_forecast_positions,
    build_forecast_time_mask,
    build_two_stream_positions,
    build_two_stream_time_mask,
    build_two_stream_variate_mask,
    compute_enc_key_validity,
    row_group_first_observed_patch,
)
from timestral.jax.scaler import (
    apply_affine_asinh_masked,
    causal_patch_loc_scale,
    instance_norm_stats,
    invert_affine_asinh,
)

ROLE_TARGET = 0
ROLE_PAST_COV = 1
ROLE_KNOWN_COV = 2


def denormalize_quantiles(preds: jax.Array, loc: jax.Array, scale: jax.Array, use_asinh: bool) -> jax.Array:
    """Invert normalized quantiles `sinh(preds) * scale + loc`."""
    return invert_affine_asinh(preds, loc, scale, use_asinh)


class PatchRepresentation(nn.Module):
    patch_size: int
    d_model: int

    def setup(self):
        self.patch_proj = ResidualBlock(
            input_dims=2 * self.patch_size,
            hidden_dims=4 * self.d_model,
            output_dims=self.d_model,
            use_bias=True,
            name="patch_proj",
        )
        self.forecast_query = self.param("forecast_query", nn.initializers.zeros, (self.d_model,))

    def __call__(self, scaled: jax.Array, observed: jax.Array) -> jax.Array:
        r, t = scaled.shape
        n = t // self.patch_size
        values = scaled.reshape(r, n, self.patch_size)
        missing = (~observed).reshape(r, n, self.patch_size).astype(scaled.dtype)
        x = jnp.concatenate([values, missing], axis=-1)
        return self.patch_proj(x)

    def forecast_tokens(self, rows: int, n_patches: int) -> jax.Array:
        query = self.forecast_query.reshape(1, 1, -1)
        return jnp.broadcast_to(query, (rows, n_patches, self.d_model))


def monotone_quantiles(raw: jax.Array, levels: Tuple[float, ...], width_factor: float) -> jax.Array:
    """Non-crossing quantiles from raw head outputs along the last axis (`monotone_cumsum`)."""
    m = levels.index(0.5)
    widths = [0.0] * len(levels)
    for i in range(m + 1, len(levels)):
        widths[i] = levels[i] - levels[i - 1]
    for i in range(m):
        widths[i] = levels[i + 1] - levels[i]
    widths_t = jnp.array(widths, dtype=raw.dtype)

    inc = width_factor * widths_t * jax.nn.softplus(raw)
    median = raw[..., m : m + 1]
    up = jnp.cumsum(inc[..., m + 1 :], axis=-1)
    down = jnp.flip(jnp.cumsum(jnp.flip(inc[..., :m], axis=-1), axis=-1), axis=-1)
    big = jnp.finfo(raw.dtype).max
    out = jnp.concatenate([median - down, median, median + up], axis=-1)
    return jnp.clip(out, -big, big)


class QuantileHead(nn.Module):
    config: TimestralConfig

    def setup(self):
        c = self.config
        self.projection = ResidualBlock(
            input_dims=c.d_model,
            hidden_dims=4 * c.d_model,
            output_dims=c.patch_size * c.num_quantiles,
            use_bias=False,
            name="projection",
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        raw = self.projection(x)
        *leading, n_patches, _ = raw.shape
        preds = raw.reshape(*leading, n_patches, self.config.patch_size, self.config.num_quantiles)
        preds = preds.astype(jnp.promote_types(preds.dtype, jnp.float32))
        preds = monotone_quantiles(preds, self.config.quantile_knots, self.config.quantile_width_factor)
        return preds.reshape(*leading, n_patches * self.config.patch_size, self.config.num_quantiles)


class TimestralModel(nn.Module):
    config: TimestralConfig

    def setup(self):
        c = self.config
        self.representation = PatchRepresentation(
            patch_size=c.patch_size,
            d_model=c.d_model,
            name="representation",
        )
        self.transformer = Transformer(
            block_schedule=c.block_schedule,
            embed_dim=c.d_model,
            num_heads=c.num_heads,
            mlp_hidden_dim=c.mlp_hidden_dim,
            rope_theta=c.rope_theta,
            rope_dim=c.rope_dim,
            norm_eps=c.norm_eps,
            attn_bias=c.attn_bias,
            mlp_bias=c.mlp_bias,
            time_attn_gate=c.time_attn_gate,
            variate_attn_gate=c.variate_attn_gate,
            name="transformer",
        )
        self.head = QuantileHead(config=c, name="head")

    def scale(self, inputs: jax.Array, observed: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Causal per-patch normalization of packed `(R, T)` rows."""
        c = self.config
        inp_exp = jnp.expand_dims(inputs, 1)
        obs_exp = jnp.expand_dims(observed, 1)
        loc, scale = causal_patch_loc_scale(
            inp_exp, obs_exp, c.patch_size, c.minimum_scale, c.stabilize_with_global
        )
        loc, scale = jnp.squeeze(loc, 1), jnp.squeeze(scale, 1)
        scaled = apply_affine_asinh_masked(inputs, loc, scale, observed, use_asinh=c.use_asinh)
        return scaled, loc, scale

    def _known_cov_scaling(
        self,
        inputs: jax.Array,
        observed: jax.Array,
        scaled: jax.Array,
        loc: jax.Array,
        scale: jax.Array,
        known_rows: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Normalize known-future covariate rows with whole-row statistics."""
        has_known = jnp.any(known_rows)

        def apply_known():
            loc_c, scale_c, is_binary = instance_norm_stats(inputs, observed, minimum_scale=self.config.minimum_scale)
            known_scaled = apply_affine_asinh_masked(inputs, loc_c, scale_c, observed, use_asinh=False)
            if self.config.use_asinh:
                work = jnp.promote_types(known_scaled.dtype, jnp.float32)
                squashed = jnp.arcsinh(known_scaled.astype(work)).astype(known_scaled.dtype)
                known_scaled = jnp.where(observed & ~is_binary, squashed, known_scaled)
            sel = known_rows[:, None]
            return (
                jnp.where(sel, known_scaled, scaled),
                jnp.where(sel, loc_c, loc),
                jnp.where(sel, scale_c, scale),
            )

        return jax.lax.cond(has_known, apply_known, lambda: (scaled, loc, scale))

    def _run_two_stream(
        self,
        x_enc: jax.Array,
        x_dec: jax.Array,
        time_mask: jax.Array,
        variate_mask: Optional[jax.Array],
        positions: jax.Array,
    ) -> jax.Array:
        """Transformer over `[Enc ; Dec]`, then the head on decoder tokens (normalized space)."""
        x = jnp.concatenate([x_enc, x_dec], axis=1)
        mask = time_mask
        if self.config.split_context_horizon_attn:
            mask = TwoStreamMask(mask=time_mask, enc_len=x_enc.shape[1])
        h = self.transformer(x, mask, variate_mask, positions)
        return self.head(h[:, x_enc.shape[1] :, :])

    def predict_block(
        self,
        inputs: jax.Array,
        observed: jax.Array,
        group_ids: jax.Array,
        horizon_patches: int,
        role_ids: Optional[jax.Array] = None,
        known_future: Optional[jax.Array] = None,
        known_future_mask: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Forecast one block of `horizon_patches` patches after a packed context."""
        assert inputs.ndim == 2, f"expected packed (R, T) inputs, got {inputs.shape}"
        all_targets = role_ids is None or bool(jnp.all(role_ids == ROLE_TARGET))
        if role_ids is None or not self.config.covariate_enabled or all_targets:
            return self._predict_context_only(inputs, observed, group_ids, horizon_patches)

        r = inputs.shape[0]
        horizon_len = horizon_patches * self.config.patch_size
        if known_future is None:
            known_future = jnp.zeros((r, horizon_len), dtype=inputs.dtype)
            known_future_mask = jnp.zeros((r, horizon_len), dtype=jnp.bool_)
        assert known_future_mask is not None

        full_inputs = jnp.concatenate([inputs, known_future], axis=-1)
        full_observed = jnp.concatenate([observed, known_future_mask], axis=-1)
        return self._predict_window(full_inputs, full_observed, group_ids, horizon_patches, role_ids)

    def _predict_context_only(
        self,
        inputs: jax.Array,
        observed: jax.Array,
        group_ids: jax.Array,
        horizon_patches: int,
    ) -> jax.Array:
        c = self.config
        ps = c.patch_size
        rows, t = inputs.shape
        ll, h = t // ps, horizon_patches

        scaled, loc, scale = self.scale(inputs, observed)
        x_enc = self.representation(scaled, observed)
        x_dec = self.representation.forecast_tokens(rows, h)

        obs_per_patch = observed.reshape(rows, ll, ps).any(axis=-1)
        key_valid = compute_enc_key_validity(obs_per_patch, c.attend_missing_patches)
        time_mask = build_forecast_time_mask(ll, h, key_valid)
        variate_mask = build_two_stream_variate_mask(group_ids, key_valid, h)
        first_obs = row_group_first_observed_patch(obs_per_patch, group_ids)
        positions = build_forecast_positions(ll, h, first_obs)

        preds = self._run_two_stream(x_enc, x_dec, time_mask, variate_mask, positions)
        anchor_loc = loc.reshape(rows, ll, ps)[..., 0][:, ll - 1 : ll]
        anchor_scale = scale.reshape(rows, ll, ps)[..., 0][:, ll - 1 : ll]
        return denormalize_quantiles(preds, jnp.expand_dims(anchor_loc, -1), jnp.expand_dims(anchor_scale, -1), c.use_asinh)

    def _predict_window(
        self,
        full_inputs: jax.Array,
        full_observed: jax.Array,
        group_ids: jax.Array,
        horizon_patches: int,
        role_ids: jax.Array,
    ) -> jax.Array:
        c = self.config
        ps = c.patch_size
        rows, w = full_inputs.shape
        h = horizon_patches
        n = w // ps
        ll = n - h
        target_rows = role_ids == ROLE_TARGET
        known_rows = role_ids == ROLE_KNOWN_COV

        scaled, loc, scale = self.scale(full_inputs, full_observed)
        scaled, loc, scale = self._known_cov_scaling(full_inputs, full_observed, scaled, loc, scale, known_rows)
        x_enc = self.representation(scaled, full_observed)
        x_dec = self.representation.forecast_tokens(rows, n)

        obs_per_patch = full_observed.reshape(rows, n, ps).any(axis=-1)
        key_valid = compute_enc_key_validity(obs_per_patch, c.attend_missing_patches)
        s_rows = jnp.full((rows,), ll, dtype=jnp.int32)
        time_mask = build_two_stream_time_mask(s_rows, n, c.block_size, key_valid, bidirectional_rows=known_rows)
        variate_mask = build_two_stream_variate_mask(
            group_ids, key_valid, n, target_rows=target_rows, bidirectional_rows=known_rows
        )
        first_obs = row_group_first_observed_patch(obs_per_patch, group_ids)
        positions = build_two_stream_positions(n, first_obs)

        preds = self._run_two_stream(x_enc, x_dec, time_mask, variate_mask, positions).reshape(rows, n, ps, -1)
        horizon_preds = preds[:, ll:n].reshape(rows, h * ps, preds.shape[-1])
        anchor_loc = loc.reshape(rows, n, ps)[..., 0][:, ll - 1 : ll]
        anchor_scale = scale.reshape(rows, n, ps)[..., 0][:, ll - 1 : ll]
        return denormalize_quantiles(horizon_preds, jnp.expand_dims(anchor_loc, -1), jnp.expand_dims(anchor_scale, -1), c.use_asinh)

    def training_forward(
        self,
        full_inputs: jax.Array,
        full_observed: jax.Array,
        group_ids: jax.Array,
        min_context: jax.Array,
        role_ids: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Full two-stream DHF forward pass for training across tiled forecast blocks.

        Args:
            full_inputs: `(R, W)` raw sequence values.
            full_observed: `(R, W)` boolean observation mask.
            group_ids: `(R,)` series index per row.
            min_context: `(R,)` integer index of first patch where forecast supervision begins.
            role_ids: `(R,)` row roles (TARGET=0, PAST_COV=1, KNOWN_COV=2).

        Returns:
            `normalized_preds`: `(R, N * patch_size, Q)` non-crossing monotone quantiles in normalized space.
            `loc`: `(R, W)` causal patch means.
            `scale`: `(R, W)` causal patch scales.
        """
        c = self.config
        ps = c.patch_size
        rows, w = full_inputs.shape
        n = w // ps
        if role_ids is None:
            role_ids = jnp.zeros((rows,), dtype=jnp.int32)

        target_rows = role_ids == ROLE_TARGET
        known_rows = role_ids == ROLE_KNOWN_COV

        scaled, loc, scale = self.scale(full_inputs, full_observed)
        scaled, loc, scale = self._known_cov_scaling(full_inputs, full_observed, scaled, loc, scale, known_rows)
        x_enc = self.representation(scaled, full_observed)
        x_dec = self.representation.forecast_tokens(rows, n)

        obs_per_patch = full_observed.reshape(rows, n, ps).any(axis=-1)
        key_valid = compute_enc_key_validity(obs_per_patch, c.attend_missing_patches)
        time_mask = build_two_stream_time_mask(min_context, n, c.block_size, key_valid, bidirectional_rows=known_rows)
        variate_mask = build_two_stream_variate_mask(
            group_ids, key_valid, n, target_rows=target_rows, bidirectional_rows=known_rows
        )
        first_obs = row_group_first_observed_patch(obs_per_patch, group_ids)
        positions = build_two_stream_positions(n, first_obs)

        normalized_preds = self._run_two_stream(x_enc, x_dec, time_mask, variate_mask, positions)
        return normalized_preds, loc, scale
