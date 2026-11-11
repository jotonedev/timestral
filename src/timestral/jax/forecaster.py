"""Probabilistic forecasting with Timestral in JAX, including clampsym test-time augmentation."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from huggingface_hub import snapshot_download

from timestral.config import TimestralConfig
from timestral.covariate_encoding import target_encode
from timestral.jax.conversion import (
    load_flax_model_from_safetensors,
    torch_to_flax_params,
)
from timestral.jax.model import (
    ROLE_KNOWN_COV,
    ROLE_PAST_COV,
    ROLE_TARGET,
    TimestralModel,
)
from timestral.jax.scaler import (
    advancing_block_frames,
    causal_patch_loc_scale,
    clamp_nonfinite,
)

ArrayLike = jax.Array | np.ndarray


@dataclass(frozen=True)
class JaxForecast:
    """Quantile forecast in JAX / NumPy."""

    quantiles: jax.Array  # (B, V, H, Q)
    quantile_levels: tuple[float, ...]

    @property
    def median(self) -> jax.Array:
        """`(B, V, H)` median forecast."""
        return self.quantiles[..., self.quantile_levels.index(0.5)]

    def quantile(self, level: float) -> jax.Array:
        """`(B, V, H)` forecast at one quantile level."""
        if level not in self.quantile_levels:
            raise ValueError(f"level {level} is not in {self.quantile_levels}")
        return self.quantiles[..., self.quantile_levels.index(level)]


def ffill_with_mask(inputs: jax.Array, mask: jax.Array) -> jax.Array:
    """Forward-fill unobserved positions; leading unobserved become 0."""
    if bool(jnp.all(mask)):
        return inputs
    idx = jnp.broadcast_to(jnp.arange(inputs.shape[-1]), inputs.shape)
    idx = jnp.where(mask, idx, 0)
    idx = jnp.maximum.accumulate(idx, axis=-1)
    return jnp.nan_to_num(jnp.take_along_axis(inputs, idx, axis=-1), nan=0.0)


def pad_to_patch(
    inputs: jax.Array, mask: jax.Array, patch_size: int
) -> tuple[jax.Array, jax.Array]:
    """Left-pad to multiple of patch_size; padding is 0 and unobserved."""
    ctx_len = inputs.shape[-1]
    if ctx_len % patch_size != 0:
        pad_len = patch_size - (ctx_len % patch_size)
        pad_cfg = [(0, 0)] * (inputs.ndim - 1) + [(pad_len, 0)]
        inputs = jnp.pad(inputs, pad_cfg, constant_values=0.0)
        mask = jnp.pad(mask, pad_cfg, constant_values=False)
    return inputs, mask


def replace_extreme_values(x: jax.Array, threshold: float, replacement: float = 0.0) -> jax.Array:
    """Clamp extreme or non-finite values to replacement value."""
    return jnp.where(jnp.isfinite(x) & (jnp.abs(x) < threshold), x, replacement)


def nonnegative_context(inputs: jax.Array, observation_mask: jax.Array) -> jax.Array:
    """`(B, V)`: whether every observed context value is non-negative (and one exists)."""
    observed = jnp.isfinite(inputs) & observation_mask.astype(jnp.bool_)
    any_obs = jnp.any(observed, axis=-1)
    all_nonneg = jnp.all((inputs >= 0) | ~observed, axis=-1)
    return any_obs & all_nonneg


class TimestralJaxForecaster:
    """Block-autoregressive quantile forecaster using Flax / JAX."""

    def __init__(
        self,
        model: TimestralModel,
        params: dict[str, Any],
        max_context: int | None = None,
        per_shot_horizon_patches: int = 32,
        quantile_real_cap_k: float = 1e4,
        extreme_value_threshold: float = 1e30,
    ) -> None:
        self.model = model
        self.params = params
        self.config = model.config
        self.max_context = max_context if max_context is not None else model.config.context_length
        self.per_shot_horizon_patches = min(per_shot_horizon_patches, model.config.block_size)
        self.quantile_real_cap_k = quantile_real_cap_k
        self.extreme_value_threshold = extreme_value_threshold
        self.quantile_levels = tuple(model.config.quantile_knots)
        self.median_index = model.config.median_index

    @property
    def patch_size(self) -> int:
        """Patch size in time steps."""
        return self.config.patch_size

    @property
    def supports_covariates(self) -> bool:
        """Whether the model architecture supports covariates."""
        return self.config.supports_known_covariates or self.config.covariate_enabled

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        max_context: int | None = None,
        per_shot_horizon_patches: int = 32,
        quantile_real_cap_k: float = 1e4,
    ) -> "TimestralJaxForecaster":
        """Load pretrained checkpoint from directory or Hub repo."""
        p = Path(path_or_repo_id)
        if not p.exists():
            p = Path(snapshot_download(str(path_or_repo_id)))
        config = TimestralConfig.load(p / "config.json")
        model = TimestralModel(config)
        params = load_flax_model_from_safetensors(p / "model.safetensors", config)
        return cls(
            model=model,
            params=params,
            max_context=max_context,
            per_shot_horizon_patches=per_shot_horizon_patches,
            quantile_real_cap_k=quantile_real_cap_k,
        )

    @classmethod
    def from_torch_state_dict(
        cls,
        state_dict: dict[str, Any],
        config: TimestralConfig,
        max_context: int | None = None,
        per_shot_horizon_patches: int = 32,
        quantile_real_cap_k: float = 1e4,
    ) -> "TimestralJaxForecaster":
        """Initialize from a PyTorch state dict and config."""
        model = TimestralModel(config)
        params = torch_to_flax_params(state_dict, config)
        return cls(
            model=model,
            params=params,
            max_context=max_context,
            per_shot_horizon_patches=per_shot_horizon_patches,
            quantile_real_cap_k=quantile_real_cap_k,
        )

    def forecast(
        self,
        context: ArrayLike,
        prediction_length: int,
        past_covariates: ArrayLike | None = None,
        known_covariates: ArrayLike | None = None,
        categorical_past_covariates: ArrayLike | None = None,
        categorical_known_covariates: ArrayLike | None = None,
        observation_mask: ArrayLike | None = None,
        symmetric_averaging: bool = True,
        clip_nonnegative: bool = True,
    ) -> JaxForecast:
        """Forecast `prediction_length` steps ahead."""
        assert prediction_length >= 1, "prediction_length must be >= 1"

        ctx_arr = jnp.asarray(context, dtype=jnp.float32)
        if ctx_arr.ndim == 1:
            ctx_arr = ctx_arr[None, None, :]
        elif ctx_arr.ndim == 2:
            ctx_arr = ctx_arr[:, None, :]
        batch, _variates, ctx_len = ctx_arr.shape

        inputs, mask = self._clean_context(ctx_arr, observation_mask)
        target_ctx = self._encoding_target(inputs, mask)

        past_enc = self._encode_categorical(
            self._as_categorical(categorical_past_covariates, batch, ctx_len, "cat_past"),
            target_ctx,
        )
        past = _cat_rows(self._as_covariates(past_covariates, batch, ctx_len, "past"), past_enc)

        known_len = ctx_len + prediction_length
        known_enc = self._encode_categorical(
            self._as_categorical(categorical_known_covariates, batch, known_len, "cat_known"),
            target_ctx,
        )
        known = _cat_rows(
            self._as_covariates(known_covariates, batch, known_len, "known"), known_enc
        )

        if symmetric_averaging:
            pos = self._forecast_raw(inputs, mask, prediction_length, past, known)
            neg = self._forecast_raw(
                -inputs, mask, prediction_length, _negate(past), _negate(known)
            )
            quantiles = 0.5 * (pos - jnp.flip(neg, axis=-1))
        else:
            quantiles = self._forecast_raw(inputs, mask, prediction_length, past, known)

        if clip_nonnegative:
            nonneg = nonnegative_context(inputs, mask)[:, :, None, None]
            quantiles = jnp.where(nonneg, jnp.maximum(quantiles, 0.0), quantiles)

        return JaxForecast(quantiles=quantiles, quantile_levels=self.quantile_levels)

    def _clean_context(
        self, x: jax.Array, observation_mask: ArrayLike | None
    ) -> tuple[jax.Array, jax.Array]:
        mask = jnp.isfinite(x)
        if observation_mask is not None:
            user_mask = jnp.asarray(observation_mask, dtype=jnp.bool_).reshape(x.shape)
            mask = mask & user_mask
        return jnp.where(mask, x, 0.0), mask

    def _encoding_target(self, inputs: jax.Array, mask: jax.Array) -> np.ndarray | None:
        if inputs.shape[1] != 1:
            return None
        t = inputs.shape[-1]
        started = jnp.maximum.accumulate(mask[:, 0].astype(jnp.int32), axis=-1).astype(jnp.bool_)
        in_window = jnp.arange(t) >= (t - self.max_context)
        ffilled = ffill_with_mask(inputs, mask)[:, 0]
        enc = jnp.where(started & in_window, ffilled, jnp.nan)
        return np.asarray(enc)

    def _encode_categorical(
        self, codes: np.ndarray | None, target_context: np.ndarray | None
    ) -> jax.Array | None:
        if codes is None or target_context is None:
            return jnp.asarray(codes, dtype=jnp.float32) if codes is not None else None
        t = target_context.shape[-1]
        out = np.empty(codes.shape, dtype=np.float32)
        for i in range(codes.shape[0]):
            for j in range(codes.shape[1]):
                future = codes[i, j, t:] if codes.shape[-1] > t else None
                past_enc, future_enc = target_encode(target_context[i], codes[i, j, :t], future)
                out[i, j, :t] = np.where(
                    np.isnan(target_context[i]) & np.isnan(codes[i, j, :t]), np.nan, past_enc
                )
                if future_enc is not None:
                    out[i, j, t:] = future_enc
        return jnp.asarray(out)

    def _as_categorical(
        self, values: ArrayLike | None, batch: int, length: int, name: str
    ) -> np.ndarray | None:
        if values is None:
            return None
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 2:
            arr = arr[:, None]
        assert arr.shape[0] == batch and arr.shape[-1] == length, f"{name} shape mismatch"
        if np.isinf(arr).any():
            raise ValueError(f"{name} must hold finite codes or NaN")
        return arr if arr.shape[1] > 0 else None

    def _as_covariates(
        self, values: ArrayLike | None, batch: int, length: int, name: str
    ) -> jax.Array | None:
        if values is None:
            return None
        arr = jnp.asarray(values, dtype=jnp.float32)
        if arr.ndim == 2:
            arr = arr[:, None]
        assert arr.shape[0] == batch and arr.shape[-1] == length, f"{name} shape mismatch"
        return arr if arr.shape[1] > 0 else None

    def _forecast_raw(
        self,
        inputs: jax.Array,
        observation_mask: jax.Array,
        prediction_length: int,
        past_covariates: jax.Array | None = None,
        known_covariates: jax.Array | None = None,
    ) -> jax.Array:
        rows, mask, role_ids, known_future, known_future_mask, n_tgt = self._assemble_rows(
            inputs, observation_mask, prediction_length, past_covariates, known_covariates
        )
        quantiles = self._block_ar_decode(
            rows, mask, prediction_length, role_ids, known_future, known_future_mask
        )
        quantiles = quantiles[:, :n_tgt]
        if self.quantile_real_cap_k > 0:
            quantiles = self._cap_quantiles(quantiles, rows[:, :n_tgt], mask[:, :n_tgt])
        return quantiles

    def _cap_quantiles(
        self, quantiles: jax.Array, inputs: jax.Array, observation_mask: jax.Array
    ) -> jax.Array:
        not_obs = ~observation_mask
        ctx_max = jnp.nan_to_num(
            jnp.where(not_obs, -float("inf"), inputs).max(axis=-1, keepdims=True), neginf=0.0
        )
        ctx_min = jnp.nan_to_num(
            jnp.where(not_obs, float("inf"), inputs).min(axis=-1, keepdims=True), posinf=0.0
        )
        c = self.config
        _, scale = causal_patch_loc_scale(
            inputs, observation_mask, c.patch_size, c.minimum_scale, c.stabilize_with_global
        )
        anchor_scale = scale[..., -1:] * self.quantile_real_cap_k
        return jnp.clip(
            quantiles, (ctx_min - anchor_scale)[..., None], (ctx_max + anchor_scale)[..., None]
        )

    def _assemble_rows(
        self,
        inputs: jax.Array,
        observation_mask: jax.Array,
        prediction_length: int,
        past_covariates: jax.Array | None,
        known_covariates: jax.Array | None,
    ):
        ps = self.patch_size
        b, n_tgt, _ctx0 = inputs.shape
        horizon = math.ceil(prediction_length / ps) * ps

        ctx_rows = [ffill_with_mask(inputs, observation_mask)]
        ctx_masks = [observation_mask]
        roles = [ROLE_TARGET] * n_tgt

        n_past = 0
        if past_covariates is not None:
            n_past = past_covariates.shape[1]
            ctx_masks.append(~jnp.isnan(past_covariates))
            ctx_rows.append(jnp.nan_to_num(past_covariates, nan=0.0))
            roles += [ROLE_PAST_COV] * n_past

        n_known = 0
        known_future_vals = None
        known_future_obs = None
        if known_covariates is not None:
            n_known = known_covariates.shape[1]
            kc_ctx = known_covariates[..., :-prediction_length]
            kc_fut = known_covariates[..., -prediction_length:]
            ctx_masks.append(~jnp.isnan(kc_ctx))
            ctx_rows.append(jnp.nan_to_num(kc_ctx, nan=0.0))
            roles += [ROLE_KNOWN_COV] * n_known
            known_future_obs = ~jnp.isnan(kc_fut)
            known_future_vals = jnp.nan_to_num(kc_fut, nan=0.0)

        rows = jnp.concatenate(ctx_rows, axis=1)
        mask = jnp.concatenate(ctx_masks, axis=1)
        if rows.shape[-1] > self.max_context:
            rows = rows[:, :, -self.max_context :]
            mask = mask[:, :, -self.max_context :]
        rows, mask = pad_to_patch(rows, mask, ps)
        v_all = rows.shape[1]
        if n_past == 0 and n_known == 0:
            return rows, mask, None, None, None, n_tgt

        role_ids = jnp.array(roles, dtype=jnp.int32).repeat(b)
        known_future = None
        known_future_mask = None
        if n_known > 0:
            assert known_future_vals is not None and known_future_obs is not None
            k0 = n_tgt + n_past
            plen = known_future_vals.shape[-1]
            kf = jnp.zeros((b, v_all, horizon), dtype=rows.dtype)
            kfm = jnp.zeros((b, v_all, horizon), dtype=jnp.bool_)
            kf = kf.at[:, k0 : k0 + n_known, :plen].set(known_future_vals)
            kfm = kfm.at[:, k0 : k0 + n_known, :plen].set(known_future_obs)
            known_future = kf.reshape(b * v_all, horizon)
            known_future_mask = kfm.reshape(b * v_all, horizon)

        return rows, mask, role_ids, known_future, known_future_mask, n_tgt

    def _block_ar_decode(
        self,
        inputs: jax.Array,
        observation_mask: jax.Array,
        prediction_length: int,
        role_ids: jax.Array | None,
        known_future: jax.Array | None,
        known_future_mask: jax.Array | None,
    ) -> jax.Array:
        ps = self.patch_size
        b, v, ctx_len = inputs.shape
        context = inputs.reshape(b * v, ctx_len)
        mask = observation_mask.reshape(b * v, ctx_len)
        group_ids = jnp.repeat(jnp.arange(b), v)

        known_rows = past_rows = target_rows = None
        if role_ids is not None:
            known_rows = role_ids == ROLE_KNOWN_COV
            past_rows = (role_ids != ROLE_TARGET) & ~known_rows
            target_rows = role_ids == ROLE_TARGET

        blocks = []
        remaining = math.ceil(prediction_length / ps)
        offset = 0
        while remaining > 0:
            block = min(self.per_shot_horizon_patches, remaining)
            blen = block * ps
            kf_block = known_future[:, offset : offset + blen] if known_future is not None else None
            kfm_block = (
                known_future_mask[:, offset : offset + blen]
                if known_future_mask is not None
                else None
            )

            quant_raw: Any = self.model.apply(
                {"params": self.params},
                context,
                mask,
                group_ids,
                block,
                role_ids,
                kf_block,
                kfm_block,
                method=self.model.predict_block,
            )
            quantiles = self._refine_block(quant_raw, context, mask, target_rows)
            blocks.append(quantiles)
            remaining -= block
            offset += blen
            if remaining <= 0:
                break

            feedback = replace_extreme_values(
                quantiles[..., self.median_index], self.extreme_value_threshold
            )
            feedback_mask = jnp.ones_like(feedback, dtype=jnp.bool_)
            if role_ids is not None:
                assert known_rows is not None and past_rows is not None
                if kf_block is not None and kfm_block is not None:
                    feedback = jnp.where(known_rows[:, None], kf_block, feedback)
                    feedback_mask = jnp.where(known_rows[:, None], kfm_block, feedback_mask)
                feedback = jnp.where(past_rows[:, None], 0.0, feedback)
                feedback_mask = feedback_mask & ~past_rows[:, None]
            context = jnp.concatenate([context, feedback], axis=-1)
            mask = jnp.concatenate([mask, feedback_mask], axis=-1)
            if context.shape[-1] > self.max_context:
                keep = max(ps, (self.max_context // ps) * ps)
                context = context[..., -keep:]
                mask = mask[..., -keep:]

        packed = jnp.concatenate(blocks, axis=-2)[..., :prediction_length, :]
        return packed.reshape(b, v, prediction_length, -1)

    def _refine_block(
        self,
        quantiles: jax.Array,
        context: jax.Array,
        mask: jax.Array,
        target_rows: jax.Array | None,
    ) -> jax.Array:
        ps = self.patch_size
        n_pred = quantiles.shape[1] // ps
        if n_pred <= 1:
            return quantiles
        scale_out: Any = self.model.apply(
            {"params": self.params}, context, mask, method=self.model.scale
        )
        _, loc, scale = scale_out
        loc0, scale0 = loc[:, -1], scale[:, -1]
        median = (
            quantiles[..., self.median_index]
            .reshape(quantiles.shape[0], n_pred, ps)
            .astype(jnp.float64)
        )
        sinh_median = (median - loc0[:, None, None]) / scale0[:, None, None]
        loc_j, scale_j = advancing_block_frames(
            context, mask, sinh_median, self.config.minimum_scale
        )

        first = jnp.zeros(loc_j.shape, dtype=jnp.bool_).at[:, 0].set(True)
        loc_j = jnp.repeat(
            jnp.where(first, loc0[:, None], loc_j.astype(quantiles.dtype)), ps, axis=-1
        )
        scale_j = jnp.repeat(
            jnp.where(first, scale0[:, None], scale_j.astype(quantiles.dtype)), ps, axis=-1
        )
        sinh_norm = (quantiles - loc0[:, None, None]) / scale0[:, None, None]
        refined = clamp_nonfinite(sinh_norm * scale_j[..., None] + loc_j[..., None])
        if target_rows is not None:
            refined = jnp.where(target_rows[:, None, None], refined, quantiles)
        return refined


def _negate(x: jax.Array | None) -> jax.Array | None:
    return -x if x is not None else None


def _cat_rows(a: jax.Array | None, b: jax.Array | None) -> jax.Array | None:
    if a is None or b is None:
        return a if b is None else b
    return jnp.concatenate([a, b], axis=1)
