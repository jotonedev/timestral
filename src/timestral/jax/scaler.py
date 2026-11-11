"""Normalization for Timestral in JAX: causal per-patch mean/std scaling with asinh squash.

Each patch is normalized with the mean and std of every observed value up to the end of
that patch, so the context never sees future statistics. Forecasts come back in the
normalized (asinh) space of the last context patch and are inverted with sinh.
"""

from typing import Tuple
import jax
import jax.numpy as jnp


def compute_causal_statistics(
    data: jax.Array,
    observation_mask: jax.Array,
    minimum_scale: float,
    stabilize_with_global: bool,
    scale_factor_exponent: float = 10.0,
) -> Tuple[jax.Array, jax.Array]:
    """Causal running mean and std along the last axis (Welford form, in float64).

    Returns `(means, scales)`, both shaped like `data`; position `t` uses the observed
    values at positions `0..t` only. Unobserved positions get zero weight. The variance
    uses Bessel's correction and the std is clamped to `minimum_scale` after the sqrt.
    """
    orig_dtype = data.dtype
    w = observation_mask.astype(jnp.float64)
    x = data.astype(jnp.float64)

    weighted = w * x
    cum_w = jnp.cumsum(w, axis=-1)
    cum_x = jnp.cumsum(weighted, axis=-1)
    denominator = jnp.maximum(cum_w, 1.0)
    means = cum_x / denominator

    shifted = jnp.pad(means[..., :-1], [(0, 0)] * (means.ndim - 1) + [(1, 0)])
    increment = (x - shifted) * (x - means) * w
    m2 = jnp.cumsum(increment, axis=-1)
    variance = m2 / jnp.maximum(denominator - 1.0, 1.0)
    scales = jnp.maximum(jnp.sqrt(variance), minimum_scale)

    if stabilize_with_global:
        lo, hi = 10.0 ** (-scale_factor_exponent), 10.0**scale_factor_exponent
        global_den = jnp.maximum(jnp.sum(w, axis=-1, keepdims=True), 1.0)
        global_means = jnp.nan_to_num(jnp.sum(weighted, axis=-1, keepdims=True) / global_den)
        global_var = jnp.sum(((x - global_means) * w) ** 2, axis=-1, keepdims=True) / global_den
        global_scale = jnp.broadcast_to(jnp.sqrt(global_var + minimum_scale), scales.shape)
        floor = jnp.maximum(jnp.array(minimum_scale, dtype=scales.dtype), global_scale * lo)
        scales = jnp.clip(scales, a_min=floor, a_max=global_scale * hi)

    return means.astype(orig_dtype), scales.astype(orig_dtype)


def causal_patch_loc_scale(
    data: jax.Array,
    observation_mask: jax.Array,
    patch_size: int,
    minimum_scale: float,
    stabilize_with_global: bool,
) -> Tuple[jax.Array, jax.Array]:
    """Per-patch `(loc, scale)` broadcast back to every time step of the patch.

    Each patch uses the causal statistics at its last time step, i.e. everything
    observed up to the end of that patch.
    """
    t = data.shape[-1]
    assert t % patch_size == 0, f"length {t} is not a multiple of patch_size {patch_size}"
    n_patches = t // patch_size

    means, scales = compute_causal_statistics(data, observation_mask, minimum_scale, stabilize_with_global)

    leading_shape = data.shape[:-1]
    patch_means = means.reshape(*leading_shape, n_patches, patch_size)[..., -1]
    patch_scales = scales.reshape(*leading_shape, n_patches, patch_size)[..., -1]

    loc = jnp.repeat(patch_means, patch_size, axis=-1)
    scale = jnp.repeat(patch_scales, patch_size, axis=-1)
    return loc, scale


def apply_affine_asinh_masked(
    data: jax.Array,
    loc: jax.Array,
    scale: jax.Array,
    observation_mask: jax.Array,
    use_asinh: bool,
) -> jax.Array:
    """`asinh((data - loc) / scale)` on observed positions; unobserved positions become 0."""
    work_dtype = jnp.promote_types(data.dtype, jnp.float32)
    work_eps = jnp.finfo(work_dtype).eps

    scaled = (data.astype(work_dtype) - loc.astype(work_dtype)) / jnp.maximum(scale.astype(work_dtype), work_eps)
    scaled = jnp.where(observation_mask, scaled, 0.0)
    if use_asinh:
        scaled = jnp.where(observation_mask, jnp.arcsinh(scaled), scaled)
    return scaled.astype(data.dtype)


def invert_affine_asinh(x: jax.Array, loc: jax.Array, scale: jax.Array, use_asinh: bool) -> jax.Array:
    """Inverse of the forward transform: `sinh(x) * scale + loc`."""
    work_dtype = jnp.promote_types(x.dtype, jnp.float32)
    xw = x.astype(work_dtype)
    if use_asinh:
        xw = jnp.sinh(xw)
    out = xw * scale.astype(work_dtype) + loc.astype(work_dtype)
    return out.astype(x.dtype)


def clamp_nonfinite(vals: jax.Array) -> jax.Array:
    """Map +/-inf (a sinh overflow) to nearest finite value along the last axis."""
    finite = jnp.isfinite(vals)
    max_finite = jnp.where(finite, vals, -float("inf")).max(axis=-1, keepdims=True)
    min_finite = jnp.where(finite, vals, float("inf")).min(axis=-1, keepdims=True)

    clamped = jnp.where(vals == float("inf"), max_finite, vals)
    clamped = jnp.where(vals == -float("inf"), min_finite, clamped)
    return jnp.where(jnp.isinf(clamped), 0.0, clamped)


def denormalize_quantiles(preds: jax.Array, loc: jax.Array, scale: jax.Array, use_asinh: bool) -> jax.Array:
    """Invert normalization and clamp infinities; monotone quantiles stay in order."""
    return clamp_nonfinite(invert_affine_asinh(preds, loc, scale, use_asinh))


def welford_combine(
    w_a: jax.Array, mean_a: jax.Array, m2_a: jax.Array,
    w_b: jax.Array, mean_b: jax.Array, m2_b: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Merge two Welford states `(weight, mean, m2)` element-wise; `(0, 0, 0)` is identity."""
    w = w_a + w_b
    w_safe = jnp.maximum(w, jnp.finfo(w.dtype).tiny)
    delta = mean_b - mean_a
    mean = mean_a + delta * w_b / w_safe
    m2 = m2_a + m2_b + jnp.square(delta) * w_a * w_b / w_safe
    is_zero = w == 0
    mean = jnp.where(is_zero, 0.0, mean)
    m2 = jnp.where(is_zero, 0.0, m2)
    return w, mean, m2


def cumulative_welford(w: jax.Array, mean: jax.Array, m2: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Inclusive prefix scan of Welford states along dim 1."""
    n = w.shape[1]
    d = 1
    while d < n:
        pad = [(0, 0), (d, 0)] + [(0, 0)] * (w.ndim - 2)
        w_l = jnp.pad(w[:, : n - d], pad)
        mean_l = jnp.pad(mean[:, : n - d], pad)
        m2_l = jnp.pad(m2[:, : n - d], pad)
        w, mean, m2 = welford_combine(w_l, mean_l, m2_l, w, mean, m2)
        d *= 2
    return w, mean, m2


def advancing_block_frames(
    context: jax.Array,
    observation_mask: jax.Array,
    sinh_median: jax.Array,
    minimum_scale: float,
) -> Tuple[jax.Array, jax.Array]:
    """Per-patch `(loc, scale)` of a decoded block in advancing refine frame."""
    r, n_pred, ps = sinh_median.shape
    blank = jnp.zeros((r, n_pred, ps), dtype=sinh_median.dtype)
    n_ctx_patches = context.shape[-1] // ps

    ctx_val = context.reshape(r, n_ctx_patches, ps).astype(jnp.float64)
    ctx_w = observation_mask.reshape(r, n_ctx_patches, ps).astype(jnp.float64)

    values = jnp.concatenate([ctx_val, blank], axis=1)
    weights = jnp.concatenate([ctx_w, blank], axis=1)

    cnt = weights.sum(axis=-1)
    mean = (values * weights).sum(axis=-1) / jnp.maximum(cnt, 1.0)
    m2 = (weights * jnp.square(values - jnp.expand_dims(mean, -1))).sum(axis=-1)
    is_zero = cnt == 0
    mean = jnp.where(is_zero, 0.0, mean)
    m2 = jnp.where(is_zero, 0.0, m2)

    cnt, mean, m2 = cumulative_welford(cnt, mean, m2)
    cnt, mean, m2 = cnt[:, -n_pred - 1], mean[:, -n_pred - 1], m2[:, -n_pred - 1]

    offset = mean
    clip = float(jnp.finfo(jnp.float32).max ** 0.5 / 1e3)
    locs, scales = [], []
    for h in range(n_pred):
        var = jnp.maximum(m2 / jnp.maximum(cnt - 1.0, 1.0), 0.0)
        scale = jnp.maximum(jnp.sqrt(var), minimum_scale)
        locs.append(mean)
        scales.append(scale)
        folded = jnp.clip(sinh_median[:, h] * scale[:, None] + mean[:, None] - offset[:, None], -clip, clip)
        folded = folded + offset[:, None]
        b_mean = folded.mean(axis=-1)
        b_m2 = jnp.square(folded - b_mean[:, None]).sum(axis=-1)
        cnt, mean, m2 = welford_combine(cnt, mean, m2, jnp.full_like(cnt, ps), b_mean, b_m2)

    return jnp.stack(locs, axis=1), jnp.stack(scales, axis=1)


def instance_norm_stats(
    values: jax.Array,
    observation_mask: jax.Array,
    minimum_scale: float = 1e-6,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Non-causal per-row mean/std over observed positions, for known-future covariates."""
    orig_dtype = values.dtype
    stat_dtype = jnp.promote_types(orig_dtype, jnp.float32)
    x = values.astype(stat_dtype)
    mask_f = observation_mask.astype(stat_dtype)
    n = jnp.maximum(mask_f.sum(axis=-1, keepdims=True), 1.0)
    loc = (x * mask_f).sum(axis=-1, keepdims=True) / n
    var = (((x - loc) * mask_f) ** 2).sum(axis=-1, keepdims=True) / n
    scale = jnp.maximum(jnp.sqrt(var), minimum_scale)

    is_zero_one = ((values == 0.0) | (values == 1.0) | ~observation_mask).all(axis=-1, keepdims=True)
    is_binary = is_zero_one & observation_mask.any(axis=-1, keepdims=True)
    loc = jnp.where(is_binary, 0.0, loc)
    scale = jnp.where(is_binary, 1.0, scale)
    return loc.astype(orig_dtype), scale.astype(orig_dtype), is_binary
