"""Attention masks and time positions for Decoupled Horizon Forecasting (DHF) in JAX.

The sequence is `[Enc ; Dec]`: encoder tokens embed the observed context patches;
decoder tokens are forecast queries for the horizon patches. All masks are boolean,
True meaning "may attend".
"""

import jax
import jax.numpy as jnp


def compute_enc_key_validity(obs_per_patch: jax.Array, attend_missing: bool) -> jax.Array:
    """`(R, N)` encoder key validity.

    A patch is a valid key if it is observed, or if `attend_missing` is set and the
    row's series has started at or before it. Leading all-unobserved patches are
    left-padding and are never valid keys.
    """
    if not attend_missing:
        return obs_per_patch
    started = jnp.maximum.accumulate(obs_per_patch, axis=-1)
    return obs_per_patch | started


def enc_causal_mask(key_valid: jax.Array) -> jax.Array:
    """`(R, L, L)` causal encoder mask over valid keys, with the diagonal always open."""
    _, ll = key_valid.shape
    causal = jnp.tril(jnp.ones((ll, ll), dtype=jnp.bool_))
    eye = jnp.eye(ll, dtype=jnp.bool_)
    return (causal[None] & key_valid[:, None, :]) | eye[None]


def build_forecast_time_mask(context_len: int, horizon: int, key_valid: jax.Array) -> jax.Array:
    """`(R, 1, L+h, L+h)` time mask for one block without covariates.

    The encoder is causal; every decoder token reads the whole valid context and its
    own causal decoder chain; the context never reads the horizon.
    """
    rows = key_valid.shape[0]
    enc = enc_causal_mask(key_valid)
    enc_dec = jnp.zeros((rows, context_len, horizon), dtype=jnp.bool_)
    dec_enc = jnp.broadcast_to(key_valid[:, None, :], (rows, horizon, context_len))
    dec_dec = jnp.broadcast_to(
        jnp.tril(jnp.ones((horizon, horizon), dtype=jnp.bool_))[None], (rows, horizon, horizon)
    )
    top = jnp.concatenate([enc, enc_dec], axis=2)
    bottom = jnp.concatenate([dec_enc, dec_dec], axis=2)
    return jnp.expand_dims(jnp.concatenate([top, bottom], axis=1), axis=1)


def build_forecast_positions(context_len: int, horizon: int, first_obs: jax.Array) -> jax.Array:
    """`(R, L+h)` time positions `0..L+h-1`, anchored at each row's first observed patch."""
    p = jnp.arange(context_len + horizon)
    return p[None, :] - first_obs[:, None]


def compute_block_layout(
    min_context: jax.Array, n_patches: int, block_size: int
) -> tuple[jax.Array, jax.Array]:
    """Per decoder position: the anchor of its block and whether it is active (`p >= s`)."""
    p = jnp.arange(n_patches)[None, :]
    s = min_context[:, None]
    active = p >= s
    offset = jnp.maximum(p - s, 0)
    anchor = s + block_size * (offset // block_size)
    anchor = jnp.where(active, anchor, 0)
    return anchor, active


def build_two_stream_time_mask(
    min_context: jax.Array,
    n_patches: int,
    block_size: int,
    key_valid: jax.Array,
    bidirectional_rows: jax.Array | None = None,
) -> jax.Array:
    """`(R, 1, 2N, 2N)` time mask over a full `[Enc ; Dec]` window.

    - Enc to Enc: causal over valid keys; full (non-causal) for `bidirectional_rows`.
    - Enc to Dec: never.
    - Dec to Enc: a decoder token in the block anchored at `b` reads valid `Enc[j]`, `j < b`.
    - Dec to Dec: causal within the same block; every decoder token reads itself.
    """
    anchor, active = compute_block_layout(min_context, n_patches, block_size)
    rows, n = active.shape
    causal = jnp.tril(jnp.ones((n, n), dtype=jnp.bool_))
    key_col = key_valid[:, None, :]
    j = jnp.arange(n).reshape(1, 1, n)
    eye = jnp.eye(n, dtype=jnp.bool_)
    enc_enc = enc_causal_mask(key_valid)
    if bidirectional_rows is not None:
        enc_bi = jnp.broadcast_to(key_valid[:, None, :], (rows, n, n)) | eye[None]
        enc_enc = jnp.where(bidirectional_rows[:, None, None], enc_bi, enc_enc)
    enc_dec = jnp.zeros((rows, n, n), dtype=jnp.bool_)
    dec_enc = active[:, :, None] & (j < anchor[:, :, None]) & key_col
    same_block = anchor[:, :, None] == anchor[:, None, :]
    dec_dec = same_block & causal[None] & active[:, :, None] & active[:, None, :]
    dec_dec = dec_dec | eye[None]
    top = jnp.concatenate([enc_enc, enc_dec], axis=2)
    bottom = jnp.concatenate([dec_enc, dec_dec], axis=2)
    return jnp.expand_dims(jnp.concatenate([top, bottom], axis=1), axis=1)


def variate_firewall(
    r: int,
    target_rows: jax.Array | None = None,
    bidirectional_rows: jax.Array | None = None,
) -> jax.Array | None:
    """`(R, R)` forbidden query-to-key row pairs, or None."""
    if target_rows is None and bidirectional_rows is None:
        return None
    forbid = jnp.zeros((r, r), dtype=jnp.bool_)
    if target_rows is not None:
        forbid = forbid | ((~target_rows[:, None]) & target_rows[None, :])
    if bidirectional_rows is not None:
        forbid = forbid | (bidirectional_rows[:, None] & (~bidirectional_rows)[None, :])
    return forbid


def build_two_stream_variate_mask(
    group_ids: jax.Array,
    enc_key_valid: jax.Array,
    n_dec: int,
    target_rows: jax.Array | None = None,
    bidirectional_rows: jax.Array | None = None,
) -> jax.Array | None:
    """`(N_enc + N_dec, 1, R, R)` variate mask, or None for a single row."""
    r, _ = enc_key_valid.shape
    if r == 1:
        return None
    row_eq = group_ids[:, None] == group_ids[None, :]
    dec_valid = jnp.ones((r, n_dec), dtype=jnp.bool_)
    slot_valid = jnp.concatenate([enc_key_valid, dec_valid], axis=1)
    key_valid = slot_valid.transpose(1, 0)
    mask = row_eq[None, None, :, :] & key_valid[:, None, None, :]
    forbid = variate_firewall(r, target_rows, bidirectional_rows)
    if forbid is not None:
        mask = mask & ~forbid[None, None, :, :]
    eye = jnp.eye(r, dtype=jnp.bool_)
    return mask | eye[None, None, :, :]


def build_two_stream_positions(n_patches: int, first_obs: jax.Array) -> jax.Array:
    """`(R, 2N)` positions: encoder and decoder token of the same patch share a time."""
    p = jnp.arange(n_patches)
    p2 = jnp.concatenate([p, p])
    return p2[None, :] - first_obs[:, None]


def row_group_first_observed_patch(obs_per_patch: jax.Array, group_ids: jax.Array) -> jax.Array:
    """`(R,)` index of earliest observed patch for the group of each row (JIT-static, no unique)."""
    rows, n = obs_per_patch.shape
    idx = jnp.broadcast_to(jnp.arange(n)[None, :], (rows, n))
    row_first = jnp.where(obs_per_patch, idx, n).min(axis=-1)  # (R,)
    same_group = group_ids[:, None] == group_ids[None, :]  # (R, R)
    candidate_first = jnp.where(same_group, row_first[None, :], n)
    group_first = candidate_first.min(axis=-1)
    return jnp.minimum(group_first, n - 1)


def group_first_observed_patch(
    obs_per_patch: jax.Array, group_inverse: jax.Array, n_groups: int
) -> jax.Array:
    """`(G,)` index of the earliest observed patch in any row of each series."""
    rows, n = obs_per_patch.shape
    idx = jnp.broadcast_to(jnp.arange(n)[None, :], (rows, n))
    row_first = jnp.where(obs_per_patch, idx, n).min(axis=-1)  # (R,)
    out = jnp.full(n_groups, n, dtype=row_first.dtype)
    out = out.at[group_inverse].min(row_first)
    return jnp.minimum(out, n - 1)
