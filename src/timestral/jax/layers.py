"""Transformer layers for Timestral in Flax / JAX.

All tensors use the packed layout: one row per variate of every series in the batch,
`(R, N, D)` with `N` patches. Time attention runs per row over patches; variate
attention runs per patch over the rows of the same series.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp

from timestral.config import LAYER_KIND_AXES


@dataclass(frozen=True)
class TwoStreamMask:
    """Time mask over the `[Enc ; Dec]` sequence plus its split point."""
    mask: jax.Array  # (R, 1, S, S) bool, True = attend
    enc_len: int


class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        scale = self.param("scale", nn.initializers.ones, (self.dim,))
        xf = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(xf), axis=-1, keepdims=True)
        normed = xf * jax.lax.rsqrt(variance + self.eps)
        return (normed * scale).astype(x.dtype)


class ResidualBlock(nn.Module):
    """`output(silu(hidden(x))) + residual(x)`, used by patch embedding and head."""
    input_dims: int
    hidden_dims: int
    output_dims: int
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        hidden_layer = nn.Dense(self.hidden_dims, use_bias=self.use_bias, name="hidden_layer")
        output_layer = nn.Dense(self.output_dims, use_bias=self.use_bias, name="output_layer")
        residual_layer = nn.Dense(self.output_dims, use_bias=self.use_bias, name="residual_layer")

        out = output_layer(jax.nn.silu(hidden_layer(x)))
        return out + residual_layer(x)


class FusedSwiGLU(nn.Module):
    """SwiGLU MLP with one fused input projection: `w3(gate * silu(up))`."""
    embed_dim: int
    hidden_dim: int
    use_bias: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        w12 = nn.Dense(2 * self.hidden_dim, use_bias=self.use_bias, name="w12")
        w3 = nn.Dense(self.embed_dim, use_bias=self.use_bias, name="w3")

        gate, up = jnp.split(w12(x), 2, axis=-1)
        return w3(gate * jax.nn.silu(up))


class SinCosRoPE(nn.Module):
    """Rotary position embedding (sin/cos concatenation) with partial rotation."""
    dim: int
    base: float
    rotary_dim: int

    def __call__(self, x: jax.Array, position_ids: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Return `(cos, sin)` tables, each `(R, N, head_dim)`."""
        inv_freq = jnp.zeros(self.dim // 2, dtype=jnp.float32)
        idx = jnp.arange(0, self.rotary_dim, 2, dtype=jnp.float32)
        inv_sub = 1.0 / (self.base ** (idx / self.rotary_dim))
        inv_freq = inv_freq.at[: self.rotary_dim // 2].set(inv_sub)

        # position_ids: (R, N)
        # freqs: (R, N, dim // 2)
        pos = position_ids.astype(jnp.float32)
        freqs = jnp.einsum("r n, d -> r n d", pos, inv_freq)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        cos = jnp.cos(emb).astype(x.dtype)
        sin = jnp.sin(emb).astype(x.dtype)
        return cos, sin


def _rotate_half(x: jax.Array) -> jax.Array:
    mid = x.shape[-1] // 2
    x1 = x[..., :mid]
    x2 = x[..., mid:]
    return jnp.concatenate([-x2, x1], axis=-1)


def _apply_rotary(
    q: jax.Array,
    k: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """Rotate `q`, `k` shaped `(batch, seq, heads, head_dim)`."""
    cos = jnp.expand_dims(cos, axis=2)
    sin = jnp.expand_dims(sin, axis=2)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


def _sdpa(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    mask: Optional[jax.Array] = None,
    is_causal: bool = False,
) -> jax.Array:
    """Scaled dot-product attention on `(batch, seq, heads, head_dim)`."""
    batch, q_len, heads, head_dim = query.shape
    k_len = key.shape[1]
    scale = 1.0 / math.sqrt(head_dim)

    # Compute scores: (batch, heads, q_len, k_len)
    scores = jnp.einsum("bqhd,bkhd->bhqk", query.astype(jnp.float32), key.astype(jnp.float32)) * scale

    if is_causal:
        causal_mask = jnp.tril(jnp.ones((q_len, k_len), dtype=jnp.bool_))
        scores = jnp.where(causal_mask[None, None, :, :], scores, -1e9)

    if mask is not None:
        # mask is broadcastable to (batch, heads, q_len, k_len) or (batch, 1, q_len, k_len)
        scores = jnp.where(mask, scores, -1e9)

    attn_weights = jax.nn.softmax(scores, axis=-1).astype(value.dtype)
    out = jnp.einsum("bhqk,bkhd->bqhd", attn_weights, value)
    return out.reshape(batch, q_len, heads * head_dim)


class Attention(nn.Module):
    """Multi-head attention over time (`is_time=True`) or variates."""
    embed_dim: int
    num_heads: int
    is_time: bool
    use_bias: bool = False
    gate: str = "sdpa_sigmoid"

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        mask: Optional[Union[jax.Array, TwoStreamMask]] = None,
        cos: Optional[jax.Array] = None,
        sin: Optional[jax.Array] = None,
    ) -> jax.Array:
        head_dim = self.embed_dim // self.num_heads
        rows, num_patches, _ = x.shape
        inputs = x if self.is_time else jnp.swapaxes(x, 0, 1)  # (R, N, D) or (N, R, D)

        wQKV = nn.Dense(3 * self.embed_dim, use_bias=self.use_bias, name="wQKV")
        wO = nn.Dense(self.embed_dim, use_bias=self.use_bias, name="wO")
        gate_layer = (
            nn.Dense(self.embed_dim, use_bias=self.use_bias, name="_gate")
            if self.gate == "sdpa_sigmoid"
            else None
        )

        batch, seq, _ = inputs.shape
        qkv = wQKV(inputs).reshape(batch, seq, 3, head_dim, self.num_heads)
        # Permute to (3, batch, seq, num_heads, head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 1, 4, 3))
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.is_time and cos is not None and sin is not None:
            q, k = _apply_rotary(q, k, cos, sin)

        if isinstance(mask, TwoStreamMask):
            assert self.is_time, "the context/horizon split applies to time attention only"
            n = mask.enc_len
            out_enc = _sdpa(q[:, :n], k[:, :n], v[:, :n], mask.mask[:, :, :n, :n])
            out_dec = _sdpa(q[:, n:], k, v, mask.mask[:, :, n:, :])
            out = jnp.concatenate([out_enc, out_dec], axis=1)
        else:
            out = _sdpa(q, k, v, mask, is_causal=self.is_time and mask is None)

        if gate_layer is not None:
            out = out * jax.nn.sigmoid(gate_layer(inputs))

        if self.is_time:
            out = out.reshape(rows, num_patches, self.embed_dim)
        else:
            out = jnp.swapaxes(out.reshape(num_patches, rows, self.embed_dim), 0, 1)

        return wO(out)


class TransformerBlock(nn.Module):
    """Attention(s) of one layer kind, each a pre-norm residual, then one MLP."""
    kind: str
    embed_dim: int
    num_heads: int
    mlp_hidden_dim: int
    norm_eps: float
    attn_bias: bool
    mlp_bias: bool
    time_attn_gate: str
    variate_attn_gate: str

    def setup(self):
        axes = LAYER_KIND_AXES[self.kind]
        self.attn_norms = [
            RMSNorm(dim=self.embed_dim, eps=self.norm_eps, name=f"attn_norms_{i}")
            for i in range(len(axes))
        ]
        self.attns = [
            Attention(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                is_time=is_time,
                use_bias=self.attn_bias,
                gate=self.time_attn_gate if is_time else self.variate_attn_gate,
                name=f"attns_{i}",
            )
            for i, is_time in enumerate(axes)
        ]
        self.mlp_norm = RMSNorm(dim=self.embed_dim, eps=self.norm_eps, name="mlp_norm")
        self.mlp = FusedSwiGLU(
            embed_dim=self.embed_dim,
            hidden_dim=self.mlp_hidden_dim,
            use_bias=self.mlp_bias,
            name="mlp",
        )

    def __call__(
        self,
        x: jax.Array,
        time_mask: Optional[Union[jax.Array, TwoStreamMask]],
        variate_mask: Optional[jax.Array],
        cos: jax.Array,
        sin: jax.Array,
    ) -> jax.Array:
        for norm, attn in zip(self.attn_norms, self.attns):
            m = time_mask if attn.is_time else variate_mask
            x = x + attn(norm(x), m, cos, sin)
        return x + self.mlp(self.mlp_norm(x))


class Transformer(nn.Module):
    block_schedule: Tuple[str, ...]
    embed_dim: int
    num_heads: int
    mlp_hidden_dim: int
    rope_theta: float
    rope_dim: int
    norm_eps: float
    attn_bias: bool
    mlp_bias: bool
    time_attn_gate: str
    variate_attn_gate: str

    def setup(self):
        self.layers = [
            TransformerBlock(
                kind=kind,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_hidden_dim=self.mlp_hidden_dim,
                norm_eps=self.norm_eps,
                attn_bias=self.attn_bias,
                mlp_bias=self.mlp_bias,
                time_attn_gate=self.time_attn_gate,
                variate_attn_gate=self.variate_attn_gate,
                name=f"layers_{i}",
            )
            for i, kind in enumerate(self.block_schedule)
        ]
        self.rope = SinCosRoPE(
            dim=self.embed_dim // self.num_heads,
            base=self.rope_theta,
            rotary_dim=self.rope_dim,
            name="rope",
        )
        self.out_norm = RMSNorm(dim=self.embed_dim, eps=self.norm_eps, name="out_norm")

    def __call__(
        self,
        x: jax.Array,
        time_mask: Optional[Union[jax.Array, TwoStreamMask]],
        variate_mask: Optional[jax.Array],
        positions: jax.Array,
    ) -> jax.Array:
        cos, sin = self.rope(x, positions)
        for layer in self.layers:
            x = layer(x, time_mask, variate_mask, cos, sin)
        return self.out_norm(x)
