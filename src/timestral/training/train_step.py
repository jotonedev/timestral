"""JIT-compiled DHF training step and loss computation for Timestral."""

from typing import Any

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from timestral.jax.model import ROLE_TARGET, TimestralModel
from timestral.jax.scaler import apply_affine_asinh_masked
from timestral.training.losses import masked_pinball_loss


def compute_dhf_loss(
    model: TimestralModel,
    params: Any,
    batch: dict[str, jax.Array],
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compute Decoupled Horizon Forecasting (DHF) pinball loss on normalized targets.

    Args:
        model: TimestralModel instance.
        params: Flax parameter pytree.
        batch: Dictionary containing:
            - "inputs": `(R, W)` raw sequence values.
            - "observed": `(R, W)` boolean observation mask.
            - "group_ids": `(R,)` series index per row.
            - "min_context": `(R,)` starting patch index for supervision.
            - "role_ids": `(R,)` row role indicators (TARGET=0, PAST_COV=1, KNOWN_COV=2).

    Returns:
        `(loss, metrics)`
    """
    full_inputs = batch["inputs"]
    full_observed = batch["observed"]
    group_ids = batch["group_ids"]
    min_context = batch["min_context"]
    role_ids = batch.get("role_ids")
    if role_ids is None:
        role_ids = jnp.zeros((full_inputs.shape[0],), dtype=jnp.int32)

    # DHF training forward pass
    fwd_out: Any = model.apply(
        {"params": params},
        full_inputs,
        full_observed,
        group_ids,
        min_context,
        role_ids,
        method=model.training_forward,
    )
    normalized_preds, loc, scale = fwd_out

    # Compute normalized ground truth targets
    y_norm = apply_affine_asinh_masked(
        full_inputs, loc, scale, full_observed, use_asinh=model.config.use_asinh
    )

    # Build supervision mask over active forecast patches on TARGET rows
    ps = model.config.patch_size
    n_patches = full_inputs.shape[-1] // ps
    p_idx = jnp.arange(n_patches)[None, :]
    active_patches = p_idx >= min_context[:, None]
    active_steps = jnp.repeat(active_patches, ps, axis=-1)
    is_target = (role_ids == ROLE_TARGET)[:, None]

    target_mask = full_observed & active_steps & is_target
    loss = masked_pinball_loss(y_norm, normalized_preds, target_mask, model.config.quantile_knots)

    metrics = {
        "loss": loss,
        "scored_points": target_mask.sum(),
    }
    return loss, metrics


def create_train_step(model: TimestralModel):
    """Create a JIT-compiled train_step function for the given model."""

    @jax.jit
    def train_step(
        state: TrainState,
        batch: dict[str, jax.Array],
    ) -> tuple[TrainState, dict[str, jax.Array]]:
        def loss_fn(params):
            return compute_dhf_loss(model, params, batch)

        (_loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, metrics

    return train_step


def create_eval_step(model: TimestralModel):
    """Create a JIT-compiled eval_step function for the given model."""

    @jax.jit
    def eval_step(
        state: TrainState,
        batch: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        _, metrics = compute_dhf_loss(model, state.params, batch)
        return metrics

    return eval_step
