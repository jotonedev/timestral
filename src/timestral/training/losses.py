"""Loss functions and probabilistic evaluation metrics for Timestral training."""

import jax
import jax.numpy as jnp


def pinball_loss(
    targets: jax.Array,
    predictions: jax.Array,
    quantile_knots: tuple[float, ...],
) -> jax.Array:
    r"""Multi-quantile pinball loss.

    $$\\rho_\\tau(e) = \\max(\\tau e, (\\tau - 1) e), \\quad e = y - \\hat{q}_\\tau$$

    Args:
        targets: `(..., T)` ground-truth values.
        predictions: `(..., T, Q)` quantile predictions.
        quantile_knots: tuple of quantile probability levels `(0.01, ..., 0.99)`.

    Returns:
        `(..., T, Q)` pinball loss values.
    """
    knots = jnp.array(quantile_knots, dtype=predictions.dtype)
    # Broadcast targets to (..., T, 1)
    diff = jnp.expand_dims(targets, -1) - predictions
    loss = jnp.maximum(knots * diff, (knots - 1.0) * diff)
    return loss


def masked_pinball_loss(
    targets: jax.Array,
    predictions: jax.Array,
    mask: jax.Array,
    quantile_knots: tuple[float, ...],
) -> jax.Array:
    """Mean pinball loss over observed target positions.

    Args:
        targets: `(R, T)` ground-truth values.
        predictions: `(R, T, Q)` quantile predictions.
        mask: `(R, T)` boolean mask (True = score this position).
        quantile_knots: tuple of quantile levels.

    Returns:
        Scalar mean pinball loss.
    """
    loss = pinball_loss(targets, predictions, quantile_knots)  # (R, T, Q)
    mask_exp = jnp.expand_dims(mask, -1).astype(loss.dtype)
    num_scored = jnp.maximum(mask_exp.sum() * len(quantile_knots), 1.0)
    return (loss * mask_exp).sum() / num_scored


def wql_metric(
    targets: jax.Array,
    predictions: jax.Array,
    mask: jax.Array,
    quantile_knots: tuple[float, ...],
) -> jax.Array:
    """Weighted Quantile Loss (WQL) / continuous ranked probability approximation.

    WQL = 2 * sum(pinball_loss) / sum(|targets|)
    """
    loss = pinball_loss(targets, predictions, quantile_knots)
    mask_exp = jnp.expand_dims(mask, -1).astype(loss.dtype)
    scored_loss = (loss * mask_exp).sum()
    scored_targets = (jnp.abs(jnp.expand_dims(targets, -1)) * mask_exp).sum()
    return (2.0 * scored_loss) / jnp.maximum(scored_targets, 1e-6)


def crps_metric(
    targets: jax.Array,
    predictions: jax.Array,
    mask: jax.Array,
    quantile_knots: tuple[float, ...],
) -> jax.Array:
    """Continuous Ranked Probability Score (CRPS) approximated from quantile predictions."""
    return 2.0 * masked_pinball_loss(targets, predictions, mask, quantile_knots)


def mase_metric(
    targets: jax.Array,
    forecasts: jax.Array,
    history: jax.Array,
    seasonality: int = 1,
) -> jax.Array:
    """Mean Absolute Scaled Error (MASE) against seasonal naive benchmark."""
    abs_err = jnp.mean(jnp.abs(targets - forecasts), axis=-1)
    naive_err = jnp.mean(jnp.abs(history[..., seasonality:] - history[..., :-seasonality]), axis=-1)
    scale = jnp.maximum(naive_err, 1e-6)
    return jnp.mean(abs_err / scale)
