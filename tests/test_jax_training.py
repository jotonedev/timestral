import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from timestral import TimestralConfig
from timestral.data.dataset import SyntheticDataset
from timestral.jax.model import TimestralModel
from timestral.training.losses import (
    crps_metric,
    masked_pinball_loss,
    pinball_loss,
    wql_metric,
)
from timestral.training.train_step import (
    create_train_step,
)
from timestral.training.trainer import Trainer, TrainingConfig, create_optimizer


def test_pinball_loss_exact_values():
    knots = (0.1, 0.5, 0.9)
    targets = jnp.array([1.0, 2.0])
    preds = jnp.array(
        [[0.5, 1.0, 1.5], [2.5, 2.0, 1.5]]
    )  # diff = [0.5, 0.0, -0.5], [-0.5, 0.0, 0.5]
    loss = pinball_loss(targets, preds, knots)
    # For target=1.0, pred=0.5 -> diff=0.5 > 0 -> tau * 0.5 = 0.1 * 0.5 = 0.05
    assert jnp.isclose(loss[0, 0], 0.05)
    # For target=1.0, pred=1.5 -> diff=-0.5 < 0 -> (tau - 1) * (-0.5) = (0.9 - 1) * (-0.5) = 0.05
    assert jnp.isclose(loss[0, 2], 0.05)


def test_masked_pinball_loss():
    knots = (0.5,)
    targets = jnp.array([[1.0, 2.0]])
    preds = jnp.array([[[0.0], [4.0]]])  # diff = [1.0, -2.0]
    mask = jnp.array([[True, False]])
    loss = masked_pinball_loss(targets, preds, mask, knots)
    # only position 0 scored: diff=1.0, loss = 0.5 * 1.0 = 0.5
    assert jnp.isclose(loss, 0.5)


def test_crps_and_wql_metrics():
    knots = (0.1, 0.5, 0.9)
    targets = jnp.array([[1.0, 2.0]])
    preds = jnp.array([[[0.9, 1.0, 1.1], [1.9, 2.0, 2.1]]])
    mask = jnp.array([[True, True]])
    crps = crps_metric(targets, preds, mask, knots)
    wql = wql_metric(targets, preds, mask, knots)
    assert crps > 0
    assert wql > 0
    assert jnp.isfinite(crps)
    assert jnp.isfinite(wql)


def test_training_step_loss_decreases():
    config = TimestralConfig(
        patch_size=8,
        d_model=32,
        num_heads=4,
        mlp_hidden_dim=64,
        block_schedule=("time", "time_variate"),
        quantile_knots=(0.1, 0.5, 0.9),
        block_size=4,
        context_length=64,
    )
    model = TimestralModel(config)
    rng = jax.random.PRNGKey(42)

    # Init params
    ps = config.patch_size
    dummy_len = ps * 4
    dummy_inp = jnp.zeros((2, dummy_len), dtype=jnp.float32)
    dummy_obs = jnp.ones((2, dummy_len), dtype=jnp.bool_)
    dummy_grp = jnp.array([0, 1])
    init_vars = model.init(
        rng, dummy_inp, dummy_obs, dummy_grp, horizon_patches=2, method=model.predict_block
    )
    params = init_vars["params"]

    train_cfg = TrainingConfig(
        learning_rate=3e-3,
        warmup_steps=5,
        total_steps=35,
        max_grad_norm=1.0,
    )
    tx = create_optimizer(train_cfg)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    train_step_fn = create_train_step(model)

    # Fixed training batch
    R, W = 4, 64
    batch = {
        "inputs": jnp.tile(jnp.sin(jnp.linspace(0, 10, W)), (R, 1)),
        "observed": jnp.ones((R, W), dtype=jnp.bool_),
        "group_ids": jnp.array([0, 0, 1, 1], dtype=jnp.int32),
        "role_ids": jnp.zeros((R,), dtype=jnp.int32),
        "min_context": jnp.array([2, 2, 2, 2], dtype=jnp.int32),
    }

    initial_loss = None
    final_loss = None

    for step in range(35):
        state, metrics = train_step_fn(state, batch)
        loss = float(metrics["loss"])
        if step == 0:
            initial_loss = loss
        final_loss = loss

    assert final_loss < initial_loss * 0.85, (
        f"Loss did not decrease: initial={initial_loss}, final={final_loss}"
    )
    assert np.isfinite(final_loss)


def test_trainer_synthetic_pipeline(tmp_path):
    config = TimestralConfig(
        patch_size=8,
        d_model=32,
        num_heads=4,
        mlp_hidden_dim=64,
        block_schedule=("time", "time_variate"),
        quantile_knots=(0.1, 0.5, 0.9),
        block_size=4,
        context_length=64,
    )
    train_cfg = TrainingConfig(
        total_steps=5,
        learning_rate=1e-3,
        warmup_steps=2,
        log_every_steps=1,
        eval_every_steps=3,
        save_every_steps=5,
        output_dir=str(tmp_path),
    )
    dataset = SyntheticDataset(
        batch_size=2,
        sequence_length=64,
        patch_size=8,
        block_size=4,
        seed=42,
    )
    model = TimestralModel(config)
    trainer = Trainer(model=model, config=train_cfg)
    results = trainer.train(train_iter=dataset)
    assert "final_loss" in results
    assert (tmp_path / "checkpoint_final" / "model.safetensors").exists()
    assert (tmp_path / "checkpoint_final" / "config.json").exists()
