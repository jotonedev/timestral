"""Trainer for Timestral time series foundation models using JAX, Flax, and Optax."""

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, Iterator, Optional

from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax

from timestral.config import TimestralConfig
from timestral.jax.conversion import save_flax_model_to_safetensors
from timestral.jax.model import TimestralModel
from timestral.training.train_step import create_eval_step, create_train_step

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Hyperparameters and configuration for Timestral training."""
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-2
    warmup_steps: int = 500
    total_steps: int = 10000
    max_grad_norm: float = 1.0
    adam_b1: float = 0.9
    adam_b2: float = 0.95
    adam_eps: float = 1e-8
    log_every_steps: int = 50
    eval_every_steps: int = 250
    save_every_steps: int = 1000
    output_dir: str = "./checkpoints"


def create_optimizer(config: TrainingConfig) -> optax.GradientTransformation:
    """Build Optax AdamW optimizer with warmup cosine decay and global norm clipping."""
    warmup_steps = min(config.warmup_steps, max(1, config.total_steps // 5))
    decay_steps = max(config.total_steps, warmup_steps + 1)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-7,
        peak_value=config.learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=config.min_learning_rate,
    )
    return optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(
            learning_rate=schedule,
            b1=config.adam_b1,
            b2=config.adam_b2,
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        ),
    )


class Trainer:
    """Manages the JAX / Optax training lifecycle."""

    def __init__(
        self,
        model: TimestralModel,
        config: TrainingConfig,
        init_params: Optional[Dict[str, Any]] = None,
        rng_seed: int = 42,
    ) -> None:
        self.model = model
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        tx = create_optimizer(config)
        rng = jax.random.PRNGKey(rng_seed)

        if init_params is None:
            # Initialize random weights with dummy input
            ps = model.config.patch_size
            dummy_len = ps * 4
            dummy_inp = jnp.zeros((2, dummy_len), dtype=jnp.float32)
            dummy_obs = jnp.ones((2, dummy_len), dtype=jnp.bool_)
            dummy_grp = jnp.array([0, 1])
            init_vars = model.init(
                rng, dummy_inp, dummy_obs, dummy_grp, horizon_patches=2, method=model.predict_block
            )
            params = init_vars["params"]
        else:
            params = init_params

        self.state = TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx,
        )

        self.train_step_fn = create_train_step(model)
        self.eval_step_fn = create_eval_step(model)

    def train(
        self,
        train_iter: Iterator[Dict[str, Any]],
        eval_iter: Optional[Iterator[Dict[str, Any]]] = None,
        num_eval_batches: int = 10,
    ) -> Dict[str, float]:
        """Execute the training loop across `total_steps`."""
        logger.info(f"Starting training for {self.config.total_steps} steps...")
        start_time = time.time()
        step_times = []

        last_loss = 0.0
        for step in range(1, self.config.total_steps + 1):
            t0 = time.time()
            batch = next(train_iter)
            self.state, metrics = self.train_step_fn(self.state, batch)
            loss_val = float(metrics["loss"])
            last_loss = loss_val
            step_times.append(time.time() - t0)

            if step % self.config.log_every_steps == 0 or step == 1:
                avg_time = np.mean(step_times[-self.config.log_every_steps :])
                elapsed = time.time() - start_time
                print(
                    f"Step {step:6d}/{self.config.total_steps} | "
                    f"Loss: {loss_val:.4f} | "
                    f"Scored pts: {int(metrics['scored_points']):d} | "
                    f"Speed: {1.0 / avg_time:.1f} steps/s | "
                    f"Elapsed: {elapsed:.1f}s"
                )

            if eval_iter is not None and step % self.config.eval_every_steps == 0:
                eval_losses = []
                for _ in range(num_eval_batches):
                    eval_batch = next(eval_iter)
                    eval_metrics = self.eval_step_fn(self.state, eval_batch)
                    eval_losses.append(float(eval_metrics["loss"]))
                mean_eval_loss = float(np.mean(eval_losses))
                print(f"--> [Evaluation] Step {step}: Val Loss = {mean_eval_loss:.4f}")

            if step % self.config.save_every_steps == 0:
                ckpt_path = self.output_dir / f"checkpoint_step_{step}"
                self.save_checkpoint(ckpt_path)

        # Final save
        final_path = self.output_dir / "checkpoint_final"
        self.save_checkpoint(final_path)
        print(f"Training complete. Final checkpoint saved to {final_path}")
        return {"final_loss": last_loss}

    def save_checkpoint(self, path: Path | str) -> None:
        """Save model configuration and weights as PyTorch-compatible safetensors."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self.model.config.save(p / "config.json")
        save_flax_model_to_safetensors(self.state.params, p / "model.safetensors", self.model.config)
        logger.info(f"Saved checkpoint to {p}")
