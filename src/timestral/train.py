"""Command line interface for training Timestral models from scratch or fine-tuning with JAX."""

import argparse
import os
from pathlib import Path

from timestral.config import TimestralConfig
from timestral.data.dataset import SyntheticDataset
from timestral.jax.conversion import load_flax_model_from_safetensors
from timestral.jax.model import TimestralModel
from timestral.training.trainer import Trainer, TrainingConfig

KNOTS = (0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
         0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Timestral from scratch with JAX/Flax")
    # Model architecture
    parser.add_argument("--patch_size", type=int, default=8, help="Time steps per patch")
    parser.add_argument("--d_model", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--mlp_hidden_dim", type=int, default=128, help="SwiGLU hidden dim")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--block_size", type=int, default=4, help="Max patches per decode block")
    parser.add_argument("--context_length", type=int, default=256, help="Trained context length cap")

    # Training hyperparams
    parser.add_argument("--steps", type=int, default=1000, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=16, help="Series per training batch")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="AdamW weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--log_every", type=int, default=50, help="Logging interval")
    parser.add_argument("--save_every", type=int, default=500, help="Checkpoint interval")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--init_from_pretrained", type=str, default=None, help="Checkpoint to initialize from")

    return parser.parse_args()


def main():
    args = parse_args()

    # Define alternating time and variate block schedule
    kinds = ["time", "variate", "time_variate", "variate_time"]
    schedule = tuple(kinds[i % len(kinds)] for i in range(args.num_layers))

    config = TimestralConfig(
        patch_size=args.patch_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        mlp_hidden_dim=args.mlp_hidden_dim,
        block_schedule=schedule,
        quantile_knots=KNOTS,
        block_size=args.block_size,
        context_length=args.context_length,
    )

    model = TimestralModel(config)
    init_params = None
    if args.init_from_pretrained:
        print(f"Loading initial weights from {args.init_from_pretrained}...")
        init_params = load_flax_model_from_safetensors(args.init_from_pretrained, config)

    train_config = TrainingConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        total_steps=args.steps,
        max_grad_norm=args.max_grad_norm,
        log_every_steps=args.log_every,
        eval_every_steps=args.steps // 5 if args.steps >= 5 else 1,
        save_every_steps=args.save_every,
        output_dir=args.output_dir,
    )

    trainer = Trainer(
        model=model,
        config=train_config,
        init_params=init_params,
        rng_seed=args.seed,
    )

    # Initialize synthetic dataset stream
    train_dataset = SyntheticDataset(
        batch_size=args.batch_size,
        sequence_length=args.context_length,
        patch_size=args.patch_size,
        block_size=args.block_size,
        seed=args.seed,
    )
    val_dataset = SyntheticDataset(
        batch_size=args.batch_size,
        sequence_length=args.context_length,
        patch_size=args.patch_size,
        block_size=args.block_size,
        seed=args.seed + 1000,
    )

    print(f"Initialized Timestral model with {args.d_model} d_model, {args.num_layers} layers.")
    trainer.train(train_iter=iter(train_dataset), eval_iter=iter(val_dataset))


if __name__ == "__main__":
    main()
