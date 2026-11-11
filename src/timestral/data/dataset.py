"""Datasets and batch iterators for Timestral training."""

import math
from collections.abc import Iterator
from typing import Any

import jax.numpy as jnp
import numpy as np

from timestral.data.synthetic import generate_multivariate_with_covariates


class SyntheticDataset:
    """Infinite iterator yielding training batches of synthetic time series."""

    def __init__(
        self,
        batch_size: int = 8,
        sequence_length: int = 256,
        patch_size: int = 8,
        block_size: int = 4,
        multivariate_prob: float = 0.3,
        past_cov_prob: float = 0.2,
        known_cov_prob: float = 0.2,
        seed: int = 42,
    ) -> None:
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.block_size = block_size
        # Round sequence_length up to multiple of patch_size
        self.sequence_length = math.ceil(sequence_length / patch_size) * patch_size
        self.n_patches = self.sequence_length // patch_size
        self.multivariate_prob = multivariate_prob
        self.past_cov_prob = past_cov_prob
        self.known_cov_prob = known_cov_prob
        self.rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return self

    def __next__(self) -> dict[str, Any]:
        return self.generate_batch()

    def generate_batch(self) -> dict[str, Any]:
        """Generate a single training batch of mixed synthetic multivariate series."""
        all_inputs = []
        all_obs = []
        all_groups = []
        all_roles = []
        all_min_ctx = []

        for b_idx in range(self.batch_size):
            num_targets = (
                int(self.rng.integers(2, 4)) if self.rng.random() < self.multivariate_prob else 1
            )
            num_past = int(self.rng.integers(1, 3)) if self.rng.random() < self.past_cov_prob else 0
            num_known = (
                int(self.rng.integers(1, 3)) if self.rng.random() < self.known_cov_prob else 0
            )

            vals, obs, roles, _ = generate_multivariate_with_covariates(
                num_targets=num_targets,
                num_past_cov=num_past,
                num_known_cov=num_known,
                length=self.sequence_length,
                rng=self.rng,
            )

            # Randomly select min_context patch index where forecast supervision begins
            # Must leave at least 1 patch as context and at least 1 patch as target
            min_ctx_patch = int(self.rng.integers(1, max(2, self.n_patches - 1)))

            n_rows = vals.shape[0]
            all_inputs.append(vals)
            all_obs.append(obs)
            all_groups.append(np.full(n_rows, b_idx, dtype=np.int32))
            all_roles.append(roles)
            all_min_ctx.append(np.full(n_rows, min_ctx_patch, dtype=np.int32))

        batch_inputs = np.concatenate(all_inputs, axis=0)
        batch_obs = np.concatenate(all_obs, axis=0)
        batch_groups = np.concatenate(all_groups, axis=0)
        batch_roles = np.concatenate(all_roles, axis=0)
        batch_min_ctx = np.concatenate(all_min_ctx, axis=0)

        return {
            "inputs": jnp.array(batch_inputs, dtype=jnp.float32),
            "observed": jnp.array(batch_obs, dtype=jnp.bool_),
            "group_ids": jnp.array(batch_groups, dtype=jnp.int32),
            "role_ids": jnp.array(batch_roles, dtype=jnp.int32),
            "min_context": jnp.array(batch_min_ctx, dtype=jnp.int32),
        }


class TimeSeriesDataset:
    """Batch generator for user-supplied real time-series data."""

    def __init__(
        self,
        data: np.ndarray | list[np.ndarray],
        batch_size: int = 8,
        sequence_length: int = 256,
        patch_size: int = 8,
        seed: int = 42,
    ) -> None:
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.sequence_length = math.ceil(sequence_length / patch_size) * patch_size
        self.n_patches = self.sequence_length // patch_size
        self.rng = np.random.default_rng(seed)

        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                self.series_list = [data]
            elif data.ndim == 2:
                self.series_list = [row for row in data]
            else:
                self.series_list = [data[i] for i in range(data.shape[0])]
        else:
            self.series_list = data

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return self

    def __next__(self) -> dict[str, Any]:
        all_inputs = []
        all_obs = []
        all_groups = []
        all_roles = []
        all_min_ctx = []

        for b_idx in range(self.batch_size):
            s = self.rng.choice(self.series_list)
            # Ensure s is 2D: (V, T)
            if s.ndim == 1:
                s = s[None, :]

            v, length = s.shape
            if length < self.sequence_length:
                # Left-pad to sequence_length
                pad_len = self.sequence_length - length
                window_vals = np.pad(s, [(0, 0), (pad_len, 0)], constant_values=0.0)
                window_obs = np.pad(np.isfinite(s), [(0, 0), (pad_len, 0)], constant_values=False)
            else:
                # Random window slice
                start = self.rng.integers(0, length - self.sequence_length + 1)
                window_vals = s[:, start : start + self.sequence_length]
                window_obs = np.isfinite(window_vals)
                window_vals = np.nan_to_num(window_vals, nan=0.0)

            min_ctx_patch = int(self.rng.integers(1, max(2, self.n_patches - 1)))

            all_inputs.append(window_vals)
            all_obs.append(window_obs)
            all_groups.append(np.full(v, b_idx, dtype=np.int32))
            all_roles.append(np.zeros(v, dtype=np.int32))
            all_min_ctx.append(np.full(v, min_ctx_patch, dtype=np.int32))

        return {
            "inputs": jnp.array(np.concatenate(all_inputs, axis=0), dtype=jnp.float32),
            "observed": jnp.array(np.concatenate(all_obs, axis=0), dtype=jnp.bool_),
            "group_ids": jnp.array(np.concatenate(all_groups, axis=0), dtype=jnp.int32),
            "role_ids": jnp.array(np.concatenate(all_roles, axis=0), dtype=jnp.int32),
            "min_context": jnp.array(np.concatenate(all_min_ctx, axis=0), dtype=jnp.int32),
        }
