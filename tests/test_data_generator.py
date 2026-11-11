import jax.numpy as jnp
import numpy as np

from timestral.data.dataset import SyntheticDataset, TimeSeriesDataset
from timestral.data.synthetic import (
    generate_multivariate_with_covariates,
    sample_kernelsynth_gp,
    sample_timesfm_components,
    tsmixup,
)


def test_kernelsynth_gp_generation():
    rng = np.random.default_rng(42)
    series = sample_kernelsynth_gp(length=128, max_kernels=3, rng=rng)
    assert series.shape == (128,)
    assert np.all(np.isfinite(series))


def test_timesfm_component_generation():
    rng = np.random.default_rng(101)
    series = sample_timesfm_components(length=200, rng=rng)
    assert series.shape == (200,)
    assert np.all(np.isfinite(series))


def test_multivariate_with_covariates_generation():
    rng = np.random.default_rng(99)
    values, obs_mask, role_ids, group_ids = generate_multivariate_with_covariates(
        num_targets=2,
        num_past_cov=1,
        num_known_cov=1,
        length=150,
        rng=rng,
    )
    assert values.shape == (4, 150)
    assert obs_mask.shape == (4, 150)
    assert role_ids.shape == (4,)
    assert group_ids.shape == (4,)
    assert np.all(np.isfinite(values))
    assert list(role_ids) == [0, 0, 1, 2]


def test_tsmixup():
    s1 = np.ones(50, dtype=np.float32)
    s2 = np.zeros(50, dtype=np.float32)
    mixed = tsmixup(s1, s2, alpha=0.5, rng=np.random.default_rng(7))
    assert mixed.shape == (50,)
    assert np.all(np.isfinite(mixed))
    assert np.all(mixed >= 0.0) and np.all(mixed <= 1.0)


def test_synthetic_dataset_iterator():
    ds = SyntheticDataset(
        batch_size=2,
        sequence_length=64,
        patch_size=8,
        block_size=4,
        multivariate_prob=0.5,
        past_cov_prob=0.5,
        known_cov_prob=0.5,
        seed=42,
    )
    batch = next(ds)
    assert "inputs" in batch
    assert "observed" in batch
    assert "group_ids" in batch
    assert "role_ids" in batch
    assert "min_context" in batch
    assert batch["inputs"].shape[-1] == 64
    assert jnp.all(jnp.isfinite(batch["inputs"]))


def test_timeseries_dataset_batches():
    series_list = [np.random.randn(2, 120).astype(np.float32) for _ in range(5)]
    ds = TimeSeriesDataset(
        data=series_list,
        batch_size=2,
        sequence_length=48,
        patch_size=8,
        seed=1,
    )
    batch = next(ds)
    assert "inputs" in batch
    assert "observed" in batch
    assert batch["inputs"].shape[-1] == 48
