import math

import numpy as np
import torch
from safetensors.torch import save_file

from timestral import TimestralConfig
from timestral.forecaster import TimestralForecaster as TorchTimestralForecaster
from timestral.jax.conversion import flax_to_torch_params, torch_to_flax_params
from timestral.jax.forecaster import TimestralJaxForecaster
from timestral.model import TimestralModel as TorchTimestralModel

KNOTS = (0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95)


def make_tiny_config(block_size: int = 4, max_context: int = 256) -> TimestralConfig:
    return TimestralConfig(
        patch_size=8,
        d_model=32,
        num_heads=4,
        mlp_hidden_dim=64,
        block_schedule=("time", "time_variate", "time", "time_variate"),
        quantile_knots=KNOTS,
        block_size=block_size,
        context_length=max_context,
    )


def test_param_conversion_roundtrip():
    config = make_tiny_config()
    torch.manual_seed(42)
    torch_model = TorchTimestralModel(config)
    for p in torch_model.parameters():
        torch.nn.init.normal_(p, std=0.05)

    orig_state = torch_model.state_dict()
    flax_params = torch_to_flax_params(orig_state, config)
    roundtrip_state = flax_to_torch_params(flax_params, config)

    assert set(orig_state.keys()) == set(roundtrip_state.keys())
    for k in orig_state:
        torch.testing.assert_close(
            orig_state[k], torch.from_numpy(roundtrip_state[k]), rtol=1e-6, atol=1e-6
        )


def test_jax_torch_forecaster_parity():
    config = make_tiny_config()
    torch.manual_seed(42)
    torch_model = TorchTimestralModel(config)
    for p in torch_model.parameters():
        torch.nn.init.normal_(p, std=0.05)
    torch_model.eval()

    torch_fc = TorchTimestralForecaster(torch_model)
    jax_fc = TimestralJaxForecaster.from_torch_state_dict(torch_model.state_dict(), config)

    # Generate synthetic input
    g = torch.Generator().manual_seed(123)
    t = torch.arange(100, dtype=torch.float32)
    x = 10 * torch.sin(2 * math.pi * t / 24) + torch.randn(2, 2, 100, generator=g).cumsum(-1)

    torch_out = torch_fc.forecast(
        x, prediction_length=16, clip_nonnegative=False, symmetric_averaging=False
    )
    jax_out = jax_fc.forecast(
        x.numpy(), prediction_length=16, clip_nonnegative=False, symmetric_averaging=False
    )

    np.testing.assert_allclose(
        torch_out.quantiles.numpy(),
        jax_out.quantiles,
        rtol=1e-4,
        atol=1e-4,
    )


def test_jax_forecaster_symmetric_averaging_and_clipping():
    config = make_tiny_config()
    torch.manual_seed(42)
    torch_model = TorchTimestralModel(config)
    torch_model.eval()

    torch_fc = TorchTimestralForecaster(torch_model)
    jax_fc = TimestralJaxForecaster.from_torch_state_dict(torch_model.state_dict(), config)

    t = torch.arange(80, dtype=torch.float32)
    x = (5 * torch.sin(2 * math.pi * t / 12)).abs() + 1.0  # strictly positive
    x = x.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1)

    torch_out = torch_fc.forecast(
        x, prediction_length=12, clip_nonnegative=True, symmetric_averaging=True
    )
    jax_out = jax_fc.forecast(
        x.numpy(), prediction_length=12, clip_nonnegative=True, symmetric_averaging=True
    )

    np.testing.assert_allclose(
        torch_out.quantiles.numpy(),
        jax_out.quantiles,
        rtol=1e-4,
        atol=1e-4,
    )


def test_jax_forecaster_from_pretrained(tmp_path):
    config = make_tiny_config()
    torch.manual_seed(42)
    torch_model = TorchTimestralModel(config)
    for p in torch_model.parameters():
        torch.nn.init.normal_(p, std=0.05)

    config.save(tmp_path / "config.json")
    save_file(torch_model.state_dict(), tmp_path / "model.safetensors")

    jax_fc = TimestralJaxForecaster.from_pretrained(str(tmp_path))
    torch_fc = TorchTimestralForecaster(torch_model.eval())

    x = np.random.randn(1, 1, 48).astype(np.float32)
    torch_out = torch_fc.forecast(
        torch.from_numpy(x), prediction_length=8, symmetric_averaging=False
    )
    jax_out = jax_fc.forecast(x, prediction_length=8, symmetric_averaging=False)

    np.testing.assert_allclose(torch_out.quantiles.numpy(), jax_out.quantiles, rtol=1e-4, atol=1e-4)
