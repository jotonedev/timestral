"""Synthetic time series generator combining Amazon Chronos (KernelSynth) and Google TimesFM."""

import functools
import math

import numpy as np
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    Kernel,
    RationalQuadratic,
    WhiteKernel,
)


# Standard KernelSynth Bank (from Chronos / Ansari et al., 2024)
def build_kernelsynth_bank(length: int = 1024) -> list[Kernel]:
    """Build standard KernelSynth GP kernel bank from Chronos."""
    return [
        ExpSineSquared(periodicity=24 / length),
        ExpSineSquared(periodicity=48 / length),
        ExpSineSquared(periodicity=96 / length),
        ExpSineSquared(periodicity=24 * 7 / length),
        ExpSineSquared(periodicity=48 * 7 / length),
        ExpSineSquared(periodicity=7 / length),
        ExpSineSquared(periodicity=14 / length),
        ExpSineSquared(periodicity=30 / length),
        ExpSineSquared(periodicity=365 / length),
        ExpSineSquared(periodicity=4 / length),
        ExpSineSquared(periodicity=12 / length),
        ExpSineSquared(periodicity=52 / length),
        DotProduct(sigma_0=0.0),
        DotProduct(sigma_0=1.0),
        DotProduct(sigma_0=10.0),
        RBF(length_scale=0.05),
        RBF(length_scale=0.1),
        RBF(length_scale=0.5),
        RBF(length_scale=1.0),
        RBF(length_scale=5.0),
        RationalQuadratic(alpha=0.1),
        RationalQuadratic(alpha=1.0),
        RationalQuadratic(alpha=10.0),
        WhiteKernel(noise_level=0.01),
        WhiteKernel(noise_level=0.1),
        ConstantKernel(),
    ]


def sample_kernelsynth_gp(
    length: int = 512,
    max_kernels: int = 4,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample from Gaussian Process prior with composite random kernels (Chronos KernelSynth)."""
    if rng is None:
        rng = np.random.default_rng()

    bank = build_kernelsynth_bank(length)
    X = np.linspace(0, 1, length)[:, None]

    for _ in range(5):
        num_k = int(rng.integers(1, max_kernels + 1))
        indices = rng.choice(len(bank), size=num_k, replace=True)
        selected = [bank[int(i)] for i in indices]

        def binary_compose(a: Kernel, b: Kernel) -> Kernel:
            return a + b if rng.random() < 0.5 else a * b

        composite_kernel = functools.reduce(binary_compose, selected)

        try:
            cov = composite_kernel(X)
            # Add small jitter for numerical stability
            cov += 1e-5 * np.eye(length)
            ts = rng.multivariate_normal(mean=np.zeros(length), cov=cov, method="eigh")
            return ts
        except (ValueError, np.linalg.LinAlgError):
            continue

    # Fallback to smooth RBF
    cov = RBF(length_scale=0.1)(X) + 1e-4 * np.eye(length)
    return rng.multivariate_normal(mean=np.zeros(length), cov=cov, method="eigh")


def sample_timesfm_components(
    length: int = 512,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample time series from TimesFM physical components (trends, harmonics, ARMA, random walk)."""
    if rng is None:
        rng = np.random.default_rng()

    t = np.linspace(0, 1, length)
    series = np.zeros(length, dtype=np.float32)

    # 1. Trend component (linear, quadratic, or logistic)
    trend_type = rng.choice(["linear", "poly", "logistic", "none"])
    if trend_type == "linear":
        slope = rng.uniform(-10, 10)
        intercept = rng.uniform(-5, 5)
        series += slope * t + intercept
    elif trend_type == "poly":
        deg = rng.integers(2, 4)
        coeffs = rng.normal(0, 2, size=deg + 1)
        series += np.polyval(coeffs, t)
    elif trend_type == "logistic":
        L = rng.uniform(5, 20)
        k = rng.uniform(5, 20)
        t0 = rng.uniform(0.2, 0.8)
        series += L / (1.0 + np.exp(-k * (t - t0)))

    # 2. Seasonality / Fourier harmonics (daily, weekly, custom)
    num_harmonics = rng.integers(1, 4)
    for _ in range(num_harmonics):
        freq = rng.uniform(2, 60)
        amp = rng.uniform(0.5, 5.0)
        phase = rng.uniform(0, 2 * math.pi)
        series += amp * np.sin(2 * math.pi * freq * t + phase)

    # 3. Autoregressive AR(1) or AR(2) process
    if rng.random() < 0.6:
        ar_series = np.zeros(length, dtype=np.float32)
        phi = rng.uniform(-0.85, 0.85)
        innov = rng.normal(0, 0.5, size=length)
        for i in range(1, length):
            ar_series[i] = phi * ar_series[i - 1] + innov[i]
        series += ar_series

    # 4. Random walk / Brownian noise
    if rng.random() < 0.5:
        rw = np.cumsum(rng.normal(0, 0.1, size=length))
        series += rw

    # 5. Structural breaks / level shifts
    if rng.random() < 0.3:
        break_pt = rng.integers(length // 4, 3 * length // 4)
        jump = rng.uniform(-5, 5)
        series[break_pt:] += jump

    # Observation noise
    series += rng.normal(0, rng.uniform(0.01, 0.3), size=length)
    return series.astype(np.float32)


def generate_multivariate_with_covariates(
    num_targets: int = 1,
    num_past_cov: int = 0,
    num_known_cov: int = 0,
    length: int = 512,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Generate coupled multivariate targets along with past and known future covariates.

    Uses the Chronos-2 multivariate mixing matrix formulation:
    Latent processes $\mathbf{Z} \in \mathbb{R}^{K \times T}$ mixed via $\mathbf{W} \in \mathbb{R}^{V \times K}$.

    Returns:
        `(values, observation_mask, role_ids, group_ids)`
        - values: `(R, T)` where `R = num_targets + num_past_cov + num_known_cov`.
        - observation_mask: `(R, T)` boolean mask.
        - role_ids: `(R,)` integers (0=TARGET, 1=PAST_COV, 2=KNOWN_COV).
        - group_ids: `(R,)` all 0 (same multivariate series).
    """
    if rng is None:
        rng = np.random.default_rng()

    total_rows = num_targets + num_past_cov + num_known_cov
    num_latents = max(3, total_rows + 1)

    # Sample latent processes (mix of KernelSynth and TimesFM)
    latents = []
    for _ in range(num_latents):
        if rng.random() < 0.5:
            s = sample_kernelsynth_gp(length=length, rng=rng)
        else:
            s = sample_timesfm_components(length=length, rng=rng)
        latents.append(s)
    Z = np.stack(latents, axis=0)  # (K, T)

    # Random coupling mixing matrix W
    W = rng.normal(0, 1.0, size=(num_targets, num_latents))
    targets = W @ Z  # (num_targets, T)

    # Past covariates: noisy / lagged copies of latents
    past_covs = []
    for _ in range(num_past_cov):
        w_past = rng.normal(0, 1.0, size=(1, num_latents))
        c = (w_past @ Z).squeeze(0) + rng.normal(0, 0.2, size=length)
        past_covs.append(c)

    # Known future covariates: calendar harmonics or binary pulse indicators
    known_covs = []
    for _ in range(num_known_cov):
        if rng.random() < 0.5:
            # Binary event / promotion indicator
            pulse = np.zeros(length, dtype=np.float32)
            n_pulses = rng.integers(2, 8)
            pulse_indices = rng.choice(length, size=n_pulses, replace=False)
            pulse[pulse_indices] = 1.0
            known_covs.append(pulse)
        else:
            # Calendar harmonic
            freq = rng.choice([7.0, 24.0, 30.0, 52.0])
            t = np.arange(length)
            har = np.sin(2 * math.pi * t / freq)
            known_covs.append(har.astype(np.float32))

    all_rows = [targets]
    roles = [0] * num_targets
    if num_past_cov > 0:
        all_rows.append(np.stack(past_covs, axis=0))
        roles += [1] * num_past_cov
    if num_known_cov > 0:
        all_rows.append(np.stack(known_covs, axis=0))
        roles += [2] * num_known_cov

    values = np.concatenate(all_rows, axis=0).astype(np.float32)
    obs_mask = _apply_missingness_augmentation(values, rng)

    role_ids = np.array(roles, dtype=np.int32)
    group_ids = np.zeros(total_rows, dtype=np.int32)
    return values, obs_mask, role_ids, group_ids


def _apply_missingness_augmentation(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply burst and dropout missingness patterns."""
    obs_mask = np.ones_like(values, dtype=bool)
    length = values.shape[-1]
    if rng.random() < 0.4:
        burst_len = rng.integers(5, min(length // 4, 64))
        burst_start = rng.integers(0, length - burst_len)
        obs_mask[:, burst_start : burst_start + burst_len] = False

    if rng.random() < 0.3:
        drop = rng.random(values.shape) < 0.05
        obs_mask = obs_mask & ~drop
    return obs_mask


def tsmixup(
    series1: np.ndarray,
    series2: np.ndarray,
    alpha: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    r"""TSMixup convex combination of two series: $\tilde{y} = \lambda y_1 + (1 - \lambda) y_2$."""
    if rng is None:
        rng = np.random.default_rng()
    lam = rng.beta(alpha, alpha)
    return lam * series1 + (1.0 - lam) * series2
