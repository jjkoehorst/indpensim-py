"""Sanity tests for pat/raman.py.

Full-spectrum validation against MATLAB CSV is *not* attempted: the
captured spectra include irreproducible per-bin random noise (MATLAB
``randi`` without a captured seed). These tests cover the deterministic
formula structure and the deterministic-mode determinism.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REF_PATH = Path(__file__).resolve().parents[1] / "indpensim" / "data" / "reference_Specra.txt"


@pytest.fixture(scope="module")
def reference():
    from indpensim.pat.raman import build_reference
    return build_reference(REF_PATH)


def test_reference_shape(reference):
    assert reference.wavelength.shape == (2200,)
    assert reference.reference.shape == (2200,)
    assert reference.intensity_shift.shape == (2200,)
    # Intensity_shift1[1] = exp(1/1100) - 0.5 ≈ 0.5009...
    assert np.isclose(reference.intensity_shift[0], np.exp(1 / 1100.0) - 0.5)
    # Intensity_shift1[2200] = exp(2200/1100) - 0.5 = exp(2) - 0.5 ≈ 6.889
    assert np.isclose(reference.intensity_shift[-1], np.exp(2.0) - 0.5)


def test_glucose_peaks_have_three_modes(reference):
    """Three peaks centered at 219, 639, 1053 (1-based)."""
    g = reference.glucose_peaks
    # 0-based centers: 218, 638, 1052
    assert g[218] > g[100] and g[218] > g[400]
    assert g[638] > g[500] and g[638] > g[700]
    assert g[1052] > g[900] and g[1052] > g[1200]


def test_simulate_spectrum_deterministic(reference):
    """Two identical inputs with no noise produce identical outputs."""
    from indpensim.pat.raman import simulate_spectrum
    args = dict(
        reference=reference, P=1.0, X=2.0, viscosity=4.0, S=10.0, PAA=1500.0,
        k=100, N_samples=1085,
    )
    a = simulate_spectrum(**args)
    b = simulate_spectrum(**args)
    assert np.allclose(a, b)
    assert a.shape == (2200,)
    assert np.all(np.isfinite(a))


def test_simulate_spectrum_reproducible_with_rng(reference):
    """Same RNG seed → same noisy spectrum."""
    from indpensim.pat.raman import simulate_spectrum
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    args = dict(
        reference=reference, P=1.0, X=2.0, viscosity=4.0, S=10.0, PAA=1500.0,
        k=100, N_samples=1085,
    )
    a = simulate_spectrum(rng=rng_a, **args)
    b = simulate_spectrum(rng=rng_b, **args)
    assert np.allclose(a, b)


def test_simulate_spectrum_replays_captured_noise(reference):
    """Passing an explicit noise vector ignores the RNG and is bit-stable."""
    from indpensim.pat.raman import simulate_spectrum
    noise = np.full(2200, 25.0)
    args = dict(
        reference=reference, P=1.0, X=2.0, viscosity=4.0, S=10.0, PAA=1500.0,
        k=100, N_samples=1085,
    )
    a = simulate_spectrum(noise=noise, **args)
    b = simulate_spectrum(noise=noise, rng=np.random.default_rng(1234), **args)
    assert np.allclose(a, b)
