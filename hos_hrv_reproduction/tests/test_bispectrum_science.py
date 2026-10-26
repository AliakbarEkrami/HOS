"""Independent numerical and scientific checks for the HOS implementation."""

from dataclasses import replace

import numpy as np
import pytest

from hos_hrv_reproduction.bispectrum import (
    BispectrumResult,
    direct_bispectrum,
    features_from_bispectrum,
)
from hos_hrv_reproduction.config import PaperConfig


def _explicit_reference(x, config):
    """Slow DFT and scalar triple products; no production estimator helpers."""
    n = config.block_size
    sample = np.arange(n)
    window = np.ones(n)
    if config.window == "hann":
        window = 0.5 - 0.5 * np.cos(2 * np.pi * sample / n)
    starts = list(range(0, len(x) - n + 1, n - config.overlap))
    spectra = []
    for start in starts:
        block = x[start:start + n].copy()
        if config.detrend:
            block -= sum(block) / n
        spectrum = []
        for frequency in range(n // 2 + 1):
            coefficients = np.exp(-2j * np.pi * frequency * sample / n)
            value = sum(block * window * coefficients)
            spectrum.append(value / n if config.fft_scaling == "length" else value)
        spectra.append(spectrum)
    expected = {}
    for k1 in range(n // 2 + 1):
        for k2 in range(k1 + 1):
            if k1 + k2 > n // 2 or (not config.include_dc and k2 == 0):
                continue
            triples, first_power, sum_power = [], [], []
            for spectrum in spectra:
                product = spectrum[k1] * spectrum[k2]
                summed = spectrum[k1 + k2]
                triples.append(product * summed.conjugate())
                first_power.append(abs(product) ** 2)
                sum_power.append(abs(summed) ** 2)
            denominator = sum(first_power) * sum(sum_power)
            coherence = abs(sum(triples)) ** 2 / denominator if denominator else 0
            expected[k1, k2] = (sum(triples) / len(starts), coherence)
    return expected, len(starts), len(x) - starts[-1] - n


@pytest.mark.parametrize("window", ["boxcar", "hann"])
@pytest.mark.parametrize("detrend", [False, True])
@pytest.mark.parametrize("scaling", ["raw", "length"])
@pytest.mark.parametrize("include_dc", [False, True])
def test_direct_estimator_matches_explicit_dft(window, detrend, scaling, include_dc):
    config = PaperConfig(block_size=16, overlap=8, window=window,
                         detrend=detrend, fft_scaling=scaling, include_dc=include_dc)
    x = np.random.default_rng(20261026).normal(size=53) + 3.0
    expected, blocks, discarded = _explicit_reference(x, config)
    actual = direct_bispectrum(x, config)
    assert set(zip(actual.k1, actual.k2)) == set(expected)
    assert actual.block_count == blocks
    assert actual.discarded_samples == discarded
    for k1, k2, value, coherence in zip(actual.k1, actual.k2, actual.values,
                                        actual.bicoherence_squared):
        expected_value, expected_coherence = expected[k1, k2]
        assert value == pytest.approx(expected_value, rel=2e-12, abs=2e-11)
        # Removing the mean leaves DC at numerical zero. Its coherence is
        # undefined analytically, so do not compare ratios of round-off errors.
        if not (detrend and window == "boxcar" and k2 == 0):
            assert coherence == pytest.approx(expected_coherence, rel=2e-11, abs=2e-11)


@pytest.mark.parametrize("log_base", ["e", "10"])
@pytest.mark.parametrize("index_origin", [0, 1])
def test_positive_amplitude_scaling_has_analytic_feature_effect(log_base, index_origin):
    config = PaperConfig(block_size=32, overlap=0, window="boxcar", detrend=False,
                         log_base=log_base, index_origin=index_origin)
    x = np.random.default_rng(125).normal(size=32 * 8) + 0.8
    factor = 3.7
    original = direct_bispectrum(x, config)
    scaled = direct_bispectrum(factor * x, config)
    assert np.min(np.abs(original.values)) > config.log_floor
    np.testing.assert_allclose(scaled.values, original.values * factor ** 3,
                               rtol=3e-12, atol=1e-10)
    np.testing.assert_allclose(scaled.bicoherence_squared,
                               original.bicoherence_squared, rtol=3e-12, atol=1e-12)
    before = features_from_bispectrum(original, config)
    after = features_from_bispectrum(scaled, config)
    for name in ("P1", "P2", "f1m", "f2m"):
        assert after[name] == pytest.approx(before[name], abs=2e-12)
    shift = 3 * (np.log(factor) if log_base == "e" else np.log10(factor))
    diagonal = original.k1 == original.k2
    assert after["H1"] - before["H1"] == pytest.approx(len(original.values) * shift)
    assert after["H2"] - before["H2"] == pytest.approx(diagonal.sum() * shift)
    diagonal_weight = (original.k1[diagonal] + index_origin).sum()
    assert after["H3"] - before["H3"] == pytest.approx(diagonal_weight * shift)


def test_bicoherence_identifies_quadratic_phase_coupling():
    n, blocks = 128, 128
    config = PaperConfig(block_size=n, overlap=0, window="boxcar",
                         detrend=False, include_dc=False)
    rng = np.random.default_rng(431)
    phase1 = rng.uniform(-np.pi, np.pi, blocks)
    phase2 = rng.uniform(-np.pi, np.pi, blocks)
    independent_phase = rng.uniform(-np.pi, np.pi, blocks)
    samples = np.arange(n)[None, :]
    first = np.cos(2 * np.pi * 11 * samples / n + phase1[:, None])
    second = np.cos(2 * np.pi * 7 * samples / n + phase2[:, None])
    coupled_sum = np.cos(2 * np.pi * 18 * samples / n
                         + (phase1 + phase2)[:, None])
    independent_sum = np.cos(2 * np.pi * 18 * samples / n
                             + independent_phase[:, None])
    coupled = direct_bispectrum((first + second + coupled_sum).ravel(), config)
    independent = direct_bispectrum((first + second + independent_sum).ravel(), config)
    selected = (coupled.k1 == 11) & (coupled.k2 == 7)
    assert selected.sum() == 1
    assert coupled.bicoherence_squared[selected][0] == pytest.approx(1, abs=1e-12)
    assert independent.bicoherence_squared[selected][0] < 0.05


@pytest.mark.parametrize("include_dc", [False, True])
def test_triangle_boundary_and_diagonal_coordinates(include_dc):
    config = PaperConfig(block_size=16, overlap=0, include_dc=include_dc,
                         index_origin=0)
    shape = direct_bispectrum(np.arange(16.0), config)
    expected = {(i, j) for i in range(9) for j in range(i + 1)
                if i + j <= 8 and (include_dc or j > 0)}
    assert set(zip(shape.k1, shape.k2)) == expected
    assert (4, 4) in expected  # diagonal endpoint at one quarter of the DFT
    assert (5, 5) not in expected
    assert ((0, 0) in expected) == include_dc
    values = np.exp(0.1 * (shape.k1 + 2 * shape.k2 + 1)).astype(complex)
    result = replace(shape, values=values)
    zero_origin = features_from_bispectrum(result, config)
    one_origin = features_from_bispectrum(result, replace(config, index_origin=1))
    diagonal_bins = np.arange(0 if include_dc else 1, 5)
    diagonal_logs = 0.1 * (3 * diagonal_bins + 1)
    assert zero_origin["H2"] == pytest.approx(diagonal_logs.sum())
    assert zero_origin["H3"] == pytest.approx(np.dot(diagonal_bins, diagonal_logs))
    assert one_origin["H3"] - zero_origin["H3"] == pytest.approx(zero_origin["H2"])
    for name in ("f1m", "f2m"):
        assert one_origin[name] - zero_origin[name] == pytest.approx(1)
    matrix = result.matrix()
    for i in range(9):
        for j in range(9):
            if (i, j) in expected:
                index = np.flatnonzero((result.k1 == i) & (result.k2 == j))[0]
                assert matrix[j, i] == pytest.approx(abs(values[index]))
            else:
                assert np.isnan(matrix[j, i])


def test_zero_series_has_zero_bispectrum_but_no_defined_features():
    config = PaperConfig(block_size=16, overlap=0)
    result = direct_bispectrum(np.zeros(48), config)
    np.testing.assert_array_equal(result.values, 0)
    np.testing.assert_array_equal(result.bicoherence_squared, 0)
    with pytest.raises(ValueError, match="undefined"):
        features_from_bispectrum(result, config)


@pytest.mark.parametrize("bad", [np.full(16, np.nan), np.full(16, np.inf),
                                  np.zeros((4, 4)), np.zeros(15), np.array([])])
def test_invalid_series_rejected(bad):
    with pytest.raises(ValueError):
        direct_bispectrum(bad, PaperConfig(block_size=16, overlap=0))


@pytest.mark.parametrize("values", [np.array([1, np.nan], complex),
                                     np.array([1, np.inf], complex),
                                     np.zeros(2, complex)])
def test_nonfinite_or_zero_feature_inputs_rejected(values):
    result = BispectrumResult(values, np.array([0, 1]), np.array([0, 1]),
                               np.zeros(2), 1, 16, 4.0, 0)
    with pytest.raises(ValueError, match="undefined"):
        features_from_bispectrum(result, PaperConfig(block_size=16, overlap=0))
