import numpy as np
import pytest

from hos_hrv_reproduction.bispectrum import direct_bispectrum, features_from_bispectrum
from hos_hrv_reproduction.config import PaperConfig


def test_triangle_and_features_are_finite():
    rng = np.random.default_rng(7)
    c = PaperConfig(block_size=64, overlap=32)
    x = rng.normal(size=64 * 3)
    result = direct_bispectrum(x, c)
    assert np.all(result.k2 <= result.k1)
    assert np.all(result.k1 + result.k2 <= c.block_size // 2)
    assert np.all(result.bicoherence_squared >= 0)
    assert np.all(result.bicoherence_squared <= 1 + 1e-12)
    features = features_from_bispectrum(result, c)
    assert set(features) == {"P1", "P2", "H1", "H2", "H3", "f1m", "f2m"}
    assert all(np.isfinite(list(features.values())))
    assert 0 <= features["P1"] <= 1
    assert 0 <= features["P2"] <= 1


def test_short_hrv_is_rejected():
    with pytest.raises(ValueError):
        direct_bispectrum(np.ones(20), PaperConfig(block_size=64, overlap=32))
