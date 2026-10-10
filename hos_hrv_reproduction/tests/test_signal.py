import numpy as np

from hos_hrv_reproduction.config import PaperConfig
from hos_hrv_reproduction.signal import resample_ecg, rr_to_hrv


def test_constant_rr_interpolates_to_constant_hr():
    times, hrv = rr_to_hrv(np.full(300, 0.8), PaperConfig())
    assert len(times) == len(hrv)
    assert len(hrv) > 100
    assert np.allclose(hrv, 75.0)


def test_resample_ecg_to_paper_rate():
    x = np.sin(2 * np.pi * 1.0 * np.arange(3600) / 360.0)
    y = resample_ecg(x, 360.0, 320.0)
    assert abs(len(y) - 3200) <= 1
    expected = np.sin(2 * np.pi * np.arange(len(y)) / 320.0)
    assert np.max(np.abs(y[32:-32] - expected[32:-32])) < 0.005
