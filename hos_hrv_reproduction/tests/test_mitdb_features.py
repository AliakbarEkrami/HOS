from types import SimpleNamespace

import numpy as np
import pytest

from hos_hrv_reproduction.bispectrum import FEATURE_NAMES
from hos_hrv_reproduction.config import PaperConfig
from hos_hrv_reproduction.mitdb_features import (
    ALL_FEATURES, BASE_FEATURES, annotated_peak_seconds, baseline_features,
    match_qrs, segment_hrv,
)
from hos_hrv_reproduction.signal import resample_ecg


def test_contextual_rr_uses_preceding_beat_without_extrapolation():
    peaks = np.array([0.3, 1.1, 2.0, 3.1, 4.0, 5.2])
    grid, hrv, rr = segment_hrv(peaks, 2, 5)
    assert np.array_equal(grid, np.arange(2, 5, 0.25))
    assert np.allclose(rr, [0.9, 1.1, 0.9])
    expected = np.interp(grid, peaks[1:], 60 / np.diff(peaks))
    assert np.allclose(hrv, expected)
    grid, _, _ = segment_hrv(peaks, 0, 10)
    assert grid[0] == 1.25
    assert grid[-1] < peaks[-1]
    assert np.all(grid * 4 == np.round(grid * 4))


@pytest.mark.parametrize("peaks", [[0, 1, 1, 2], [0, 2, 1, 3], [0, np.nan, 2, 3]])
def test_invalid_peak_sequences_are_rejected(peaks):
    with pytest.raises(ValueError):
        segment_hrv(peaks, 0, 3)


def test_native_and_resampled_indices_represent_the_same_seconds():
    native = SimpleNamespace(sample=[0, 360, 720, 900, 1080], symbol=["+", "N", "V", "~", "L"])
    assert np.array_equal(annotated_peak_seconds(native, 360), [1, 2, 3])
    resampled = np.array([320, 640, 960]) / 320
    assert np.array_equal(annotated_peak_seconds(native, 360), resampled)
    x = np.zeros(4 * 360)
    x[2 * 360] = 1
    y = resample_ecg(x, 360, 320)
    assert len(y) == 4 * 320
    assert np.argmax(y) / 320 == 2


def test_qrs_matching_is_one_to_one_and_includes_150ms_boundary():
    result = match_qrs([1, 2, 3], [0.85, 1.05, 2.15, 4])
    assert result["tp"] == 2
    assert result["fp"] == 2
    assert result["fn"] == 1
    assert result["sensitivity"] == pytest.approx(2 / 3)
    assert result["ppv"] == 0.5
    assert result["timing_mae_ms"] == pytest.approx(150)
    assert match_qrs([1], [1.15001])["tp"] == 0
    assert match_qrs([], [1])["sensitivity"] is None
    assert match_qrs([1], [])["fn"] == 1
    with pytest.raises(ValueError):
        match_qrs([1, 1], [1])


def test_baseline_names_and_statistics_preserve_ectopic_intervals():
    c = PaperConfig(block_size=64, overlap=32)
    rr = np.array([0.8, 0.8, 0.4, 1.2, 0.8])
    t = np.arange(512) / c.hrv_fs
    hrv = 75 + 4 * np.sin(2 * np.pi * 0.1 * t) + np.sin(2 * np.pi * 0.3 * t)
    features = baseline_features(rr, hrv, c)
    assert tuple(features) == BASE_FEATURES
    assert features["mean_rr"] == pytest.approx(0.8)
    assert features["sd_rr"] == pytest.approx(rr.std(ddof=1))
    assert features["rmssd_rr"] == pytest.approx(np.sqrt(np.mean(np.diff(rr) ** 2)))
    assert features["prr50"] == 0.75
    assert np.isfinite(list(features.values())).all()
    assert 0 <= features["psd_entropy"] <= 1
    assert ALL_FEATURES == (*FEATURE_NAMES, *BASE_FEATURES)
    assert len(ALL_FEATURES) == len(set(ALL_FEATURES)) == 15
    assert not {"pvc_fraction", "n_v", "n_beats", "y", "label", "subject_id"} & set(ALL_FEATURES)


def test_zero_power_baseline_and_nonpositive_rr_fail_explicitly():
    c = PaperConfig(block_size=64, overlap=32)
    with pytest.raises(ValueError, match="PSD entropy"):
        baseline_features([1, 1, 1], np.ones(128), c)
    with pytest.raises(ValueError, match="positive"):
        baseline_features([1, 0, 1], np.arange(128), c)
