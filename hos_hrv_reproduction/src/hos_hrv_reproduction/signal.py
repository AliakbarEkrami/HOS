"""ECG cleaning, an offline Pan-Tompkins-style detector, and HR resampling.

This is NOT a byte-for-byte port of the 1985 integer/real-time detector. It uses
zero-phase filters, derivative energy, moving integration, adaptive thresholds,
search-back, and local QRS localization. Detector output must be checked for each
new cohort. Curated R-peak timestamps are also accepted by the pipeline.
"""
import numpy as np
from scipy import signal
from scipy.ndimage import median_filter
from scipy.interpolate import PchipInterpolator
from .config import PaperConfig


def finite_vector(values, name="signal", min_size=3):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < min_size or not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must be a finite 1-D array of at least {min_size} values")
    return x


def preprocess_ecg(ecg, fs: float, mains_hz=50.0, config: PaperConfig | None = None):
    c = config or PaperConfig()
    x = finite_vector(ecg)
    if not np.isfinite(fs) or fs <= 2 * c.lowpass_hz:
        raise ValueError("ECG sampling rate must exceed twice the low-pass cutoff")
    for cutoff, kind in [(c.lowpass_hz, "lowpass"), (c.highpass_hz, "highpass")]:
        sos = signal.butter(c.filter_order, cutoff, btype=kind, fs=fs, output="sos")
        x = signal.sosfiltfilt(sos, x)
    if mains_hz:
        if not 0 < mains_hz < fs / 2:
            raise ValueError("mains_hz must be below Nyquist, or 0 to disable")
        b, a = signal.iirnotch(mains_hz, c.notch_q, fs=fs)
        x = signal.filtfilt(b, a, x)
    if c.median_seconds:
        width = int(round(c.median_seconds * fs)) | 1
        x = x - median_filter(x, size=width, mode="reflect")
    return x


def resample_ecg(ecg, original_fs: float, target_fs: float):
    """Band-limited rational resampling for external datasets.

    The paper's recordings were digitized at 320 Hz. Public comparators use
    other rates, so this conversion can be applied before the paper's filters.
    """
    x = finite_vector(ecg)
    if not np.isfinite(original_fs) or not np.isfinite(target_fs):
        raise ValueError("Sampling rates must be finite")
    if original_fs <= 0 or target_fs <= 0:
        raise ValueError("Sampling rates must be positive")
    if np.isclose(original_fs, target_fs):
        return x.copy()
    from fractions import Fraction
    ratio = Fraction(float(target_fs) / float(original_fs)).limit_denominator(10000)
    return signal.resample_poly(x, ratio.numerator, ratio.denominator)


def detect_rpeaks(ecg, fs: float, config: PaperConfig | None = None):
    c = config or PaperConfig()
    x = finite_vector(ecg)
    if fs <= 30 or not np.isfinite(fs):
        raise ValueError("Detector requires fs > 30 Hz")
    band = signal.sosfiltfilt(signal.butter(2, [5, 15], fs=fs,
                              btype="bandpass", output="sos"), x)
    derivative = np.convolve(band, np.array([1, 2, 0, -2, -1]) * fs / 8, mode="same")
    width = max(1, int(round(0.15 * fs)))
    integrated = np.convolve(derivative ** 2, np.ones(width) / width, mode="same")
    candidates, _ = signal.find_peaks(integrated, distance=max(1, int(0.08 * fs)))
    if not candidates.size or integrated.max() == 0:
        raise ValueError("No QRS candidates detected")
    initial = integrated[:int(2 * fs)]
    signal_level, noise_level = 0.25 * initial.max(), 0.5 * initial.mean()
    accepted = []
    refractory = max(1, int(c.refractory_seconds * fs))
    for peak in candidates:
        threshold = noise_level + 0.25 * (signal_level - noise_level)
        # Look for a missed low-amplitude beat when the RR interval is unusually long.
        if len(accepted) >= 3:
            rr = np.diff(accepted[-9:])
            if peak - accepted[-1] > 1.66 * np.median(rr):
                eligible = candidates[(candidates > accepted[-1] + refractory)
                                      & (candidates < peak - refractory)]
                if eligible.size:
                    best = eligible[np.argmax(integrated[eligible])]
                    if integrated[best] > threshold * 0.5:
                        accepted.append(int(best))
        far_enough = not accepted or peak - accepted[-1] >= refractory
        is_twave = False
        if accepted and 0.2 * fs < peak - accepted[-1] < 0.36 * fs:
            radius = max(1, int(0.075 * fs))
            current_slope = np.max(np.abs(derivative[max(0, peak-radius):peak+radius+1]))
            last = accepted[-1]
            previous_slope = np.max(np.abs(derivative[max(0, last-radius):last+radius+1]))
            is_twave = current_slope < 0.5 * previous_slope
        if far_enough and not is_twave and integrated[peak] > threshold:
            accepted.append(int(peak))
            signal_level = 0.125 * integrated[peak] + 0.875 * signal_level
        else:
            noise_level = 0.125 * integrated[peak] + 0.875 * noise_level
    # Centered offline filters have no fixed delay. Find the largest local deflection.
    radius = max(1, int(0.1 * fs))
    localized = []
    for peak in accepted:
        left, right = max(0, peak-radius), min(len(x), peak+radius+1)
        candidate = left + int(np.argmax(np.abs(x[left:right])))
        if not localized or candidate - localized[-1] >= refractory:
            localized.append(candidate)
        elif abs(x[candidate]) > abs(x[localized[-1]]):
            localized[-1] = candidate
    if len(localized) < 3:
        raise ValueError("Fewer than three QRS complexes detected")
    return np.asarray(localized, dtype=int)


def rpeaks_to_hrv(rpeaks_seconds, config: PaperConfig | None = None):
    c = config or PaperConfig()
    peaks = finite_vector(rpeaks_seconds, "R-peak times")
    rr = np.diff(peaks)
    if np.any(rr <= 0):
        raise ValueError("R-peak times must be strictly increasing")
    # Attach each interval to its ending beat. No extrapolation or automatic
    # ectopic removal: PVC/pauses may be the disease information of interest.
    times, rate = peaks[1:], 60.0 / rr
    grid = np.arange(times[0], times[-1], 1 / c.hrv_fs)
    if c.interpolation == "pchip":
        hrv = PchipInterpolator(times, rate, extrapolate=False)(grid)
    else:
        hrv = np.interp(grid, times, rate)
    return grid, hrv


def rr_to_hrv(rr_seconds, config: PaperConfig | None = None):
    rr = finite_vector(rr_seconds, "RR intervals")
    if np.any(rr <= 0):
        raise ValueError("RR intervals must be positive seconds")
    return rpeaks_to_hrv(np.r_[0, np.cumsum(rr)], config)
