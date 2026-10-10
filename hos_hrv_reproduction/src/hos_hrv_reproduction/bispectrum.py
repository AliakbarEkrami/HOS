"""Direct ensemble bispectrum and the seven paper features, from equations.

Each result is for ONE recording: complex products are averaged over overlapping
blocks before taking magnitudes. Window-wise samples never become SVM subjects.
The retained triangle follows Figure 1 (no aliased sum frequencies).
"""
from dataclasses import dataclass
import numpy as np
from scipy.signal import get_window
from .config import PaperConfig

FEATURE_NAMES = ("P1", "P2", "H1", "H2", "H3", "f1m", "f2m")


@dataclass
class BispectrumResult:
    values: np.ndarray
    k1: np.ndarray
    k2: np.ndarray
    bicoherence_squared: np.ndarray
    block_count: int
    nfft: int
    fs: float
    discarded_samples: int

    def matrix(self, quantity="magnitude"):
        vals = {"magnitude": np.abs(self.values),
                "bicoherence_squared": self.bicoherence_squared}[quantity]
        out = np.full((self.nfft // 2 + 1,) * 2, np.nan)
        out[self.k2, self.k1] = vals  # horizontal k1, vertical k2
        return out


def direct_bispectrum(signal, config: PaperConfig | None = None):
    c = config or PaperConfig()
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("HRV must be a finite 1-D array")
    if len(x) < c.block_size:
        raise ValueError(f"Need {c.block_size} HRV samples ({c.block_size/c.hrv_fs:g} s); got {len(x)}")
    starts = np.arange(0, len(x) - c.block_size + 1, c.block_size - c.overlap)
    # Compact storage: only the triangle from Figure 1.
    a, b = np.indices((c.block_size // 2 + 1,) * 2)
    mask = (b <= a) & (a + b <= c.block_size // 2)
    if not c.include_dc:
        mask &= b > 0
    k1, k2 = a[mask], b[mask]
    total = np.zeros(k1.size, dtype=complex)
    den1 = np.zeros(k1.size)
    den2 = np.zeros(k1.size)
    window = get_window(c.window, c.block_size, fftbins=True)
    for start in starts:
        block = x[start:start + c.block_size].copy()
        if c.detrend:
            block -= block.mean()
        fft = np.fft.rfft(block * window)
        if c.fft_scaling == "length":
            fft /= c.block_size
        product = fft[k1] * fft[k2]
        summed = fft[k1 + k2]
        total += product * summed.conj()
        den1 += np.abs(product) ** 2
        den2 += np.abs(summed) ** 2
    denominator = den1 * den2
    bicoh = np.divide(np.abs(total) ** 2, denominator,
                       out=np.zeros_like(denominator), where=denominator > 0)
    bicoh = np.clip(bicoh, 0, 1)  # clip roundoff only (Cauchy-Schwarz bound)
    return BispectrumResult(total / len(starts), k1, k2, bicoh,
                            len(starts), c.block_size, c.hrv_fs,
                            int(len(x) - starts[-1] - c.block_size))


def features_from_bispectrum(result: BispectrumResult, config: PaperConfig | None = None):
    c = config or PaperConfig()
    amplitude = np.abs(result.values)
    if amplitude.size < 2 or not np.all(np.isfinite(amplitude)) or amplitude.max() == 0:
        raise ValueError("Bispectrum is empty, nonfinite, or zero; features are undefined")
    # Rescale for probability calculations only, to avoid squaring huge values.
    weights = amplitude / amplitude.max()

    def entropy(power):
        p = weights ** power
        p /= p.sum()
        positive = p > 0
        h = -np.sum(p[positive] * np.log(p[positive]))
        if c.normalize_entropy:
            h /= np.log(len(p))
        elif c.log_base == "10":
            h /= np.log(10)
        return float(h)

    logs = np.log(np.maximum(amplitude, c.log_floor))
    if c.log_base == "10":
        logs /= np.log(10)
    diagonal = result.k1 == result.k2
    if not diagonal.any():
        raise ValueError("No diagonal bins in the selected bispectral region")
    # Coordinates in bins, not Hz. The origin convention is explicitly exposed.
    i, j = result.k1 + c.index_origin, result.k2 + c.index_origin
    values = (entropy(1), entropy(2), logs.sum(), logs[diagonal].sum(),
              np.dot(i[diagonal], logs[diagonal]),
              np.dot(i, weights) / weights.sum(), np.dot(j, weights) / weights.sum())
    return dict(zip(FEATURE_NAMES, map(float, values)))
