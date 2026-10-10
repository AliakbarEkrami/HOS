"""Explicit paper settings and assumptions. See docs/paper_audit.md."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class PaperConfig:
    # Reported by the article.
    ecg_target_fs: float | None = 320.0
    hrv_fs: float = 4.0
    block_size: int = 1024
    overlap: int = 512
    lowpass_hz: float = 35.0
    highpass_hz: float = 0.3
    # Assumptions below; the article does not uniquely determine these.
    filter_order: int = 4
    notch_q: float = 30.0
    median_seconds: float = 0.6
    refractory_seconds: float = 0.2
    interpolation: str = "linear"
    window: str = "hann"
    detrend: bool = True
    fft_scaling: str = "raw"
    include_dc: bool = True
    normalize_entropy: bool = True
    log_base: str = "e"
    log_floor: float = 1e-12
    index_origin: int = 1

    def __post_init__(self):
        if self.ecg_target_fs is not None and self.ecg_target_fs <= 2 * self.lowpass_hz:
            raise ValueError("ecg_target_fs must exceed twice the low-pass cutoff")
        if self.hrv_fs <= 0 or self.block_size < 8 or self.block_size % 2:
            raise ValueError("Require positive HRV sampling rate and even block_size >= 8")
        if not 0 <= self.overlap < self.block_size:
            raise ValueError("overlap must be in [0, block_size)")
        if not 0 < self.highpass_hz < self.lowpass_hz:
            raise ValueError("Require 0 < highpass_hz < lowpass_hz")
        if self.filter_order < 1 or self.notch_q <= 0 or self.median_seconds < 0:
            raise ValueError("Invalid ECG filter settings")
        if self.refractory_seconds <= 0:
            raise ValueError("refractory_seconds must be positive")
        if self.interpolation not in ("linear", "pchip"):
            raise ValueError("interpolation must be linear or pchip")
        if self.window not in ("hann", "boxcar"):
            raise ValueError("window must be hann or boxcar")
        if self.fft_scaling not in ("raw", "length"):
            raise ValueError("fft_scaling must be raw or length")
        if self.log_base not in ("e", "10") or not 0 < self.log_floor < 1:
            raise ValueError("Invalid logarithm settings")
        if self.index_origin not in (0, 1):
            raise ValueError("index_origin must be 0 or 1")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path | None):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8"))) if path else cls()
