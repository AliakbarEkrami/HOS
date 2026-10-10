"""Record-to-feature orchestration."""
from pathlib import Path
import csv
import numpy as np
from .bispectrum import direct_bispectrum, features_from_bispectrum
from .config import PaperConfig
from .io import load_array, Record
from .signal import preprocess_ecg, detect_rpeaks, rpeaks_to_hrv, resample_ecg


def process_record(record: Record, config: PaperConfig | None = None,
                   peaks_path: str | Path | None = None,
                   peaks_units: str = "seconds"):
    c = config or PaperConfig()
    if record.kind == "rr":
        rr = load_array(record.path, record.channel)
        times, hrv = rpeaks_to_hrv(np.r_[0, np.cumsum(rr)], c)
        detected = len(rr) + 1
    elif record.kind == "hrv":
        hrv = load_array(record.path, record.channel)
        times = np.arange(len(hrv)) / c.hrv_fs
        detected = None
    else:
        ecg = load_array(record.path, record.channel)
        ecg_fs = record.fs
        if c.ecg_target_fs is not None:
            ecg = resample_ecg(ecg, record.fs, c.ecg_target_fs)
            ecg_fs = c.ecg_target_fs
        ecg = preprocess_ecg(ecg, ecg_fs, record.mains_hz, c)
        if peaks_path:
            peaks = load_array(peaks_path)
            if peaks_units not in ("seconds", "samples"):
                raise ValueError("peaks_units must be seconds or samples")
            # Curated annotation samples refer to the original recording rate,
            # regardless of the optional ECG resampling step above.
            peaks_seconds = peaks / record.fs if peaks_units == "samples" else peaks
        else:
            peaks = detect_rpeaks(ecg, ecg_fs, c)
            peaks_seconds = peaks / ecg_fs
        times, hrv = rpeaks_to_hrv(peaks_seconds, c)
        detected = len(peaks)
    result = direct_bispectrum(hrv, c)
    features = features_from_bispectrum(result, c)
    return {"record_id": record.record_id, "subject_id": record.subject_id,
            "label": record.label, "path": str(record.path),
            "n_rpeaks": detected, "n_hrv": len(hrv),
            "n_blocks": result.block_count, **features}


def extract_manifest(records, output: str | Path, config=None):
    c = config or PaperConfig()
    rows = []
    for i, record in enumerate(records, 1):
        try:
            rows.append(process_record(record, c))
        except Exception as exc:
            raise RuntimeError(f"Failed on {record.record_id} ({record.path}): {exc}") from exc
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["record_id", "subject_id", "label", "path", "n_rpeaks", "n_hrv", "n_blocks",
              "P1", "P2", "H1", "H2", "H3", "f1m", "f2m"]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    return rows
