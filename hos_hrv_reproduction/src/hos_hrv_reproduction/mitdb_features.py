"""Paired reference/detector extraction without using annotation labels in X."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.integrate import trapezoid
import wfdb

from .bispectrum import FEATURE_NAMES, direct_bispectrum, features_from_bispectrum
from .config import PaperConfig
from .mitdb_common import provenance, read_csv, sha256_file, write_csv, write_json
from .mitdb_protocol import BEAT_SYMBOLS, MitdbProtocol
from .signal import detect_rpeaks, finite_vector, preprocess_ecg, resample_ecg

BASE_FEATURES = ("mean_rr", "sd_rr", "rmssd_rr", "prr50", "log_vlf_power",
                 "log_lf_power", "log_hf_power", "psd_entropy")
ALL_FEATURES = (*FEATURE_NAMES, *BASE_FEATURES)
FEATURE_FIELDS = ("segment_id", "record_id", "subject_id", "label", "y", "source",
                  "status", "error", "n_rpeaks", "n_hrv", "n_blocks",
                  "hrv_start_s", "hrv_end_s", "discarded_hrv_samples", *ALL_FEATURES)
QRS_FIELDS = ("segment_id", "record_id", "subject_id", "status", "error", "tp", "fp",
              "fn", "sensitivity", "ppv", "f1", "timing_mae_ms", "timing_median_ms",
              "tolerance_ms")


def segment_hrv(peak_seconds, start_s, end_s, config=None):
    """Use contextual beats, global quarter-second alignment, no extrapolation.

    The RR series includes real ectopic intervals. Only intervals whose ending
    beat lies within [start_s, end_s) contribute to the RR baseline statistics.
    """
    c = config or PaperConfig()
    peaks = finite_vector(peak_seconds, "R-peak times")
    intervals = np.diff(peaks)
    if np.any(intervals <= 0):
        raise ValueError("R-peak times must be strictly increasing")
    if not end_s > start_s:
        raise ValueError("Segment end must follow start")
    times, rates = peaks[1:], 60.0 / intervals
    left, right = max(start_s, times[0]), min(end_s, times[-1])
    grid = np.arange(np.ceil(left * c.hrv_fs), np.ceil(right * c.hrv_fs)) / c.hrv_fs
    grid = grid[(grid >= times[0]) & (grid <= times[-1]) & (grid < end_s)]
    if c.interpolation == "pchip":
        from scipy.interpolate import PchipInterpolator
        hrv = PchipInterpolator(times, rates, extrapolate=False)(grid)
    else:
        hrv = np.interp(grid, times, rates)
    rr = intervals[(times >= start_s) & (times < end_s)]
    return grid, hrv, finite_vector(rr, "segment RR intervals")


def baseline_features(rr, hrv, config=None):
    c = config or PaperConfig()
    rr = finite_vector(rr, "RR intervals")
    hrv = finite_vector(hrv, "HR series", c.block_size)
    if np.any(rr <= 0):
        raise ValueError("RR intervals must be positive")
    differences = np.diff(rr)
    frequencies, psd = signal.welch(hrv, fs=c.hrv_fs, window=c.window,
                                    nperseg=c.block_size, noverlap=c.overlap,
                                    detrend="constant", scaling="density")
    bands = ((0.0033, 0.04), (0.04, 0.15), (0.15, 0.4))
    powers = []
    for low, high in bands:
        selected = (frequencies >= low) & (frequencies <= high)
        power = trapezoid(psd[selected], frequencies[selected])
        powers.append(float(np.log(max(float(power), c.log_floor))))
    selected = (frequencies >= 0.0033) & (frequencies <= 0.4)
    probabilities = psd[selected]
    if len(probabilities) < 2 or probabilities.sum() <= 0:
        raise ValueError("PSD entropy undefined: zero power or insufficient frequency bins")
    probabilities = probabilities / probabilities.sum()
    positive = probabilities > 0
    entropy = -np.sum(probabilities[positive] * np.log(probabilities[positive]))
    entropy /= np.log(len(probabilities))
    values = (rr.mean(), rr.std(ddof=1), np.sqrt(np.mean(differences ** 2)),
              np.mean(np.abs(differences) > 0.05), *powers, entropy)
    result = dict(zip(BASE_FEATURES, map(float, values)))
    if not np.all(np.isfinite(list(result.values()))):
        raise ValueError("Nonfinite baseline feature")
    return result


def match_qrs(reference, detected, tolerance=0.15):
    """Chronological one-to-one matching; no reference beat is reused."""
    reference, detected = np.asarray(reference, float), np.asarray(detected, float)
    if tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError("QRS tolerance must be positive and finite")
    for x in (reference, detected):
        if x.ndim != 1 or not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
            raise ValueError("QRS times must be finite and strictly increasing")
    i, j, errors = 0, 0, []
    while i < len(reference) and j < len(detected):
        delta = detected[j] - reference[i]
        if abs(delta) <= tolerance + 1e-12:
            errors.append(abs(float(delta)))
            i += 1
            j += 1
        elif delta < -tolerance:
            j += 1
        else:
            i += 1
    tp, fp, fn = len(errors), len(detected) - len(errors), len(reference) - len(errors)
    return {"tp": tp, "fp": fp, "fn": fn,
            "sensitivity": tp / (tp + fn) if tp + fn else None,
            "ppv": tp / (tp + fp) if tp + fp else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "timing_mae_ms": float(np.mean(errors) * 1000) if errors else None,
            "timing_median_ms": float(np.median(errors) * 1000) if errors else None,
            "tolerance_ms": tolerance * 1000}


def annotated_peak_seconds(annotation, fs):
    """Exclude rhythm changes, quality markers and ventricular flutter waves."""
    return np.asarray([sample / fs for sample, symbol in zip(annotation.sample, annotation.symbol)
                       if symbol in BEAT_SYMBOLS], dtype=float)


def extract_mitdb(run_dir, config=None, peak_source="both", save_examples=True):
    if peak_source not in ("both", "reference", "detector"):
        raise ValueError("peak_source must be both, reference, or detector")
    c, run_dir = config or PaperConfig(), Path(run_dir).resolve()
    if (run_dir / "evaluation.json").exists():
        raise ValueError("Run already evaluated; choose a new run directory")
    if (run_dir / "extraction.json").exists():
        raise ValueError("Run already extracted; prepare a fresh run directory")
    p = MitdbProtocol.from_json(run_dir / "protocol.json")
    rows = read_csv(run_dir / "manifest.csv")
    prepared_path = run_dir / "provenance.json"
    if prepared_path.exists():
        prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
        for key, name in (("protocol_sha256", "protocol.json"), ("manifest_sha256", "manifest.csv")):
            if prepared.get(key) and prepared[key] != sha256_file(run_dir / name):
                raise ValueError("Prepared input changed: " + name)
    # Recheck actual files rather than trusting an old list of hashes.
    recorded = {item["file"]: item for item in json.loads(
        (run_dir / "source_files.json").read_text(encoding="utf-8"))}
    for record in sorted({r["record_id"] for r in rows}):
        base = Path(next(r["path"] for r in rows if r["record_id"] == record))
        for extension in ("hea", "dat", "atr"):
            file = base.with_suffix("." + extension)
            if file.name not in recorded or sha256_file(file) != recorded[file.name]["sha256"]:
                raise ValueError("Source file changed since preparation: " + file.name)
    config_path = run_dir / "paper_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != c.to_dict():
        raise ValueError("Extraction configuration differs from frozen run settings")
    write_json(config_path, c.to_dict())
    sources = ("reference", "detector") if peak_source == "both" else (peak_source,)
    outputs, qrs_rows, examples_done = {s: [] for s in sources}, [], set()
    records = sorted({r["record_id"] for r in rows})
    for number, record in enumerate(records, 1):
        record_rows = [r for r in rows if r["record_id"] == record]
        first = record_rows[0]
        fs, channel = float(first["fs"]), int(first["channel"])
        ref, det, ecg = None, None, None
        source_errors = {}
        try:
            header = wfdb.rdheader(first["path"])
            if header.fs != fs or header.sig_name[channel] != first["channel_name"]:
                raise ValueError("Manifest/header sampling rate or channel mismatch")
            annotation = wfdb.rdann(first["path"], "atr")
            ref = annotated_peak_seconds(annotation, fs)
        except Exception as exc:
            source_errors["reference"] = f"reference_read_error:{exc}"
        if "detector" in sources or save_examples:
            try:
                samples, fields = wfdb.rdsamp(first["path"], channels=[channel])
                if fields["fs"] != fs:
                    raise ValueError("Signal and manifest sampling rates differ")
                ecg = samples[:, 0]
                if "detector" in sources:
                    detector_fs = c.ecg_target_fs or fs
                    cleaned = resample_ecg(ecg, fs, detector_fs)
                    cleaned = preprocess_ecg(cleaned, detector_fs, 60.0, c)
                    det = detect_rpeaks(cleaned, detector_fs, c) / detector_fs
            except Exception as exc:
                source_errors["detector"] = f"detector_error:{exc}"
        for segment in record_rows:
            start, end = float(segment["start_s"]), float(segment["end_s"])
            example_result = None
            for source in sources:
                row = dict.fromkeys(FEATURE_FIELDS, "")
                row.update({key: segment[key] for key in ("segment_id", "record_id", "subject_id",
                                                         "label", "y")})
                row.update(source=source, status="failed")
                try:
                    if source in source_errors:
                        raise ValueError(source_errors[source])
                    peaks = ref if source == "reference" else det
                    grid, hrv, rr = segment_hrv(peaks, start, end, c)
                    spectrum = direct_bispectrum(hrv, c)
                    row.update(n_rpeaks=int(np.sum((peaks >= start) & (peaks < end))),
                               n_hrv=len(hrv), n_blocks=spectrum.block_count,
                               hrv_start_s=float(grid[0]), hrv_end_s=float(grid[-1]),
                               discarded_hrv_samples=spectrum.discarded_samples)
                    if spectrum.block_count < p.min_blocks:
                        raise ValueError(f"insufficient_blocks:{spectrum.block_count}<{p.min_blocks}")
                    row.update(features_from_bispectrum(spectrum, c),
                               **baseline_features(rr, hrv, c), status="ok")
                    if not np.all(np.isfinite([row[name] for name in ALL_FEATURES])):
                        raise ValueError("nonfinite_features")
                    if source == "reference":
                        example_result = (grid, hrv, spectrum)
                except Exception as exc:
                    row.update(status="failed", error=str(exc))
                    for name in ALL_FEATURES:
                        row[name] = ""
                outputs[source].append(row)
            if "detector" in sources:
                qrs = dict.fromkeys(QRS_FIELDS, "")
                qrs.update({key: segment[key] for key in ("segment_id", "record_id", "subject_id")})
                qrs.update(status="failed", error="", tolerance_ms=p.qrs_tolerance_seconds * 1000)
                if ref is not None:
                    reference = ref[(ref >= start) & (ref < end)]
                    detected = det[(det >= start) & (det < end)] if det is not None else np.array([])
                    qrs.update(match_qrs(reference, detected, p.qrs_tolerance_seconds))
                    qrs["status"] = "ok" if det is not None else "failed"
                qrs["error"] = ";".join(source_errors.values())
                qrs_rows.append(qrs)
            if (save_examples and segment["label"] not in examples_done and ecg is not None
                    and example_result is not None):
                grid, hrv, spectrum = example_result
                trace_end = min(start + 12, end)
                trace = ecg[round(start * fs):round(trace_end * fs)]
                times = start + np.arange(len(trace)) / fs
                sample_peaks = lambda peaks: (peaks[(peaks >= start) & (peaks < trace_end)]
                                               if peaks is not None else np.array([]))
                example_dir = run_dir / "examples"
                example_dir.mkdir(exist_ok=True)
                np.savez_compressed(example_dir / (segment["segment_id"] + ".npz"),
                                    times_s=times, ecg=trace,
                                    reference_peaks_s=sample_peaks(ref),
                                    detector_peaks_s=sample_peaks(det),
                                    hrv_times_s=grid, hrv=hrv,
                                    bispectrum_magnitude=spectrum.matrix(),
                                    f_axis=np.fft.rfftfreq(spectrum.nfft, 1 / spectrum.fs),
                                    label=segment["label"], segment_id=segment["segment_id"])
                examples_done.add(segment["label"])
        print(f"extracted {number}/{len(records)} records ({record})", flush=True)
    for source in sources:
        write_csv(run_dir / f"features_{source}.csv", outputs[source], FEATURE_FIELDS)
    if qrs_rows:
        write_csv(run_dir / "qrs_metrics.csv", qrs_rows, QRS_FIELDS)
    summary = {"sources": {s: dict(Counter(r["status"] for r in output))
                           for s, output in outputs.items()},
               "protocol_id": p.protocol_id, "paper_config": c.to_dict(),
               "manifest_sha256": sha256_file(run_dir / "manifest.csv"),
               "protocol_sha256": sha256_file(run_dir / "protocol.json"),
               "paper_config_sha256": sha256_file(config_path),
               "source_files_sha256": sha256_file(run_dir / "source_files.json"),
               "feature_files": {s: sha256_file(run_dir / f"features_{s}.csv") for s in sources},
               "provenance": provenance()}
    write_json(run_dir / "extraction.json", summary)
    return summary
