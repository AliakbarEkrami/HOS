"""Frozen, annotation-defined cohort for the LOW_PVC/HIGH_PVC experiment.

Labels are research burden bands, not patient diagnoses. Waveforms and reference
annotations are never selected according to downstream model performance.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields
from fractions import Fraction
from http.client import HTTPException
import json
from pathlib import Path
from urllib.request import Request, urlopen
import time
import re

import numpy as np
import wfdb

from .mitdb_common import provenance, sha256_file, write_csv, write_json

RECORDS = tuple("100 101 102 103 104 105 106 107 108 109 111 112 113 114 115 116 "
                "117 118 119 121 122 123 124 200 201 202 203 205 207 208 209 210 "
                "212 213 214 215 217 219 220 221 222 223 228 230 231 232 233 234".split())
BEAT_SYMBOLS = frozenset(("N", "L", "R", "A", "a", "J", "S", "V", "F",
                          "e", "j", "E", "/", "f", "Q"))
LABELS = ("LOW_PVC", "HIGH_PVC")
DATA_URL = "https://physionet.org/files/mitdb/1.0.0/"
SEGMENT_FIELDS = ("segment_id", "record_id", "subject_id", "path", "fs", "channel",
                  "channel_name", "start_sample", "end_sample", "start_s", "end_s",
                  "label", "y", "included", "exclusion_reasons", "n_beats", "n_v",
                  "pvc_fraction", "purity_fraction", "allowed_rhythm_fraction",
                  "beat_counts", "quality_flags", "flatline_seconds")


@dataclass(frozen=True)
class MitdbProtocol:
    protocol_id: str = "mitdb_pvc_v1"
    dataset_version: str = "1.0.0"
    segment_seconds: int = 900
    record_seconds: int = 1800
    expected_fs: float = 360.0
    low_pvc_max: float = 0.01
    high_pvc_min: float = 0.05
    min_beat_purity: float = 0.99
    min_rhythm_coverage: float = 0.95
    allowed_beats: tuple = ("N", "L", "R", "V")
    allowed_rhythms: tuple = ("(N", "(B", "(T", "(SBR")
    excluded_records: tuple = ("102", "104", "107", "217")
    max_flatline_seconds: float = 2.0
    min_blocks: int = 5
    min_subjects_per_class: int = 5
    qrs_tolerance_seconds: float = 0.15
    seed: int = 20261026

    def __post_init__(self):
        for name in ("allowed_beats", "allowed_rhythms", "excluded_records"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not re.fullmatch(r"mitdb_pvc_[a-z0-9_]+", self.protocol_id) or self.dataset_version != "1.0.0":
            raise ValueError("Unsupported experiment protocol or dataset version")
        if self.protocol_id == "mitdb_pvc_v1":
            modified = [field.name for field in fields(self)
                        if getattr(self, field.name) != field.default]
            if modified:
                raise ValueError("mitdb_pvc_v1 is frozen; use a different protocol_id for changes: "
                                 + ", ".join(modified))
        if not 0 <= self.low_pvc_max < self.high_pvc_min <= 1:
            raise ValueError("Require 0 <= low_pvc_max < high_pvc_min <= 1")
        if not 0 < self.min_beat_purity <= 1 or not 0 < self.min_rhythm_coverage <= 1:
            raise ValueError("Purity and rhythm coverage must be in (0, 1]")
        if self.segment_seconds <= 0 or self.record_seconds % self.segment_seconds:
            raise ValueError("Record duration must be divisible by segment duration")
        if self.expected_fs <= 0 or self.max_flatline_seconds < 0 or self.min_blocks < 2:
            raise ValueError("Invalid sampling, quality, or ensemble settings")
        if not set(self.allowed_beats) <= BEAT_SYMBOLS:
            raise ValueError("Unknown allowed beat symbol")
        if self.min_subjects_per_class < 3 or self.qrs_tolerance_seconds <= 0:
            raise ValueError("Invalid group minimum or QRS tolerance")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_json(cls, path=None):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8"))) if path else cls()


def subject_id(record_id):
    return "201" if str(record_id) == "202" else str(record_id)


def burden_label(n_v, n_beats, protocol=None):
    p = protocol or MitdbProtocol()
    if n_beats <= 0 or not 0 <= n_v <= n_beats:
        return "", None
    burden = Fraction(int(n_v), int(n_beats))
    if burden <= Fraction(str(p.low_pvc_max)):
        return LABELS[0], 0
    if burden >= Fraction(str(p.high_pvc_min)):
        return LABELS[1], 1
    return "", None


def annotation_statistics(annotation, start_sample, end_sample, protocol=None):
    p = protocol or MitdbProtocol()
    samples = np.asarray(annotation.sample)
    symbols = np.asarray(annotation.symbol)
    in_segment = (samples >= start_sample) & (samples < end_sample)
    counts = Counter(str(s) for s in symbols[in_segment] if s in BEAT_SYMBOLS)
    flags = Counter(str(s) for s in symbols[in_segment] if s not in BEAT_SYMBOLS)
    changes = sorted((int(sample), str(aux).strip("\x00 "))
                     for sample, aux in zip(samples, annotation.aux_note)
                     if str(aux).startswith("("))
    state, cursor, allowed_duration = "unknown", start_sample, 0
    for sample, rhythm in changes:
        if sample <= start_sample:
            state = rhythm
            continue
        if sample >= end_sample:
            break
        if state in p.allowed_rhythms:
            allowed_duration += sample - cursor
        cursor, state = sample, rhythm
    if state in p.allowed_rhythms:
        allowed_duration += end_sample - cursor
    n = sum(counts.values())
    purity = sum(counts[s] for s in p.allowed_beats) / n if n else 0.0
    coverage = allowed_duration / (end_sample - start_sample)
    label, y = burden_label(counts["V"], n, p)
    reasons = []
    if not n:
        reasons.append("no_reference_beats")
    if purity < p.min_beat_purity:
        reasons.append("beat_purity")
    if coverage < p.min_rhythm_coverage:
        reasons.append("rhythm_coverage")
    if not label and n:
        reasons.append("intermediate_pvc_burden")
    return {"label": label, "y": "" if y is None else y,
            "n_beats": n, "n_v": counts["V"],
            "pvc_fraction": counts["V"] / n if n else None,
            "purity_fraction": purity, "allowed_rhythm_fraction": coverage,
            "beat_counts": json.dumps(dict(sorted(counts.items()))),
            "quality_flags": json.dumps(dict(sorted(flags.items()))),
            "reasons": reasons}


def flatline_seconds(signal, fs):
    x = np.asarray(signal)
    if len(x) < 2:
        return 0.0
    equal = np.diff(x) == 0
    edges = np.diff(np.r_[False, equal, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return float(np.max(ends - starts) / fs) if len(starts) else 0.0


def _request_bytes(url):
    last_error = None
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={"User-Agent": "HOS-academic-reproduction/0.2"}),
                         timeout=30) as stream:
                return stream.read()
        except (OSError, HTTPException) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(attempt + 1)
    raise RuntimeError(f"Download failed: {url}: {last_error}") from last_error


def _download_verified_file(url, target, checksum):
    """Keep interrupted transfers and resume HTTP ranges; publish only verified bytes."""
    target = Path(target)
    if target.exists() and sha256_file(target) == checksum:
        return
    partial = target.with_suffix(target.suffix + ".part")
    last_error = None
    for attempt in range(5):
        if partial.exists() and sha256_file(partial) == checksum:
            partial.replace(target)
            return
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "HOS-academic-reproduction/0.2"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with urlopen(Request(url, headers=headers), timeout=30) as response:
                if response.status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",
                                         response.headers.get("Content-Range", ""))
                    if not match or int(match[1]) != offset:
                        raise OSError("Server returned an inconsistent byte range")
                    total = int(match[3])
                elif response.status == 200:
                    # A server may ignore Range. Safely restart, without appending.
                    offset = 0
                    length = response.headers.get("Content-Length")
                    total = int(length) if length else None
                else:
                    raise OSError(f"Unexpected HTTP status {response.status}")
                with partial.open("ab" if offset else "wb") as stream:
                    while True:
                        chunk = response.read(64 * 1024)
                        if not chunk:
                            break
                        stream.write(chunk)
            if total is not None and partial.stat().st_size < total:
                raise OSError("Connection ended before the advertised file length")
            if sha256_file(partial) != checksum:
                partial.unlink()
                raise ValueError("Checksum mismatch downloading " + target.name)
            partial.replace(target)
            return
        except (OSError, HTTPException, ValueError) as exc:
            last_error = exc
            if getattr(exc, "code", None) == 416 and partial.exists():
                partial.unlink()
            if attempt < 4:
                time.sleep(min(attempt + 1, 3))
    raise RuntimeError(f"Download failed: {url}: {last_error}") from last_error


def download_mitdb(data_dir, records=None, workers=4):
    """Download only version-pinned public files; verify official SHA256 values."""
    data_dir = Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    records = tuple(records or RECORDS)
    if not set(records) <= set(RECORDS):
        raise ValueError("Unknown MIT-BIH record requested")
    sums = _request_bytes(DATA_URL + "SHA256SUMS.txt")
    expected = {}
    for line in sums.decode("utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            expected[parts[1].lstrip("* ./")] = parts[0]
    (data_dir / "SHA256SUMS.txt").write_bytes(sums)
    (data_dir / "RECORDS").write_bytes(_request_bytes(DATA_URL + "RECORDS"))
    names = [f"{r}.{extension}" for r in records for extension in ("hea", "dat", "atr")]

    def fetch(name):
        target = data_dir / name
        checksum = expected.get(name)
        if checksum is None:
            raise ValueError(f"No official checksum for {name}")
        _download_verified_file(DATA_URL + name, target, checksum)
        return name

    completed, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, name): name for name in names}
        for future in as_completed(futures):
            try:
                completed.append(future.result())
                if len(completed) % 12 == 0 or len(completed) == len(names):
                    print(f"verified {len(completed)}/{len(names)} MIT-BIH files", flush=True)
            except Exception as exc:
                failures.append({"file": futures[future], "error": str(exc)})
    write_json(data_dir / "download_metadata.json",
               {"dataset": "mitdb", "version": "1.0.0", "url": DATA_URL,
                "records": list(records), "verified_files": sorted(completed),
                "failures": failures, **provenance()})
    if failures:
        raise RuntimeError(f"{len(failures)} downloads failed; see download_metadata.json")
    return data_dir


def prepare_mitdb(data_dir, run_dir, protocol=None, records=None):
    p = protocol or MitdbProtocol()
    data_dir, run_dir = Path(data_dir).resolve(), Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    # Prevent accidental reuse of stale downstream predictions after preparation.
    if (run_dir / "evaluation.json").exists():
        raise ValueError("Run directory already evaluated; choose a fresh output directory")
    if (run_dir / "extraction.json").exists():
        raise ValueError("Run already extracted; choose a fresh output directory")
    records = tuple(records or RECORDS)
    rows, source_files = [], []
    checksum_path = data_dir / "SHA256SUMS.txt"
    official = {}
    if checksum_path.exists():
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                official[parts[1].lstrip("* ./")] = parts[0]
    for record in records:
        path = data_dir / record
        header, annotation, waveform = None, None, None
        record_errors = []
        for extension in ("hea", "dat", "atr"):
            file = path.with_suffix("." + extension)
            digest = sha256_file(file) if file.exists() else None
            source_files.append({"file": file.name, "exists": file.exists(),
                                 "bytes": file.stat().st_size if file.exists() else None,
                                 "sha256": digest,
                                 "official_sha256": official.get(file.name),
                                 "official_match": digest == official[file.name] if file.name in official else None})
            if not file.exists():
                record_errors.append("missing_" + extension)
            elif file.name in official and digest != official[file.name]:
                record_errors.append("checksum_mismatch_" + extension)
        if record in p.excluded_records:
            record_errors.append("paced_record")
        channel = -1
        if not record_errors:
            try:
                header = wfdb.rdheader(str(path))
                if header.fs != p.expected_fs:
                    record_errors.append("unexpected_sampling_rate")
                if "MLII" not in header.sig_name:
                    record_errors.append("missing_MLII")
                else:
                    channel = header.sig_name.index("MLII")
                if not record_errors:
                    annotation = wfdb.rdann(str(path), "atr")
                    waveform, fields = wfdb.rdsamp(str(path), channels=[channel])
                    if fields["fs"] != header.fs:
                        record_errors.append("header_signal_sampling_mismatch")
                    waveform = waveform[:, 0]
            except Exception as exc:
                record_errors.append("read_error:" + str(exc))
        fs = header.fs if header else p.expected_fs
        for index in range(p.record_seconds // p.segment_seconds):
            start_s, end_s = index * p.segment_seconds, (index + 1) * p.segment_seconds
            start, end = round(start_s * fs), round(end_s * fs)
            row = dict.fromkeys(SEGMENT_FIELDS, "")
            row.update(segment_id=f"{record}_{start_s:04d}_{end_s:04d}", record_id=record,
                       subject_id=subject_id(record), path=str(path), fs=fs,
                       channel=channel, channel_name="MLII" if channel >= 0 else "",
                       start_sample=start, end_sample=end, start_s=start_s, end_s=end_s)
            reasons = list(record_errors)
            if annotation is not None:
                stats = annotation_statistics(annotation, start, end, p)
                reasons.extend(stats.pop("reasons"))
                row.update(stats)
            if waveform is not None:
                segment = waveform[start:end]
                if len(segment) != end - start:
                    reasons.append("incomplete_segment")
                elif not np.all(np.isfinite(segment)):
                    reasons.append("nonfinite_signal")
                else:
                    row["flatline_seconds"] = flatline_seconds(segment, fs)
                    if row["flatline_seconds"] > p.max_flatline_seconds:
                        reasons.append("flatline")
            row["included"] = int(not reasons)
            row["exclusion_reasons"] = ";".join(reasons)
            rows.append(row)
    included = [r for r in rows if r["included"]]
    counts = {label: {"segments": sum(r["label"] == label for r in included),
                      "subjects": len({r["subject_id"] for r in included if r["label"] == label})}
              for label in LABELS}
    exclusions = Counter(reason for r in rows for reason in r["exclusion_reasons"].split(";") if reason)
    audit = {"protocol_id": p.protocol_id, "dataset_version": p.dataset_version,
             "candidate_segments": len(rows), "included_segments": len(included),
             "excluded_segments": len(rows) - len(included), "class_counts": counts,
             "n_subjects": len({r["subject_id"] for r in included}),
             "exclusion_counts": dict(exclusions),
             "status": "ready" if all(c["subjects"] >= p.min_subjects_per_class
                                       for c in counts.values()) else "insufficient_data"}
    write_csv(run_dir / "cohort.csv", rows, SEGMENT_FIELDS)
    write_csv(run_dir / "manifest.csv", included, SEGMENT_FIELDS)
    write_json(run_dir / "protocol.json", p.to_dict())
    write_json(run_dir / "audit.json", audit)
    write_json(run_dir / "source_files.json", source_files)
    write_json(run_dir / "provenance.json", {**provenance(), "dataset_url": DATA_URL,
               "protocol_sha256": sha256_file(run_dir / "protocol.json"),
               "manifest_sha256": sha256_file(run_dir / "manifest.csv")})
    return audit
