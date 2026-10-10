"""Small, dependency-light record and manifest loaders."""
from dataclasses import dataclass
from pathlib import Path
import csv
import numpy as np


@dataclass(frozen=True)
class Record:
    record_id: str
    subject_id: str
    path: Path
    label: str
    kind: str = "ecg"
    fs: float = 320.0
    mains_hz: float = 50.0
    channel: int = 0


def load_array(path: str | Path, channel=0):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        x = np.load(path, allow_pickle=False)
    elif suffix in (".csv", ".txt", ".tsv"):
        x = np.loadtxt(path, delimiter="\t" if suffix == ".tsv" else ",")
    else:
        # WFDB records are normally referenced by their basename (for example
        # ``data/mitdb/100``) and consist of a .hea header plus one or more
        # companion files.  Keep this dependency optional so NumPy-only users
        # can still run the HRV/bispectrum code.
        try:
            import wfdb
        except ImportError as exc:
            raise ValueError(
                f"Unsupported raw signal file: {path}. Use .npy/.csv/.txt/.tsv "
                "or install the optional wfdb dependency for WFDB records"
            ) from exc
        record_path = path.with_suffix("") if suffix in (".hea", ".dat") else path
        try:
            # Read one requested channel when the backend supports it.  WFDB
            # returns an ``(n_samples, 1)`` matrix in that case, which is
            # flattened below without applying the channel index a second time.
            x, _ = wfdb.rdsamp(str(record_path), channels=[int(channel)])
        except Exception as exc:
            raise ValueError(f"Could not read WFDB record {record_path}: {exc}") from exc
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == 1 and suffix not in (".npy", ".csv", ".txt", ".tsv"):
        x = x[:, 0]
    elif x.ndim == 2:
        if not 0 <= channel < x.shape[1]:
            raise ValueError(f"channel {channel} is outside {x.shape[1]} columns in {path}")
        x = x[:, channel]
    if x.ndim != 1:
        raise ValueError(f"Signal in {path} must be 1-D or a channel matrix")
    return x.astype(float, copy=False)


def read_manifest(path: str | Path):
    path = Path(path)
    records = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row_number, row in enumerate(csv.DictReader(f), 2):
            if not row.get("path") or not row.get("label"):
                raise ValueError(f"Manifest row {row_number} needs path and label")
            raw_path = Path(row["path"])
            if not raw_path.is_absolute():
                raw_path = path.parent.parent / raw_path if str(raw_path).startswith("data") else path.parent / raw_path
            record_id = row.get("record_id") or raw_path.stem
            subject_id = row.get("subject_id") or record_id
            records.append(Record(record_id, subject_id, raw_path,
                                  row["label"].strip(), row.get("kind", "ecg"),
                                  float(row.get("fs") or 320),
                                  float(row.get("mains_hz") or 50),
                                  int(row.get("channel") or 0)))
    if not records:
        raise ValueError(f"Manifest has no records: {path}")
    return records
