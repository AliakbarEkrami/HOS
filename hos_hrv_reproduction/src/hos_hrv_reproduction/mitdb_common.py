"""Small artifact/provenance helpers for the versioned MIT-BIH experiment."""
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import subprocess


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), ensure_ascii=False,
                               indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def provenance(root=None):
    root = Path(root or __file__).resolve()
    cwd = root if root.is_dir() else root.parent
    git = {}
    for name, args in (("commit", ["rev-parse", "HEAD"]),
                       ("status", ["status", "--porcelain"])):
        try:
            result = subprocess.run(["git", *args], cwd=cwd, check=True,
                                    capture_output=True, text=True, timeout=10)
            git[name] = result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            git[name] = None
    versions = {}
    for package in ("hos-hrv-reproduction", "numpy", "scipy", "scikit-learn",
                    "wfdb", "matplotlib"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return {"created_utc": datetime.now(timezone.utc).isoformat(),
            "git": git, "packages": versions}
