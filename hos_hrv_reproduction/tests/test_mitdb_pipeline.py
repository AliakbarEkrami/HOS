"""Actual offline WFDB files exercise the complete public-data workflow."""
import json

import numpy as np
import pytest
import wfdb

from hos_hrv_reproduction import mitdb_evaluation as evaluation
from hos_hrv_reproduction import mitdb_features as extraction
from hos_hrv_reproduction.config import PaperConfig
from hos_hrv_reproduction.mitdb_common import read_csv
from hos_hrv_reproduction.mitdb_protocol import MitdbProtocol, prepare_mitdb
from hos_hrv_reproduction.mitdb_report import report_mitdb


def _synthetic_records(data_dir):
    records = ("100", "101", "103", "105", "106", "108", "109", "111", "112", "113")
    peaks_by_record = {}
    fs, duration = 360, 240
    t = np.arange(fs * duration) / fs
    for i, record in enumerate(records):
        rng = np.random.default_rng(800 + i)
        peaks, symbols, time, count = [], [], 0.6, 0
        while time < duration - 0.3:
            peaks.append(round(time * fs))
            symbols.append("V" if i >= 5 and count % 10 == 5 else "N")
            interval = 0.8 + 0.035 * np.sin(0.12 * count + i / 7)
            if i >= 5 and count % 10 == 4:
                interval = 0.45
            elif i >= 5 and count % 10 == 5:
                interval = 1.15
            time += interval
            count += 1
        ecg = 0.025 * np.sin(2 * np.pi * 0.6 * t) + rng.normal(0, 0.008, len(t))
        kernel_t = np.arange(-25, 26) / fs
        kernel = np.exp(-(kernel_t / 0.012) ** 2)
        for sample in peaks:
            ecg[sample - 25:sample + 26] += kernel
        wfdb.wrsamp(record, fs=fs, units=["mV", "mV"], sig_name=["V1", "MLII"],
                    p_signal=np.column_stack((ecg / 2, ecg)), fmt=["16", "16"],
                    write_dir=str(data_dir))
        wfdb.wrann(record, "atr", sample=np.r_[0, peaks], symbol=["+", *symbols],
                   aux_note=["(N", *([""] * len(peaks))], write_dir=str(data_dir))
        peaks_by_record[record] = np.asarray(peaks) / fs
    return records, peaks_by_record


def test_offline_wfdb_prepare_extract_evaluate_report(tmp_path, monkeypatch):
    data_dir, run_dir = tmp_path / "data", tmp_path / "run"
    data_dir.mkdir()
    records, peaks = _synthetic_records(data_dir)
    p = MitdbProtocol(protocol_id="mitdb_pvc_test", segment_seconds=120,
                      record_seconds=240, min_blocks=2)
    audit = prepare_mitdb(data_dir, run_dir, p, records=records)
    assert audit["status"] == "ready"
    assert audit["included_segments"] == 20
    assert audit["class_counts"] == {"LOW_PVC": {"segments": 10, "subjects": 5},
                                     "HIGH_PVC": {"segments": 10, "subjects": 5}}
    queue = iter(records)
    def controlled_detector(cleaned, fs, config):
        # Isolate dataflow from detector accuracy; real conversion and filtering
        # still execute, and the public matching function has independent tests.
        assert fs == 320
        assert len(cleaned) == 240 * 320
        record = next(queue)
        if record == records[0]:
            raise ValueError("deliberate first-record detector failure")
        return np.round(peaks[record] * fs).astype(int)
    monkeypatch.setattr(extraction, "detect_rpeaks", controlled_detector)
    config = PaperConfig(block_size=64, overlap=32)
    result = extraction.extract_mitdb(run_dir, config=config)
    assert result["sources"] == {"reference": {"ok": 20}, "detector": {"failed": 2, "ok": 18}}
    for source in ("reference", "detector"):
        rows = read_csv(run_dir / f"features_{source}.csv")
        assert len(rows) == 20
        assert all(np.isfinite([float(row[name]) for name in extraction.ALL_FEATURES]).all()
                   for row in rows if row["status"] == "ok")
        assert sum(row["status"] == "failed" for row in rows) == (2 if source == "detector" else 0)
    qrs_rows = read_csv(run_dir / "qrs_metrics.csv")
    assert all(float(row["sensitivity"]) == (0 if row["record_id"] == records[0] else 1)
               for row in qrs_rows)
    monkeypatch.setattr(evaluation, "PARAM_GRID", ({"C": 1, "gamma": "scale"},))
    evaluated = evaluation.evaluate_mitdb(run_dir, bootstrap_repeats=20)
    assert evaluated["status"] == "complete"
    assert len(evaluated["results"]) == 8
    predictions = read_csv(run_dir / "predictions.csv")
    assert len(predictions) == 20 * 8
    failures = [row for row in predictions if row["source"] == "detector"
                and row["record_id"] == records[0]]
    assert len(failures) == 8
    assert all(row["prediction"] == "ABSTAIN" and row["y_pred"] == "-1"
               for row in failures)
    assert all(item["metrics"]["coverage"] == (0.9 if item["source"] == "detector" else 1)
               for item in evaluated["results"])
    splits = json.loads((run_dir / "splits.json").read_text())
    assert len(splits["folds"]) == 3
    text = report_mitdb(run_dir).read_text(encoding="utf-8")
    assert "mitdb_pvc_test" in text
    assert "آزمایش تغییریافته" in text
    assert "LOW_PVC" in text and "HIGH_PVC" in text
    for name in ("signal_examples.png", "bispectrum_examples.png", "hos_features.png", "performance.png", "confusion.png"):
        assert (run_dir / "figures" / name).stat().st_size > 1000


def test_failed_detector_segments_and_qrs_misses_are_retained(tmp_path, monkeypatch):
    data_dir, run_dir = tmp_path / "data", tmp_path / "run"
    data_dir.mkdir()
    records, _ = _synthetic_records(data_dir)
    p = MitdbProtocol(protocol_id="mitdb_pvc_test", segment_seconds=120,
                      record_seconds=240, min_blocks=2)
    prepare_mitdb(data_dir, run_dir, p, records=records[:1])
    def failed_detector(*args):
        raise ValueError("deliberate offline detector failure")
    monkeypatch.setattr(extraction, "detect_rpeaks", failed_detector)
    result = extraction.extract_mitdb(run_dir, PaperConfig(block_size=64, overlap=32))
    assert result["sources"] == {"reference": {"ok": 2}, "detector": {"failed": 2}}
    failed = read_csv(run_dir / "features_detector.csv")
    assert len(failed) == 2
    assert all("deliberate offline detector failure" in row["error"] for row in failed)
    assert all(all(row[name] == "" for name in extraction.ALL_FEATURES) for row in failed)
    qrs = read_csv(run_dir / "qrs_metrics.csv")
    assert all(row["status"] == "failed" and int(row["tp"]) == 0
               and int(row["fn"]) > 0 for row in qrs)
    text = report_mitdb(run_dir).read_text(encoding="utf-8")
    assert "deliberate offline detector failure" in text


@pytest.mark.parametrize("modified", ["manifest.csv", "protocol.json", "paper_config.json", "100.dat"])
def test_extraction_rejects_changed_prepared_inputs(tmp_path, modified):
    data_dir, run_dir = tmp_path / "data", tmp_path / "run"
    data_dir.mkdir()
    records, _ = _synthetic_records(data_dir)
    p = MitdbProtocol(protocol_id="mitdb_pvc_test", segment_seconds=120,
                      record_seconds=240, min_blocks=2)
    prepare_mitdb(data_dir, run_dir, p, records=records[:1])
    target = (data_dir if modified.endswith(".dat") else run_dir) / modified
    if modified == "paper_config.json":
        target.write_text(json.dumps(PaperConfig().to_dict()))
    else:
        with target.open("ab") as stream:
            stream.write(b"\n")
    with pytest.raises(ValueError, match="changed|differs"):
        extraction.extract_mitdb(run_dir, PaperConfig(block_size=64, overlap=32))


def test_insufficient_block_count_is_a_recorded_failure(tmp_path):
    data_dir, run_dir = tmp_path / "data", tmp_path / "run"
    data_dir.mkdir()
    records, _ = _synthetic_records(data_dir)
    p = MitdbProtocol(protocol_id="mitdb_pvc_test", segment_seconds=120,
                      record_seconds=240, min_blocks=5)
    prepare_mitdb(data_dir, run_dir, p, records=records[:1])
    # 120 seconds at 4 Hz gives fewer than five 256-sample blocks.
    result = extraction.extract_mitdb(run_dir, PaperConfig(block_size=256, overlap=128),
                                     peak_source="reference", save_examples=False)
    assert result["sources"] == {"reference": {"failed": 2}}
    rows = read_csv(run_dir / "features_reference.csv")
    assert len(rows) == 2
    assert all("insufficient_blocks" in row["error"] for row in rows)
