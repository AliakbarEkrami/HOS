import csv
import json

import numpy as np
import pytest

from hos_hrv_reproduction.mitdb_report import report_mitdb


def _csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_report_handles_missing_evaluation_and_failed_features(tmp_path):
    _json(tmp_path / "audit.json", {
        "protocol_id": "mitdb_pvc_v1", "candidate_segments": 2,
        "included_segments": 1, "excluded_segments": 1, "n_subjects": 1,
        "class_counts": {"LOW_PVC": {"segments": 1, "subjects": 1}, "HIGH_PVC": {"segments": 0, "subjects": 0}},
        "exclusion_counts": {"quality_flatline": 1},
    })
    _csv(tmp_path / "features_reference.csv", [{"segment_id": "100_0", "subject_id": "100", "label": "LOW_PVC", "status": "failed", "error": "Too few blocks"}])
    path = report_mitdb(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert path.name == "report_fa.md"
    assert "ارزیابی هنوز" in text
    assert "Too few blocks" in text
    assert "quality_flatline" in text
    assert "ثبت نشده" in text
    assert "0.0%" in text
    assert not any(char in text for char in ("\u202a", "\u202b", "\u202c", "\u2066", "\u2067", "\u2069"))
    assert "<table" not in text
    assert len(list((tmp_path / "figures").glob("*.png"))) == 3


def test_report_complete_uses_saved_metrics_and_example_assets(tmp_path):
    rows = []
    for i, label in enumerate(("LOW_PVC", "HIGH_PVC")):
        row = {"segment_id": f"s{i}", "subject_id": str(i), "label": label, "status": "ok", "error": ""}
        row.update({name: float(i + 1) for name in ("P1", "P2", "H1", "H2", "H3", "f1m", "f2m")})
        rows.append(row)
    _csv(tmp_path / "manifest.csv", [{"segment_id": "s0", "subject_id": "0", "label": "LOW_PVC"}, {"segment_id": "s1", "subject_id": "1", "label": "HIGH_PVC"}])
    for source in ("reference", "detector"):
        _csv(tmp_path / f"features_{source}.csv", rows)
    _json(tmp_path / "evaluation.json", {
        "protocol": "mitdb_pvc_v1", "status": "complete",
        "cohort": {"manifest_count": 4, "reference_valid_count": 3, "detector_success_count": 2},
        "results": [{"source": "reference", "model": "hos", "metrics": {
            "balanced_accuracy": 0.75, "sensitivity": 0.5, "specificity": 1,
            "f1_positive": 0.667, "coverage": 0.75, "confusion": [[2, 0, 0], [0, 1, 1]],
        }, "ci": {"balanced_accuracy": [0.5, 1]}}],
        "comparisons": {
            "hos_minus_baseline": [{"source": "reference", "balanced_accuracy_difference": -0.1, "ci": [-0.2, 0.1]}],
            "paired_sources": [{"model": "hos", "n_segments": 2, "reference_metrics": {"balanced_accuracy": 0.75}, "detector_metrics": {"balanced_accuracy": 0.5}}],
        },
    })
    _csv(tmp_path / "qrs_metrics.csv", [{"segment_id": "s0", "tp": 4, "fp": 1, "fn": 1, "timing_mae_ms": 10}])
    examples_dir = tmp_path / "examples"
    examples_dir.mkdir()
    times = np.arange(400) / 40
    for label in ("LOW_PVC", "HIGH_PVC"):
        np.savez(examples_dir / f"{label}.npz", times_s=times, ecg=np.sin(times),
                 reference_peaks_s=[1.0, 2.0], detector_peaks_s=[1.01, 2.01],
                 hrv_times_s=times, hrv=np.full(times.size, 70.0),
                 bispectrum_magnitude=np.array([[1, 2], [np.nan, 3]]), f_axis=[0.0, 1.0])
    path = report_mitdb(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "75.0%" in text
    assert "50.0% تا 100.0%" in text
    assert "-10.0 واحد درصد" in text
    assert "10.00 ms" in text
    assert "مشترک" in text
    assert "شکست استخراج در هر یک از مسیرهای مرجع و آشکارساز" in text
    assert "مخرج ارزیابی اصلی را تغییر نمی‌دهد" in text
    assert "85.79" not in text
    for figure in ("signal_examples.png", "bispectrum_examples.png", "hos_features.png", "performance.png", "confusion.png"):
        asset = tmp_path / "figures" / figure
        assert asset.stat().st_size > 1000
        assert f"figures/{figure}" in text
    # Running the report again reads fixed artifacts and retains the same claims.
    assert report_mitdb(tmp_path).read_text(encoding="utf-8") == text


def test_report_states_insufficient_data_and_works_with_empty_run(tmp_path):
    _json(tmp_path / "evaluation.json", {"status": "insufficient_data", "reason": "fewer than five subjects"})
    text = report_mitdb(tmp_path).read_text(encoding="utf-8")
    assert "insufficient_data" in text
    assert "fewer than five subjects" in text
    assert "ارزیابی کامل نشده" in text
    assert "ممیزی کامل در دسترس نیست" in text


def test_report_requires_existing_run_and_does_not_hide_corrupt_json(tmp_path):
    with pytest.raises(FileNotFoundError):
        report_mitdb(tmp_path / "absent")
    (tmp_path / "audit.json").write_text("{invalid", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        report_mitdb(tmp_path)


def test_report_displays_custom_protocol_levels_and_recorded_processing(tmp_path):
    _json(tmp_path / "protocol.json", {
        "protocol_id": "mitdb_pvc_test", "low_pvc_max": 0.02, "high_pvc_min": 0.07,
        "segment_seconds": 600, "min_blocks": 3, "min_subjects_per_class": 4,
        "qrs_tolerance_seconds": 0.1,
    })
    _json(tmp_path / "paper_config.json", {
        "block_size": 64, "overlap": 32, "window": "boxcar", "log_base": "10",
        "include_dc": False, "normalize_entropy": False, "index_origin": 0,
    })
    _json(tmp_path / "extraction.json", {"sources": {"reference": {"failed": 1}}})
    _json(tmp_path / "provenance.json", {})
    _json(tmp_path / "source_files.json", {})
    _json(tmp_path / "evaluation.json", {"status": "insufficient_data", "seed": 9,
        "split_config": {"outer_splits": 2, "inner_splits": 2},
        "bootstrap": {"requested_draws": 10, "usable_draws": 9}})
    _csv(tmp_path / "cohort.csv", [{"segment_id": "s0", "quality_flags": '{"+": 2, "~": 1}'}])
    text = report_mitdb(tmp_path).read_text(encoding="utf-8")
    assert "آزمایش تغییریافته" in text
    assert "حداکثر 2.0%" in text
    assert "حداقل 7.0%" in text
    assert "قطعات 10 دقیقه‌ای" in text
    assert "| اندازهٔ بلوک / NFFT | 64 |" in text
    assert "| پنجره | boxcar |" in text
    assert "| پایهٔ لگاریتم | 10 |" in text
    assert "| حضور مؤلفهٔ DC | غیرفعال |" in text
    assert "| مبدأ شاخص H3 و مرکز وزنی | 0 |" in text
    assert "حداقل 3 بلوک معتبر" in text
    assert "10 بازنمونه‌گیری" in text
    assert "تعداد قابل استفاده: 9" in text
    assert "هر دو کلاس را ندارند" in text
    assert "نشانه‌های غیرضربان" in text
    assert "[paper_config.json]" in text
    assert "[extraction.json]" in text
    assert "[source_files.json]" in text
    assert "[provenance.json]" in text
