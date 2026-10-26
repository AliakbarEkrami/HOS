import csv
import json

import numpy as np
import pytest

from hos_hrv_reproduction import mitdb_evaluation as evaluation


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_run(path, subjects_per_class=9, segments_per_subject=2):
    rng = np.random.default_rng(444)
    manifest, rows = [], []
    for y, label in enumerate(evaluation.LABELS):
        for subject in range(subjects_per_class):
            for segment in range(segments_per_subject):
                identity = {"segment_id": f"{y}_{subject}_{segment}",
                            "subject_id": f"person_{y}_{subject}",
                            "record_id": f"record_{y}_{subject}",
                            "label": label, "included": "1"}
                manifest.append(identity)
                row = {key: value for key, value in identity.items() if key != "included"}
                row.update(y=str(y), status="ok", error="", pvc_count=str(1000 * y),
                           pvc_fraction=str(y), unknown_predictor="999999")
                for feature in evaluation.HOS_FEATURES + evaluation.BASE_FEATURES:
                    row[feature] = str(rng.normal(y * 4, 0.2))
                rows.append(row)
    _write_csv(path / "manifest.csv", manifest)
    _write_csv(path / "features_reference.csv", rows)
    _write_csv(path / "features_detector.csv", rows)
    return manifest, rows


def test_abstention_metrics_are_visible_and_wrong():
    result = evaluation.binary_metrics([0, 0, 1, 1], [0, -1, 1, -1])
    assert result["confusion"] == [[1, 0, 1], [0, 1, 1]]
    assert result["coverage"] == result["accuracy"] == 0.5
    assert result["balanced_accuracy"] == 0.5
    assert result["sensitivity"] == result["specificity"] == 0.5
    assert result["f1_positive"] == pytest.approx(2 / 3)
    assert evaluation.binary_metrics([0, 1], [-1, -1])["f1_macro"] == 0
    with pytest.raises(ValueError):
        evaluation.binary_metrics([0, 1], [0, 2])


def test_nested_folds_are_reproducible_and_subject_disjoint():
    y = np.repeat([0, 1], 18)
    subjects = np.array([f"{label}_{person}" for label in (0, 1)
                         for person in range(9) for _ in range(2)])
    folds, seed = evaluation.nested_subject_splits(y, subjects)
    other, other_seed = evaluation.nested_subject_splits(y, subjects)
    assert seed == other_seed
    seen = []
    for first, second in zip(folds, other):
        train, test = first["train"], first["test"]
        assert np.array_equal(test, second["test"])
        assert not set(subjects[train]) & set(subjects[test])
        assert set(y[test]) == {0, 1}
        for a, b in first["inner"]:
            assert not set(subjects[train[a]]) & set(subjects[train[b]])
            assert set(y[train[a]]) == set(y[train[b]]) == {0, 1}
        seen.extend(test)
    assert sorted(seen) == list(range(len(y)))


def test_small_data_stops_before_tuning(tmp_path, monkeypatch):
    _make_run(tmp_path, subjects_per_class=2)
    def fail(*args, **kwargs):
        raise AssertionError("Tuning must not run on an insufficient cohort")
    monkeypatch.setattr(evaluation, "_tune", fail)
    result = evaluation.evaluate_mitdb(tmp_path, bootstrap_repeats=2)
    assert result["status"] == "insufficient_data"
    assert result["cohort"]["subject_counts"] == {"LOW_PVC": 2, "HIGH_PVC": 2}
    assert json.loads((tmp_path / "splits.json").read_text())["folds"] == []
    with (tmp_path / "predictions.csv").open() as stream:
        assert list(csv.DictReader(stream)) == []


def test_input_metadata_and_linked_subjects_are_validated(tmp_path):
    manifest, rows = _make_run(tmp_path, subjects_per_class=2)
    rows[0]["y"] = "1"
    _write_csv(tmp_path / "features_detector.csv", rows)
    with pytest.raises(ValueError, match="Label/y mismatch"):
        evaluation.evaluate_mitdb(tmp_path)
    rows[0]["y"] = "0"
    for i, record in enumerate(("201", "202")):
        manifest[i]["record_id"] = rows[i]["record_id"] = record
        manifest[i]["subject_id"] = rows[i]["subject_id"] = f"different_person_{i}"
    _write_csv(tmp_path / "manifest.csv", manifest)
    _write_csv(tmp_path / "features_reference.csv", rows)
    _write_csv(tmp_path / "features_detector.csv", rows)
    with pytest.raises(ValueError, match="201 and 202"):
        evaluation.evaluate_mitdb(tmp_path)
    manifest[1]["subject_id"] = rows[1]["subject_id"] = rows[0]["subject_id"]
    _write_csv(tmp_path / "manifest.csv", manifest)
    _write_csv(tmp_path / "features_reference.csv", rows)
    _write_csv(tmp_path / "features_detector.csv", rows)
    assert evaluation.evaluate_mitdb(tmp_path)["status"] == "insufficient_data"


def test_feature_allowlist_excludes_annotation_leaks(tmp_path):
    _, rows = _make_run(tmp_path, subjects_per_class=2)
    matrix, valid, _ = evaluation._feature_matrix(rows)
    assert valid.all()
    assert matrix.shape[1] == 15
    for row in rows:
        row["pvc_count"] = "nan"
        row["pvc_fraction"] = "999999999"
        row["label"] = "not_a_feature"
    other, other_valid, _ = evaluation._feature_matrix(rows)
    assert np.array_equal(matrix, other)
    assert np.array_equal(valid, other_valid)
    assert set(evaluation.FEATURE_SETS["hos"]) == set(evaluation.HOS_FEATURES)
    assert not {"pvc_count", "pvc_fraction", "y", "subject_id"} & set(evaluation.FEATURE_SETS["combined"])


def test_complete_oof_shared_folds_and_detector_failures_retained(tmp_path, monkeypatch):
    manifest, rows = _make_run(tmp_path)
    reference = [dict(row) for row in rows]
    reference[0].update(status="failed", error="reference example failure")
    detector = [dict(row) for row in rows]
    detector[1].update(status="failed", error="detector example failure")
    _write_csv(tmp_path / "features_reference.csv", reference)
    _write_csv(tmp_path / "features_detector.csv", detector)
    monkeypatch.setattr(evaluation, "PARAM_GRID", ({"C": 1, "gamma": "scale"},))
    result = evaluation.evaluate_mitdb(tmp_path, bootstrap_repeats=20)
    assert result["status"] == "complete"
    assert result["cohort"]["reference_valid_count"] == len(manifest) - 1
    assert result["cohort"]["reference_failures"][0]["segment_id"] == rows[0]["segment_id"]
    assert len(result["results"]) == 8
    with (tmp_path / "predictions.csv").open(newline="") as stream:
        predictions = list(csv.DictReader(stream))
    assert len(predictions) == len(manifest) * 8
    seen_folds = {}
    for row in predictions:
        seen_folds.setdefault(row["segment_id"], set()).add(row["fold"])
        assert int(row["y_true"]) == evaluation.LABELS.index(row["label"])
        if row["source"] == "reference" and row["segment_id"] == rows[0]["segment_id"]:
            assert row["prediction"] == "ABSTAIN"
            assert row["error"] == "reference example failure"
        if row["source"] == "detector" and row["segment_id"] == rows[1]["segment_id"]:
            assert row["prediction"] == "ABSTAIN"
            assert row["error"] == "detector example failure"
    assert all(len(value) == 1 for value in seen_folds.values())
    for item in result["results"]:
        expected = (len(manifest) - 1) / len(manifest)
        assert item["metrics"]["coverage"] == pytest.approx(expected)
        assert item["ci"]["balanced_accuracy"][0] is not None
        assert set(item["feature_names"]) <= set(evaluation.HOS_FEATURES + evaluation.BASE_FEATURES)
    assert len(result["comparisons"]["paired_sources"]) == 4
    for comparison in result["comparisons"]["paired_sources"]:
        assert comparison["n_segments"] == len(manifest) - 2
    first_predictions = (tmp_path / "predictions.csv").read_text()
    first_splits = (tmp_path / "splits.json").read_text()
    repeated = evaluation.evaluate_mitdb(tmp_path, bootstrap_repeats=20)
    result.pop("provenance")
    repeated.pop("provenance")
    assert result == repeated
    assert (tmp_path / "predictions.csv").read_text() == first_predictions
    assert (tmp_path / "splits.json").read_text() == first_splits


def test_detector_training_with_missing_class_abstains_without_label_invention():
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.array([0, 0, 1, 1, 0, 1])
    valid = np.array([True, True, False, False, True, True])
    pred, score, error = evaluation._predict(
        X, y, valid, np.array([0, 1, 2, 3]), np.array([4, 5]), {"C": 1, "gamma": "scale"})
    assert list(pred) == [-1, -1]
    assert np.isnan(score).all()
    assert error == "training_has_fewer_than_two_classes"


def test_subject_bootstrap_preserves_cluster_and_reuses_seed():
    subjects = np.array(["a", "a", "b", "b", "c", "c"])
    first = list(evaluation._cluster_draws(subjects, 20, 41))
    second = list(evaluation._cluster_draws(subjects, 20, 41))
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    for indices in first:
        assert len(indices) == 6
        for i in range(0, len(indices), 2):
            assert subjects[indices[i]] == subjects[indices[i + 1]]


def test_custom_protocol_defaults_are_recorded_and_override_is_explicit(tmp_path):
    _make_run(tmp_path, subjects_per_class=2)
    protocol = {"protocol_id": "mitdb_pvc_custom_test", "seed": 13,
                "min_subjects_per_class": 6, "low_pvc_max": 0.02}
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    result = evaluation.evaluate_mitdb(tmp_path, bootstrap_repeats=2)
    assert result["protocol"] == "mitdb_pvc_custom_test"
    assert result["protocol_settings"] == protocol
    assert result["seed"] == 13
    assert result["reason"].startswith("Fewer than 6")
    assert "seed" not in result["protocol_deviations"]
    changed = evaluation.evaluate_mitdb(tmp_path, seed=99, min_subjects_per_class=3,
                                         bootstrap_repeats=2)
    assert changed["protocol_deviations"]["seed"] == {"protocol": 13, "effective": 99}
    assert changed["protocol_deviations"]["min_subjects_per_class"]["effective"] == 3


def _freeze_run(path):
    for name in ("protocol", "paper_config", "source_files"):
        (path / (name + ".json")).write_text("{}")
    frozen = {f"{name}_sha256": evaluation.sha256_file(path / filename)
              for name, filename in (("manifest", "manifest.csv"), ("protocol", "protocol.json"),
              ("paper_config", "paper_config.json"), ("source_files", "source_files.json"))}
    frozen["feature_files"] = {source: evaluation.sha256_file(path / f"features_{source}.csv")
                               for source in ("reference", "detector")}
    (path / "extraction.json").write_text(json.dumps(frozen))


def test_changed_frozen_artifacts_and_incomplete_sources_are_rejected(tmp_path):
    _make_run(tmp_path, subjects_per_class=2)
    _freeze_run(tmp_path)
    result = evaluation.evaluate_mitdb(tmp_path)
    assert result["frozen_extraction_verified"] is True
    assert set(result["input_hashes"]) == {"manifest", "protocol", "paper_config", "source_files",
                                           "features_reference", "features_detector", "extraction"}
    with (tmp_path / "features_detector.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="features_detector"):
        evaluation.evaluate_mitdb(tmp_path)
    _freeze_run(tmp_path)
    frozen = json.loads((tmp_path / "extraction.json").read_text())
    del frozen["feature_files"]["reference"]
    (tmp_path / "extraction.json").write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="both reference and detector"):
        evaluation.evaluate_mitdb(tmp_path)


@pytest.mark.parametrize("artifact,filename", [
    ("manifest", "manifest.csv"), ("protocol", "protocol.json"),
    ("paper_config", "paper_config.json"), ("source_files", "source_files.json"),
    ("features_reference", "features_reference.csv"), ("features_detector", "features_detector.csv"),
])
def test_each_frozen_input_hash_detects_stale_artifacts(tmp_path, artifact, filename):
    _make_run(tmp_path, subjects_per_class=2)
    _freeze_run(tmp_path)
    with (tmp_path / filename).open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match=artifact):
        evaluation.evaluate_mitdb(tmp_path)
    assert not (tmp_path / "evaluation.json").exists()
