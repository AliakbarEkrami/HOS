"""Subject-disjoint nested evaluation for the frozen MIT-BIH PVC protocol.

Annotation-derived PVC counts never enter a feature matrix. Every QC-included
segment remains in the analysis cohort; extraction failures are abstentions.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .mitdb_common import provenance, sha256_file

LABELS = ("LOW_PVC", "HIGH_PVC")
HOS_FEATURES = ("P1", "P2", "H1", "H2", "H3", "f1m", "f2m")
BASE_FEATURES = ("mean_rr", "sd_rr", "rmssd_rr", "prr50", "log_vlf_power",
                 "log_lf_power", "log_hf_power", "psd_entropy")
FEATURE_SETS = {"majority": (), "baseline": BASE_FEATURES, "hos": HOS_FEATURES,
                "combined": HOS_FEATURES + BASE_FEATURES}
CI_METRICS = ("balanced_accuracy", "sensitivity", "specificity", "f1_positive", "f1_macro")
PARAM_GRID = tuple({"C": c, "gamma": g} for c in (1, 10, 100, 190)
                   for g in ("scale", 0.01, 0.1))


def _frozen_metadata(run_dir):
    """Reject stale or edited artifacts when extraction metadata is available."""
    paths = {"manifest": "manifest.csv", "protocol": "protocol.json",
             "paper_config": "paper_config.json", "source_files": "source_files.json",
             "features_reference": "features_reference.csv",
             "features_detector": "features_detector.csv"}
    hashes = {name: sha256_file(run_dir / filename) for name, filename in paths.items()
              if (run_dir / filename).exists()}
    extraction_path = run_dir / "extraction.json"
    frozen = {}
    if extraction_path.exists():
        frozen = json.loads(extraction_path.read_text(encoding="utf-8"))
        feature_hashes = frozen.get("feature_files", {})
        if set(feature_hashes) != {"reference", "detector"}:
            raise ValueError("Frozen extraction must include both reference and detector sources")
        required = {"manifest": frozen.get("manifest_sha256"),
                    "protocol": frozen.get("protocol_sha256"),
                    "paper_config": frozen.get("paper_config_sha256"),
                    "source_files": frozen.get("source_files_sha256"),
                    **{f"features_{source}": digest for source, digest in feature_hashes.items()}}
        for name, expected in required.items():
            if not expected or hashes.get(name) != expected:
                raise ValueError(f"Frozen extraction artifact changed or missing checksum: {name}")
        hashes["extraction"] = sha256_file(extraction_path)
    protocol_path = run_dir / "protocol.json"
    protocol = (json.loads(protocol_path.read_text(encoding="utf-8"))
                if protocol_path.exists() else {"protocol_id": "mitdb_pvc_v1",
                                               "seed": 20261026, "min_subjects_per_class": 5})
    return protocol, hashes, frozen


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False), encoding="utf-8")


def _read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = [row.get("segment_id", "") for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"Missing or duplicate segment_id in {path}")
    return rows


def _load_inputs(run_dir):
    manifest = _read_csv(run_dir / "manifest.csv")
    manifest = [row for row in manifest
                if row.get("included", "1").lower() in ("1", "true", "yes")]
    expected = {row["segment_id"]: row for row in manifest}
    sources = {}
    for source in ("reference", "detector"):
        rows = _read_csv(run_dir / f"features_{source}.csv")
        indexed = {row["segment_id"]: row for row in rows}
        if set(indexed) != set(expected):
            raise ValueError(f"{source} segment IDs differ from manifest cohort")
        for segment_id, row in indexed.items():
            for key in ("record_id", "subject_id", "label"):
                if row.get(key) != expected[segment_id].get(key):
                    raise ValueError(f"{source} metadata mismatch: {segment_id}/{key}")
            if row.get("label") not in LABELS:
                raise ValueError(f"Unknown binary label in {segment_id}")
            label_y = LABELS.index(row["label"])
            if row.get("y", "") != "" and float(row["y"]) != label_y:
                raise ValueError(f"Label/y mismatch in {segment_id}")
        sources[source] = [indexed[row["segment_id"]] for row in manifest]
    linked = {row["subject_id"] for row in manifest if row["record_id"] in ("201", "202")}
    if len(linked) > 1:
        raise ValueError("Records 201 and 202 must share one subject_id")
    return manifest, sources


def _feature_matrix(rows):
    """Use a hard allowlist: PVC counts, labels and IDs cannot become predictors."""
    names = HOS_FEATURES + BASE_FEATURES
    matrix = np.full((len(rows), len(names)), np.nan)
    valid = np.zeros(len(rows), dtype=bool)
    reasons = []
    for i, row in enumerate(rows):
        error = row.get("error", "")
        if row.get("status") == "ok":
            try:
                values = np.array([float(row[name]) for name in names], dtype=float)
                if np.all(np.isfinite(values)):
                    matrix[i] = values
                    valid[i] = True
                else:
                    error = "nonfinite_features"
            except (ValueError, TypeError, KeyError):
                error = "missing_or_invalid_features"
        else:
            error = error or "feature_extraction_failed"
        reasons.append(error)
    return matrix, valid, reasons


def binary_metrics(y_true, y_pred):
    """All abstentions (-1) count wrong; their column stays visible in confusion."""
    y, pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    if y.shape != pred.shape or not np.all(np.isin(y, (0, 1))):
        raise ValueError("Expected equal binary true/prediction vectors")
    if not np.all(np.isin(pred, (-1, 0, 1))):
        raise ValueError("Prediction must be 0, 1, or -1 (abstention)")
    cm = np.array([[np.sum((y == a) & (pred == b)) for b in (0, 1, -1)]
                   for a in (0, 1)], dtype=int)
    recalls, f1 = [], []
    for label in (0, 1):
        actual = int(np.sum(y == label))
        correct = int(np.sum((y == label) & (pred == label)))
        false_positive = int(np.sum((y != label) & (pred == label)))
        false_negative = actual - correct
        recalls.append(correct / actual if actual else None)
        denominator = 2 * correct + false_positive + false_negative
        f1.append(2 * correct / denominator if denominator else 0.0)
    return {
        "n_segments": len(y), "accuracy": float(np.mean(y == pred)) if len(y) else None,
        "balanced_accuracy": float(np.mean(recalls)) if None not in recalls else None,
        "sensitivity": recalls[1], "specificity": recalls[0],
        "f1_positive": f1[1], "f1_macro": float(np.mean(f1)),
        "coverage": float(np.mean(pred != -1)) if len(y) else None,
        "confusion": cm.tolist(), "confusion_rows": list(LABELS),
        "confusion_columns": [*LABELS, "ABSTAIN"],
    }


def _valid_folds(folds, y, groups):
    return all(set(y[train]) == {0, 1} and set(y[test]) == {0, 1}
               and not set(groups[train]) & set(groups[test])
               for train, test in folds)


def _find_folds(y, groups, n_splits, seed, attempts=100):
    if n_splits < 2:
        raise ValueError("At least two CV splits are required")
    for trial_seed in range(seed, seed + attempts):
        try:
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                      random_state=trial_seed)
            folds = list(cv.split(np.zeros((len(y), 1)), y, groups))
        except ValueError:
            continue
        if _valid_folds(folds, y, groups):
            return folds, trial_seed
    raise ValueError("No feasible group-disjoint folds with both classes")


def nested_subject_splits(y, groups, seed=20261026, outer_splits=3, inner_splits=3):
    """Search only label/group feasibility; never inspect feature values or scores."""
    y, groups = np.asarray(y, int), np.asarray(groups, str)
    for trial_seed in range(seed, seed + 100):
        try:
            outer, used_seed = _find_folds(y, groups, outer_splits, trial_seed, attempts=1)
            nested = []
            for fold, (train, test) in enumerate(outer):
                inner, inner_seed = _find_folds(y[train], groups[train], inner_splits,
                                                seed + 1000 * (fold + 1))
                nested.append({"train": train, "test": test, "inner": inner,
                               "inner_seed": inner_seed})
            return nested, used_seed
        except ValueError:
            continue
    raise ValueError("No feasible outer/inner subject-grouped split within 100 seeds")


def _membership(indices, manifest):
    return {"segment_ids": [manifest[i]["segment_id"] for i in indices],
            "subject_ids": sorted({manifest[i]["subject_id"] for i in indices})}


def _serialize_splits(folds, manifest, seed):
    serialized = {"outer_seed": seed, "folds": []}
    for fold, spec in enumerate(folds):
        train, test = spec["train"], spec["test"]
        inner = [{"fold": i, "train": _membership(train[a], manifest),
                  "validation": _membership(train[b], manifest)}
                 for i, (a, b) in enumerate(spec["inner"])]
        serialized["folds"].append({"fold": fold, "train": _membership(train, manifest),
                                    "test": _membership(test, manifest),
                                    "inner_seed": spec["inner_seed"], "inner": inner})
    return serialized


def _predict(X, y, valid, train, test, params, majority=False):
    pred = np.full(len(test), -1, dtype=int)
    scores = np.full(len(test), np.nan)
    fitting = train[valid[train]]
    testing = np.flatnonzero(valid[test])
    if set(y[fitting]) != {0, 1}:
        return pred, scores, "training_has_fewer_than_two_classes"
    if majority:
        counts = np.bincount(y[fitting], minlength=2)
        pred[testing] = int(np.argmax(counts))
        return pred, scores, ""
    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", **params))
    try:
        model.fit(X[fitting], y[fitting])
        if len(testing):
            score = model.decision_function(X[test[testing]])
            scores[testing] = score
            pred[testing] = (score > 0).astype(int)
    except (ValueError, FloatingPointError) as exc:
        return np.full(len(test), -1, int), scores, f"fit_failed: {exc}"
    return pred, scores, ""


def _tune(X, y, valid, spec):
    train = spec["train"]
    best_score, best = -1.0, None
    for params in PARAM_GRID:
        scores = []
        for inner_train, inner_test in spec["inner"]:
            a, b = train[inner_train], train[inner_test]
            pred, _, _ = _predict(X, y, valid, a, b, params)
            scores.append(binary_metrics(y[b], pred)["balanced_accuracy"])
        score = float(np.mean(scores))
        if score > best_score:
            best_score, best = score, dict(params)
    return best, best_score


def _cluster_draws(subjects, repeats, seed):
    unique = np.unique(subjects)
    indices = {subject: np.flatnonzero(subjects == subject) for subject in unique}
    rng = np.random.default_rng(seed)
    for _ in range(repeats):
        sampled = rng.choice(unique, len(unique), replace=True)
        yield np.concatenate([indices[subject] for subject in sampled])


def _interval(values):
    return [float(x) for x in np.percentile(values, (2.5, 97.5))] if values else [None, None]


def _bootstrap(y, subjects, predictions, repeats, seed):
    """Reuse subject draws for every model and all within-source differences."""
    values = {key: {metric: [] for metric in CI_METRICS} for key in predictions}
    differences = {source: [] for source in ("reference", "detector")}
    usable_draws = 0
    for indices in _cluster_draws(subjects, repeats, seed):
        if len(set(y[indices])) != 2:
            continue
        usable_draws += 1
        draw_metrics = {}
        for key, pred in predictions.items():
            metrics = binary_metrics(y[indices], pred[indices])
            draw_metrics[key] = metrics
            for name in CI_METRICS:
                values[key][name].append(metrics[name])
        for source in differences:
            differences[source].append(draw_metrics[(source, "hos")]["balanced_accuracy"] -
                                       draw_metrics[(source, "baseline")]["balanced_accuracy"])
    intervals = {key: {name: _interval(samples) for name, samples in data.items()}
                 for key, data in values.items()}
    return intervals, {source: _interval(samples) for source, samples in differences.items()}, usable_draws


def _paired_sources(y, subjects, predictions, repeats, seed):
    comparisons = []
    for model in FEATURE_SETS:
        reference, detector = predictions[("reference", model)], predictions[("detector", model)]
        common = (reference != -1) & (detector != -1)
        yc, sc = y[common], subjects[common]
        refc, detc = reference[common], detector[common]
        rmetrics, dmetrics = binary_metrics(yc, refc), binary_metrics(yc, detc)
        values = []
        if len(yc):
            for indices in _cluster_draws(sc, repeats, seed):
                if len(set(yc[indices])) != 2:
                    continue
                values.append(binary_metrics(yc[indices], detc[indices])["balanced_accuracy"] -
                              binary_metrics(yc[indices], refc[indices])["balanced_accuracy"])
        point = None
        if rmetrics["balanced_accuracy"] is not None and dmetrics["balanced_accuracy"] is not None:
            point = dmetrics["balanced_accuracy"] - rmetrics["balanced_accuracy"]
        comparisons.append({"model": model, "n_segments": int(common.sum()),
                            "n_subjects": len(set(sc)), "reference_metrics": rmetrics,
                            "detector_metrics": dmetrics,
                            "difference_detector_minus_reference": point,
                            "ci": _interval(values)})
    return comparisons


def evaluate_mitdb(run_dir, seed=None, bootstrap_repeats=2000,
                   min_subjects_per_class=None, outer_splits=3, inner_splits=3):
    """Evaluate all feature sets using the same nested subject-disjoint splits.

    Writes evaluation.json, splits.json and predictions.csv. No feature selection
    or segmentation decisions depend on model performance.
    """
    run_dir = Path(run_dir)
    protocol, hashes, extraction = _frozen_metadata(run_dir)
    protocol_seed = int(protocol.get("seed", 20261026))
    protocol_minimum = int(protocol.get("min_subjects_per_class", 5))
    seed = protocol_seed if seed is None else int(seed)
    min_subjects_per_class = protocol_minimum if min_subjects_per_class is None else int(min_subjects_per_class)
    if bootstrap_repeats < 0 or min_subjects_per_class < 1:
        raise ValueError("Invalid bootstrap count or minimum subject count")
    manifest, rows = _load_inputs(run_dir)
    matrices, validity, errors = {}, {}, {}
    for source in rows:
        matrices[source], validity[source], errors[source] = _feature_matrix(rows[source])
    cohort_indices = np.arange(len(manifest))
    exclusions = [{"segment_id": manifest[i]["segment_id"], "error": errors["reference"][i]}
                  for i in np.flatnonzero(~validity["reference"])]
    cohort = [manifest[i] for i in cohort_indices]
    y = np.array([LABELS.index(row["label"]) for row in cohort], int)
    subjects = np.array([row["subject_id"] for row in cohort], str)
    deviations = {name: {"protocol": expected, "effective": effective}
                  for name, expected, effective in (("seed", protocol_seed, seed),
                  ("min_subjects_per_class", protocol_minimum, min_subjects_per_class),
                  ("bootstrap_repeats", 2000, bootstrap_repeats),
                  ("outer_splits", 3, outer_splits), ("inner_splits", 3, inner_splits))
                  if expected != effective}
    result = {"protocol": protocol.get("protocol_id", "mitdb_pvc_v1"),
              "protocol_settings": protocol, "protocol_deviations": deviations,
              "input_hashes": hashes, "provenance": provenance(),
              "frozen_extraction_verified": bool(extraction),
              "status": "insufficient_data", "seed": seed,
              "cohort": {"manifest_count": len(manifest), "reference_valid_count": int(validity["reference"].sum()),
                         "reference_failures": exclusions,
                         "class_counts": {label: int(np.sum(y == i)) for i, label in enumerate(LABELS)},
                         "subject_counts": {label: len(set(subjects[y == i]))
                                            for i, label in enumerate(LABELS)},
                         "n_subjects": len(set(subjects)),
                         "detector_success_count": int(validity["detector"][cohort_indices].sum())},
              "split_config": {"outer_splits": outer_splits, "inner_splits": inner_splits,
                               "max_seed_attempts": 100, "selection": "label/group feasibility only"},
              "bootstrap": {"unit": "subject", "requested_draws": bootstrap_repeats,
                            "confidence": 0.95, "conditional_on_oof_predictions": True},
              "results": [], "comparisons": {"hos_minus_baseline": [], "paired_sources": []},
              "notes": ["Positive class is HIGH_PVC (y=1); threshold is zero.",
                        "All QC-included manifest segments define the primary intention-to-process cohort.",
                        "Reference and detector extraction failures remain abstentions and count as incorrect.",
                        "Feature matrices use a fixed allowlist; annotation counts are never predictors.",
                        "Cluster bootstrap uncertainty is conditional on these OOF model fits."]}
    fieldnames = ("source", "model", "segment_id", "record_id", "subject_id", "label", "y_true",
                  "fold", "prediction", "y_pred", "status", "error", "decision_score")
    def save(prediction_rows, splits):
        _write_json(run_dir / "evaluation.json", result)
        _write_json(run_dir / "splits.json", splits)
        with (run_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(prediction_rows)

    if any(value < min_subjects_per_class for value in result["cohort"]["subject_counts"].values()):
        result["reason"] = f"Fewer than {min_subjects_per_class} distinct QC-included subjects in a class"
        save([], {"status": "insufficient_data", "folds": []})
        return result
    try:
        folds, outer_seed = nested_subject_splits(y, subjects, seed, outer_splits, inner_splits)
    except ValueError as exc:
        result["reason"] = str(exc)
        save([], {"status": "insufficient_data", "folds": []})
        return result
    splits = _serialize_splits(folds, cohort, outer_seed)
    predictions, prediction_rows = {}, []
    all_names = HOS_FEATURES + BASE_FEATURES
    for source in ("reference", "detector"):
        valid = validity[source][cohort_indices]
        source_errors = [errors[source][i] for i in cohort_indices]
        for name, feature_names in FEATURE_SETS.items():
            columns = [all_names.index(feature) for feature in feature_names]
            X = matrices[source][cohort_indices][:, columns]
            oof = np.full(len(y), -1, int)
            fold_results = []
            for fold_number, spec in enumerate(folds):
                params, tuning_score = ({}, None) if name == "majority" else _tune(X, y, valid, spec)
                test = spec["test"]
                pred, scores, fit_error = _predict(X, y, valid, spec["train"], test, params,
                                                   majority=name == "majority")
                oof[test] = pred
                score_valid = np.isfinite(scores) & (pred != -1)
                auc = None
                if set(y[test][score_valid]) == {0, 1}:
                    auc = float(roc_auc_score(y[test][score_valid], scores[score_valid]))
                fold_results.append({"fold": fold_number, "status": "failed" if fit_error else "ok",
                                     "error": fit_error, "best_params": params,
                                     "inner_balanced_accuracy": tuning_score,
                                     "training_success_count": int(valid[spec["train"]].sum()),
                                     "test_success_count": int(np.sum(pred != -1)), "roc_auc": auc})
                for j, index in enumerate(test):
                    row = {key: cohort[index][key] for key in ("segment_id", "record_id", "subject_id", "label")}
                    row.update(source=source, model=name, y_true=int(y[index]), fold=fold_number,
                               prediction=LABELS[pred[j]] if pred[j] >= 0 else "ABSTAIN",
                               y_pred=int(pred[j]), status="ok" if pred[j] >= 0 else "failed",
                               error=(source_errors[index] or fit_error) if pred[j] < 0 else "",
                               decision_score=float(scores[j]) if np.isfinite(scores[j]) else "")
                    prediction_rows.append(row)
            predictions[(source, name)] = oof
            metrics = binary_metrics(y, oof)
            metrics["fold_auc"] = [fold["roc_auc"] for fold in fold_results]
            result["results"].append({"source": source, "model": name,
                                      "feature_names": list(feature_names), "metrics": metrics,
                                      "folds": fold_results})
    intervals, differences, usable_draws = _bootstrap(y, subjects, predictions, bootstrap_repeats, seed)
    result["bootstrap"]["usable_draws"] = usable_draws
    for item in result["results"]:
        item["ci"] = intervals[(item["source"], item["model"])]
    for source in ("reference", "detector"):
        difference = (binary_metrics(y, predictions[(source, "hos")])["balanced_accuracy"] -
                      binary_metrics(y, predictions[(source, "baseline")])["balanced_accuracy"])
        result["comparisons"]["hos_minus_baseline"].append(
            {"source": source, "balanced_accuracy_difference": difference, "ci": differences[source]})
    result["comparisons"]["paired_sources"] = _paired_sources(
        y, subjects, predictions, bootstrap_repeats, seed)
    result["status"] = "complete"
    prediction_rows.sort(key=lambda row: (row["source"], row["model"], row["segment_id"]))
    save(prediction_rows, splits)
    return result
