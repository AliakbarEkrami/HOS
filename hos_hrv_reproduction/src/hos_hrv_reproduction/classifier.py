"""Paper-like repeated random split and RBF SVM evaluation."""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

LABELS = ("NSR", "PVC", "CHB", "SSS", "CHF")


def abnormal_metrics(confusion, labels=LABELS):
    """Compute binary metrics with actual rows, predicted columns, NSR negative."""
    cm = np.asarray(confusion)
    labels = tuple(labels)
    if "NSR" not in labels or cm.shape != (len(labels), len(labels)):
        raise ValueError("Need a square confusion matrix and an NSR label")
    normal = labels.index("NSR")
    abnormal = [i for i in range(len(labels)) if i != normal]
    tn = cm[normal, normal]
    fp = cm[normal, abnormal].sum()
    fn = cm[abnormal, normal].sum()
    tp = cm[np.ix_(abnormal, abnormal)].sum()
    return {
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "ppv": float(tp / (tp + fp)) if tp + fp else float("nan"),
    }


@dataclass(frozen=True)
class Evaluation:
    labels: tuple[str, ...]
    confusion: np.ndarray
    repeat_confusions: tuple[np.ndarray, ...]
    class_accuracy: dict[str, float]
    macro_accuracy: float
    overall_accuracy: float
    sensitivity_abnormal: float
    specificity_abnormal: float
    positive_predictive_value: float
    params: dict

    def to_dict(self):
        out = asdict(self)
        out["labels"] = list(self.labels)
        out["confusion"] = self.confusion.tolist()
        out["repeat_confusions"] = [x.tolist() for x in self.repeat_confusions]
        return out


def _check_inputs(X, y, labels):
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    if X.ndim != 2 or not np.all(np.isfinite(X)):
        raise ValueError("X must be a finite 2-D feature matrix")
    if y.ndim != 1 or len(X) != len(y) or len(y) == 0:
        raise ValueError("X and y lengths must match and be nonzero")
    unknown = sorted(set(y) - set(labels))
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}")


def repeated_random_split(X, y, labels=LABELS, repeats=5, test_size=100,
                          train_fraction=2/3, C=190.0, gamma="scale",
                          seed=20261004):
    """Evaluate record-level vectors, preserving class balance when possible.

    `test_size=100` matches the number of test predictions implied by the paper's
    five-repeat aggregate matrix. Set `test_size=None` to use one-third per class.
    """
    _check_inputs(X, y, labels)
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if test_size is not None and test_size < len(labels):
        raise ValueError("test_size must be at least the number of classes")
    if C <= 0:
        raise ValueError("C must be positive")
    X, y, labels = np.asarray(X, float), np.asarray(y), tuple(labels)
    rng = np.random.default_rng(seed)
    confs = []
    for repeat in range(repeats):
        train_idx, test_idx = [], []
        for label in labels:
            idx = np.flatnonzero(y == label)
            if len(idx) < 2:
                raise ValueError(f"Need at least two records in class {label}")
            perm = rng.permutation(idx)
            n_test = min(len(idx) - 1, max(1, int(round(len(idx) * (1 - train_fraction)))))
            test_idx.extend(perm[:n_test])
            train_idx.extend(perm[n_test:])
        if test_size is not None:
            # The article is ambiguous about its 500 aggregate test vectors; take
            # a stratified subsample from the one-third pool to match a requested total.
            # Its confusion matrix is strongly imbalanced by class, so equal quotas
            # would contradict the published row totals.
            test_idx = np.asarray(test_idx)
            sizes = np.asarray([np.sum(y == label) for label in labels], dtype=float)
            ideal = test_size * sizes / sizes.sum()
            quotas = np.floor(ideal).astype(int)
            for i in np.argsort(-(ideal - quotas))[:test_size - quotas.sum()]:
                quotas[i] += 1
            chosen = []
            for label, quota in zip(labels, quotas):
                pool = test_idx[y[test_idx] == label]
                if quota > len(pool):
                    raise ValueError(f"test_size={test_size} exceeds available {label} holdout pool")
                chosen.extend(rng.choice(pool, quota, replace=False))
            test_idx = np.asarray(chosen)
            # Records left out of the requested test-size subsample return to the
            # training set; each record is therefore in exactly one partition.
            train_idx = np.setdiff1d(np.arange(len(y)), test_idx, assume_unique=False)
        model = make_pipeline(StandardScaler(),
                              OneVsRestClassifier(SVC(C=C, gamma=gamma, kernel="rbf")))
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        cm = confusion_matrix(y[test_idx], pred, labels=labels)
        confs.append(cm)
    confusion = np.sum(confs, axis=0)
    diag = np.diag(confusion).astype(float)
    row_sum = confusion.sum(axis=1)
    class_accuracy = {label: float(diag[i] / row_sum[i]) if row_sum[i] else float("nan")
                      for i, label in enumerate(labels)}
    metrics = abnormal_metrics(confusion, labels)
    return Evaluation(labels, confusion, tuple(confs), class_accuracy,
                      float(np.nanmean(list(class_accuracy.values()))),
                      float(np.trace(confusion) / confusion.sum()),
                      metrics["sensitivity"], metrics["specificity"], metrics["ppv"],
                      {"repeats": repeats, "test_size": test_size, "train_fraction": train_fraction,
                       "C": C, "gamma": gamma, "seed": seed,
                       "multiclass": "one-vs-rest", "standardize": True})
def save_evaluation(evaluation: Evaluation, path: str | Path):
    Path(path).write_text(json.dumps(evaluation.to_dict(), indent=2), encoding="utf-8")
