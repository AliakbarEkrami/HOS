"""The article's one-way ANOVA and follow-up pairwise tests."""
from itertools import combinations
import numpy as np
from scipy.stats import f_oneway, ttest_ind


FEATURE_NAMES = ("P1", "P2", "H1", "H2", "H3", "f1m", "f2m")


def anova_by_class(X, labels, feature_names=FEATURE_NAMES):
    X, labels = np.asarray(X, float), np.asarray(labels)
    if X.ndim != 2 or X.shape[1] != len(feature_names) or len(X) != len(labels):
        raise ValueError("X must have one column per feature and one label per row")
    classes = tuple(dict.fromkeys(labels.tolist()))
    if len(classes) < 2:
        raise ValueError("ANOVA requires at least two classes")
    rows = {}
    for j, name in enumerate(feature_names):
        groups = [X[labels == cls, j] for cls in classes]
        if any(len(g) < 2 for g in groups):
            raise ValueError(f"Each class needs at least two values for {name}")
        statistic, p_value = f_oneway(*groups)
        rows[name] = {"F": float(statistic), "p_value": float(p_value),
                      "n": {cls: int(len(g)) for cls, g in zip(classes, groups)}}
    return {"classes": list(classes), "features": rows}


def pairwise_welch_ttests(X, labels, feature_names=FEATURE_NAMES):
    X, labels = np.asarray(X, float), np.asarray(labels)
    classes = tuple(dict.fromkeys(labels.tolist()))
    rows = {}
    for j, name in enumerate(feature_names):
        tests = {}
        for left, right in combinations(classes, 2):
            stat, p_value = ttest_ind(X[labels == left, j], X[labels == right, j],
                                      equal_var=False, nan_policy="raise")
            tests[f"{left}__{right}"] = {"t": float(stat), "p_value": float(p_value)}
        rows[name] = tests
    return rows
