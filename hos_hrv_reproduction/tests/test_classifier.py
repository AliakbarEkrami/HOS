import numpy as np
import pytest

from hos_hrv_reproduction.classifier import LABELS, abnormal_metrics, repeated_random_split


def test_binary_metrics_match_published_confusion_matrix():
    cm = np.array([[255, 17, 13, 0, 5], [12, 37, 0, 1, 0],
                   [5, 0, 48, 0, 7], [0, 1, 0, 64, 0], [4, 0, 0, 0, 31]])
    metrics = abnormal_metrics(cm)
    assert metrics["sensitivity"] == pytest.approx(0.9)
    assert metrics["specificity"] == pytest.approx(255 / 290)
    assert metrics["ppv"] == pytest.approx(189 / 224)


def test_repeat_evaluation_is_reproducible_and_counts_records():
    rng = np.random.default_rng(9)
    x = np.vstack([rng.normal(i * 5, 0.2, (12, 7)) for i in range(5)])
    y = np.repeat(LABELS, 12)
    first = repeated_random_split(x, y, repeats=2, test_size=None, seed=8)
    second = repeated_random_split(x, y, repeats=2, test_size=None, seed=8)
    assert np.array_equal(first.confusion, second.confusion)
    assert first.confusion.sum() == 40
    assert first.overall_accuracy > 0.9
    assert first.params["multiclass"] == "one-vs-rest"
