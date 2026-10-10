"""Command line entry points used by README examples."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

from .classifier import LABELS, repeated_random_split, save_evaluation
from .config import PaperConfig
from .io import read_manifest
from .pipeline import extract_manifest
from .statistics import anova_by_class, pairwise_welch_ttests


def _config_arg(parser):
    parser.add_argument("--config", type=Path,
                        help="JSON file overriding PaperConfig assumptions")


def _gamma_arg(value):
    if value in ("scale", "auto"):
        return value
    try:
        gamma = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("gamma must be scale, auto, or a positive number") from exc
    if not np.isfinite(gamma) or gamma <= 0:
        raise argparse.ArgumentTypeError("gamma must be positive and finite")
    return gamma


def extract_main(argv=None):
    p = argparse.ArgumentParser(description="Extract the seven HOS features from a manifest")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    _config_arg(p)
    args = p.parse_args(argv)
    rows = extract_manifest(read_manifest(args.manifest), args.output,
                            PaperConfig.from_json(args.config))
    print(f"wrote {len(rows)} records to {args.output}")
    return 0


def evaluate_main(argv=None):
    p = argparse.ArgumentParser(description="Run the paper-like repeated RBF-SVM evaluation")
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--test-size", type=int, default=100,
                   help="total test records per repeat; use 0 for the full one-third pool")
    p.add_argument("--train-fraction", type=float, default=2/3)
    p.add_argument("--C", type=float, default=190.0)
    p.add_argument("--gamma", default="scale", type=_gamma_arg)
    p.add_argument("--seed", type=int, default=20261004)
    args = p.parse_args(argv)
    with args.features.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("feature CSV has no rows")
    missing = [x for x in ("label", "P1", "P2", "H1", "H2", "H3", "f1m", "f2m")
               if x not in rows[0]]
    if missing:
        raise SystemExit(f"feature CSV missing columns: {missing}")
    X = np.asarray([[float(row[name]) for name in ("P1", "P2", "H1", "H2", "H3", "f1m", "f2m")]
                    for row in rows])
    y = np.asarray([row["label"] for row in rows])
    evaluation = repeated_random_split(X, y, labels=LABELS, repeats=args.repeats,
                                       test_size=None if args.test_size == 0 else args.test_size,
                                       train_fraction=args.train_fraction, C=args.C,
                                       gamma=args.gamma, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_evaluation(evaluation, args.output)
    print(f"wrote {args.output}")
    print(f"overall_accuracy={evaluation.overall_accuracy:.6f} "
          f"macro_class_accuracy={evaluation.macro_accuracy:.6f}")
    return 0


def stats_main(argv=None):
    p = argparse.ArgumentParser(description="Run the paper-like ANOVA on a feature CSV")
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args(argv)
    names = ("P1", "P2", "H1", "H2", "H3", "f1m", "f2m")
    with args.features.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("feature CSV has no rows")
    missing = [name for name in ("label", *names) if name not in rows[0]]
    if missing:
        raise SystemExit(f"feature CSV missing columns: {missing}")
    X = np.asarray([[float(row[name]) for name in names] for row in rows])
    y = np.asarray([row["label"] for row in rows])
    result = {"anova": anova_by_class(X, y, names),
              "pairwise_welch_t": pairwise_welch_ttests(X, y, names)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(extract_main())
