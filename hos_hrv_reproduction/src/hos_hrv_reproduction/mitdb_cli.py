"""Versioned public-data experiment, separate from the legacy paper commands."""
import argparse
import json
from pathlib import Path

from .config import PaperConfig
from .mitdb_protocol import MitdbProtocol, download_mitdb, prepare_mitdb


def prepare_main(argv=None):
    parser = argparse.ArgumentParser(description="Freeze the annotation-defined MIT-BIH PVC cohort")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="Download and verify public version 1.0.0")
    parser.add_argument("--protocol", type=Path, help="JSON protocol; modified rules require a different protocol_id")
    args = parser.parse_args(argv)
    protocol = MitdbProtocol.from_json(args.protocol)
    if args.download:
        download_mitdb(args.data_dir)
    audit = prepare_mitdb(args.data_dir, args.output_dir, protocol)
    print(json.dumps(audit, indent=2))
    return 0


def extract_main(argv=None):
    from .mitdb_features import extract_mitdb
    parser = argparse.ArgumentParser(description="Extract paired HR bispectrum and baseline features")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="PaperConfig JSON; saved with the run")
    parser.add_argument("--peak-source", choices=("both", "reference", "detector"), default="both")
    args = parser.parse_args(argv)
    summary = extract_mitdb(args.run_dir, PaperConfig.from_json(args.config), args.peak_source)
    print(json.dumps(summary["sources"], indent=2))
    return 0


def evaluate_main(argv=None):
    from .mitdb_evaluation import evaluate_mitdb
    parser = argparse.ArgumentParser(description="Nested subject-grouped SVM evaluation with cluster uncertainty")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000, help="Subject bootstrap draws (protocol default: 2000)")
    args = parser.parse_args(argv)
    protocol = MitdbProtocol.from_json(args.run_dir / "protocol.json")
    result = evaluate_mitdb(args.run_dir, seed=protocol.seed,
                           min_subjects_per_class=protocol.min_subjects_per_class,
                           bootstrap_repeats=args.bootstrap)
    print(result["status"])
    if result.get("reason"):
        print(result["reason"])
    for row in result["results"]:
        print(row["source"], row["model"], "balanced_accuracy=", row["metrics"]["balanced_accuracy"])
    return 0


def report_main(argv=None):
    from .mitdb_report import report_mitdb
    parser = argparse.ArgumentParser(description="Generate the Persian experiment report and scientific figures")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(report_mitdb(args.run_dir))
    return 0
