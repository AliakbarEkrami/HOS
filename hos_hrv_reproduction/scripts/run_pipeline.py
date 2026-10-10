"""Run directly from a checkout without installing the package."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hos_hrv_reproduction.cli import extract_main, evaluate_main  # noqa: E402


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("extract", "evaluate", "stats"):
        raise SystemExit("usage: run_pipeline.py {extract|evaluate|stats} [options]")
    command = sys.argv.pop(1)
    if command == "extract":
        raise SystemExit(extract_main())
    if command == "evaluate":
        raise SystemExit(evaluate_main())
    from hos_hrv_reproduction.cli import stats_main
    raise SystemExit(stats_main())
