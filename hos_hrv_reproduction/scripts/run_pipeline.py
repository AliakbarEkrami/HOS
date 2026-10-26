"""Run directly from a checkout without installing the package."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hos_hrv_reproduction.cli import extract_main, evaluate_main  # noqa: E402


if __name__ == "__main__":
    from hos_hrv_reproduction import mitdb_cli
    commands = {"extract": extract_main, "evaluate": evaluate_main,
                "mitdb-prepare": mitdb_cli.prepare_main, "mitdb-extract": mitdb_cli.extract_main,
                "mitdb-evaluate": mitdb_cli.evaluate_main, "mitdb-report": mitdb_cli.report_main}
    from hos_hrv_reproduction.cli import stats_main
    commands["stats"] = stats_main
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        raise SystemExit("usage: run_pipeline.py {" + "|".join(commands) + "} [options]")
    command = sys.argv.pop(1)
    raise SystemExit(commands[command]())
