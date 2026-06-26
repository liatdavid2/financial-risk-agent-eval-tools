from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from financial_risk_agent.config import GOLDEN_DATA_PATH, PROJECT_ROOT
from financial_risk_agent.evaluation import run_golden_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GenAI evaluation on golden dataset.")
    parser.add_argument("--golden", default=str(GOLDEN_DATA_PATH))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "reports" / "evals" / "eval_results.json"))
    args = parser.parse_args()

    result = run_golden_eval(args.golden, args.output)
    summary = {k: v for k, v in result.items() if k != "rows"}
    print(json.dumps(summary, indent=2))
    print(f"\nSaved detailed eval report: {Path(args.output)}")


if __name__ == "__main__":
    main()
