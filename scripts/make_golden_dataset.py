from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from financial_risk_agent.config import GOLDEN_DATA_PATH, PROJECT_ROOT
from financial_risk_agent.golden import build_golden_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Create golden dataset for GenAI/tool evaluation.")
    parser.add_argument("--csv", default=str(PROJECT_ROOT / "data" / "raw" / "UCI_Credit_Card.csv"))
    parser.add_argument("--output", default=str(GOLDEN_DATA_PATH))
    parser.add_argument("--limit", type=int, default=300)
    args = parser.parse_args()

    count = build_golden_dataset(args.csv, args.output, args.limit)
    print(f"Golden dataset created: {Path(args.output)}")
    print(f"Rows: {count}")


if __name__ == "__main__":
    main()
