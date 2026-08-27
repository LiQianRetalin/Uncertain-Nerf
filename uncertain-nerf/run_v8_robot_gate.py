"""CLI for the robot-oriented Gaussian short-screen gate."""

import argparse
import json

from v8_robot.gate import evaluate_candidate, load_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, help="Candidate metrics JSON")
    parser.add_argument(
        "--limits",
        default="configs/v8_robot_screen_limits.json",
        help="Admission limits JSON",
    )
    parser.add_argument("--output", help="Optional decision JSON path")
    args = parser.parse_args()

    decision = evaluate_candidate(load_json(args.metrics), load_json(args.limits))
    rendered = json.dumps(decision, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    return 0 if decision["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
