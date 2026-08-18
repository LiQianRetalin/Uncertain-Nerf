"""Aggregate at least three random seeds and build a method comparison CSV."""

import argparse
import csv
import json
import math
import os
import statistics


T_975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


def summarize(paths):
    if len(paths) < 3:
        raise ValueError("SCI protocol requires at least three independent random seeds")
    records = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            records.append(json.load(handle))
    seeds = [record.get("seed") for record in records]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Seed identifiers must be distinct, got {seeds}")
    numeric_keys = sorted(
        set.intersection(
            *[
                {key for key, value in record.items() if isinstance(value, (int, float)) and value is not None and key != "seed"}
                for record in records
            ]
        )
    )
    rows = []
    for key in numeric_keys:
        values = [float(record[key]) for record in records]
        mean = statistics.mean(values)
        std = statistics.stdev(values)
        critical = T_975.get(len(values), 1.96)
        half = critical * std / math.sqrt(len(values))
        rows.append({
            "metric": key,
            "n": len(values),
            "mean": mean,
            "std": std,
            "ci95_low": mean - half,
            "ci95_high": mean + half,
        })
    return {"method": records[0].get("method", "unknown"), "seeds": seeds, "metrics": rows}


def main(args):
    summary = summarize(args.metrics)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    csv_path = os.path.splitext(args.output)[0] + ".csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "metric", "n", "mean", "std", "ci95_low", "ci95_high"])
        writer.writeheader()
        for row in summary["metrics"]:
            writer.writerow({"method": summary["method"], **row})
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved {args.output} and {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
