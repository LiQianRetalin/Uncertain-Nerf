"""Merge per-method, multi-seed summaries into a paper-ready comparison CSV."""

import argparse
import csv
import json


def main(args):
    rows = []
    all_metrics = set()
    for path in args.summaries:
        with open(path, encoding="utf-8") as handle:
            summary = json.load(handle)
        row = {"method": summary["method"], "seeds": ",".join(map(str, summary["seeds"]))}
        for metric in summary["metrics"]:
            name = metric["metric"]
            all_metrics.add(name)
            row[name] = metric["mean"]
            row[name + "_ci95"] = 0.5 * (metric["ci95_high"] - metric["ci95_low"])
        rows.append(row)
    fields = ["method", "seeds"]
    for metric in sorted(all_metrics):
        fields.extend([metric, metric + "_ci95"])
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved method comparison to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
