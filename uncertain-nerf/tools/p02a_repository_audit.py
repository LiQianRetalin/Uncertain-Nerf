#!/usr/bin/env python3
"""Close P02 runtime repository/source provenance without rewriting historical snapshots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True, encoding="utf-8").strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--robust-source", type=Path, required=True)
    parser.add_argument("--spotless-source", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve(); work = args.work_root.resolve()
    robust = args.robust_source.resolve(); spotless = args.spotless_source.resolve()
    head = command("git", "rev-parse", "HEAD", cwd=repo)
    branch = command("git", "branch", "--show-current", cwd=repo)
    origin = command("git", "rev-parse", "origin/ru-part", cwd=repo)
    if branch != "ru-part" or head != args.expected_commit or origin != head:
        raise RuntimeError(f"repository mismatch: branch={branch} head={head} origin={origin}")
    formal = read_csv(work / "state/formal_ledger.csv")
    starts = [int(row["started_epoch"]) for row in formal if row["started_epoch"]]
    ends = [int(row["ended_epoch"]) for row in formal if row["ended_epoch"]]
    commit_epoch = int(command("git", "show", "-s", "--format=%ct", head, cwd=repo))
    preflight = json.loads((work / "report/preflight.json").read_text(encoding="utf-8"))
    comparisons = []
    source_map = {"robustsplat": robust, "sls_mlp": spotless}
    for source_entry in preflight.get("sources", []):
        source = source_map[source_entry["method"]]
        for relative, expected_hash in source_entry.get("required_file_sha256", {}).items():
            current_hash = sha256_file(source / relative)
            comparisons.append({
                "method": source_entry["method"], "file": relative,
                "preflight_sha256": expected_hash, "current_sha256": current_hash,
                "match": current_hash == expected_hash,
            })
    result = {
        "schema": "puri-gs-p02a-repository-provenance-v1", "status": "PASS",
        "branch": branch, "head": head, "origin_ru_part": origin,
        "head_commit_epoch": commit_epoch,
        "formal_started_epoch_min": min(starts), "formal_ended_epoch_max": max(ends),
        "head_predates_formal_runs": commit_epoch <= min(starts),
        "head_subject": command("git", "show", "-s", "--format=%s", head, cwd=repo),
        "head_history": command("git", "log", "-8", "--pretty=format:%H|%ct|%s", cwd=repo).splitlines(),
        "p02_runner_sha256": sha256_file(repo / "uncertain-nerf/scripts/p02_server_pipeline.sh"),
        "robust_source_commit": command("git", "rev-parse", "HEAD", cwd=robust),
        "spotless_source_commit": command("git", "rev-parse", "HEAD", cwd=spotless),
        "preflight_current_file_hash_comparison": comparisons,
        "all_preflight_source_hashes_still_match": bool(comparisons) and all(row["match"] for row in comparisons),
        "historical_scope": "formal ledger time is after the verified runtime commit; exact current external source hashes are compared to the frozen preflight",
    }
    if not result["head_predates_formal_runs"] or not result["all_preflight_source_hashes_still_match"]:
        result["status"] = "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("P02A_REPOSITORY_PROVENANCE_" + result["status"])
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
