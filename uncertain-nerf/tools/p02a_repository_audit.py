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
    if not starts or not ends:
        raise RuntimeError("formal ledger has no complete timing boundary")
    formal_started_epoch_min = min(starts)
    formal_ended_epoch_max = max(ends)
    audit_commit_epoch = int(command("git", "show", "-s", "--format=%ct", head, cwd=repo))

    # The checked-out HEAD contains the P02-A audit tools, so it is expected to
    # postdate the already completed P02 runs.  Recover the last first-parent
    # repository commit that existed before the first formal run and then prove
    # that every P02 execution-identity file is unchanged between that commit
    # and the current audit commit.
    runtime_commit = command(
        "git", "rev-list", "--first-parent", "--max-count=1",
        f"--before=@{formal_started_epoch_min + 1}", head, cwd=repo,
    )
    if not runtime_commit:
        raise RuntimeError("cannot resolve a repository commit before formal training")
    runtime_commit_epoch = int(command("git", "show", "-s", "--format=%ct", runtime_commit, cwd=repo))
    p02_identity_files = [
        "uncertain-nerf/scripts/p02_server_pipeline.sh",
        "uncertain-nerf/patches/p02_robustsplat_seed_float_timing.patch",
        "uncertain-nerf/patches/p02_robustsplat_pin_dinov2.patch",
        "uncertain-nerf/patches/p02_spotless_seed_float_timing.patch",
        "uncertain-nerf/tools/p02_extract_sls_features.py",
        "uncertain-nerf/tools/p02_evaluate_predictions.py",
        "uncertain-nerf/tools/p02_finalize_report.py",
        "uncertain-nerf/tools/p02_select_gpu.py",
        "uncertain-nerf/tools/p02_validate_common_inputs.py",
    ]
    changed_identity_files = command(
        "git", "diff", "--name-only", f"{runtime_commit}..{head}", "--",
        *p02_identity_files, cwd=repo,
    ).splitlines()
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
        "audit_head_commit_epoch": audit_commit_epoch,
        "audit_head_predates_formal_runs": audit_commit_epoch <= formal_started_epoch_min,
        "formal_started_epoch_min": formal_started_epoch_min,
        "formal_ended_epoch_max": formal_ended_epoch_max,
        "p02_runtime_commit": runtime_commit,
        "p02_runtime_commit_epoch": runtime_commit_epoch,
        "p02_runtime_commit_predates_formal_runs": runtime_commit_epoch <= formal_started_epoch_min,
        "p02_identity_files_checked": p02_identity_files,
        "p02_identity_files_changed_after_runtime_commit": changed_identity_files,
        "p02_identity_files_unchanged_since_runtime_commit": not changed_identity_files,
        "head_subject": command("git", "show", "-s", "--format=%s", head, cwd=repo),
        "p02_runtime_commit_subject": command("git", "show", "-s", "--format=%s", runtime_commit, cwd=repo),
        "head_history": command("git", "log", "-8", "--pretty=format:%H|%ct|%s", cwd=repo).splitlines(),
        "p02_runner_sha256": sha256_file(repo / "uncertain-nerf/scripts/p02_server_pipeline.sh"),
        "robust_source_commit": command("git", "rev-parse", "HEAD", cwd=robust),
        "spotless_source_commit": command("git", "rev-parse", "HEAD", cwd=spotless),
        "preflight_current_file_hash_comparison": comparisons,
        "all_preflight_source_hashes_still_match": bool(comparisons) and all(row["match"] for row in comparisons),
        "historical_scope": "the audit HEAD may postdate training; the last first-parent commit before formal start is accepted only when all P02 identity files stayed unchanged and exact current external source hashes match the frozen preflight",
    }
    if (
        not result["p02_runtime_commit_predates_formal_runs"]
        or not result["p02_identity_files_unchanged_since_runtime_commit"]
        or not result["all_preflight_source_hashes_still_match"]
    ):
        result["status"] = "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("P02A_REPOSITORY_PROVENANCE_" + result["status"])
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
