from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str | Path, cwd: Path, env: dict[str, str] | None = None) -> str:
    return subprocess.check_output(
        [str(arg) for arg in args], cwd=cwd, env=env, text=True, encoding="utf-8"
    ).strip()


def _commit(repo: Path, message: str, epoch: int) -> str:
    env = os.environ.copy()
    timestamp = f"@{epoch} +0000"
    env["GIT_AUTHOR_DATE"] = timestamp
    env["GIT_COMMITTER_DATE"] = timestamp
    _run("git", "add", ".", cwd=repo, env=env)
    _run("git", "commit", "-m", message, cwd=repo, env=env)
    return _run("git", "rev-parse", "HEAD", cwd=repo)


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True)
    _run("git", "init", cwd=path)
    _run("git", "config", "user.email", "p02a@example.invalid", cwd=path)
    _run("git", "config", "user.name", "P02A Test", cwd=path)


def test_audit_separates_pretraining_runtime_commit_from_later_audit_head(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _run("git", "checkout", "-b", "ru-part", cwd=repo)
    identity_files = [
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
    for relative in identity_files:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen {relative}\n", encoding="utf-8")
    runtime_commit = _commit(repo, "P02 runtime", 1_000_000_000)

    audit_file = repo / "uncertain-nerf/tools/p02a_repository_audit.py"
    audit_file.write_text("P02-A only\n", encoding="utf-8")
    audit_head = _commit(repo, "P02-A audit", 1_000_000_200)
    _run("git", "update-ref", "refs/remotes/origin/ru-part", audit_head, cwd=repo)

    robust = tmp_path / "robust"
    spotless = tmp_path / "spotless"
    required = {}
    for method, source in (("robustsplat", robust), ("sls_mlp", spotless)):
        _init_repo(source)
        tracked = source / "tracked.py"
        tracked.write_text(f"{method}\n", encoding="utf-8")
        _commit(source, method, 999_999_000)
        required[method] = hashlib.sha256(tracked.read_bytes()).hexdigest()

    work = tmp_path / "work"
    (work / "state").mkdir(parents=True)
    (work / "report").mkdir(parents=True)
    with (work / "state/formal_ledger.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["run_id", "started_epoch", "ended_epoch"]
        )
        writer.writeheader()
        writer.writerow(
            {"run_id": "formal", "started_epoch": 1_000_000_100, "ended_epoch": 1_000_000_150}
        )
    (work / "report/preflight.json").write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "method": method,
                        "required_file_sha256": {"tracked.py": required[method]},
                    }
                    for method in ("robustsplat", "sls_mlp")
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "repository_provenance.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/p02a_repository_audit.py"),
            "--repo-root", str(repo),
            "--work-root", str(work),
            "--robust-source", str(robust),
            "--spotless-source", str(spotless),
            "--expected-commit", audit_head,
            "--output", str(output),
        ],
        check=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "PASS"
    assert result["audit_head_predates_formal_runs"] is False
    assert result["p02_runtime_commit"] == runtime_commit
    assert result["p02_runtime_commit_predates_formal_runs"] is True
    assert result["p02_identity_files_unchanged_since_runtime_commit"] is True


def test_server_audit_uses_original_internal_eval_protocol_and_gpu_override() -> None:
    script = (ROOT / "scripts/p02a_server_audit.sh").read_text(encoding="utf-8")
    assert 'android_internal_data="$code_root/data/nerf_robustnerf/robustnerf/android"' in script
    assert 'patio_internal_data="$code_root/data/nerf_on-the-go/patio_high_v1"' in script
    assert '--data-dir "$data" --result-dir "$result" --gpu "$gpu" --data-factor 4' in script
    assert '"$patio_internal_data" ontogo-patio-high' in script
    assert '"$android_internal_data" colmap' in script
    assert "P02A_GPU_OVERRIDE" in script
    assert 'data["selection_policy"] = "explicit_user_override_for_this_p02a_run"' in script
