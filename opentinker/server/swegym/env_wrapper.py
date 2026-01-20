from __future__ import annotations

import os
import stat
import time
from pathlib import Path
from shutil import rmtree
from typing import Any, Mapping

try:
    from sweagent.run.batch_instances import BatchInstance
    from sweagent.environment.swe_env import EnvironmentConfig
    from sweagent.environment.repo import GithubRepoConfig
    from sweagent.environment.conda import CondaDeploymentConfig
    from sweagent.agent.problem_statement import TextProblemStatement

    SWEAGENT_AVAILABLE = True
except ImportError:  # pragma: no cover - handled at runtime when sweagent isn't installed
    SWEAGENT_AVAILABLE = False


def batch_instance_from_dict(
    d: Mapping[str, Any],
    *,
    deployment_cfg: CondaDeploymentConfig | None = None,
) -> "BatchInstance":
    """Build a SWE-agent BatchInstance from a dict.

    Required keys in `d`:
      - instance_id, repo, base_commit, problem_statement
    Optional keys:
      - test_patch, eval_script, FAIL_TO_PASS, PASS_TO_PASS
    """
    if not SWEAGENT_AVAILABLE:
        raise ImportError(
            "sweagent is not installed. Install SWE-agent and its dependencies to use swegym."
        )

    base_commit = str(d.get("base_commit", "HEAD"))
    repo = str(d.get("repo", "")).strip()
    github_url = f"https://github.com/{repo}" if repo else ""

    repo_cfg = GithubRepoConfig(github_url=github_url, base_commit=base_commit)

    if deployment_cfg is None:
        deployment_cfg = CondaDeploymentConfig()  # default python=3.11

    env_cfg = EnvironmentConfig(
        deployment=deployment_cfg.model_copy(deep=True),
        repo=repo_cfg,
    )

    ps = TextProblemStatement(
        text=str(d.get("problem_statement", "")),
        id=str(d.get("instance_id", "")),
        extra_fields={"base_commit": base_commit},
    )

    return BatchInstance(
        env=env_cfg,
        problem_statement=ps,
        test_patch=d.get("test_patch"),
        eval_script=d.get("eval_script"),
        FAIL_TO_PASS=[str(x) for x in d.get("FAIL_TO_PASS", [])],
        PASS_TO_PASS=[str(x) for x in d.get("PASS_TO_PASS", [])],
    )


def _is_under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def remove_runtime_root(
    runtime_root: Path, traj_root: Path, retries: int = 3, delay: float = 0.2
):
    """Remove a runtime root safely, guarding against path traversal."""
    if not runtime_root.exists() or not runtime_root.is_dir():
        return
    if not _is_under(runtime_root, traj_root):
        raise RuntimeError(
            f"Refusing to delete {runtime_root}; not under {traj_root}"
        )

    for i in range(retries):
        try:
            rmtree(runtime_root)
            return
        except Exception:
            for root, dirs, files in os.walk(runtime_root, topdown=False):
                for name in files + dirs:
                    path = Path(root, name)
                    try:
                        path.chmod(stat.S_IRWXU)
                    except Exception:
                        pass
            if i == retries - 1:
                raise
            time.sleep(delay)
