#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEST_DIR="${1:-${DEST_DIR:-${ROOT}/data/AgentGym-RL-Data-ID}}"
TASK_NAME="${TASK_NAME:-sciworld}"
PYTHON_BIN="${PYTHON_BIN:-python}"
REPO_ID="${REPO_ID:-AgentGym/AgentGym-RL-Data-ID}"

mkdir -p "${DEST_DIR}"

echo "Downloading dataset repo: ${REPO_ID}"
echo "Target directory: ${DEST_DIR}"
echo "Task selection: ${TASK_NAME}"

REPO_ID="${REPO_ID}" DEST_DIR="${DEST_DIR}" TASK_NAME="${TASK_NAME}" "${PYTHON_BIN}" - <<'PY'
import os
import sys

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:
    raise SystemExit(
        "huggingface_hub is not installed in the current Python environment. "
        "Install it first with: pip install huggingface_hub"
    ) from exc

repo_id = os.environ["REPO_ID"]
dest_dir = os.environ["DEST_DIR"]
task_name = os.environ["TASK_NAME"]
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

allow_patterns = None
if task_name.lower() != "all":
    allow_patterns = [
        f"train/{task_name}_train.json",
        f"eval/{task_name}_test.json",
    ]

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=dest_dir,
    local_dir_use_symlinks=False,
    allow_patterns=allow_patterns,
    token=token,
)

print("\nDownload completed.")
if allow_patterns is None:
    print(f"Full dataset saved under: {dest_dir}")
else:
    train_src = os.path.join(dest_dir, "train", f"{task_name}_train.json")
    eval_src = os.path.join(dest_dir, "eval", f"{task_name}_test.json")

    compat_train_dir = os.path.join(dest_dir, "AgentItemId")
    compat_eval_dir = os.path.join(dest_dir, "AgentEval", task_name)
    os.makedirs(compat_train_dir, exist_ok=True)
    os.makedirs(compat_eval_dir, exist_ok=True)

    if os.path.exists(train_src):
        import shutil
        shutil.copy2(train_src, os.path.join(compat_train_dir, f"{task_name}_train.json"))
    if os.path.exists(eval_src):
        import shutil
        shutil.copy2(eval_src, os.path.join(compat_eval_dir, f"{task_name}_test.json"))

    print(f"Downloaded train file: {train_src}")
    print(f"Downloaded eval file: {eval_src}")
    print(f"Compatibility train file: {os.path.join(dest_dir, 'AgentItemId', f'{task_name}_train.json')}")
    print(f"Compatibility eval dir: {os.path.join(dest_dir, 'AgentEval', task_name)}")
PY
