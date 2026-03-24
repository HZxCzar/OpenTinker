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

"${PYTHON_BIN}" - <<'PY'
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
        f"AgentItemId/{task_name}_train.json",
        f"AgentEval/{task_name}/**",
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
    print(f"Expected train file: {os.path.join(dest_dir, 'AgentItemId', f'{task_name}_train.json')}")
    print(f"Expected eval dir: {os.path.join(dest_dir, 'AgentEval', task_name)}")
PY
