#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_sci.sh base
#   bash run_sci.sh wmc-erc
#   bash run_sci.sh eval /path/to/ckpt_or_model_dir

MODE="${1:-}"
USER_MODEL_PATH="${2:- /inspire/hdd/project/qproject-fundationmodel/public/wxxu/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 }"

if [[ -z "${MODE}" ]]; then
  echo "Usage: bash run_sci.sh {base|wmc-erc|eval} [model_path_for_eval]"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${HF_HOME:-}" ]]; then
  export HF_HOME="${ROOT_DIR}/.cache/huggingface"
fi

# Offline runtime guardrails
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_ENTITY="${WANDB_ENTITY:-hz-czar-uiuc}"
export WANDB_PROJECT="${WANDB_PROJECT:-co-evolve}"
export WANDB_DISABLED="${WANDB_DISABLED:-false}"

# Defaults (edit here if your machine setup differs)
MODEL_REPO_CACHE_DIR="${HF_HOME}/hub/models--Qwen--Qwen3-8B/snapshots"
SCHEDULER_PORT="${SCHEDULER_PORT:-8780}"
TRAIN_ENV_PORT="${TRAIN_ENV_PORT:-8092}"
TRAIN_ENV_SHARDS="${TRAIN_ENV_SHARDS:-8}"
DEV_ENV_PORT="${DEV_ENV_PORT:-8093}"
TEST_ENV_PORT="${TEST_ENV_PORT:-8094}"
ENV_HOST="${ENV_HOST:-127.0.0.1}"
AVAILABLE_GPUS="${AVAILABLE_GPUS:-[0,1,2,3,4,5,6,7]}"
DEV_THREAD_BASE="${DEV_THREAD_BASE:-40000}"
TEST_THREAD_BASE="${TEST_THREAD_BASE:-50000}"

timestamp="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${ROOT_DIR}/logs/${timestamp}_${MODE}"
mkdir -p "${LOG_ROOT}"

SCHED_PID=""
ENV_PID=""
DEV_ENV_PID=""
TEST_ENV_PID=""
LAST_BG_PID=""

spawn_bg() {
  if command -v setsid >/dev/null 2>&1; then
    setsid "$@" &
  else
    "$@" &
  fi
  LAST_BG_PID="$!"
}

terminate_process_group() {
  local pid="${1:-}"
  if [[ -z "${pid}" ]]; then
    return 0
  fi

  if kill -0 "${pid}" 2>/dev/null; then
    kill -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
  fi
}

force_terminate_process_group() {
  local pid="${1:-}"
  if [[ -z "${pid}" ]]; then
    return 0
  fi

  if kill -0 "${pid}" 2>/dev/null; then
    kill -9 -- "-${pid}" 2>/dev/null || kill -9 "${pid}" 2>/dev/null || true
  fi
}

find_sciworld_shard_pids() {
  local split_name="$1"
  local port="$2"
  ps -eo pid=,args= | awk -v split_name="${split_name}" -v port="${port}" '
    $0 ~ /opentinker\/environment\/sciworld\/sciworld_server\.py/ &&
    $0 ~ ("--port " port "([[:space:]]|$)") &&
    $0 ~ ("--split " split_name "([[:space:]]|$)") &&
    $0 ~ /--shards 1([[:space:]]|$)/ {
      print $1
    }
  '
}

cleanup_sciworld_shards() {
  local split_name="$1"
  local start_port="$2"
  local shard_count="$3"
  local -a pids=()
  local -A seen=()

  for ((i=0; i<shard_count; i++)); do
    local port=$((start_port + i))
    while IFS= read -r pid; do
      if [[ -n "${pid}" && -z "${seen[${pid}]:-}" ]]; then
        pids+=("${pid}")
        seen["${pid}"]=1
      fi
    done < <(find_sciworld_shard_pids "${split_name}" "${port}")
  done

  if (( ${#pids[@]} == 0 )); then
    return 0
  fi

  echo "[cleanup] Found stale ScienceWorld ${split_name} shard(s): ${pids[*]}"
  kill "${pids[@]}" 2>/dev/null || true
  sleep 2

  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -9 "${pid}" 2>/dev/null || true
    fi
  done
}

cleanup() {
  local pids=("${SCHED_PID}" "${ENV_PID}" "${DEV_ENV_PID}" "${TEST_ENV_PID}")
  for pid in "${pids[@]}"; do
    terminate_process_group "${pid}"
  done

  sleep 1

  for pid in "${pids[@]}"; do
    force_terminate_process_group "${pid}"
  done

  cleanup_sciworld_shards train "${TRAIN_ENV_PORT}" "${TRAIN_ENV_SHARDS}"
  cleanup_sciworld_shards dev "${DEV_ENV_PORT}" 1
  cleanup_sciworld_shards test "${TEST_ENV_PORT}" 1
}
trap cleanup EXIT INT TERM

resolve_model_path() {
  if [[ -n "${USER_MODEL_PATH}" ]]; then
    echo "${USER_MODEL_PATH}"
    return 0
  fi

  if [[ -d "${MODEL_REPO_CACHE_DIR}" ]]; then
    local latest
    latest="$(ls -1t "${MODEL_REPO_CACHE_DIR}" 2>/dev/null | head -n 1 || true)"
    if [[ -n "${latest}" ]]; then
      echo "${MODEL_REPO_CACHE_DIR}/${latest}"
      return 0
    fi
  fi

  echo ""
}

MODEL_PATH="$(resolve_model_path)"

if [[ "${MODE}" != "eval" ]]; then
  if [[ -z "${MODEL_PATH}" ]]; then
    echo "Cannot find model path."
    echo "Expected cache under: ${MODEL_REPO_CACHE_DIR}"
    echo "Please set HF_HOME correctly and make sure model is downloaded."
    exit 1
  fi
fi

echo "============================================================"
echo "Mode: ${MODE}"
echo "Root: ${ROOT_DIR}"
echo "HF_HOME: ${HF_HOME}"
echo "Model Path: ${MODEL_PATH:-<not set>}"
echo "Log Dir: ${LOG_ROOT}"
echo "============================================================"

start_scheduler() {
  echo "[1/3] Starting scheduler..."
  spawn_bg python opentinker/scheduler/launch_scheduler_kill.py \
    available_gpus="${AVAILABLE_GPUS}" \
    scheduler_port="${SCHEDULER_PORT}" \
    > "${LOG_ROOT}/scheduler.log" 2>&1
  SCHED_PID="${LAST_BG_PID}"
  sleep 5
}

start_train_env() {
  echo "[2/3] Starting ScienceWorld train env..."
  cleanup_sciworld_shards train "${TRAIN_ENV_PORT}" "${TRAIN_ENV_SHARDS}"
  spawn_bg python -m opentinker.environment.sciworld.sciworld_server \
    --host 0.0.0.0 \
    --port "${TRAIN_ENV_PORT}" \
    --shards "${TRAIN_ENV_SHARDS}" \
    --split train \
    > "${LOG_ROOT}/env_train.log" 2>&1
  ENV_PID="${LAST_BG_PID}"
  sleep 5
}

run_base() {
  echo "[3/3] Starting BASE training..."
  python opentinker/client/sciworld_rl.py \
    tokenizer_path="${MODEL_PATH}" \
    project_name="${WANDB_PROJECT}" \
    scheduler_url="http://${ENV_HOST}:${SCHEDULER_PORT}" \
    interaction.config.env_host="${ENV_HOST}" \
    interaction.config.env_port="${TRAIN_ENV_PORT}" \
    logger_backends='["console"]' \
    enable_tracing=false \
    | tee "${LOG_ROOT}/train_base.log"
}

run_wmc_erc() {
  echo "[3/3] Starting WMC-ERC training..."
  python opentinker/client/sciworld_rl.py \
    --config-name sciworld_wmc_erc_param \
    tokenizer_path="${MODEL_PATH}" \
    project_name="${WANDB_PROJECT}" \
    scheduler_url="http://${ENV_HOST}:${SCHEDULER_PORT}" \
    interaction.config.env_host="${ENV_HOST}" \
    interaction.config.env_port="${TRAIN_ENV_PORT}" \
    logger_backends='["console"]' \
    enable_tracing=false \
    | tee "${LOG_ROOT}/train_wmc_erc.log"
}

start_eval_envs() {
  echo "[1/2] Starting ScienceWorld dev env..."
  cleanup_sciworld_shards dev "${DEV_ENV_PORT}" 1
  spawn_bg python -m opentinker.environment.sciworld.sciworld_server \
    --host 0.0.0.0 \
    --port "${DEV_ENV_PORT}" \
    --split dev \
    --shards 1 \
    --thread-base "${DEV_THREAD_BASE}" \
    > "${LOG_ROOT}/env_dev.log" 2>&1
  DEV_ENV_PID="${LAST_BG_PID}"

  echo "[2/2] Starting ScienceWorld test env..."
  cleanup_sciworld_shards test "${TEST_ENV_PORT}" 1
  spawn_bg python -m opentinker.environment.sciworld.sciworld_server \
    --host 0.0.0.0 \
    --port "${TEST_ENV_PORT}" \
    --split test \
    --shards 1 \
    --thread-base "${TEST_THREAD_BASE}" \
    > "${LOG_ROOT}/env_test.log" 2>&1
  TEST_ENV_PID="${LAST_BG_PID}"
  sleep 5
}

run_eval() {
  if [[ -z "${MODEL_PATH}" ]]; then
    echo "Eval requires model path."
    echo "Usage: bash run_sci.sh eval /path/to/ckpt_or_model_dir"
    echo "Or download model under HF_HOME and let script auto-detect snapshot."
    exit 1
  fi

  python opentinker/client/sciworld_eval.py \
    --model-path "${MODEL_PATH}" \
    --tokenizer-path "${MODEL_PATH}" \
    --split both \
    --dev-env-endpoint "http://${ENV_HOST}:${DEV_ENV_PORT}" \
    --test-env-endpoint "http://${ENV_HOST}:${TEST_ENV_PORT}" \
    --dev-output-jsonl "${LOG_ROOT}/sciworld_eval_dev.jsonl" \
    --test-output-jsonl "${LOG_ROOT}/sciworld_eval_test.jsonl" \
    --summary-json "${LOG_ROOT}/sciworld_eval_both.summary.json" \
    | tee "${LOG_ROOT}/eval.log"
}

case "${MODE}" in
  base)
    start_scheduler
    start_train_env
    run_base
    ;;
  wmc-erc)
    start_scheduler
    start_train_env
    run_wmc_erc
    ;;
  eval)
    start_eval_envs
    run_eval
    ;;
  *)
    echo "Unknown mode: ${MODE}"
    echo "Usage: bash run_sci.sh {base|wmc-erc|eval} [model_path_for_eval]"
    exit 1
    ;;
esac

echo "Done. Logs saved to: ${LOG_ROOT}"

echo "Starting GPU occupy program: python test.py"
python test.py | tee "${LOG_ROOT}/post_test.log"
