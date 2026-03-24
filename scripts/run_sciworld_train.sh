#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
TRAIN_ENV="${TRAIN_ENV:-${ROOT}/envs/agentgym-rl}"
SCIWORLD_ENV="${SCIWORLD_ENV:-${ROOT}/envs/agentenv-sciworld}"

MODEL_PATH="${MODEL_PATH:-}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data/AgentGym-RL-Data-ID}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs}"
RUN_NAME="${RUN_NAME:-sciworld_run1}"

SCIWORLD_HOST="${SCIWORLD_HOST:-0.0.0.0}"
SCIWORLD_PORT="${SCIWORLD_PORT:-36005}"
START_SERVER="${START_SERVER:-1}"

NUM_GPUS="${NUM_GPUS:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
ROLLOUT_SAMPLE_NUM="${ROLLOUT_SAMPLE_NUM:-8}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-10}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
MAX_ROUNDS="${MAX_ROUNDS:-20}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TOKENS_PER_ROUND="${MAX_TOKENS_PER_ROUND:-200}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"

KL_COEF="${KL_COEF:-0.001}"
POLICY_LR="${POLICY_LR:-1e-6}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
PPO_INNER_EPOCHS="${PPO_INNER_EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:-25}"

SERVER_URL="http://127.0.0.1:${SCIWORLD_PORT}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
SERVER_LOG="${RUN_DIR}/sciworld_server.log"
SERVER_PID_FILE="${RUN_DIR}/sciworld_server.pid"
TRAIN_LOG="${RUN_DIR}/train.log"
TRAIN_FILE="${DATA_ROOT}/AgentItemId/sciworld_train.json"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is required."
  echo "Example:"
  echo "  MODEL_PATH=/mnt/share/OpenTinker/models/Qwen2.5-7B-Instruct bash scripts/run_sciworld_train.sh"
  exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Cannot find conda activation script: ${CONDA_SH}"
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH does not exist: ${MODEL_PATH}"
  exit 1
fi

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "Training file does not exist: ${TRAIN_FILE}"
  echo "Run scripts/download_agentgym_rl_data.sh first, or set DATA_ROOT correctly."
  exit 1
fi

mkdir -p "${RUN_DIR}"
mkdir -p "${ROOT}/.cache/huggingface" "${ROOT}/.cache/wandb"

export XDG_CACHE_HOME="${ROOT}/.cache"
export HF_HOME="${ROOT}/.cache/huggingface"
export TRANSFORMERS_CACHE="${ROOT}/.cache/huggingface"
export WANDB_DIR="${ROOT}/.cache/wandb"

export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

source "${CONDA_SH}"

if [[ "${START_SERVER}" == "1" ]]; then
  if curl -fsS "${SERVER_URL}/" >/dev/null 2>&1; then
    echo "SciWorld server is already reachable at ${SERVER_URL}"
  else
    echo "Starting SciWorld server at ${SERVER_URL}"
    conda activate "${SCIWORLD_ENV}"
    nohup sciworld --host "${SCIWORLD_HOST}" --port "${SCIWORLD_PORT}" >"${SERVER_LOG}" 2>&1 &
    echo $! > "${SERVER_PID_FILE}"
    sleep 5

    if ! curl -fsS "${SERVER_URL}/" >/dev/null 2>&1; then
      echo "SciWorld server failed to start. Check log: ${SERVER_LOG}"
      exit 1
    fi
  fi
fi

conda activate "${TRAIN_ENV}"
cd "${ROOT}/AgentGym-RL"

echo "Training run directory: ${RUN_DIR}"
echo "Train log: ${TRAIN_LOG}"
echo "Checkpoint root: ${RUN_DIR}"
echo "Checkpoints will appear as:"
echo "  ${RUN_DIR}/global_step_*/actor"
echo "  ${RUN_DIR}/global_step_*/critic"
echo "  ${RUN_DIR}/latest_checkpointed_iteration.txt"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE="${WANDB_MODE}" \
python -m verl.agent_trainer.main_ppo \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node="${NUM_GPUS}" \
  algorithm.adv_estimator=grpo \
  algorithm.rounds_ctrl.type=fixed \
  algorithm.rounds_ctrl.rounds="${MAX_ROUNDS}" \
  data.train_file="${TRAIN_FILE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  actor_rollout_ref.agentgym.task_name=sciworld \
  actor_rollout_ref.agentgym.env_addr="${SERVER_URL}" \
  actor_rollout_ref.agentgym.timeout=600 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="${KL_COEF}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_SAMPLE_NUM}" \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.max_tokens="${MAX_TOKENS_PER_ROUND}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.actor.ppo_epochs="${PPO_INNER_EPOCHS}" \
  actor_rollout_ref.actor.optim.lr="${POLICY_LR}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.rollout.rollout_log_dir="${RUN_DIR}/executer_logs" \
  algorithm.kl_ctrl.kl_coef="${KL_COEF}" \
  trainer.default_local_dir="${RUN_DIR}" \
  trainer.project_name=sciworld \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" | tee "${TRAIN_LOG}"
