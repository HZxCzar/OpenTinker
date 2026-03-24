#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/inspire/hdd/project/qproject-fundationmodel/public/wxxu/OpenTinker

MODEL_PATH=/inspire/hdd/project/qproject-fundationmodel/public/wxxu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
DATA_ROOT=/inspire/hdd/project/qproject-fundationmodel/public/wxxu/OpenTinker/data
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs}"
RUN_NAME="${RUN_NAME:-sciworld_$(date +%Y%m%d_%H%M%S)}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:36005}"

NUM_GPUS="${NUM_GPUS:-8}"
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

WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_PROJECT="${WANDB_PROJECT:-co-evolve}"
WANDB_ENTITY="${WANDB_ENTITY:-hz-czar-uiuc}"

TRAIN_FILE_OLD="${DATA_ROOT}/AgentItemId/sciworld_train.json"
TRAIN_FILE_NEW="${DATA_ROOT}/train/sciworld_train.json"

if [[ -f "${TRAIN_FILE_OLD}" ]]; then
  TRAIN_FILE="${TRAIN_FILE_OLD}"
elif [[ -f "${TRAIN_FILE_NEW}" ]]; then
  TRAIN_FILE="${TRAIN_FILE_NEW}"
else
  TRAIN_FILE="${TRAIN_FILE_OLD}"
fi

if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is required."
  echo "Example:"
  echo "  MODEL_PATH=/path/to/Qwen2.5-7B-Instruct bash scripts/train_sciworld.sh"
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH does not exist: ${MODEL_PATH}"
  exit 1
fi

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "Training file does not exist."
  echo "Checked:"
  echo "  ${TRAIN_FILE_OLD}"
  echo "  ${TRAIN_FILE_NEW}"
  exit 1
fi

RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_FILE="${RUN_DIR}/train.log"

mkdir -p "${RUN_DIR}"
mkdir -p "${ROOT}/.cache/huggingface" "${ROOT}/.cache/wandb"

export XDG_CACHE_HOME="${ROOT}/.cache"
export HF_HOME="${ROOT}/.cache/huggingface"
export TRANSFORMERS_CACHE="${ROOT}/.cache/huggingface"
export WANDB_DIR="${ROOT}/.cache/wandb"
export WANDB_MODE="${WANDB_MODE}"
export WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_ENTITY="${WANDB_ENTITY}"

export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

cd "${ROOT}/AgentGym-RL"

echo "Run name: ${RUN_NAME}"
echo "Server URL: ${SERVER_URL}"
echo "Train file: ${TRAIN_FILE}"
echo "Model path: ${MODEL_PATH}"
echo "Log file: ${LOG_FILE}"
echo "Checkpoint root: ${RUN_DIR}"
echo "Wandb mode/project/entity: ${WANDB_MODE} / ${WANDB_PROJECT} / ${WANDB_ENTITY}"
echo "Checkpoints will be saved under:"
echo "  ${RUN_DIR}/global_step_*/actor"
echo "  ${RUN_DIR}/global_step_*/critic"
echo "  ${RUN_DIR}/latest_checkpointed_iteration.txt"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
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
  trainer.project_name="${WANDB_PROJECT}" \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" | tee "${LOG_FILE}"
