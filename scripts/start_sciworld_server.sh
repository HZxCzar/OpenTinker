#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-36005}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs}"

mkdir -p "${OUTPUT_ROOT}"

LOG_FILE="${OUTPUT_ROOT}/sciworld_server_${PORT}.log"
PID_FILE="${OUTPUT_ROOT}/sciworld_server_${PORT}.pid"

echo "Starting SciWorld server on ${HOST}:${PORT}"
echo "Log file: ${LOG_FILE}"

nohup sciworld --host "${HOST}" --port "${PORT}" > "${LOG_FILE}" 2>&1 &
echo $! > "${PID_FILE}"

sleep 3
echo "PID: $(cat "${PID_FILE}")"
echo "Health check URL: http://127.0.0.1:${PORT}/"
