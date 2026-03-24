#!/usr/bin/env bash
set -euo pipefail

# Launch randomized evaluation simulation + CollectDemos in two terminals (tmux windows):
# - Window 0 (sim): distrobox eval container running /entrypoint.sh with randomized config
# - Window 1 (policy): host terminal running aic_model with CollectDemos
#
# Example:
#   ./scripts/run_randomized_collect_demos.sh --num-trials 40 --dataset-name aic_cable_r512 --seed 7

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NUM_TRIALS=20
SEED=0
TASK_MODE="mixed"        # mixed | sfp_only | sc_only
SESSION_NAME="aic_collect"
DATASET_NAME="aic_cable_train_r512"
POLICY_CLASS="aic_lewm_policies.ros.CollectDemos"
POLICY_DELAY_SEC=15
GROUND_TRUTH="true"

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --num-trials N          Number of randomized trials in generated engine config (default: ${NUM_TRIALS})
  --seed N                RNG seed (default: ${SEED})
  --task-mode MODE        mixed | sfp_only | sc_only (default: ${TASK_MODE})
  --dataset-name NAME     AIC_DATASET_NAME for CollectDemos (default: ${DATASET_NAME})
  --session-name NAME     tmux session name (default: ${SESSION_NAME})
  --policy-delay-sec N    Delay before policy starts (default: ${POLICY_DELAY_SEC})
  --ground-truth BOOL     true/false for sim ground truth TF (default: ${GROUND_TRUTH})
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-trials) NUM_TRIALS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --task-mode) TASK_MODE="$2"; shift 2 ;;
    --dataset-name) DATASET_NAME="$2"; shift 2 ;;
    --session-name) SESSION_NAME="$2"; shift 2 ;;
    --policy-delay-sec) POLICY_DELAY_SEC="$2"; shift 2 ;;
    --ground-truth) GROUND_TRUTH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required. Install tmux and retry."
  exit 1
fi

if ! command -v distrobox >/dev/null 2>&1; then
  echo "distrobox is required to run the eval container."
  exit 1
fi

RANDOM_CFG="/tmp/aic_random_engine_${SEED}_$(date +%Y%m%d_%H%M%S).yaml"
SAMPLE_CFG="${AIC_ROOT}/aic_engine/config/sample_config.yaml"

python3 "${SCRIPT_DIR}/generate_random_engine_config.py" \
  --sample-config "${SAMPLE_CFG}" \
  --output-config "${RANDOM_CFG}" \
  --num-trials "${NUM_TRIALS}" \
  --seed "${SEED}" \
  --task-mode "${TASK_MODE}"

SIM_CMD="distrobox enter -r aic_eval -- /entrypoint.sh \
ground_truth:=${GROUND_TRUTH} \
start_aic_engine:=true \
shutdown_on_aic_engine_exit:=true \
aic_engine_config_file:=${RANDOM_CFG}"

POLICY_CMD="cd '${AIC_ROOT}' && \
export AIC_DATASET_NAME='${DATASET_NAME}' && \
export AIC_CAPTURE_PIXEL_MAX='512' && \
sleep '${POLICY_DELAY_SEC}' && \
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=${POLICY_CLASS}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session '${SESSION_NAME}' already exists."
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
  echo "Or kill it with: tmux kill-session -t ${SESSION_NAME}"
  exit 1
fi

tmux new-session -d -s "${SESSION_NAME}" -n sim "${SIM_CMD}"
tmux new-window -t "${SESSION_NAME}:1" -n policy "${POLICY_CMD}"

echo "Started tmux session '${SESSION_NAME}'"
echo "  Sim window:    ${SIM_CMD}"
echo "  Policy window: ${POLICY_CMD}"
echo "  Random config: ${RANDOM_CFG}"
echo
echo "Attach with:"
echo "  tmux attach -t ${SESSION_NAME}"

tmux attach -t "${SESSION_NAME}"
