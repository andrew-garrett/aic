#!/usr/bin/env bash
set -euo pipefail

# Launch randomized evaluation simulation + CollectDemos in tmux (sim + policy windows).
#
# When --num-trials > --batch-size (default 5), the full randomized config is split into
# multiple YAML files; each batch runs in a fresh tmux session (new sim + policy pair).
# Between batches, this script removes ~/aic_results before launching the next tmux session.
#
# Example:
#   ./scripts/run_randomized_collect_demos.sh --num-trials 40 --dataset-name aic_cable_r512 --seed 7
#   AIC_GCS_UPLOAD_URI=gs://my-bucket/aic/ ./scripts/run_randomized_collect_demos.sh --dataset-name foo
#   ./scripts/run_randomized_collect_demos.sh --gcs-upload gs://my-bucket/aic/full_name.h5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NUM_TRIALS=20
SEED=0
TASK_MODE="mixed"        # mixed | sfp_only | sc_only
SESSION_NAME="aic_collect"
DATASET_NAME="aic_cable_train_r512"
POLICY_CLASS="aic_lewm_policies.ros.CollectDemos"
POLICY_DELAY_SEC=10
GROUND_TRUTH="true"
BATCH_SIZE=5
LOG_ROOT=""
# CollectDemos writes "${STABLEWM_HOME}/${AIC_DATASET_NAME}.h5" (default ~/.stable-wm).
STABLEWM_HOME="${STABLEWM_HOME:-${HOME}/.stable-wm}"
GCS_UPLOAD_URI="${AIC_GCS_UPLOAD_URI:-}"

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --num-trials N          Number of randomized trials in generated engine config (default: ${NUM_TRIALS})
  --seed N                RNG seed (default: ${SEED})
  --task-mode MODE        mixed | sfp_only | sc_only (default: ${TASK_MODE})
  --dataset-name NAME     AIC_DATASET_NAME for CollectDemos (default: ${DATASET_NAME})
  --session-name NAME     Base tmux session name (default: ${SESSION_NAME}); batches append _bN_<stamp>
  --policy-delay-sec N    Delay before policy starts (default: ${POLICY_DELAY_SEC})
  --ground-truth BOOL     true/false for sim ground truth TF (default: ${GROUND_TRUTH})
  --batch-size N          Split trials into YAML batches of N (default: ${BATCH_SIZE})
  --log-root DIR          Parent dir for per-session logs (default: /tmp/aic_collect_logs_<timestamp>)
  --gcs-upload URI        After run, gsutil cp the dataset .h5 to GCS. URI may be gs://bucket/prefix/
                          (uploads as prefix/<dataset>_<run_stamp>.h5) or gs://bucket/path/file.h5
  --no-attach             Do not tmux attach (wait for each sim to exit; cleans up session)
  -h, --help              Show this help

Environment:
  STABLEWM_HOME           HDF5 directory (default: ${HOME}/.stable-wm), must match CollectDemos
  AIC_GCS_UPLOAD_URI      Same as --gcs-upload if the flag is not passed
EOF
}

ATTACH_LAST=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-trials) NUM_TRIALS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --task-mode) TASK_MODE="$2"; shift 2 ;;
    --dataset-name) DATASET_NAME="$2"; shift 2 ;;
    --session-name) SESSION_NAME="$2"; shift 2 ;;
    --policy-delay-sec) POLICY_DELAY_SEC="$2"; shift 2 ;;
    --ground-truth) GROUND_TRUTH="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --gcs-upload) GCS_UPLOAD_URI="$2"; shift 2 ;;
    --no-attach) ATTACH_LAST=0; shift ;;
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

if [[ "${NUM_TRIALS}" -lt 1 ]]; then
  echo "--num-trials must be >= 1"
  exit 1
fi

if [[ "${BATCH_SIZE}" -lt 1 ]]; then
  echo "--batch-size must be >= 1"
  exit 1
fi

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RANDOM_CFG="/tmp/aic_random_engine_${SEED}_${RUN_STAMP}.yaml"
SAMPLE_CFG="${AIC_ROOT}/aic_engine/config/sample_config.yaml"
BATCH_DIR="/tmp/aic_engine_batches_${SEED}_${RUN_STAMP}"
if [[ -z "${LOG_ROOT}" ]]; then
  LOG_ROOT="/tmp/aic_collect_logs_${RUN_STAMP}"
fi
mkdir -p "${LOG_ROOT}"

python3 "${SCRIPT_DIR}/generate_random_engine_config.py" \
  --sample-config "${SAMPLE_CFG}" \
  --output-config "${RANDOM_CFG}" \
  --num-trials "${NUM_TRIALS}" \
  --seed "${SEED}" \
  --task-mode "${TASK_MODE}"

python3 "${SCRIPT_DIR}/split_engine_config_batches.py" \
  --input "${RANDOM_CFG}" \
  --out-dir "${BATCH_DIR}" \
  --batch-size "${BATCH_SIZE}"

shopt -s nullglob
mapfile -t BATCH_FILES < <(printf '%s\n' "${BATCH_DIR}"/batch_*.yaml | sort)
shopt -u nullglob
if [[ "${#BATCH_FILES[@]}" -eq 0 ]]; then
  echo "No batch_*.yaml files in ${BATCH_DIR}"
  exit 1
fi

TOTAL_BATCHES="${#BATCH_FILES[@]}"

POLICY_CMD="cd '${AIC_ROOT}' && \
export STABLEWM_HOME='${STABLEWM_HOME}' && \
export AIC_DATASET_NAME='${DATASET_NAME}' && \
export AIC_CAPTURE_PIXEL_MAX='512' && \
sleep '${POLICY_DELAY_SEC}' && \
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=${POLICY_CLASS}"

wait_sim_then_cleanup() {
  local session="$1"
  local log_dir="$2"
  # Window 0 = sim: wait until its pane exits (aic_engine / entrypoint finished).
  while tmux has-session -t "${session}" 2>/dev/null; do
    local dead
    dead="$(tmux list-panes -t "${session}:0" -F '#{pane_dead}' 2>/dev/null || echo 1)"
    if [[ "${dead}" == "1" ]]; then
      break
    fi
    sleep 2
  done
  capture_session_logs "${session}" "${log_dir}"
  tmux kill-session -t "${session}" 2>/dev/null || true
}

start_session_logging() {
  local session="$1"
  local log_dir="$2"
  local cfg="$3"
  local sim_cmd="$4"
  local policy_cmd="$5"

  mkdir -p "${log_dir}"
  cp "${cfg}" "${log_dir}/engine_config.yaml"
  cat > "${log_dir}/meta.txt" <<EOF
run_stamp=${RUN_STAMP}
session=${session}
engine_config=${cfg}
dataset=${DATASET_NAME}
seed=${SEED}
task_mode=${TASK_MODE}
ground_truth=${GROUND_TRUTH}
policy_class=${POLICY_CLASS}
policy_delay_sec=${POLICY_DELAY_SEC}
EOF
  printf '%s\n' "${sim_cmd}" > "${log_dir}/sim_command.txt"
  printf '%s\n' "${policy_cmd}" > "${log_dir}/policy_command.txt"

  # Stream full pane output to files in real time.
  tmux pipe-pane -o -t "${session}:0" "cat >> \"${log_dir}/sim.log\""
  tmux pipe-pane -o -t "${session}:1" "cat >> \"${log_dir}/policy.log\""
}

capture_session_logs() {
  local session="$1"
  local log_dir="$2"
  # Final pane snapshots (useful if a process exited before pipe flush or for quick tail review).
  tmux capture-pane -p -S - -t "${session}:0" > "${log_dir}/sim_capture.txt" 2>/dev/null || true
  tmux capture-pane -p -S - -t "${session}:1" > "${log_dir}/policy_capture.txt" 2>/dev/null || true
}

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session '${SESSION_NAME}' already exists."
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
  echo "Or kill it with: tmux kill-session -t ${SESSION_NAME}"
  exit 1
fi

BATCH_NUM=0
for CFG in "${BATCH_FILES[@]}"; do
  BATCH_NUM=$((BATCH_NUM + 1))
  SESSION_BATCH="${SESSION_NAME}_b${BATCH_NUM}_${RUN_STAMP}"
  LOG_DIR="${LOG_ROOT}/batch_${BATCH_NUM}"

  if tmux has-session -t "${SESSION_BATCH}" 2>/dev/null; then
    echo "tmux session '${SESSION_BATCH}' already exists."
    echo "Attach with: tmux attach -t ${SESSION_BATCH}"
    echo "Or kill it with: tmux kill-session -t ${SESSION_BATCH}"
    exit 1
  fi

  SIM_CMD="distrobox enter -r aic_eval -- /entrypoint.sh \
ground_truth:=${GROUND_TRUTH} \
start_aic_engine:=true \
shutdown_on_aic_engine_exit:=true \
launch_rviz:=false \
gazebo_gui:=false \
aic_engine_config_file:=${CFG}"

  echo "=== Batch ${BATCH_NUM}/${TOTAL_BATCHES}  session=${SESSION_BATCH} ==="
  echo "  Sim window:    ${SIM_CMD}"
  echo "  Policy window: ${POLICY_CMD}"
  echo "  Config:        ${CFG}"
  echo "  Logs:          ${LOG_DIR}/"

  tmux new-session -d -s "${SESSION_BATCH}" -n sim "${SIM_CMD}"
  tmux new-window -t "${SESSION_BATCH}:1" -n policy "${POLICY_CMD}"
  start_session_logging "${SESSION_BATCH}" "${LOG_DIR}" "${CFG}" "${SIM_CMD}" "${POLICY_CMD}"

  if [[ "${BATCH_NUM}" -lt "${TOTAL_BATCHES}" ]]; then
    if [[ "${ATTACH_LAST}" -eq 1 ]]; then
      echo "  Optional attach: tmux attach -t ${SESSION_BATCH}"
    fi
    echo "  Waiting for batch ${BATCH_NUM} sim to exit..."
    wait_sim_then_cleanup "${SESSION_BATCH}" "${LOG_DIR}"
    echo "  Cleaning stale results before next batch: rm -rf ~/aic_results"
    rm -rf ~/aic_results
    echo "  Batch ${BATCH_NUM} complete."
  else
    if [[ "${ATTACH_LAST}" -eq 1 ]]; then
      echo "  Attaching to final batch (detach with Ctrl-b d)..."
      tmux attach -t "${SESSION_BATCH}"
      capture_session_logs "${SESSION_BATCH}" "${LOG_DIR}"
    else
      echo "  Waiting for final batch sim to exit..."
      wait_sim_then_cleanup "${SESSION_BATCH}" "${LOG_DIR}"
    fi
  fi
done

echo
echo "Run complete."
echo "  Full trial list: ${RANDOM_CFG}"
echo "  Batch configs:   ${BATCH_DIR}/"
echo "  Logs:            ${LOG_ROOT}/"
echo "  Dataset (local): ${STABLEWM_HOME}/${DATASET_NAME}.h5"

if [[ -n "${GCS_UPLOAD_URI}" ]]; then
  H5_LOCAL="${STABLEWM_HOME}/${DATASET_NAME}.h5"
  if ! command -v gsutil >/dev/null 2>&1; then
    echo "  GCS upload skipped: gsutil not found (install Google Cloud SDK)."
  elif [[ ! -f "${H5_LOCAL}" ]]; then
    echo "  GCS upload skipped: file missing: ${H5_LOCAL}"
  else
    if [[ "${GCS_UPLOAD_URI}" == *.h5 ]]; then
      GCS_DEST="${GCS_UPLOAD_URI}"
    else
      GCS_DEST="${GCS_UPLOAD_URI%/}/${DATASET_NAME}_${RUN_STAMP}.h5"
    fi
    echo "  Uploading to GCS: ${GCS_DEST}"
    gsutil cp "${H5_LOCAL}" "${GCS_DEST}"
    echo "  Upload complete."
  fi
fi
