#!/usr/bin/env bash

set -euo pipefail

RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MOVE_CODE_ROOT="$(cd -- "${RUN_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${MOVE_CODE_ROOT}/../.." && pwd)"

export MODEL_PATH="Qwen/Qwen3-VL-32B-Instruct"
export MODEL_ADAPTER="generic"
export COORD_OUTPUT_MODE="norm1000"
export NOT_IN_SUBSET="${WORKSPACE_ROOT}/position/norm1000/outputs_qwen32b_sspro_not_in_worst_cells_32b.jsonl"
export WORST_SUBSET="${WORKSPACE_ROOT}/blind_zone/find_zone/32b/worst_cells_32b.jsonl"
export REGION_BBOX="${WORKSPACE_ROOT}/gui-blind_area-test/worst_cells_32b.jsonl"
export OUTPUT_DIR="${WORKSPACE_ROOT}/blind_zone/find_zone/32b"
export OUTPUT_TAG="qwen32b"
export MASTER_PORT="${MASTER_PORT:-29500}"

# Specify only arguments that differ from the Python defaults.
export RANDOM_REL_ARGS=""
export IN_REGION_ARGS="--max_new_tokens 32 --min_pixels 3136 --max_pixels 8847360 --center_inset_px 50 --bbox_edge_margin_px 50"
export OUT_ARGS="--canvas_edge_margin_px 300"
export RANDOM_IN_ARGS=""

exec bash "${MOVE_CODE_ROOT}/scripts/run_strategy.sh" "$@"
