#!/usr/bin/env bash

set -euo pipefail

RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MOVE_CODE_ROOT="$(cd -- "${RUN_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${MOVE_CODE_ROOT}/../.." && pwd)"

export MODEL_PATH="inclusionAI/UI-Venus-Ground-7B"
export MODEL_ADAPTER="venus"
export COORD_OUTPUT_MODE="venus_bbox"
export NOT_IN_SUBSET="${WORKSPACE_ROOT}/gui-blind_area-test/uivenus_ground_7b_not_in_worst_cells_venus.jsonl"
export WORST_SUBSET="${WORKSPACE_ROOT}/blind_zone/find_zone/venus/worst_cells_venus.jsonl"
export REGION_BBOX="${WORKSPACE_ROOT}/gui-blind_area-test/worst_cells_venus.jsonl"
export OUTPUT_DIR="${WORKSPACE_ROOT}/blind_zone/find_zone/venus"
export OUTPUT_TAG="venus"
export MASTER_PORT="${MASTER_PORT:-29213}"

# 只保留与 Python 默认值不同的参数。
export RANDOM_REL_ARGS=""
export IN_REGION_ARGS="--center_inset_px 50 --bbox_edge_margin_px 50"
export OUT_ARGS="--away_margin_px 100 --canvas_edge_margin_px 300"
export RANDOM_IN_ARGS=""

exec bash "${MOVE_CODE_ROOT}/scripts/run_strategy.sh" "$@"
