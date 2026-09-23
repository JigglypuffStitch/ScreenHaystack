#!/usr/bin/env bash

set -euo pipefail

RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MOVE_CODE_ROOT="$(cd -- "${RUN_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${MOVE_CODE_ROOT}/../.." && pwd)"

export MODEL_PATH="HelloKKMe/GTA1-7B"
export MODEL_ADAPTER="generic"
export COORD_OUTPUT_MODE="pixel"
export NOT_IN_SUBSET="${WORKSPACE_ROOT}/position/norm/outputs_gta_sspro_not_in_worst_cells_gta.jsonl"
export WORST_SUBSET="${WORKSPACE_ROOT}/blind_zone/find_zone/gta/worst_cells_gta.jsonl"
export REGION_BBOX="${WORKSPACE_ROOT}/gui-blind_area-test/worst_cells_gta.jsonl"
export OUTPUT_DIR="${WORKSPACE_ROOT}/blind_zone/find_zone/gta"
export OUTPUT_TAG="gta"
export MASTER_PORT="${MASTER_PORT:-29518}"

# Specify only arguments that differ from the Python defaults.
export RANDOM_REL_ARGS=""
export IN_REGION_ARGS="--batch_size 2 --center_inset_px 50 --bbox_edge_margin_px 50"
export OUT_ARGS="--canvas_edge_margin_px 300"
export RANDOM_IN_ARGS=""

exec bash "${MOVE_CODE_ROOT}/scripts/run_strategy.sh" "$@"
