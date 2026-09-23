#!/usr/bin/env bash

set -euo pipefail

SHARED_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage: bash run.sh <strategy> [additional Python arguments]

strategy:
  random_rel  Randomly move non-blind-zone samples into blind zones
  in_region   Move non-blind-zone samples into a specified region
  out         Move blind-zone samples out of a specified region
  random_in   Randomly move blind-zone samples within a specified region
  all         Run all four strategies in sequence

Examples:
  bash run.sh random_rel --max_samples 10
  CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash run.sh all
EOF
}

STRATEGY="${1:-}"
if [[ -z "${STRATEGY}" || "${STRATEGY}" == "-h" || "${STRATEGY}" == "--help" ]]; then
  usage
  exit 0
fi
shift

for required_name in MODEL_PATH MODEL_ADAPTER COORD_OUTPUT_MODE NOT_IN_SUBSET WORST_SUBSET REGION_BBOX OUTPUT_DIR OUTPUT_TAG; do
  if [[ -z "${!required_name:-}" ]]; then
    printf 'Missing model configuration variable: %s\n' "${required_name}" >&2
    exit 2
  fi
done

case "${MODEL_ADAPTER}" in
  generic)
    SCRIPT_SUFFIX=""
    ;;
  venus)
    SCRIPT_SUFFIX="_venus"
    ;;
  *)
    printf 'Unsupported MODEL_ADAPTER: %s\n' "${MODEL_ADAPTER}" >&2
    exit 2
    ;;
esac

for input_path in "${NOT_IN_SUBSET}" "${WORST_SUBSET}" "${REGION_BBOX}"; do
  if [[ ! -f "${input_path}" ]]; then
    printf 'Input file not found: %s\n' "${input_path}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_DIR}"

run_one() {
  local strategy_name="$1"
  shift
  local script_name
  local subset_path
  local output_path
  local strategy_args_name
  local strategy_args_raw
  local -a strategy_args=()

  case "${strategy_name}" in
    random_rel)
      script_name="move_random_rel${SCRIPT_SUFFIX}.py"
      subset_path="${NOT_IN_SUBSET}"
      output_path="${OUTPUT_DIR}/move_random_rel_${OUTPUT_TAG}_not_in_worst_cells.jsonl"
      strategy_args_name="RANDOM_REL_ARGS"
      ;;
    in_region)
      script_name="move_in_region_rel${SCRIPT_SUFFIX}.py"
      subset_path="${NOT_IN_SUBSET}"
      output_path="${OUTPUT_DIR}/move_in_region_rel_${OUTPUT_TAG}_not_in_worst_cells.jsonl"
      strategy_args_name="IN_REGION_ARGS"
      ;;
    out)
      script_name="move_out_rel${SCRIPT_SUFFIX}.py"
      subset_path="${WORST_SUBSET}"
      output_path="${OUTPUT_DIR}/move_out_rel_${OUTPUT_TAG}_worst_cells.jsonl"
      strategy_args_name="OUT_ARGS"
      ;;
    random_in)
      script_name="move_random_in${SCRIPT_SUFFIX}.py"
      subset_path="${WORST_SUBSET}"
      output_path="${OUTPUT_DIR}/move_random_in_${OUTPUT_TAG}_worst_cells.jsonl"
      strategy_args_name="RANDOM_IN_ARGS"
      ;;
    *)
      printf 'Unknown strategy: %s\n' "${strategy_name}" >&2
      usage >&2
      exit 2
      ;;
  esac

  strategy_args_raw="${!strategy_args_name:-}"
  if [[ -n "${strategy_args_raw}" ]]; then
    read -r -a strategy_args <<< "${strategy_args_raw}"
  fi

  printf 'Model: %s\nStrategy: %s\nOutput: %s\n' "${MODEL_PATH}" "${strategy_name}" "${output_path}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    torchrun \
      --nproc_per_node="${NPROC_PER_NODE:-1}" \
      --master_port="${MASTER_PORT:-29500}" \
      "${SHARED_DIR}/${script_name}" \
      --model_path "${MODEL_PATH}" \
      --subset_path "${subset_path}" \
      --region_bbox_path "${REGION_BBOX}" \
      --coord_output_mode "${COORD_OUTPUT_MODE}" \
      --output_path "${output_path}" \
      "${strategy_args[@]}" \
      "$@"
}

case "${STRATEGY}" in
  all)
    run_one random_rel "$@"
    run_one in_region "$@"
    run_one out "$@"
    run_one random_in "$@"
    ;;
  random_rel|in_region|out|random_in)
    run_one "${STRATEGY}" "$@"
    ;;
  *)
    printf 'Unknown strategy: %s\n' "${STRATEGY}" >&2
    usage >&2
    exit 2
    ;;
esac
