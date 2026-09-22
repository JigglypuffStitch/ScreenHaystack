#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# No arguments: retain the original Qwen/clock entry point.
if [[ "$#" -eq 0 ]]; then
  set -- qwen3-8b clock --batch_size 16 --attn_impl flash_attention_2
fi
exec "${PYTHON:-python}" "${SCRIPT_DIR}/run.py" "$@"
