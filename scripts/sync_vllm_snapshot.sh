#!/usr/bin/env bash
set -euo pipefail

# Sync upstream vllm source tree into CFIE local snapshot.
# Usage:
#   scripts/sync_vllm_snapshot.sh [UPSTREAM_VLLM_DIR] [TARGET_DIR]
#
# Defaults:
#   UPSTREAM_VLLM_DIR=../vllm
#   TARGET_DIR=./vllm

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFIE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
UPSTREAM_DIR="${1:-${CFIE_ROOT}/../vllm}"
TARGET_DIR="${2:-${CFIE_ROOT}/vllm}"

if [[ ! -d "${UPSTREAM_DIR}" ]]; then
  echo "upstream vllm dir not found: ${UPSTREAM_DIR}" >&2
  exit 1
fi

if [[ ! -d "${TARGET_DIR}" ]]; then
  mkdir -p "${TARGET_DIR}"
fi

rsync -a --delete \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  --exclude "*.pyo" \
  "${UPSTREAM_DIR}/" "${TARGET_DIR}/"

echo "synced vllm snapshot: ${UPSTREAM_DIR} -> ${TARGET_DIR}"

