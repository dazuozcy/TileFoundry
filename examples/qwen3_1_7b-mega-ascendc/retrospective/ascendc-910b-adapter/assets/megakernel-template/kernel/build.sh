#!/usr/bin/env bash
# 910B/CANN 8.5 mix recipe. Do not add -cce-enable-mix or replace the <<<>>> stub.
set -euo pipefail
cd -- "$(dirname -- "$0")"
: "${ASCEND_HOME_PATH:?source the existing CANN set_env.sh first}"
python ../tools/check_contract.py
SRC=${1:-mega.cpp}
OUT=${2:-libmegak.so}
bisheng --npu-arch=dav-2201 -std=c++17 -xasc -I. \
  -DMEGA_TIMELINE="${MEGA_TIMELINE:-0}" \
  -DMEGA_LM_TILE="${MEGA_LM_TILE:-96}" \
  -DMEGA_LM_STREAM="${MEGA_LM_STREAM:-1}" \
  -DMEGA_COMPACT_A="${MEGA_COMPACT_A:-1}" \
  -DMEGA_GU_INTERLEAVE="${MEGA_GU_INTERLEAVE:-1}" \
  -I"$ASCEND_HOME_PATH/x86_64-linux/asc" -I"$ASCEND_HOME_PATH/include" \
  -L"$ASCEND_HOME_PATH/lib64" -lruntime -lascendcl -lplatform -lc_sec -ldl -lm \
  -fPIC --shared "$SRC" -o "$OUT"
# Avoid grep -q + pipefail SIGPIPE on large readelf output.
sections=$(readelf -SW "$OUT")
symbols=$(readelf -sW "$OUT")
[[ "$sections" == *aicore_binary* ]] || { echo 'missing .aicore_binary' >&2; exit 1; }
[[ "$symbols" == *run_megak* && "$symbols" == *megak_layout* ]] || { echo 'missing ABI exports' >&2; exit 1; }
[[ "$symbols" != *mix_aic_mix_aic* ]] || { echo 'doubled mix symbol; audit compiler flags' >&2; exit 1; }
sha256sum "$OUT"
echo 'BUILD PASS (compilation/ELF only; run the NPU gates separately)'
