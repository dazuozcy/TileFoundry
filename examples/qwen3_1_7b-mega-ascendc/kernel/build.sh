#!/bin/bash
# Build the Qwen3 mega AscendC MIX kernel (hardware-verified recipe v2).
#   ./build.sh [src.cpp] [output.so] (default: mega.cpp libmegak.so)
# Optional MEGA_TIMELINE, MEGA_LM_TILE, MEGA_LM_STREAM, MEGA_COMPACT_A,
# MEGA_GU_INTERLEAVE settings are compiled into the binary. Production defaults
# match mega.cpp; diagnostic builds use their own output library.
# Produces: libmegak.so -- device binary (.aicore_binary, auto-identified mix)
#          + host launch stub run_megak() (bisheng <<<>>> plugin does
#          rtDevBinaryRegister + rtFunctionRegister + rtKernelLaunchWithFlagV2).
#
# RULES (see smoke3/NOTES.md):
#  * NEVER pass -cce-enable-mix (emits doubled _mix_aic_mix_aic symbols ->
#    rtDevBinaryRegister fails 107000). The mix type is auto-identified via
#    the KERNEL_TYPE_MIX_AIC_1_2 global + cube ops in the AIC branch.
#  * Launch ONLY through the <<<>>> stub (manual rtKernelLaunch hangs for mix).
#  * Run with ASCEND_RT_VISIBLE_DEVICES=4 (chips 0~3 have reported faults).
set -e
cd "$(dirname "$0")"
SRC=${1:-mega.cpp}
OUT=${2:-libmegak.so}
: "${ASCEND_HOME_PATH:?source CANN set_env first}"
bisheng --npu-arch=dav-2201 -std=c++17 -xasc \
  -DMEGA_TIMELINE="${MEGA_TIMELINE:-0}" \
  -DMEGA_LM_TILE="${MEGA_LM_TILE:-96}" \
  -DMEGA_LM_STREAM="${MEGA_LM_STREAM:-1}" \
  -DMEGA_COMPACT_A="${MEGA_COMPACT_A:-1}" \
  -DMEGA_GU_INTERLEAVE="${MEGA_GU_INTERLEAVE:-1}" \
  -I$ASCEND_HOME_PATH/x86_64-linux/asc -I$ASCEND_HOME_PATH/include \
  -L$ASCEND_HOME_PATH/lib64 -lruntime -lascendcl -lplatform -lc_sec -ldl -lm \
  -fPIC --shared "$SRC" -o "$OUT"
# sanity: .aicore_binary section present and stub exported
readelf -SW "$OUT" | grep -q aicore_binary || { echo "no .aicore_binary"; exit 1; }
readelf -sW "$OUT" | grep -q run_megak || { echo "run_megak stub missing"; exit 1; }
echo OK
