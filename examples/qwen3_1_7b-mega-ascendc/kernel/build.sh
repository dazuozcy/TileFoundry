#!/bin/bash
# Build the Qwen3 mega AscendC MIX kernel (hardware-verified recipe v2).
#   ./build.sh [src.cpp]     (default: mega.cpp)
# Produces: libmegak.so -- device binary (.aicore_binary, auto-identified mix)
#          + host launch stub run_megak() (bisheng <<<>>> plugin does
#          rtDevBinaryRegister + rtFunctionRegister + rtKernelLaunchWithFlagV2).
#
# RULES (see smoke3/NOTES.md):
#  * NEVER pass -cce-enable-mix (emits doubled _mix_aic_mix_aic symbols ->
#    rtDevBinaryRegister fails 107000). The mix type is auto-identified via
#    the KERNEL_TYPE_MIX_AIC_1_2 global + cube ops in the AIC branch.
#  * Launch ONLY through the <<<>>> stub (manual rtKernelLaunch hangs for mix).
#  * Run with ASCEND_RT_VISIBLE_DEVICES=0 (chip 3 has MTE ROB ECC faults).
set -e
cd "$(dirname "$0")"
SRC=${1:-mega.cpp}
: "${ASCEND_HOME_PATH:?source CANN set_env first}"
bisheng --npu-arch=dav-2201 -std=c++17 -xasc \
  -I$ASCEND_HOME_PATH/x86_64-linux/asc -I$ASCEND_HOME_PATH/include \
  -L$ASCEND_HOME_PATH/lib64 -lruntime -lascendcl -lplatform -lc_sec -ldl -lm \
  -fPIC --shared "$SRC" -o libmegak.so
# sanity: .aicore_binary section present and stub exported
readelf -SW libmegak.so | grep -q aicore_binary || { echo "no .aicore_binary"; exit 1; }
readelf -sW libmegak.so | grep -q run_megak || { echo "run_megak stub missing"; exit 1; }
echo OK
