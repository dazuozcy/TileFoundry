"""Shared environment bootstrap for the qwen3_mega_tl scripts.

Spike-1 conclusion: the ONE interpreter that runs both tilelang-npuir and
torch_npu is the tilefoundry env python (3.12): the npuir checkout's
libtilelangir.so native module is built for 3.12 (the `zuo` env is 3.11 and
its import fails with "Python version mismatch"), while torch_npu in the
tilefoundry env needs the LD_LIBRARY_PATH bootstrap below (re-exec pattern
inherited from qwen3_mega/_bootstrap.py, which the CANN/torch_npu layout on
this machine requires).
"""

import os
import sys

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

_TF_LIB = os.environ.get("MEGA_TF_LIB", "/home/tilelang/miniconda3/envs/tilefoundry/lib")
_CANN = os.environ.get("ASCEND_HOME_PATH", "/home/tilelang/lxn50063176/Ascend/cann-8.5.0")
_TNPU_LIB = os.environ.get(
    "MEGA_TNPU_LIB",
    "/home/tilelang/miniconda3/envs/tilefoundry/lib/python3.12/site-packages/torch_npu/lib",
)

# project-local tilelang cache (npuir.md interop 6: suspect the cache first)
os.environ.setdefault(
    "TILELANG_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tlcache"),
)

_ld = os.environ.get("LD_LIBRARY_PATH", "")
_have = set(p for p in _ld.split(":") if p)
_missing = [p for p in (_TF_LIB, os.path.join(_CANN, "lib64"), _TNPU_LIB) if p and p not in _have]
if _missing and not os.environ.get("MEGA_ENV_FIXED"):
    os.environ["LD_LIBRARY_PATH"] = ":".join(_missing + ([_ld] if _ld else []))
    os.environ["MEGA_ENV_FIXED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)
