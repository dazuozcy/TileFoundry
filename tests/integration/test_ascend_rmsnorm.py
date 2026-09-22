"""Ascend RMSNorm authored-fixture runtime coverage.

Compiles the TIR fixture through the full pipeline -- AscendC device unit,
CPU host unit, one bisheng link -- and checks the result against torch on the
NPU. Skips when no NPU is visible, the same contract the CUDA tests hold with
``torch.cuda``.
"""

from __future__ import annotations

import pytest
import torch
import torch_npu  # noqa: F401  (registers the npu device with torch)

import tilefoundry
from tests.fixtures.tir.rmsnorm_ascend import TirRmsnormAscend
from tilefoundry.target import AscendTarget


def _npu_available() -> bool:
    try:
        return torch.npu.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _npu_available(), reason="requires an Ascend NPU")
def test_rmsnorm_ascend_matches_torch() -> None:
    torch.manual_seed(1)
    x = torch.randn(1, 128, dtype=torch.float32, device="npu")
    weight = torch.randn(128, dtype=torch.float32, device="npu")
    out = torch.empty_like(x)
    expected = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-5) * weight

    runtime = tilefoundry.compile(TirRmsnormAscend, target=AscendTarget("huawei.ascend910b2c"))
    runtime(x, weight, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, expected, rtol=2e-5, atol=2e-5)
