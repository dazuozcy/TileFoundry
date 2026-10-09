"""Ascend 910B2C device resources.

Every value is built from the installed ``huawei.ascend910b2c`` document;
this module holds the shape of the device value, never a copy of its numbers.
The unified buffer and vector-lane geometry are ISA facts and belong to the
architecture, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.types import DType
from tilefoundry.target.base import Device


@dataclass(frozen=True)
class AscendDevice(Device):
    """One Ascend NPU package: its cores, HBM, and unit rates."""

    name: str

    sm_count: int
    """How many AI Core blocks a launch of one kernel may occupy."""

    vector_core_count: int
    cube_core_count: int
    hbm_capacity_bytes: int
    gmem_owner: str
    hbm_bandwidth_bytes_per_second: int
    l2_capacity_bytes: int

    _dense_flops: tuple[tuple[DType, int], ...] = ()

    def _python_import_module(self) -> str:
        if type(self) is AscendDevice:
            return "tilefoundry.target.ascend"
        return super()._python_import_module()

    @property
    def dense_flops_per_second(self) -> dict[DType, int]:
        """Return the dense compute-throughput map by dtype."""
        return dict(self._dense_flops)

    def throughput_for(self, dtype: DType) -> int:
        """Return dense throughput for a ``dtype``."""
        try:
            return self.dense_flops_per_second[dtype]
        except KeyError:
            raise ValueError(
                f"{self.name}: no dense compute-throughput entry for dtype "
                f"{getattr(dtype, 'name', dtype)!r}"
            ) from None


__all__ = ["AscendDevice"]
