"""AscendC (Ascend NPU) compilation capabilities.

Every value is built from the installed ``huawei.dav2201`` document; this
module holds the shape of an Ascend architecture value, never a copy of its
numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.types import DType
from tilefoundry.target.base import Architecture


@dataclass(frozen=True)
class AscendArchitecture(Architecture):
    """What one Ascend architecture states about itself.

    The per-core limits live here rather than on a device: they are
    properties of the microarchitecture, so every product built on it shares
    them.
    """

    name: str
    supported_compute_dtypes: tuple[DType, ...]
    instruction_capabilities: tuple[str, ...]
    max_vector_lanes: int
    unified_buffer_per_core_bytes: int
    ub_owner: str

    def _python_import_module(self) -> str:
        if type(self) is AscendArchitecture:
            return "tilefoundry.target.ascend"
        return super()._python_import_module()

    def supports_compute_dtype(self, dtype: DType) -> bool:
        """Return whether this architecture has a compute instruction for ``dtype``."""
        return dtype in self.supported_compute_dtypes

    def topology_limit(self, name: str) -> int:
        """Return the structural limit for an Ascend topology level.

        ``thread`` names the vector-lane level a mesh's finest axes declare;
        the count is a program-logical ceiling, not a hardware width.
        """
        if name == "thread":
            return self.max_vector_lanes
        raise ValueError(
            f"{self.name}: no architecture limit for topology level {name!r}"
        )


__all__ = ["AscendArchitecture"]
