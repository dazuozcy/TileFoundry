"""Ascend NPU compilation target package."""

from __future__ import annotations

from tilefoundry.target.ascend.architecture import AscendArchitecture
from tilefoundry.target.ascend.device import AscendDevice
from tilefoundry.target.ascend.spec import (
    ASCEND910B2C_ID,
    DAV2201_ID,
)
from tilefoundry.target.ascend.target import AscendTarget

__all__ = [
    "ASCEND910B2C_ID",
    "AscendArchitecture",
    "AscendDevice",
    "AscendTarget",
    "DAV2201_ID",
]
