"""Typed Ascend hardware schema builders."""

from __future__ import annotations

from tilefoundry.ir.types import DType
from tilefoundry.target.ascend.architecture import AscendArchitecture
from tilefoundry.target.ascend.device import AscendDevice
from tilefoundry.target.facts import TARGET_MEMORY_OWNER
from tilefoundry.target.hardware.envelope import (
    HardwareDocument,
    SchemaValidationError,
)
from tilefoundry.target.hardware.schema import SchemaReader

ARCHITECTURE_SCHEMA = "tilefoundry.ascend.architecture/v1"
DEVICE_SCHEMA = "tilefoundry.ascend.device/v1"

DAV2201_ID = "huawei.dav2201"
ASCEND910B2C_ID = "huawei.ascend910b2c"

_THROUGHPUT_DTYPE_NAMES = ("f32", "f16", "bf16")


def _memory_owner(reader: SchemaReader, path: str) -> str:
    """Read an owner in the Ascend target's topology vocabulary."""
    owner = reader.text(path)
    allowed = (TARGET_MEMORY_OWNER, "cta", "thread")
    if owner not in allowed:
        raise SchemaValidationError(
            f"{reader.document.id}: memory owner {owner!r} at {path!r} must be "
            f"one of {list(allowed)}"
        )
    return owner


def _dtypes(names: tuple[str, ...], document: HardwareDocument) -> tuple[DType, ...]:
    """Resolve recorded dtype names against the IR dtype table."""
    resolved = []
    for name in names:
        dtype = getattr(DType, name, None)
        if dtype is None:
            raise SchemaValidationError(
                f"{document.id}: unknown compute dtype {name!r}"
            )
        resolved.append(dtype)
    return tuple(resolved)


def build_ascend_architecture(document: HardwareDocument) -> AscendArchitecture:
    """Build the immutable Ascend architecture value from its document."""
    reader = SchemaReader(document)
    architecture = AscendArchitecture(
        name=reader.text("identity.name"),
        supported_compute_dtypes=_dtypes(
            reader.names("instruction.compute_dtypes"), document
        ),
        instruction_capabilities=reader.names("instruction.capabilities"),
        max_vector_lanes=reader.integer("compute.max_vector_lanes", unit="count"),
        unified_buffer_per_core_bytes=reader.integer(
            "memory.unified_buffer.per_core", unit="byte"
        ),
        ub_owner=_memory_owner(reader, "memory.unified_buffer.owner"),
    )
    reader.declared_unavailable("memory.unified_buffer.bandwidth")
    reader.close()
    return architecture


def build_ascend_device(document: HardwareDocument) -> AscendDevice:
    """Build the immutable Ascend device value from its document."""
    reader = SchemaReader(document)
    dense_flops: list[tuple[DType, int]] = []
    for dtype_name in _THROUGHPUT_DTYPE_NAMES:
        if f"throughput.{dtype_name}" not in document.facts:
            continue
        peak = reader.optional_integer(f"throughput.{dtype_name}", unit="flop/s")
        if peak is not None:
            dense_flops.append((getattr(DType, dtype_name), peak))
    device = AscendDevice(
        name=reader.text("identity.name"),
        sm_count=reader.integer("compute.ai_core_count", unit="count"),
        vector_core_count=reader.integer("compute.vector_core_count", unit="count"),
        cube_core_count=reader.integer("compute.cube_core_count", unit="count"),
        hbm_capacity_bytes=reader.integer("memory.hbm.capacity", unit="byte"),
        gmem_owner=_memory_owner(reader, "memory.hbm.owner"),
        hbm_bandwidth_bytes_per_second=reader.integer(
            "memory.hbm.bandwidth", unit="byte/s"
        ),
        l2_capacity_bytes=reader.optional_integer("memory.l2.capacity", unit="byte"),
        _dense_flops=tuple(dense_flops),
    )
    reader.declared_unavailable("memory.l2.bandwidth")
    reader.close()

    if device.sm_count != device.vector_core_count:
        raise SchemaValidationError(
            f"{document.id}: {device.sm_count} AI Core blocks must equal the "
            f"{device.vector_core_count} vector cores they schedule"
        )
    return device


__all__ = [
    "ASCEND910B2C_ID",
    "ARCHITECTURE_SCHEMA",
    "DAV2201_ID",
    "DEVICE_SCHEMA",
    "build_ascend_architecture",
    "build_ascend_device",
]
