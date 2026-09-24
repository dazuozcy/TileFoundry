"""What an Ascend target tells the analysis families.

Each conversion restates the installed hardware documents in the shape the
asking analysis declared, and nothing here decides anything. Two honest gaps
are preserved rather than papered over: no document states a register-file
capacity, so the ``rmem`` level carries ``None`` and stays advisory; and no
document states a per-unit service rate, so ``unit_ops`` stays empty and the
performance analysis refuses unrated work by name instead of pricing it at
nothing.
"""

from __future__ import annotations

from tilefoundry.analysis.facts import (
    ExplicitMemoryLevelFacts,
    ImplicitMemoryLevelFacts,
    MemoryHierarchyFacts,
    MemoryLevelRelation,
    MemoryRelationKind,
    PerformanceServiceFacts,
    ThroughputFacts,
)

from ..base import UnsupportedCapabilityError
from .target import AscendTarget


def memory_hierarchy(target: AscendTarget, query: object = None) -> MemoryHierarchyFacts:
    """The Ascend memory levels and how they are related.

    The Unified Buffer is the level an AscendC kernel stages through, so it is
    the ``smem`` a program places values in: one allocation per AI Core block.
    The L2 sits in front of HBM as one device-wide cache; no per-core cache is
    published, so the implicit list names no other level. Nothing shares a
    physical block here, so the relation list is a plain cache chain.
    """
    architecture = target.architecture
    device = target.device
    return MemoryHierarchyFacts(
        explicit_levels=(
            ExplicitMemoryLevelFacts(
                name="gmem",
                capacity_bytes=device.hbm_capacity_bytes,
                scope="npu",
                owner=device.gmem_owner,
            ),
            ExplicitMemoryLevelFacts(
                name="smem",
                capacity_bytes=architecture.unified_buffer_per_core_bytes,
                scope="cta",
                owner=architecture.ub_owner,
            ),
            ExplicitMemoryLevelFacts(
                name="rmem",
                capacity_bytes=None,
                scope="cta",
                owner="cta",
            ),
        ),
        implicit_levels=(
            ImplicitMemoryLevelFacts(
                name="l2", capacity_bytes=device.l2_capacity_bytes, scope="npu"
            ),
        ),
        relations=(MemoryLevelRelation(kind=MemoryRelationKind.CACHES, near="l2", far="gmem"),),
    )


def _cards(target: AscendTarget) -> int:
    """How many cards the deployment runs on; one when nobody said."""
    return 1 if target.device_count is None else target.device_count


def parallel_units(target: AscendTarget, unit: str) -> int:
    """How many of *unit* the deployment runs at once.

    Both the divisor a per-unit rate is the deployment peak over, and the
    answer to how many instances of a level run together -- the same number,
    asked once. The ``thread`` count restates the architecture's vector-lane
    ceiling, which the document itself calls a program-logical bound: no
    hardware lane width is published to divide by instead.
    """
    cards = _cards(target)
    try:
        return {
            "npu": cards,
            "cta": cards * target.device.sm_count,
            "thread": (cards * target.device.sm_count * target.architecture.max_vector_lanes),
        }[unit]
    except KeyError:
        raise UnsupportedCapabilityError(
            f"ascend: no per-unit rate for topology level {unit!r}; the levels this "
            "target divides its peaks among are ('npu', 'cta', 'thread')"
        ) from None


def throughput(target: AscendTarget, query: object = None) -> ThroughputFacts:
    """The deployment rates a roofline divides work by.

    The bandwidth is HBM's, so the memory side of the bound is computed from
    global traffic alone. The Unified Buffer and the register file publish no
    static bandwidth, and inventing one would put a number on the bound that no
    document supports.
    """
    device = target.device
    cards = _cards(target)
    peaks = tuple(
        (dtype, peak * cards)
        for dtype, peak in sorted(
            device.dense_flops_per_second.items(), key=lambda item: item[0].name
        )
    )
    return ThroughputFacts(
        peak_flops_per_second=peaks,
        memory_bandwidth_bytes_per_second=device.hbm_bandwidth_bytes_per_second * cards,
        bandwidth_level="gmem",
    )


def performance_service(target: AscendTarget, query: object = None) -> PerformanceServiceFacts:
    """What one unit of the level asked about gets through, by kind of work.

    The float rates are the deployment's peaks divided among however many of
    that unit run at once, the same division the roofline's bound starts from.
    No document states a per-unit rate for the non-float services, so
    ``unit_ops`` states none: a program that asks for such work is refused by
    name rather than priced at nothing.
    """
    device = target.device
    cards = _cards(target)
    unit = "cta" if query is None else str(query)
    share = parallel_units(target, unit)
    return PerformanceServiceFacts(
        unit_flops=tuple(
            (dtype, peak * cards // share)
            for dtype, peak in sorted(
                device.dense_flops_per_second.items(), key=lambda item: item[0].name
            )
        ),
        unit_ops=(),
        unit_bandwidth=(("gmem", device.hbm_bandwidth_bytes_per_second * cards // share),),
        unit=unit,
    )


__all__ = [
    "memory_hierarchy",
    "parallel_units",
    "performance_service",
    "throughput",
]
