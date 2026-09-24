"""Facts are selected by the exact Target value, not a global registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest

from tilefoundry.analysis.facts import (
    MemoryHierarchyFacts,
    PerformanceServiceFacts,
    ThroughputFacts,
)
from tilefoundry.ir.types import DType
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import (
    AmxTarget,
    AscendTarget,
    CudaTarget,
    Target,
    TopologyFacts,
    TopologyLevelFacts,
    UnsupportedCapabilityError,
)
from tilefoundry.target.facts import TargetFactsError, facts_result


def test_builtin_targets_own_their_facts_projections() -> None:
    cuda = CudaTarget("nvidia.h200_sxm")

    throughput = cuda.get_facts(ThroughputFacts)
    memory = cuda.get_facts(MemoryHierarchyFacts)

    assert throughput.memory_bandwidth_bytes_per_second == 4_800_000_000_000
    assert memory.explicit("gmem").capacity_bytes == cuda.device.hbm_capacity_bytes
    assert {level.name: level.owner for level in memory.explicit_levels} == {
        "gmem": "target",
        "smem": "cta",
        "rmem": "thread",
    }
    assert AmxTarget().get_facts(ThroughputFacts).bandwidth_level == "gmem"
    assert {
        level.name: level.owner
        for level in AmxTarget().get_facts(MemoryHierarchyFacts).explicit_levels
    } == {"host": "target", "gmem": "target", "rmem": "amx"}


def test_two_cuda_products_project_the_hardware_each_one_is() -> None:
    """One projection serves both products.

    One projection serves both products, so what separates them is what their
    documents record rather than a branch per architecture.

    Tensor memory is the case that matters: an absent level says the architecture
    has no such store, and a capacity says a CTA's accumulators live in one. A
    projection that reported the same for both would price a Blackwell kernel
    against Hopper's registers.
    """
    hopper = CudaTarget("nvidia.h200_sxm")
    blackwell = CudaTarget("nvidia.b200_sxm")

    assert hopper.get_facts(MemoryHierarchyFacts).explicit("tmem") is None
    tmem = blackwell.get_facts(MemoryHierarchyFacts).explicit("tmem")
    assert (tmem.capacity_bytes, tmem.scope) == (262_144, "cta")

    throughput = blackwell.get_facts(ThroughputFacts)
    peaks = dict(throughput.peak_flops_per_second)
    assert throughput.memory_bandwidth_bytes_per_second == 7_672_320_000_000
    assert peaks[DType.f4e2m1] == 9_000_000_000_000_000
    assert DType.f4e2m1 not in dict(hopper.get_facts(ThroughputFacts).peak_flops_per_second)
    assert blackwell.get_facts(TopologyFacts).parallel().max_physical_units == 148


def test_ascend_target_projects_its_installed_documents() -> None:
    """The Ascend projections restate the 910B2C documents, gaps included.

    What the documents do not state shows up as honestly absent: no register
    capacity, and no per-unit service rate, so ``rmem`` stays advisory and
    ``unit_ops`` stays empty.
    """
    ascend = AscendTarget("huawei.ascend910b2c")

    topology = ascend.get_facts(TopologyFacts)
    assert topology.level("npu") == TopologyLevelFacts("npu", None, 1, from_target=True)
    assert topology.level("cta") == TopologyLevelFacts("cta", None, 48)
    assert topology.level("thread").max_logical_units == 1024
    assert topology.level("thread").max_physical_units == 48 * 1024
    assert topology.parallel() is topology.level("cta")

    memory = ascend.get_facts(MemoryHierarchyFacts)
    assert {level.name: level.owner for level in memory.explicit_levels} == {
        "gmem": "target",
        "smem": "cta",
        "rmem": "cta",
    }
    assert memory.explicit("gmem").capacity_bytes == ascend.device.hbm_capacity_bytes
    assert (
        memory.explicit("smem").capacity_bytes == ascend.architecture.unified_buffer_per_core_bytes
    )
    assert memory.explicit("rmem").capacity_bytes is None
    assert memory.implicit("l2").capacity_bytes == ascend.device.l2_capacity_bytes
    assert memory.backing_level("l2") == "gmem"

    throughput = ascend.get_facts(ThroughputFacts)
    peaks = dict(throughput.peak_flops_per_second)
    assert peaks[DType.f32] == 100_000_000_000_000
    assert peaks[DType.f16] == 200_000_000_000_000
    assert throughput.memory_bandwidth_bytes_per_second == 1_600_000_000_000
    assert throughput.bandwidth_level == "gmem"

    service = ascend.get_facts(PerformanceServiceFacts)
    assert service.unit == "cta"
    assert service.unit_flops == (
        (DType.bf16, 200_000_000_000_000 // 48),
        (DType.f16, 200_000_000_000_000 // 48),
        (DType.f32, 100_000_000_000_000 // 48),
    )
    assert service.unit_ops == ()
    assert service.unit_bandwidth == (("gmem", 1_600_000_000_000 // 48),)


def test_ascend_device_count_scales_the_deployment_rates() -> None:
    """A deployment that names its cards divides its peaks over that many."""

    def cards(target: AscendTarget) -> int:
        return target.device_count or 1

    two_cards = AscendTarget("huawei.ascend910b2c", device_count=2)
    one_card = AscendTarget("huawei.ascend910b2c")

    assert two_cards.get_facts(TopologyLevelFacts, "npu") == TopologyLevelFacts(
        "npu", 2, 2, from_target=True
    )
    assert two_cards.get_facts(TopologyLevelFacts, "cta").max_physical_units == 2 * 48
    assert (
        two_cards.get_facts(ThroughputFacts).memory_bandwidth_bytes_per_second
        == cards(two_cards) * 1_600_000_000_000
    )
    assert two_cards.get_facts(ThroughputFacts).peak_flops_per_second == tuple(
        (dtype, peak * cards(two_cards))
        for dtype, peak in one_card.get_facts(ThroughputFacts).peak_flops_per_second
    )
    assert (
        two_cards.get_facts(PerformanceServiceFacts).unit_flops
        == one_card.get_facts(PerformanceServiceFacts).unit_flops
    )


def test_a_target_without_a_requested_projection_fails_closed() -> None:
    @dataclass(frozen=True)
    class _UnknownFacts:
        units: int

    with pytest.raises(
        UnsupportedCapabilityError, match=r"CudaTarget \(cuda\): no Facts projection"
    ):
        CudaTarget("nvidia.h200_sxm").get_facts(_UnknownFacts)


def test_topology_limits_are_target_facts_and_base_validation_is_inherited() -> None:
    cuda = CudaTarget("nvidia.h200_sxm")
    amx = AmxTarget()

    cuda_levels = cuda.get_facts(TopologyFacts)
    amx_levels = amx.get_facts(TopologyFacts)
    assert cuda_levels.level("cta") == TopologyLevelFacts("cta", None, 132)
    assert cuda_levels.level("thread").max_logical_units == 1024
    assert cuda_levels.parallel() == cuda_levels.level("cta")
    assert amx_levels.level("core") == TopologyLevelFacts("core", 8, 8)
    assert amx_levels.level("amx") == TopologyLevelFacts("amx", 1, None)
    assert amx_levels.parallel() == amx_levels.level("core")
    assert cuda.topology_limit("cta") == cuda.device.sm_count == 132
    assert cuda.topology_limit("thread") == cuda.architecture.max_threads_per_cta

    @dataclass(frozen=True)
    class _DirectTarget(Target):
        name: ClassVar[str] = "test.direct-topology"

        def get_facts(self, facts_type: type, query: object | None = None):
            if facts_type is TopologyFacts and query is None:
                return TopologyFacts((TopologyLevelFacts("unit", 4, 2),))
            if facts_type is TopologyLevelFacts and query == "unit":
                return TopologyLevelFacts("unit", 4, 2)
            return super().get_facts(facts_type, query)

    _DirectTarget().validate_program_topology(Topology("unit", 4))
    with pytest.raises(ValueError, match="1 <= extent <= 4"):
        _DirectTarget().validate_program_topology(Topology("unit", 5))


def test_projection_results_are_still_immutable_aggregates_of_the_requested_type() -> None:
    @dataclass(frozen=True)
    class _Facts:
        units: int

    @dataclass
    class _MutableFacts:
        units: int

    @dataclass(frozen=True)
    class _CustomTarget(Target):
        name: ClassVar[str] = "test.custom"

        def get_facts(self, facts_type: type, query: object | None = None):
            if facts_type is _Facts:
                return facts_result(self, facts_type, _Facts(4))
            return super().get_facts(facts_type, query)

    assert _CustomTarget().get_facts(_Facts) == _Facts(4)
    with pytest.raises(TargetFactsError, match="must be a frozen dataclass"):
        facts_result(_CustomTarget(), _MutableFacts, _MutableFacts(4))
    with pytest.raises(TargetFactsError, match="returned int"):
        facts_result(_CustomTarget(), _Facts, 4)
