"""Ascend NPU compilation target composition.

The target mirrors the CUDA target's shape: one device document plus the
architecture it runs, resolved through installed hardware documents. The
topology vocabulary maps CUDA's program levels onto Ascend execution:

- ``npu`` — which card, told at launch the way CUDA's ``gpu`` id is (no
  card can read which of them it is);
- ``cta`` — one AI Core block, the unit ``GetBlockIdx()`` numbers and a
  launch's ``grid_x`` counts;
- ``thread`` — the vector-lane level of one core. There is no SIMT
  register for it: a mesh at this level states data-parallel lanes the
  kernel covers inside the core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from tilefoundry.target.ascend.architecture import AscendArchitecture
from tilefoundry.target.ascend.device import AscendDevice
from tilefoundry.target.ascend.spec import (
    ARCHITECTURE_SCHEMA,
    ASCEND910B2C_ID,
    DEVICE_SCHEMA,
    build_ascend_architecture,
    build_ascend_device,
)
from tilefoundry.target.base import (
    Architecture,
    Device,
    HardwareSpec,
    Target,
    _architecture_of,
    _available_device_ids,
    check_compatible,
    register_target,
    select,
)
from tilefoundry.target.facts import (
    TopologyFacts,
    TopologyLimitFacts,
    facts_result,
)
from tilefoundry.target.hardware.envelope import HardwareDocument
from tilefoundry.target.services import CodeGenerator
from tilefoundry.utils.python_source import PythonExpr


@register_target
@dataclass(frozen=True, init=False)
class AscendTarget(Target):
    """Ascend target composed from one device and the architecture it runs."""

    name: ClassVar[str] = "ascend"
    hardware: ClassVar[HardwareSpec] = HardwareSpec(
        package="tilefoundry.target.ascend.hardware",
        schemas={
            ARCHITECTURE_SCHEMA: build_ascend_architecture,
            DEVICE_SCHEMA: build_ascend_device,
        },
    )
    gmem_device_type: ClassVar[str] = "kDLExtDev"
    """The DLPack device_type constant an NPU GMEM tensor carries.

    torch_npu hands tensors out as ``kDLExtDev`` (12), so that is the
    placement a host entry checks for.
    """

    architecture: Architecture = field(init=False)
    device: Device = field(init=False)

    device_count: int | None = field(default=None, init=False)
    """How many cards a program of this target may name at its ``npu`` level."""

    architecture_id: str | None = field(default=None, init=False, compare=False)
    device_id: str | None = field(default=None, init=False, compare=False)
    architecture_digest: str | None = field(default=None, init=False, compare=False)
    device_digest: str | None = field(default=None, init=False, compare=False)
    _architecture_document: HardwareDocument | None = field(
        default=None, init=False, compare=False, repr=False
    )
    _device_document: HardwareDocument | None = field(
        default=None, init=False, compare=False, repr=False
    )

    @property
    def identity(self) -> str:
        return self.device_id or self.name

    @classmethod
    def available(cls) -> tuple[AscendTarget, ...]:
        return tuple(cls(device_id) for device_id in _available_device_ids(cls.hardware))

    def __init__(
        self,
        device: Device | str | Path | None = None,
        architecture: Architecture | str | Path | None = None,
        *,
        device_count: int | None = None,
    ) -> None:
        if device_count is not None and (
            isinstance(device_count, bool)
            or not isinstance(device_count, int)
            or device_count < 1
        ):
            raise ValueError(
                f"AscendTarget: device_count {device_count!r} must be a positive "
                f"int or None, which admits any extent at the npu level"
            )
        device = ASCEND910B2C_ID if device is None else device
        if architecture is None:
            architecture = _architecture_of(
                device,
                device_type=AscendDevice,
                role="AscendTarget.device",
                hardware=self.hardware,
            )
        architecture = select(
            architecture,
            AscendArchitecture,
            role="AscendTarget.architecture",
            hardware=self.hardware,
        )
        device = select(
            device, AscendDevice, role="AscendTarget.device", hardware=self.hardware
        )
        if architecture.id is not None and device.id is not None:
            check_compatible(architecture, device)
        object.__setattr__(self, "device_count", device_count)
        object.__setattr__(self, "architecture", architecture.value)
        object.__setattr__(self, "device", device.value)
        object.__setattr__(self, "architecture_id", architecture.id)
        object.__setattr__(self, "device_id", device.id)
        object.__setattr__(self, "architecture_digest", architecture.digest)
        object.__setattr__(self, "device_digest", device.digest)
        object.__setattr__(self, "_architecture_document", architecture.document)
        object.__setattr__(self, "_device_document", device.document)

    def _topology_facts(self) -> TopologyFacts:
        """The three Ascend levels, coarsest first.

        Only ``npu`` comes from the target instance: how many cards a
        deployment has is stated by whoever constructs the target, and no card
        can read which of them it is. ``cta`` is decided by the launch (its
        ``grid_x``), so it states no static ceiling.
        """
        return TopologyFacts(
            (
                TopologyLimitFacts("npu", self.device_count, from_target=True),
                TopologyLimitFacts("cta", None),
                TopologyLimitFacts(
                    "thread", self.architecture.topology_limit("thread")
                ),
            )
        )

    def get_facts(self, facts_type: type, query: object | None = None):
        """Project Ascend hardware through the facts this Target owns."""
        if facts_type is TopologyFacts and query is None:
            return facts_result(self, facts_type, self._topology_facts())
        if facts_type is TopologyLimitFacts:
            for level in self._topology_facts().topologies:
                if level.name == query:
                    return facts_result(self, facts_type, level)
            return super().get_facts(facts_type, query)
        return super().get_facts(facts_type, query)

    def get_code_generator(self) -> CodeGenerator:
        from tilefoundry.codegen.ascend.emit import (  # noqa: PLC0415
            register_ascend_emitters,
        )
        from tilefoundry.codegen.ascend.module import (  # noqa: PLC0415
            ASCEND_CODE_GENERATOR,
        )

        register_ascend_emitters()
        return ASCEND_CODE_GENERATOR

    def _python_import_module(self) -> str:
        if type(self) is AscendTarget:
            return "tilefoundry.target.ascend"
        return super()._python_import_module()

    def to_python(self) -> PythonExpr:
        if type(self) is AscendTarget and self.device_id and self.architecture_id:
            count = (
                "" if self.device_count is None else f", device_count={self.device_count}"
            )
            return PythonExpr(
                ("from tilefoundry.target import AscendTarget",),
                f'AscendTarget("{self.device_id}"{count})',
            )
        return super().to_python()

    @property
    def arch(self) -> str:
        """Return the architecture name bisheng's ``--npu-arch`` takes."""
        return self.architecture.name

    def topology_limit(self, name: str) -> int:
        """Return the physical parallel limit for one Ascend topology level."""
        if name == "npu":
            return self.device_count or 1
        if name == "cta":
            return self.device.sm_count
        return self.architecture.topology_limit(name)


__all__ = ["AscendTarget"]
