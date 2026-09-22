"""How an Ascend function is called: the symbols it exports and what it takes.

A caller cannot name a device function, only the shim that launches it, so
that is what Ascend puts in the symbol table. The convention mirrors the CUDA
one a host unit already speaks -- untyped pointers, wide extents, the launch
geometry last -- so a CPU entry calls into either device target the same way.
"""

from __future__ import annotations

from tilefoundry.codegen.ascend.context import AscendCodegenContext
from tilefoundry.codegen.context import EmitContext
from tilefoundry.codegen.signature import (
    CallableSignature,
    LaunchSignature,
    ProgramIdSignature,
    Signature,
    TensorSignature,
    tensor_signature_of,
)
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.target import AscendTarget
from tilefoundry.visitor_registry.registries import Role, register_codegen

PROGRAM_ID_CTYPE = "long long"
"""What a level's id travels as, wide enough for any count a target states."""

_C_ABI = ("void*", "long long")
"""What a buffer and an extent are called where a host unit has to spell them.

A host translation unit compiles without AscendC's headers, so it cannot name
``bfloat16_t``; the shim takes what plain C states and puts the type back on
inside the unit that has it.
"""


def _identifier(name: str) -> str:
    """*name* as a plain C identifier -- a mangled variant's ``$`` is not one."""
    return name.replace("$", "__")


def launch_shim(name: str) -> str:
    """The ``extern "C"`` symbol of the shim that launches kernel *name*."""
    return f"tilefoundry_{_identifier(name)}_launch"


def device_kernel(name: str) -> str:
    """The ``__global__ __aicore__`` symbol of the kernel compiled from *name*."""
    return f"tilefoundry_{_identifier(name)}_kernel"


def kernel_signature(shim: CallableSignature, name: str) -> CallableSignature:
    """How the kernel compiled from *name* is called inside its own unit.

    It takes *shim*'s own parameters and, where the shim was told ids no card
    can read, those same ids spelled in the device unit's types. The launch
    geometry stops at the shim, which is what spends it.
    """
    return CallableSignature(
        name=device_kernel(name),
        params=shim.params,
        output_count=shim.output_count,
        leading=shim.leading,
    )


def kernel_arguments(kernel: CallableSignature, ctx: AscendCodegenContext) -> str:
    """What the shim hands the kernel: its C ABI values, typed as the kernel declared.

    The boundary is crossed here and nowhere else -- the shim holds what any
    unit can spell, and the kernel body indexes device types.
    """
    written: list[str] = []
    for param in kernel.all_params:
        if not isinstance(param, TensorSignature):
            written.append(param.name)
            continue
        written.append(f"static_cast<uint8_t*>({param.name})")
        written.extend(param.extent_name(axis) for axis in param.dynamic_axes)
    return ", ".join(written)


@register_codegen(AscendTarget, Role.CALLEE, PrimFunction)
def _called_as_shim(
    fn: PrimFunction, program_ids: tuple[ProgramIdSignature, ...]
) -> CallableSignature:
    """The one convention of *fn* a caller can reach: its launch shim.

    The kernel's own convention is not this one, because nobody outside the
    translation unit can name it. A program that names a level the target
    names such a level is told its id first, and the launch geometry comes last.
    """
    return CallableSignature(
        name=launch_shim(fn.name),
        params=tuple(tensor_signature_of(var) for var in fn.params),
        output_count=fn.output_count,
        leading=program_ids,
    )


@register_codegen(AscendTarget, Role.CALLEE, TensorSignature)
def _declare_tensor(sig: TensorSignature, ctx: EmitContext) -> tuple[str, ...]:
    """A tensor arrives as a pointer, then one extent per open axis.

    A static extent is a constant on both sides of the call and costs the
    convention nothing; an axis the type leaves open travels with the pointer.
    Where another unit reads the declaration the pointer is untyped, because a
    host unit cannot spell a device dtype.
    """
    if ctx.exported:
        pointer, extent = _C_ABI
    else:
        pointer, extent = "GM_ADDR", "int64_t"
    return (
        f"{pointer} {sig.name}",
        *(f"{extent} {sig.extent_name(axis)}" for axis in sig.dynamic_axes),
    )


@register_codegen(AscendTarget, Role.CALLEE, LaunchSignature)
def _declare_launch(sig: LaunchSignature, ctx: AscendCodegenContext) -> tuple[str, ...]:
    """One geometry argument, in the C type the convention states for it."""
    return (f"{sig.ctype} {sig.name}",)


@register_codegen(AscendTarget, Role.CALLEE, ProgramIdSignature)
def _declare_program_id(sig: ProgramIdSignature, ctx: AscendCodegenContext) -> tuple[str, ...]:
    """The id of a level no card can read for itself, told to it instead."""
    return (f"{PROGRAM_ID_CTYPE} {sig.name}",)


@register_codegen(AscendTarget, Role.CALLER, TensorSignature)
def _pass_tensor(sig: TensorSignature, ctx: EmitContext) -> tuple[str, ...]:
    """The pointer, then the extents, as the calling scope spells them.

    How many arguments there are and in what order is Ascend's, since Ascend
    declared them; what each is called belongs to whoever is calling.
    """
    return (
        ctx.local_value(sig),
        *(ctx.local_extent(sig, axis) for axis in sig.dynamic_axes),
    )


@register_codegen(AscendTarget, Role.CALLER, LaunchSignature)
@register_codegen(AscendTarget, Role.CALLER, ProgramIdSignature)
def _pass_value(sig: Signature, ctx: EmitContext) -> tuple[str, ...]:
    """One argument the caller already holds, spelled in its own scope."""
    return (ctx.local_value(sig),)


__all__ = [
    "PROGRAM_ID_CTYPE",
    "device_kernel",
    "kernel_arguments",
    "kernel_signature",
    "launch_shim",
]
