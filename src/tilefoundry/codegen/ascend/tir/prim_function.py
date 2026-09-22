"""Emit the body of one AscendC kernel: its statements, then its statements' ops.

A kernel is handed ``GM_ADDR`` pointers and the extents its parameter types
leave open; the ops that index them wrap what they need (an AscendC
``GlobalTensor``) at the view site, so the kernel body starts with nothing to
do but its own statements.
"""

from __future__ import annotations

from tilefoundry.codegen.ascend.context import AscendCodegenContext
from tilefoundry.codegen.ascend.emitter import AscendEmitter
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.target import AscendTarget
from tilefoundry.visitor_registry.registries import Role, register_codegen


@register_codegen(AscendTarget, Role.EMIT, PrimFunction)
def _emit(fn: PrimFunction, ctx: AscendCodegenContext) -> None:
    """Write what runs inside one ``__global__ __aicore__``."""
    if fn.variants:
        raise NotImplementedError(
            f"Ascend codegen: {fn.name!r} is a specialization prototype with "
            f"{len(fn.variants)} variants; Ascend dispatch-on-extent is not "
            "supported yet"
        )
    for param in fn.params:
        ctx.register_kernel_param(param)
    AscendEmitter(context=ctx).visit(fn.body)
