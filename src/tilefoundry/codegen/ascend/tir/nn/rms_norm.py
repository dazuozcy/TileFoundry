"""Emitter for ``tir.nn.RMSNorm`` (fused RMS normalization stmt) on Ascend.

Emits one call to ``tilefoundry::ops::rmsnorm(src, dst, weight, M, K, eps)``;
the row-wise reduction, rescaling, and the UB staging that orders them live
in the device runtime header.
"""

from __future__ import annotations

from tilefoundry.codegen.ascend.context import AscendCodegenContext
from tilefoundry.ir.core import Var
from tilefoundry.ir.tir.nn.rms_norm import RMSNorm
from tilefoundry.ir.types.shape_helpers import static_dim_value
from tilefoundry.target import AscendTarget
from tilefoundry.visitor_registry.registries import Role, register_codegen


@register_codegen(AscendTarget, Role.EMIT, RMSNorm)
def _emit(call, ctx: AscendCodegenContext) -> None:
    src, dst, weight = call.args[0], call.args[1], call.args[2]
    if not isinstance(src, Var) or not isinstance(dst, Var) or not isinstance(weight, Var):
        raise RuntimeError("tir.nn.RMSNorm: demo path expects Var operands for src/dst/weight")
    src_name = ctx.name_for(src)
    dst_name = ctx.name_for(dst)
    weight_name = ctx.name_for(weight)

    shape = tuple(static_dim_value(dim) for dim in src.type.shape)
    if len(shape) != 2 or any(dim is None for dim in shape):
        raise NotImplementedError(
            "tir.nn.RMSNorm on Ascend: source must be a rank-2 tensor with "
            "static extents; the row count and width travel as call arguments"
        )
    rows, width = shape
    weight_shape = tuple(static_dim_value(dim) for dim in weight.type.shape)
    if weight_shape not in ((width,), (1, width), (width, 1)):
        raise NotImplementedError(
            "tir.nn.RMSNorm on Ascend: weight must be a vector whose length "
            f"equals the reduced width {width}; got {weight_shape}"
        )

    eps = call.target.eps
    ctx.emit(
        f"tilefoundry::ops::rmsnorm({src_name}, {dst_name}, {weight_name}, "
        f"{rows}, {width}, {eps}f);"
    )
