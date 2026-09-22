"""What the shared codegen context does not know: Ascend's types.

Everything a compile carries regardless of target lives in
:class:`tilefoundry.codegen.context.EmitContext`; what is added here is
Ascend's alone -- the AscendC dtype spelling. Handler registration lives in
``tilefoundry.visitor_registry``, keyed by this target.
"""

from __future__ import annotations

from collections.abc import Mapping

from tilefoundry.codegen.context import EmitContext
from tilefoundry.codegen.signature import CallableSignature
from tilefoundry.ir.tir.abort import Abort
from tilefoundry.ir.tir.stmts import (
    Evaluate,
    For,
    If,
    LetStmt,
    MeshScope,
    Return,
    Sequential,
    While,
)
from tilefoundry.target import AscendTarget
from tilefoundry.target.base import Target
from tilefoundry.visitor_registry.registries import codegen_registry

_ASCEND_CPP: dict[str, str] = {
    "f32": "float",
    "f16": "half_t",
    "bf16": "bfloat16_t",
    "i32": "int32_t",
    "i64": "int64_t",
}
"""The AscendC spelling of each IR dtype, on the device side of the ABI."""


class AscendCodegenContext(EmitContext):
    """A compile writing AscendC: the shared context plus what only Ascend states.

    Unlike the CUDA context it carries no launch geometry: an Ascend unit
    states no geometry of its own, because every launch spends its own
    ``grid_x`` through its own shim rather than through a unit-level program
    dimension.
    """

    target_kind = AscendTarget

    def __init__(
        self,
        *,
        symbols: Mapping[int, CallableSignature] | None = None,
        target: Target | None = None,
        codegen_context: object | None = None,
    ) -> None:
        super().__init__(
            codegen_registry,
            symbols=symbols,
            target=target,
            codegen_context=codegen_context,
        )

    def dtype_to_cpp(self, dtype_name: str) -> str:
        t = _ASCEND_CPP.get(dtype_name)
        if t is None:
            raise ValueError(f"unsupported dtype for Ascend codegen: {dtype_name!r}")
        return t

    def emit_node(self, node) -> None:
        """Write *node* through this target's statement emitter or op handlers."""
        if isinstance(
            node, (Abort, Evaluate, For, If, LetStmt, MeshScope, Return, Sequential, While)
        ):
            from tilefoundry.codegen.ascend.emitter import (  # noqa: PLC0415
                AscendEmitter,
            )

            AscendEmitter(context=self).visit(node)
            return
        self.handler_for(type(node))(node, self)


__all__ = ["AscendCodegenContext"]
