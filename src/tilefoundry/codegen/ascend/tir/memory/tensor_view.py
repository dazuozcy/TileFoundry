"""Emitter for ``tir.memory.TensorView`` on Ascend.

A view over a kernel parameter becomes an AscendC ``GlobalTensor`` bound to
the parameter's ``__gm__`` address; the ops that consume it read the shape
off the IR type, so the C++ value carries only the typed pointer. This
emitter supports the whole-buffer case a single-position mesh (or an all-
broadcast sharding) states; splitting a buffer across a multi-position mesh
needs core/lane offsets the AscendC path does not model yet.
"""

from __future__ import annotations

from tilefoundry.codegen.ascend.context import AscendCodegenContext
from tilefoundry.ir.core import Var
from tilefoundry.ir.tir.memory.tensor_view import TensorView
from tilefoundry.ir.tir.stmts import LetStmt
from tilefoundry.ir.types.shard.mesh import Mesh
from tilefoundry.ir.types.shard.shard_layout import (
    Broadcast,
    shard_layout_of,
)
from tilefoundry.target import AscendTarget
from tilefoundry.visitor_registry.registries import Role, register_codegen


def _single_position(mesh: Mesh) -> bool:
    """Whether *mesh* states exactly one program position."""
    for topology in mesh.topologies:
        size = topology.size if not isinstance(topology, str) else topology
        if size is None or size != 1:
            return False
    return True


def _whole_buffer(layout) -> bool:
    """Whether *layout* states a view of the whole source buffer.

    An all-broadcast sharding, or any sharding over a single-position mesh,
    leaves every position holding the entire tensor, which is what the one
    core this kernel runs on should see.
    """
    shard = shard_layout_of(layout)
    if shard is None:
        return True
    if _single_position(shard.mesh):
        return True
    return all(isinstance(attr, Broadcast) for attr in shard.attrs)


@register_codegen(AscendTarget, Role.EMIT, TensorView)
def _emit(let: LetStmt, ctx: AscendCodegenContext) -> None:
    call = let.value
    if len(call.args) != 1:
        raise NotImplementedError(
            "Ascend tensor_view: a sliced view (with offsets) is not supported "
            "yet; only a whole-buffer view over a kernel parameter is"
        )
    source = call.args[0]
    if not isinstance(source, Var) or not ctx.is_kernel_param(source):
        raise NotImplementedError(
            "Ascend tensor_view: the memory source must be a kernel parameter; "
            f"got {type(source).__name__}"
        )
    if not _whole_buffer(call.target.layout):
        raise NotImplementedError(
            "Ascend tensor_view: sharding a buffer across a multi-position mesh "
            "needs core/lane offsets the AscendC path does not model yet; "
            "declare the sharding broadcast, or the mesh single-position"
        )
    view_name = ctx.name_for(let.var)
    element = ctx.dtype_to_cpp(let.var.type.dtype.name)
    source_name = ctx.name_for(source)
    ctx.emit(f"GlobalTensor<{element}> {view_name};")
    ctx.emit(f"{view_name}.SetGlobalBuffer((__gm__ {element}*){source_name});")
