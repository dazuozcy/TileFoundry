"""Emitter for the ``tir.Sync`` op on Ascend.

``classify`` still runs for its refusals, the same ones CUDA makes: a partial
grid sync and a ragged subset are deadlocks. What survives classifies into
two Ascend answers: a barrier over thread-level (vector-lane) positions is
ordered by the AscendC queue discipline inside one core -- nothing to emit --
while a grid-level (cross-core) barrier has no tested lowering yet and is
refused rather than emitted wrong.
"""

from __future__ import annotations

from tilefoundry.codegen.ascend.context import AscendCodegenContext
from tilefoundry.ir.tir.sync import Sync, SyncBarrier, classify
from tilefoundry.target import AscendTarget
from tilefoundry.visitor_registry.registries import Role, register_codegen


@register_codegen(AscendTarget, Role.EMIT, Sync)
def _emit(call, ctx: AscendCodegenContext) -> None:
    """Emit the barrier the mesh's scope needs, or refuse what has no lowering."""
    mesh = call.target.mesh
    barrier = classify(mesh)
    if barrier is SyncBarrier.GRID:
        raise NotImplementedError(
            "T.sync on Ascend: a cross-core (grid) barrier is not supported "
            "yet; AscendC SyncAll would need every launched core participating, "
            "which this backend does not model"
        )
    ctx.emit(
        f"// T.sync: thread-scope barrier over "
        f"{getattr(mesh.layout, 'shape', ())}; AscendC queue discipline orders it"
    )
