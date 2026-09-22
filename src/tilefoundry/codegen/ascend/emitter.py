"""The AscendC statement emitter: shared C++ traversal plus mesh-scope lowering.

Control flow and scalar predicates are C++ and are shared with the CUDA
emitter (:class:`tilefoundry.codegen.emitter.CppEmitter`); only the
``MeshScope`` is this target's own -- on an AI Core a mesh states which data
each core position holds, not which SIMT lane runs the body, so the scope
draws no hardware boundary of its own.
"""

from __future__ import annotations

from tilefoundry.codegen.emitter import CppEmitter
from tilefoundry.ir.tir.stmts import MeshScope
from tilefoundry.ir.types.shard.layout import ComposedLayout
from tilefoundry.ir.types.shard.mesh import Mesh
from tilefoundry.target.facts import TopologyFacts


def _validate_topology(mesh: Mesh, target) -> None:
    """Validate that the target supports every program topology level.

    Each program topology level a mesh binds must be one this target
    supports. Defense-in-depth alongside the declared-topology check at
    lowering entry.
    """
    supported = tuple(
        level.name for level in target.get_facts(TopologyFacts).topologies
    )
    for topology in mesh.topologies:
        name = topology.name if not isinstance(topology, str) else topology
        if name not in supported:
            raise ValueError(
                f"ascend target supports {{{', '.join(supported)}}} topology "
                f"levels; got {name!r}"
            )


class AscendEmitter(CppEmitter):
    """AscendC emitter, including mesh scope lowering."""

    def visit_MeshScope(self, node: MeshScope) -> None:
        ctx = self.context
        if ctx.target is None:
            raise RuntimeError("Ascend MeshScope emission requires its Target")
        _validate_topology(node.mesh, ctx.target)
        names = tuple(
            topology.name if not isinstance(topology, str) else topology
            for topology in node.mesh.topologies
        )
        ctx.emit(f"// mesh scope: {', '.join(names)}")
        if isinstance(node.mesh.layout, ComposedLayout):
            raise NotImplementedError(
                "Ascend mesh emission: a sliced (sub-box) mesh is not supported "
                "yet; the participating core would have to be derived from the "
                "slice and the launch geometry"
            )
        ctx.emit("{")
        ctx.indent()
        self.visit(node.body)
        ctx.dedent()
        ctx.emit("}")


__all__ = ["AscendEmitter"]
