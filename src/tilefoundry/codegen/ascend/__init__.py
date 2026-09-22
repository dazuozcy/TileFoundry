"""AscendC codegen: the ABI registrations import as a side effect.

Importing this package registers Ascend's call-side handlers (how a function
of this target is called), because a symbol table is read for a whole tree at
once and a function in it may belong to any target. The per-op emitter
handlers register separately and lazily, from
``AscendTarget.get_code_generator``: they subclass the shared
``tilefoundry.codegen.emitter`` module, whose own top-level imports pull in
CUDA's emitters, so they must not execute while another target's package init
is mid-flight.
"""

from __future__ import annotations

from tilefoundry.codegen.ascend import abi as _abi  # noqa: F401 -- registers Ascend's side
