"""Ascend emitter handler registration.

Loading every registered per-Op emitter under ``ascend/tir/`` registers its
handler against this target. What a function is called and what it takes are
answered in ``codegen.ascend.abi``, not here.

Registration runs on demand (``register_ascend_emitters``), not at this
package's import: these emitters subclass the shared
``tilefoundry.codegen.emitter`` module, whose top-level imports pull in
CUDA's emitters, so executing them while CUDA's own package init is
mid-flight would make CUDA's autodiscovery silently drop a handler. By the
time a target is asked for its code generator, every package init is done.
"""

from __future__ import annotations

import importlib
import logging
import os
import pkgutil

_log = logging.getLogger(__name__)
_tir_path = os.path.dirname(__file__)
_registered = False


def _discover(subdir: str, prefix: str) -> None:
    full = os.path.join(_tir_path, subdir)
    if not os.path.isdir(full):
        return
    for _finder, _name, _ispkg in pkgutil.iter_modules([full], prefix=prefix):
        try:
            importlib.import_module(_name)
        except Exception:
            _log.debug("codegen autodiscovery: skip %s", _name, exc_info=True)


def register_ascend_emitters() -> None:
    """Import every emitter module under ``ascend/tir/`` once, registering each."""
    global _registered
    if _registered:
        return
    _registered = True
    _discover("tir/stmts", "tilefoundry.codegen.ascend.tir.stmts.")
    _discover("tir/memory", "tilefoundry.codegen.ascend.tir.memory.")
    _discover("tir/nn", "tilefoundry.codegen.ascend.tir.nn.")
    _discover("tir", "tilefoundry.codegen.ascend.tir.")
