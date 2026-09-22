"""Linker — the final stage of codegen.

Separately compiles each target's :class:`~tilefoundry.codegen.linkable.LinkableModule`
translation unit and links them into one host-callable shared library, returning
a ``LinkedModule`` (artifact + the entry's signature) for the runtime loader
to turn into a ``RuntimeModule``.

The CUDA pipeline is split (nvcc for device units, g++ for the host unit,
CMake to drive both); the Ascend pipeline is not — bisheng compiles the host
unit and the AscendC units in one invocation, because the host and device
code share the one dialect bisheng speaks.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from tilefoundry.codegen.signature import CallableSignature
from tilefoundry.dump import DumpFlags, DumpScope, dump

_TILEFOUNDRY_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_INCLUDE = _TILEFOUNDRY_ROOT / "include"
_DEFAULT_CUTLASS_INCLUDE = _TILEFOUNDRY_ROOT / "third_party" / "cutlass" / "include"
_CMAKE_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates" / "cmake"

_ASCEND_HOME_FALLBACKS = (
    Path("/usr/local/Ascend/ascend-toolkit/latest"),
)


@dataclass(frozen=True)
class LinkedModule:
    """Linked .so + the entry's signature. Consumed by the runtime loader."""
    library_path: Path
    source: str
    entry: CallableSignature


def _render_cmakelists(
    *,
    name: str,
    includes: list[str],
    device_options: str,
    cuda_arch: str,
    device_sources: list[str],
) -> str:
    """Render the split-pipeline CMake project from its Jinja template."""
    # noqa lazy: jinja2 is already a codegen dep; import here keeps the linker

    from jinja2 import (  # noqa: PLC0415
        Environment,
        FileSystemLoader,
        StrictUndefined,
    )

    env = Environment(
        loader=FileSystemLoader(str(_CMAKE_TEMPLATE_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    return env.get_template("CMakeLists.txt.j2").render(
        name=name,
        includes=includes,
        device_options=device_options,
        cuda_arch=cuda_arch,
        device_sources=device_sources,
    )


def _tvm_ffi_include() -> Path:
    # noqa lazy: tvm_ffi is an optional runtime dep; only required when

    import tvm_ffi  # noqa: PLC0415
    return Path(tvm_ffi.__file__).resolve().parent / "include"


def _ascend_home() -> Path:
    """Where the installed CANN toolkit lives, or a refusal that says how to fix it."""
    from_env = os.environ.get("ASCEND_HOME_PATH")
    if from_env:
        return Path(from_env)
    for fallback in _ASCEND_HOME_FALLBACKS:
        if (fallback / "bin" / "bisheng").exists():
            return fallback
    raise RuntimeError(
        "link_modules: no CANN toolkit found; source its set_env.sh (it puts "
        "bisheng on PATH and exports ASCEND_HOME_PATH), or install CANN"
    )


def _run_logged(step: str, cmd: list[str], timeout: int = 300) -> None:
    """Run one link step, keeping its command and output in the build log."""
    with DumpScope("build"):
        dump(f"{step}.cmd.txt", " ".join(cmd) + "\n", DumpFlags.BUILD_LOG)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        dump(f"{step}.stdout", proc.stdout, DumpFlags.BUILD_LOG)
        dump(f"{step}.stderr", proc.stderr, DumpFlags.BUILD_LOG)
        if proc.returncode != 0:
            raise RuntimeError(
                f"link_modules: {step} failed (rc={proc.returncode})\n"
                f"cmd: {' '.join(cmd)}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )


def _link_cuda(
    cu,
    cpp,
    *,
    workdir: Path,
    lib_name: str,
    entry: CallableSignature,
    nvcc: str,
    host_cxx: str,
    extra_nvcc_flags: tuple[str, ...],
    cuda_arch: str,
    include_dirs: tuple[Path, ...],
) -> LinkedModule:
    """Compile the device units with nvcc and the host unit separately, then link both."""
    for tool in (nvcc, host_cxx, "cmake"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"link_modules: {tool!r} not on PATH")

    for index, device in enumerate(cu):
        (workdir / f"device_{index}.cu").write_text(device.source)
    (workdir / "host.cpp").write_text(cpp[0].source)

    tvm_inc = _tvm_ffi_include()
    includes = [_DEFAULT_INCLUDE, _DEFAULT_CUTLASS_INCLUDE, tvm_inc, *include_dirs]
    device_options = ";".join(("-Wno-deprecated-gpu-targets", *extra_nvcc_flags))
    cmake_text = _render_cmakelists(
        name=lib_name,
        includes=[str(p) for p in includes],
        device_options=device_options,
        cuda_arch=cuda_arch,
        device_sources=[f"device_{i}.cu" for i in range(len(cu))],
    )
    (workdir / "CMakeLists.txt").write_text(cmake_text)

    build_dir = workdir / "build"

    configure_cmd = [
        "cmake", "-G", "Ninja", "-S", str(workdir), "-B", str(build_dir),
        f"-DCMAKE_CUDA_COMPILER={nvcc}", f"-DCMAKE_CXX_COMPILER={host_cxx}",
    ]
    build_cmd = ["cmake", "--build", str(build_dir), "--verbose"]
    lib = build_dir / f"lib{lib_name}.so"

    with DumpScope("build"):
        dump("CMakeLists.txt", cmake_text, DumpFlags.BUILD_LOG)
        for index, device in enumerate(cu):
            dump(f"module.device_{index}.cu", device.source, DumpFlags.BUILD_LOG)
        dump("module.host.cpp", cpp[0].source, DumpFlags.BUILD_LOG)
    _run_logged("cmake-configure", configure_cmd)
    _run_logged("cmake-build", build_cmd)
    if not lib.exists():
        raise RuntimeError(f"link_modules: cmake build did not produce {lib}")

    source = (
        f"// cpu module\n{cpp[0].source}\n"
        f"// cuda modules\n{''.join(m.source for m in cu)}"
    )
    return LinkedModule(
        library_path=lib,
        source=source,
        entry=entry,
    )


def _link_ascend(
    asc,
    cpp,
    *,
    workdir: Path,
    lib_name: str,
    entry: CallableSignature,
    bisheng: str,
    extra_bisheng_flags: tuple[str, ...],
    npu_arch: str,
    include_dirs: tuple[Path, ...],
) -> LinkedModule:
    """Compile host and AscendC units in one bisheng invocation, then link.

    bisheng speaks the host dialect and the AscendC dialect in the same
    invocation (``-xasc``), so the host unit and the device units compile
    together; there is no split pipeline to drive and no CMake project to
    generate. The host unit selects the CPU runtime umbrella with a define,
    and the device units include their own runtime header directly, so one
    define for the whole invocation is correct.
    """
    if shutil.which(bisheng) is None:
        raise RuntimeError(
            f"link_modules: {bisheng!r} not on PATH; source the CANN toolkit's "
            "set_env.sh (it puts bisheng on PATH) or install CANN"
        )
    ascend_home = _ascend_home()

    for index, device in enumerate(asc):
        (workdir / f"device_{index}.cpp").write_text(device.source)
    (workdir / "host.cpp").write_text(cpp[0].source)

    sources = [str(workdir / "host.cpp")] + [
        str(workdir / f"device_{index}.cpp") for index in range(len(asc))
    ]
    lib = workdir / f"lib{lib_name}.so"
    cmd = [
        bisheng,
        f"--npu-arch={npu_arch}",
        "-std=c++17",
        "-xasc",
        f"-I{ascend_home / 'include'}",
        f"-I{ascend_home / 'include' / 'experiment' / 'runtime'}",
        f"-I{_DEFAULT_INCLUDE}",
        f"-I{_tvm_ffi_include()}",
        *(f"-I{d}" for d in include_dirs),
        f"-L{ascend_home / 'lib64'}",
        "-lruntime",
        "-lascendcl",
        "-lplatform",
        "-lc_sec",
        "-ldl",
        "-lm",
        "-fPIC",
        "--shared",
        "-DTILEFOUNDRY_TARGET_CPU",
        *extra_bisheng_flags,
        *sources,
        "-o",
        str(lib),
    ]

    with DumpScope("build"):
        for index, device in enumerate(asc):
            dump(f"module.device_{index}.cpp", device.source, DumpFlags.BUILD_LOG)
        dump("module.host.cpp", cpp[0].source, DumpFlags.BUILD_LOG)
    _run_logged("bisheng", cmd, timeout=600)
    if not lib.exists():
        raise RuntimeError(f"link_modules: bisheng did not produce {lib}")

    source = (
        f"// cpu module\n{cpp[0].source}\n"
        f"// ascend modules\n{''.join(m.source for m in asc)}"
    )
    return LinkedModule(
        library_path=lib,
        source=source,
        entry=entry,
    )


def link_modules(
    modules,
    *,
    workdir: str | Path,
    lib_name: str,
    entry: CallableSignature,
    nvcc: str = "nvcc",
    host_cxx: str = "g++",
    bisheng: str = "bisheng",
    extra_nvcc_flags: tuple[str, ...] = (),
    extra_bisheng_flags: tuple[str, ...] = (),
    cuda_arch: str = "90",
    npu_arch: str = "dav-2201",
    include_dirs: tuple[Path, ...] = (),
) -> LinkedModule:
    """Link modules into one host-callable shared library.

    Requires exactly one ``cpp`` module (the host unit) and at least one
    device module: ``cu`` for the CUDA target or ``asc`` for the Ascend
    target, never both in one library. A CUDA library compiles its device
    units with nvcc and its host unit with a plain host compiler; an Ascend
    library compiles everything with bisheng in one invocation.
    """
    modules = tuple(modules)
    if len(modules) < 2 or sum(m.language == "cpp" for m in modules) != 1:
        received = ", ".join(f"({m.target}, {m.language})" for m in modules)
        raise ValueError(
            f"link_modules: requires one 'cpp' host module and at least one "
            f"device module, got [{received}]"
        )
    cu = [m for m in modules if m.language == "cu"]
    asc = [m for m in modules if m.language == "asc"]
    cpp = [m for m in modules if m.language == "cpp"]
    if cu and asc:
        received = ", ".join(f"({m.target}, {m.language})" for m in modules)
        raise ValueError(
            f"link_modules: a library links one device target, not CUDA and "
            f"Ascend together, got [{received}]"
        )
    if not cu and not asc:
        received = ", ".join(f"({m.target}, {m.language})" for m in modules)
        raise ValueError(
            f"link_modules: requires at least one 'cu' or 'asc' device "
            f"module, got [{received}]"
        )

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if cu:
        return _link_cuda(
            cu,
            cpp,
            workdir=workdir,
            lib_name=lib_name,
            entry=entry,
            nvcc=nvcc,
            host_cxx=host_cxx,
            extra_nvcc_flags=extra_nvcc_flags,
            cuda_arch=cuda_arch,
            include_dirs=include_dirs,
        )
    return _link_ascend(
        asc,
        cpp,
        workdir=workdir,
        lib_name=lib_name,
        entry=entry,
        bisheng=bisheng,
        extra_bisheng_flags=extra_bisheng_flags,
        npu_arch=npu_arch,
        include_dirs=include_dirs,
    )


__all__ = ["LinkedModule", "link_modules"]
