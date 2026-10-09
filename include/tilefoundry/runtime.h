/// tilefoundry runtime umbrella header.
///
/// Selects the target-specific runtime by the build-injected target macro.
/// Exactly one of TILEFOUNDRY_TARGET_CUDA / TILEFOUNDRY_TARGET_CPU must be
/// defined (the build sets it per translation unit); both or neither is an
/// error. Ascend device units do not pass through here: they include
/// <tilefoundry/runtime/ascend/runtime.h> directly, because its ops are
/// AscendC and compile only under bisheng, which a host unit never sees.
#pragma once

#if defined(TILEFOUNDRY_TARGET_CUDA) && defined(TILEFOUNDRY_TARGET_CPU)
#error                                                                         \
    "tilefoundry/runtime.h: TILEFOUNDRY_TARGET_CUDA and TILEFOUNDRY_TARGET_CPU are mutually exclusive"
#elif defined(TILEFOUNDRY_TARGET_CUDA)
#include <tilefoundry/runtime/cuda/runtime.cuh>
#elif defined(TILEFOUNDRY_TARGET_CPU)
#include <tilefoundry/runtime/cpu/runtime.h>
#else
#error                                                                         \
    "tilefoundry/runtime.h: define exactly one TILEFOUNDRY_TARGET_* (CUDA or CPU)"
#endif
