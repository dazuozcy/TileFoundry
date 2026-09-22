/// tilefoundry Ascend-target device runtime surface.
///
/// AscendC op implementations included in-context from the generated device
/// unit (compiled by bisheng ``-xasc``). Host-side code never includes this
/// header: the CPU host unit crosses into the device through the raw
/// ``extern "C"`` launch shims only.
#pragma once
