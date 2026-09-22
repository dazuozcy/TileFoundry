from __future__ import annotations

from tilefoundry import module, prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.types.shard import B, Layout, Mesh, Topology
from tilefoundry.target import AscendTarget, CpuTarget


@module(entry="rmsnorm_ascend_host", target=AscendTarget("huawei.ascend910b2c"))
class TirRmsnormAscend:
    @prim_func(target=AscendTarget("huawei.ascend910b2c"))
    def rmsnorm_ascend_device(x: Tensor[(1, 128), "f32"], weight: Tensor[(128,), "f32"], out: Tensor[(1, 128), "f32"]):
        with Mesh((Topology("thread", 1),), Layout((1,), (1,)), names=('t',)) as thread:
            x_view = T.tensor_view(x, layout=((1, 128), (128, 1), {thread.t @ B()}))
            weight_view = T.tensor_view(weight, layout=((128,), (1,), {thread.t @ B()}))
            out_view = T.tensor_view(out, layout=((1, 128), (128, 1), {thread.t @ B()}))
            T.rms_norm(x_view, out_view, weight_view, eps=1e-05)
            T.sync(thread)

    @prim_func(target=CpuTarget())
    def rmsnorm_ascend_host(x: Tensor[(1, 128), "f32"], weight: Tensor[(128,), "f32"], out: Tensor[(1, 128), "f32"]):
        launch(rmsnorm_ascend_device, x, weight, out, grid=(1, 1, 1), block=(1, 1, 1))  # noqa: F821
