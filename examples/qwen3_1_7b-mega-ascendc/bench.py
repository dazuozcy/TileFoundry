import ctypes, os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_mega as T   # sets up buffers/weights/launch

print("warmup...")
T.launch(0, 28); torch.npu.synchronize()

for sl in (0, 100, 1000, 5000, 20000, 40000):
    # prefill caches once
    T.KcP.zero_(); T.VcP.zero_()
    if sl > 0:
        pf = (torch.randn(28, T.KVH, sl, T.D, device=T.DEV) * 0.05).to(torch.bfloat16)
        vf = (torch.randn(28, T.KVH, sl, T.D, device=T.DEV) * 0.05).to(torch.bfloat16)
        for li in range(28):
            for j in range(T.KVH):
                kk = torch.zeros(T.SPADG, T.D, dtype=torch.bfloat16, device=T.DEV)
                vv = torch.zeros(T.SPADG, T.D, dtype=torch.bfloat16, device=T.DEV)
                kk[:sl] = pf[li, j]; vv[:sl] = vf[li, j]
                T.KcP[(li*T.KVH+j)*(8*(T.SPADG//16)*256):(li*T.KVH+j+1)*(8*(T.SPADG//16)*256)] = T.pack_kc(kk)
                T.VcP[(li*T.KVH+j)*((T.SPADG//16)*8*256):(li*T.KVH+j+1)*((T.SPADG//16)*8*256)] = T.pack_vc(vv)
    for _ in range(3):
        T.launch(sl, 28)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    N = 10
    for _ in range(N):
        T.launch(sl, 28)
    torch.npu.synchronize()
    dt = (time.perf_counter() - t0) / N
    print(f"sl={sl:6d}: {dt*1e3:8.3f} ms/token   ({1/dt:7.1f} tok/s)")
