"""Greedy decode of Qwen3-1.7B with ONE mega kernel launch per token.

The whole decoder -- 28 layers, attention over the full live context, the LM
head -- runs inside a single tilelang kernel (kernel/mega.py); Python only
feeds the next token and reads back the argmax.

  python run.py --prompt "..." --max-new-tokens 128
  python run.py --prompt-file README.md --max-new-tokens 2048
  python run.py --bench ctx            # ms/token + tok/s across the range

MEGA_S (default 40960) is the padded context the kernel is cut for; positions
0..MEGA_S-1 are valid.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "kernel")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("MEGA_S", "40960")

import _bootstrap  # noqa: F401  -- E402 (LD_LIBRARY_PATH, before torch_npu)

import torch  # noqa: E402
import torch_npu  # noqa: E402

import runtime_model as rm  # noqa: E402
from prepare_weights import build  # noqa: E402

WDIR = "/home/tilelang/zuochuanuong/weights/qwen3_1_7b"
L, KVH, D = 28, 8, 128
MAXPOS = 40960


def rope_tables_kernel(rows, d, base=1e6):
    """The kernel's table form: doubled cos, signed sin."""
    inv = base ** (-torch.arange(0, d, 2).float() / d)
    freqs = torch.outer(torch.arange(rows).float(), inv)
    cos = freqs.cos()
    sin = freqs.sin()
    return torch.cat([cos, cos], dim=1), torch.cat([-sin, sin], dim=1)


class MegaDecoder:
    def __init__(self):
        self.rt = rt = rm._Runtime()
        rt.set_weights(build())
        cos, sin = rope_tables_kernel(rm.SPADG, D)
        rt.cos_tab[:] = cos.to(rm._DEV)
        rt.sin_tab[:] = sin.to(rm._DEV)
        self.scale = float(torch.tensor(D**-0.5, dtype=torch.bfloat16).float())
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(WDIR)

    def step(self, token: int, pos: int) -> int:
        rt = self.rt
        rt.kern(*rt.args, token, pos, self.scale)
        torch.npu.synchronize()
        pm = rt.part_max.view(rm.NB, rm.NCH).cpu()
        pi = rt.part_idx.view(rm.NB, rm.NCH).cpu()
        flat = int(torch.argmax(pm))
        b, c = flat // rm.NCH, flat % rm.NCH
        return int(pi[b, c]) + b * rm.SPAN

    def generate(self, ids, max_new, echo=False):
        """Greedy; returns the generated ids and per-step times (ms)."""
        out, times = [], []
        pos = len(ids) - 1
        # teacher-forced prefill: one launch per prompt token, the last one
        # already proposes the first generated token
        t0 = time.perf_counter()
        for i, t in enumerate(ids):
            nxt = self.step(int(t), i)
        prefill = time.perf_counter() - t0
        out.append(nxt)
        for _ in range(max_new - 1):
            pos += 1
            if pos >= MAXPOS:
                break
            t0 = time.perf_counter()
            nxt = self.step(out[-1], pos)
            times.append((time.perf_counter() - t0) * 1e3)
            out.append(nxt)
            if echo:
                print(self.tok.decode([nxt]), end="", flush=True)
            if nxt == self.tok.eos_token_id:
                break
        return out, prefill, times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--greedy", action="store_true", default=True,
                    help="greedy decoding (the only mode; kept for CLI parity)")
    ap.add_argument("--bench", default=None, metavar="WHAT",
                    help="'ctx': sweep context length; 'steps': steady-state")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    dec = MegaDecoder()
    if args.prompt is None and args.prompt_file is None:
        args.prompt = "The capital of France is"
    text = args.prompt if args.prompt is not None else open(args.prompt_file).read()
    ids = dec.tok.encode(text)
    assert len(ids) + args.max_new_tokens <= MAXPOS, "prompt+generation exceeds the kernel's context"

    if args.bench == "ctx":
        bench_ctx(dec)
        return

    t_all = time.perf_counter()
    out, prefill, times = dec.generate(ids, args.max_new_tokens, echo=True)
    wall = time.perf_counter() - t_all
    gen = dec.tok.decode(out)
    print()
    steady = sum(times) / len(times) if times else float("nan")
    print(f"\n[prefill {len(ids)} tok in {prefill:.2f}s | "
          f"{args.max_new_tokens} gen tok | steady {steady:.2f} ms/tok "
          f"({1000.0 / steady:.1f} tok/s) | wall {wall:.2f}s]")
    print("--- generated ---")
    print(gen)


def bench_ctx(dec):
    """One timed step at each context length: the cost profile across the
    whole range the kernel is cut for."""
    import numpy as np

    rt = dec.rt
    lens = [64, 1024, 2048, 4096, 8192, 13824, 27648, 40959]
    print(f"{'ctx':>7} {'ms/step':>9} {'tok/s':>8}")
    for ctx in lens:
        # warm the cache with ctx live rows (values irrelevant for timing)
        for pos in range(0, ctx, 2048):
            rt.kern(*rt.args, 5, min(pos, ctx - 1), dec.scale)
        torch.npu.synchronize()
        reps = 5
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            rt.kern(*rt.args, 5, ctx - 1, dec.scale)
            torch.npu.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        med = float(np.median(ts))
        print(f"{ctx:>7} {med:>9.2f} {1000.0 / med:>8.1f}")


if __name__ == "__main__":
    main()
