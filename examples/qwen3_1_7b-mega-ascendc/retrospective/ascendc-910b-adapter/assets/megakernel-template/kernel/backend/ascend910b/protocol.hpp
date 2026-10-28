#pragma once
// ADAPT ONLY AFTER PROBES: epochs, sentinel and diagnostic trace.
// seq workspace layout (int32), carved out of p_syncws:
//   [0, 1536)                SyncAll GM ws (48 x 8) -- unused in v2
//   [4096 ...]               rows 0..23 AIV slots, 24..47 AIC slots;
//                           slot = 16 int32 at SEQ_WANT(row, r)
//                           = 4096 + (row*256 + r)*16  (max 196608)
constexpr int SEQ_OFF = 4096;
constexpr int SEQ_ROUNDS = 256;        // 8*28 + finals headroom
// Double-buffer the previous layer's head maxima. A current-layer
// slice must never overwrite the shift before another slice has read it.
constexpr int SHIFT_HISTORY = SEQ_OFF + 48 * SEQ_ROUNDS * 16 + 4096;
__aicore__ inline int SEQ_WANT(int cid, int r) { return SEQ_OFF + (cid * 256 + r) * 16; }

// ---- v7 fine-grained producer/consumer progress words (see OPTLOG R6) ----
// One dedicated seq round per channel.  The producer stores
//     value = tag + li*PROG_SPAN + count
// and the consumer tests `>= want`, so a consumer on a COARSE grain (e.g. a
// 1024-position softmax chunk = 4 score tiles) can proceed the instant that
// chunk's producer step lands, instead of waiting for the whole stage.  This
// turns a phase barrier into a streaming hand-off (Mirage-style grain
// interleave) and is what lets gemm2's PV stream overlap the softmax tail.
// PROG_SPAN=16 keeps li*PROG_SPAN+count < 512 = the per-launch tag step, so
// each word stays strictly monotonic across launches: a stale word from a
// previous token can never satisfy `>= want` (the old count is >512 behind).
constexpr int PROG_SPAN = 16;          // grains/layer (softmax chunks <= 14)
constexpr int PR_PROBS = 232;          // round idx: AIV->AIC probs chunks landed
constexpr int PR_SCORES = 233;         // round idx: AIC->AIV score groups landed
// (232/233 sit above the per-layer rounds 0..223 and the 224 finals round)
// R13 (2a): the scores->softmax hand-off is streamed per 4096-position group
// (16 score tiles) instead of one whole-slice barrier, so pass 1's max overlaps
// gemm1's K stream.  Numerics are unchanged: the slice max is the max of the
// same 4096-wide reductions, merged in the same order.

// ---- v2 sentinel gates (replace poll+2xSyncAll; see OPTLOG/README) ----
// AIC arms a DEAD-ROW word in its own fixpipe region with SENT at layer
// start, then (after the gate's fixpipes) self-polls it until the data lands
// (!= SENT), then publishes its seq word at SEQ_WANT(24+cid, r). Consumers
// poll the seq word(s) -- single word for 1:1 gates, the [c0,c1) range for
// all-24 gates. No plane SyncAll anywhere.
// R27 generality test hook: build with -DMEGA_SCORE_BUMP=<int> to add that
// constant to every scaled score.  Softmax is shift invariant, so the token
// must be unchanged -- but a shift-0 build with a bump > 88 MUST produce inf
// (that is the whole point: it proves the carried shift, not a lucky constant,
// is what keeps the exponent in range).  Off by default, zero cost.
#ifndef MEGA_SCORE_BUMP
#define MEGA_SCORE_BUMP 0
#endif
constexpr int SENT = 0x7FC00000;       // qNaN bit pattern: never a real f32 GEMV value

// Optional diagnostic timestamps. Each traced core owns complete 128B lines.
// Stored in the existing extra sync workspace; disabled builds emit no stamps.
#ifndef MEGA_TIMELINE
#define MEGA_TIMELINE 0
#endif
#if MEGA_TIMELINE
constexpr int TRACE_BASE = SEQ_OFF + 48 * SEQ_ROUNDS * 16;
__aicore__ inline void mega_stamp(GM_ADDR workspace, int plane, int layer, int event) {
    volatile __gm__ int32_t* p = reinterpret_cast<volatile __gm__ int32_t*>(workspace)
        + TRACE_BASE + (plane * (L + 1) + layer) * 32 + event;
    *p = static_cast<int32_t>(GetSystemCycle());
    __asm__ __volatile__("" ::: "memory");
    dcci(reinterpret_cast<__gm__ uint64_t*>((__gm__ int32_t*)p),
         cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
    __asm__ __volatile__("" ::: "memory");
}
#define TRACE_C(layer, event) do { if ((cid % 8) == 0) mega_stamp(p_syncws, cid / 8, layer, event); } while (0)
#define TRACE_V(layer, event) do { if (cid == 0 && sub == 0) mega_stamp(p_syncws, 3, layer, event); } while (0)
#else
#define TRACE_C(layer, event) do {} while (0)
#define TRACE_V(layer, event) do {} while (0)
#endif
