#pragma once
// ADAPT HARDWARE/PLACEMENT: these values encode 24 AIC and current tiling.
constexpr int NB = 24;
constexpr int KTILE = 256;
constexpr int SLT = 256;
constexpr int NSL = 3;
constexpr int STEP = 13824;      // ceil(40961/3) rounded to 256
constexpr int SPADG = 41472;     // STEP*NSL
constexpr int NQKV = 4096;
constexpr int NQKV_PAD = 4224;   // NB*176
constexpr int NC_QKV = 96;       // tiles 96 + 80
constexpr int NO_PAD = 2304;     // NB*96
constexpr int NC_O = 96;
constexpr int NGU = 12288;       // NB*512
constexpr int NC_GU = 128;
constexpr int ND_PAD = 2304;     // NB*96
constexpr int NC_D = 96;
constexpr int SPAN = 6336;       // ceil(VV/NB) = 6331 -> 16-mult
#ifndef MEGA_LM_TILE
#define MEGA_LM_TILE 96
#endif
#ifndef MEGA_LM_STREAM
#define MEGA_LM_STREAM 1
#endif
constexpr int PR_LM = 234;       // final-head chunk progress; tag + chunk + 1
#ifndef MEGA_COMPACT_A
#define MEGA_COMPACT_A 1
#endif
#ifndef MEGA_GU_INTERLEAVE
#define MEGA_GU_INTERLEAVE 1
#endif
constexpr int A_PACK_CHUNK = MEGA_COMPACT_A ? 512 : 4096;
constexpr int A_PACK_ROWS = MEGA_COMPACT_A ? 2 : 16;
constexpr int NC_LM = MEGA_LM_TILE;
static_assert(NC_LM > 0 && NC_LM <= 128 && NC_LM % 16 == 0,
              "LM tile must fit the 64KB weight ring slot");
constexpr int NRD_LM = (SPAN + NC_LM - 1) / NC_LM;
constexpr int SPANP = 8192;
constexpr int CH = 2048;
constexpr int NCH = SPAN / CH + 1;    // 4 (last chunk 192 valid)
constexpr int NTM = STEP / SLT;       // 54 max tiles per slice
constexpr float EPS = 1e-6f;
constexpr float NEG_INF = -1e30f;
// chunk-major weight pack: elements per layer per gemv
//   = NB * nrd * nk * (16 kfrac * maxncc nfrac * 256)
constexpr int QKV_LAYER = NB * 2 * (H / KTILE) * 16 * (NC_QKV / 16) * 256;   // 9437184
constexpr int O_LAYER = NB * 1 * (H / KTILE) * 16 * (NC_O / 16) * 256;       // = NO_PAD*H
constexpr int GU_LAYER = NB * 4 * (H / KTILE) * 16 * (NC_GU / 16) * 256;     // = NGU*H
constexpr int D_LAYER = NB * 1 * (II / KTILE) * 16 * (NC_D / 16) * 256;      // = ND_PAD*II
constexpr int LM_LAYER = NB * NRD_LM * (H / KTILE) * 16 * (NC_LM / 16) * 256;  // includes per-slab N-tile padding
