// HOST-ONLY shared-memory / occupancy probe for the persistent-B grouped MoE
// decode-in-mainloop kernel (ROADMAP K1.1).
//
// A standalone main() developer tool, NOT a serving source: nothing loads it,
// and everything under csrc/tools/ is kept in the repo and the sdist but
// excluded from the wheel (see pyproject.toml / MANIFEST.in). Its OUTPUT is
// what ships -- the smem/occupancy table in cb_moe_persistent_b.cu's file-top
// comment and in docs/KERNELS.md -- so re-run it whenever a tile shape, the
// K-stage width, or the packed-row padding changes.
//
// It recomputes the SAME arithmetic `cb_moe_persistent_b.cu`'s
// `cfg_smem_bytes` / `ts_padded` perform, for the compiled ladder crossed with
// the FP4-CB v2 rung ladder, and reports:
//
//   * the three shared-memory regions (A double-buffered stages, the decoded
//     B tile, the packed superblock staging),
//   * CTAs per SM against the 102,400-byte sm_120 budget, counting the ~1 KiB
//     the hardware reserves per CTA -- the term that is easy to forget and
//     that cost 1.9x when a resident-codebook experiment tripped over it (see
//     `kSmemReservedPerCta` in the kernel),
//   * the accumulator register count per thread, which is the OTHER bound on
//     how large TM can grow.
//
// There is NO kernel launch and NO CUDA runtime call, so this binary runs on a
// GPU-less box and needs no CUTLASS and no torch.
//
// build (from the repo root):
//   nvcc -std=c++17 -O3 gridbook/csrc/tools/persistent_b_probe.cu -o pbprobe
// or, since it is plain host C++:
//   c++ -std=c++17 -O2 -x c++ gridbook/csrc/tools/persistent_b_probe.cu -o pbprobe
#include <cstdio>
#include <initializer_list>

namespace {

// ---- MIRRORED FROM cb_moe_persistent_b.cu -------------------------------
// Kept as literals rather than an #include because that translation unit
// pulls in torch/extension.h. Any divergence shows up immediately as a table
// that disagrees with `cb_moe_persistent_b_configs()`, which the test suite
// pins against the kernel itself.
constexpr int kTK = 64;                       // BF16 columns per K stage
constexpr int kPublicNvfp4MaxK = 25;
constexpr long kSm120SmemCapacity = 101376;   // 99 KiB opt-in per CTA
constexpr long kSmemPerSm = 102400;           // 100 KiB per SM
constexpr long kSmemReservedPerCta = 1024;

struct TileCfg {
  int tm;
  int tn;
  int warps;
};

constexpr TileCfg kCfgs[] = {
    {128, 64, 8},
    {64, 64, 4},
    {128, 32, 4},
    {64, 128, 8},
};
constexpr int kNumCfgs = int(sizeof(kCfgs) / sizeof(kCfgs[0]));

// Tiles that were compiled, measured and DROPPED. Reported here so the
// occupancy reason stays reproducible rather than remembered.
constexpr TileCfg kRejected[] = {
    {128, 128, 8},
    {256, 64, 8},
};
constexpr int kNumRejected = int(sizeof(kRejected) / sizeof(kRejected[0]));

int ts_padded(int type_size) { return ((type_size + 3) / 4) * 4 + 8; }

long a_bytes(TileCfg c) { return 2L * c.tm * kTK * 2; }
long b_bytes(TileCfg c) { return (long)c.tn * kTK * 2; }
long pk_bytes(TileCfg c, int type_size) {
  return (long)c.tn * ts_padded(type_size);
}
long smem_bytes(TileCfg c, int type_size) {
  return a_bytes(c) + b_bytes(c) + pk_bytes(c, type_size);
}

int ctas_per_sm(long smem) {
  return int(kSmemPerSm / (smem + kSmemReservedPerCta));
}

// Warp grid: WN = TN/32 columns by WM = WARPS/WN rows, so every warp owns a
// 32x32 output patch -- 2 M-atoms x 4 N-atoms x 4 accumulators.
int accum_regs(TileCfg c) {
  const int wn = c.tn / 32;
  const int wm = c.warps / wn;
  const int matom = (c.tm / wm) / 16;
  const int natom = (c.tn / wn) / 8;
  return matom * natom * 4;
}

void report(const char* title, const TileCfg* cfgs, int n, int type_size,
            int k_bits) {
  std::printf("\n%s  (k=%d, type_size=%d, ts_pad=%d)\n", title, k_bits,
              type_size, ts_padded(type_size));
  std::printf("  cfg   TM   TN  warps  thr        A       B      pk      "
              "smem  CTAs/SM  accum/thr  fits99K\n");
  for (int i = 0; i < n; ++i) {
    const TileCfg c = cfgs[i];
    const long s = smem_bytes(c, type_size);
    std::printf("  %3d  %3d  %3d  %5d  %3d  %7ld %7ld %7ld  %8ld  %5d    "
                "%7d      %s\n",
                i + 1, c.tm, c.tn, c.warps, c.warps * 32, a_bytes(c),
                b_bytes(c), pk_bytes(c, type_size), s, ctas_per_sm(s),
                accum_regs(c), s <= kSm120SmemCapacity ? "yes" : "NO");
  }
}

}  // namespace

int main() {
  std::printf("persistent-B grouped MoE: shared-memory / occupancy budget\n");
  std::printf("TK=%d BF16 columns per stage; sm_120 budget %ld B/CTA, "
              "%ld B/SM, %ld B reserved per CTA\n",
              kTK, kSm120SmemCapacity, kSmemPerSm, kSmemReservedPerCta);
  std::printf("CTAs/SM = floor(%ld / (smem + %ld)); the reservation is part "
              "of the divisor -- omitting it is what silently costs the\n"
              "second CTA (see kSmemReservedPerCta in the kernel).\n",
              kSmemPerSm, kSmemReservedPerCta);

  // The public FP4-CB v2 rung ladder: type_size == 4*k + 9. K25 is the
  // widest public packed superblock and therefore the sizing authority.
  for (int k : {1, 12, 16, 20, 24, kPublicNvfp4MaxK}) {
    report("COMPILED LADDER", kCfgs, kNumCfgs, 4 * k + 9, k);
  }
  report("COMPILED, MEASURED, DROPPED (1 CTA/SM; never won a sweep cell)",
         kRejected, kNumRejected, 4 * kPublicNvfp4MaxK + 9,
         kPublicNvfp4MaxK);

  std::printf("\nEvery compiled config must show CTAs/SM >= 2 at public "
              "ceiling k=%d; run_persistent_b TORCH_CHECKs exactly that.\n",
              kPublicNvfp4MaxK);
  return 0;
}
