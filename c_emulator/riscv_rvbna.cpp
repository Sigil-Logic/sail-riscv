#include "riscv_rvbna.h"
#include "riscv_rvbna_core.h"

bv5_bv32 softfloat_f32addroundodd(uint64_t v1, uint64_t v2) {
  rvbna::Result r = rvbna::add_odd(uint32_t(v1), uint32_t(v2));
  return {r.flags, r.bits};
}

namespace {

bv5_bv32 pack(const rvbna::Result &r) {
  return {r.flags, r.bits};
}

void split(uint64_t x, int w, int n, uint32_t *out) {
  for (int k = 0; k < n; k++)
    out[k] = uint32_t((x >> (w * k)) & ((uint64_t(1) << w) - 1));
}

} // namespace

bv5_bv32 rvbna_dot_ofp8(uint64_t a_e5m2, uint64_t b_e5m2, uint64_t a, uint64_t b) {
  uint32_t av[4], bv[4];
  split(a, 8, 4, av);
  split(b, 8, 4, bv);
  return pack(rvbna::dot(a_e5m2 ? rvbna::E5M2 : rvbna::E4M3, b_e5m2 ? rvbna::E5M2 : rvbna::E4M3, av, bv, 4));
}

bv5_bv32 rvbna_dot_fp16(uint64_t a, uint64_t b) {
  uint32_t av[2], bv[2];
  split(a, 16, 2, av);
  split(b, 16, 2, bv);
  return pack(rvbna::dot(rvbna::FP16, rvbna::FP16, av, bv, 2));
}

bv5_bv32 rvbna_dot_bf16(uint64_t a, uint64_t b) {
  uint32_t av[2], bv[2];
  split(a, 16, 2, av);
  split(b, 16, 2, bv);
  return pack(rvbna::dot(rvbna::BF16, rvbna::BF16, av, bv, 2));
}
