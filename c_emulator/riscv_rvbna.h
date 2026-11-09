#pragma once

#include "riscv_softfloat.h"

// Zvt (the RISC-V Vector Matrix Extension, VME): the SEW <= 16 floating-point
// dot products, by the RISC-V Bulk Normalization Algorithm (RVBNA), and the
// FP32 add rounding to odd that accumulates them.  See riscv_rvbna.cpp.

// The FP32 sum v1 + v2, rounded to odd with RVBNA's two divergences: an
// overflow gives an infinity, and a zero result is +0.  The flags are
// SoftFloat's (NV, DZ, OF, UF, NX from bit 4 down).
bv5_bv32 softfloat_f32addroundodd(uint64_t v1, uint64_t v2);

// The RVBNA dot product, into FP32, of n = 4 OFP8 products: a and b each
// hold four 8-bit elements, element k in bits 8k+7..8k.  a_e5m2 and b_e5m2
// select E5M2 (1) or E4M3 (0) for each side.
bv5_bv32 rvbna_dot_ofp8(uint64_t a_e5m2, uint64_t b_e5m2, uint64_t a, uint64_t b);

// The RVBNA dot product, into FP32, of n = 2 FP16 (resp. BF16) products: a
// and b each hold two 16-bit elements, element k in bits 16k+15..16k.
bv5_bv32 rvbna_dot_fp16(uint64_t a, uint64_t b);
bv5_bv32 rvbna_dot_bf16(uint64_t a, uint64_t b);
