#pragma once

// Zvt (the RISC-V Vector Matrix Extension, VME): the RISC-V Bulk
// Normalization Algorithm (RVBNA) dot product, into FP32, for the SEW <= 16
// floating-point matrix multiplies.
//
// [RVBNA] is src/unpriv/rvbna.adoc in riscv/riscv-isa-manual at commit
// e03ff979b15bf11f17afe2f6619835afd4e346b4 (CC-BY-4.0); [VME] (zvt.adoc at
// 98b0e96fb240a0b6a54bf814fc2f6a139a409e10, Matrix Arithmetic Instructions)
// requires it for SEW <= 16.  This header has no dependency on the generated
// model, so that it can be tested on its own.
//
// The choices this model makes where [VME] and [RVBNA] leave something
// open, each marked "Choice" below:
//   1. n, the static dimension, is KMAX: 4 at SEW = 8 and 2 at SEW = 16.
//      Elements k >= tk are passed as +0, the identity product that the
//      [RVBNA] note on n describes for inactive elements.
//   2. The final normalization rounds to odd, as the [RVBNA] prose says.
//      Two lines of the [RVBNA] pseudocode, read literally, do not: the jam
//      bit of a normal result (line 213) tests bits that are always zero,
//      and the denormalization shift of a subnormal result (lines 233-234)
//      shrinks as the exponent falls.
//   3. OFP8 special values follow the OCP 8-bit floating point
//      specification: E4M3 has no infinities, and S.1111.111 is its only NaN;
//      E5M2 has IEEE 754 infinities and NaNs.  An E5M2 NaN is signaling when
//      its mantissa MSB is 0, as in IEEE 754; an E4M3 NaN is quiet.

#include <cstdint>

// softfloat.h is missing #ifdef __cplusplus etc.
extern "C" {
#include "softfloat.h"
}

namespace rvbna {

// An operand format: exponent and mantissa widths, and whether it is E4M3.
struct Format {
  int e;     // biased exponent width
  int m;     // mantissa (trailing significand) width
  bool e4m3; // OCP E4M3 special values
};

constexpr Format E4M3 = {4, 3, true};
constexpr Format E5M2 = {5, 2, false};
constexpr Format FP16 = {5, 10, false};
constexpr Format BF16 = {8, 7, false};

// Flag bits as sail-riscv's SoftFloat interface returns them.
constexpr uint8_t FLAG_NV = 0x10;
constexpr uint8_t FLAG_OF = 0x04;

struct Result {
  uint32_t bits;
  uint8_t flags;
};

// Classification of one operand.
struct Operand {
  uint32_t sign, exp, mant;
  bool sub, zero, inf, nan, snan;
};

inline Operand classify(const Format &fmt, uint32_t x) {
  Operand o;
  uint32_t maskExp = (1u << fmt.e) - 1;
  uint32_t maskMant = (1u << fmt.m) - 1;
  o.exp = (x >> fmt.m) & maskExp;
  o.mant = x & maskMant;
  o.sign = (x >> (fmt.e + fmt.m)) & 1;
  o.sub = o.exp == 0;
  o.zero = o.sub && o.mant == 0;
  if (fmt.e4m3) {
    // Choice 3: E4M3 has no infinities; S.1111.111 is its only NaN, quiet.
    o.inf = false;
    o.nan = o.exp == maskExp && o.mant == maskMant;
    o.snan = false;
  } else {
    o.inf = o.exp == maskExp && o.mant == 0;
    o.nan = o.exp == maskExp && o.mant != 0;
    o.snan = o.nan && (o.mant & (1u << (fmt.m - 1))) == 0;
  }
  return o;
}

// x >> s for any s >= 0, and whether any bit shifted out was 1.
inline uint64_t shift_right_jam(uint64_t x, int s, bool &jam) {
  if (s <= 0) {
    jam = false;
    return x;
  }
  if (s >= 64) {
    jam = x != 0;
    return 0;
  }
  jam = (x & ((uint64_t(1) << s) - 1)) != 0;
  return x >> s;
}

inline int log2_exact(int n) {
  int k = 0;
  while ((1 << k) < n)
    k++;
  return k;
}

// The [RVBNA] BulkNormalizedDotProduct of a[0..n-1] (format fa) and
// b[0..n-1] (format fb), into FP32 (q = 24, f = 8).  n is a power of 2.
inline Result dot(const Format &fa, const Format &fb, const uint32_t *a, const uint32_t *b, int n) {
  const int q = 24, f = 8;
  const int g = log2_exact(n), o = g;
  const int p_l = fa.m + 1, p_r = fb.m + 1;
  const int64_t overflowExp = (1 << f) - 1;
  const int64_t lhs_bias = (1 << (fa.e - 1)) - 1;
  const int64_t rhs_bias = (1 << (fb.e - 1)) - 1;
  const int64_t res_bias = (1 << (f - 1)) - 1;
  const int64_t prodOpBias = lhs_bias + rhs_bias;

  bool nanResult = false, infiniteResult = false, invalidFlag = false;
  uint32_t infiniteSign = 0;
  int64_t maxExp = 0;
  uint32_t prodSigns[8];
  uint64_t prodSigs[8];
  int64_t prodRefExps[8];

  // The maximum reference exponent, and the special cases.
  for (int i = 0; i < n; i++) {
    Operand A = classify(fa, a[i]);
    Operand B = classify(fb, b[i]);
    prodSigns[i] = A.sign ^ B.sign;
    bool prod_isZero = A.zero || B.zero;
    bool invalidProd = (A.inf && B.zero) || (B.inf && A.zero);
    bool infiniteProd = (A.inf && !B.nan && !B.zero) || (B.inf && !A.nan && !A.zero);
    bool invalidSum = infiniteResult && infiniteProd && (infiniteSign != prodSigns[i]);
    infiniteResult = infiniteResult || infiniteProd;
    invalidFlag = invalidFlag || invalidProd || invalidSum || A.snan || B.snan;
    if (infiniteProd)
      infiniteSign = prodSigns[i];
    nanResult = nanResult || A.nan || B.nan || invalidProd || invalidSum;
    uint64_t A_sig = (uint64_t(!A.sub) << (p_l - 1)) | A.mant;
    uint64_t B_sig = (uint64_t(!B.sub) << (p_r - 1)) | B.mant;
    prodSigs[i] = A_sig * B_sig;
    int64_t A_ref = A.sub ? 1 : A.exp;
    int64_t B_ref = B.sub ? 1 : B.exp;
    prodRefExps[i] = prod_isZero ? 0 : A_ref + B_ref;
    if (prodRefExps[i] > maxExp)
      maxExp = prodRefExps[i];
  }

  if (nanResult)
    return {(uint32_t(overflowExp) << (q - 1)) | (1u << (q - 2)), uint8_t(invalidFlag ? FLAG_NV : 0)};
  if (infiniteResult)
    return {(infiniteSign << (q + f - 1)) | (uint32_t(overflowExp) << (q - 1)), 0};

  // Aligning each product on maxExp, rounding it to odd at q - 1 + g
  // fractional bits, and accumulating.
  const int padRight = q + 1 + g - (p_l + p_r);
  int64_t accumulator = 0;
  for (int i = 0; i < n; i++) {
    int64_t alignShift = maxExp - prodRefExps[i];
    bool jam;
    uint64_t aligned = shift_right_jam(prodSigs[i] << padRight, int(alignShift > 64 ? 64 : alignShift), jam);
    aligned |= jam ? 1 : 0;
    accumulator += prodSigns[i] ? -int64_t(aligned) : int64_t(aligned);
  }

  if (accumulator == 0)
    return {0, 0}; // a zero result is always +0
  uint32_t accSign = accumulator < 0;
  uint64_t accAbs = accSign ? uint64_t(-accumulator) : uint64_t(accumulator);

  // Normalizing and rounding to odd (Choice 2).  The accumulator's least
  // significant bit weighs 2^(maxExp - prodOpBias - (q - 1 + g)); W is the
  // accumulator width g + q + 1 + o of the [RVBNA] leading-zero count.
  const int W = g + q + 1 + o;
  int msb = 63;
  while (!((accAbs >> msb) & 1))
    msb--;
  int lzc = W - 1 - msb;
  int64_t resExp = (maxExp + o + 1 - lzc) - prodOpBias + res_bias;

  if (resExp >= overflowExp)
    return {(accSign << (q + f - 1)) | (uint32_t(overflowExp) << (q - 1)), FLAG_OF};

  if (resExp >= 1) {
    // q significant bits, the leading one at bit q - 1.
    bool jam;
    uint64_t sig = shift_right_jam(accAbs, msb - (q - 1), jam);
    if (msb < q - 1)
      sig = accAbs << (q - 1 - msb);
    sig |= jam ? 1 : 0;
    uint32_t mant = uint32_t(sig) & ((1u << (q - 1)) - 1);
    return {(accSign << (q + f - 1)) | (uint32_t(resExp) << (q - 1)) | mant, 0};
  }

  // A subnormal result: the significand is the value in units of the
  // smallest subnormal, 2^(1 - res_bias - (q - 1)); with the weight of the
  // accumulator's LSB, that is accAbs >> (msb - (q - 1) + 1 - resExp).
  int64_t s = int64_t(msb) - (q - 1) + 1 - resExp;
  bool jam;
  uint64_t sig = s < 0 ? accAbs << -s : shift_right_jam(accAbs, int(s > 64 ? 64 : s), jam);
  if (s < 0)
    jam = false;
  sig |= jam ? 1 : 0;
  return {(accSign << (q + f - 1)) | uint32_t(sig), 0};
}

// The FP32 sum a + b rounded to odd, with the two divergences of [RVBNA]
// Rounding to Odd behavior in dot product mode: an overflow gives an
// infinity (SoftFloat's round to odd gives the largest finite value), and a
// zero result is always +0.  The flags are SoftFloat's.
inline Result add_odd(uint32_t x, uint32_t y) {
  softfloat_exceptionFlags = 0;
  softfloat_roundingMode = softfloat_round_odd;
  float32_t a, b;
  a.v = x;
  b.v = y;
  uint32_t r = f32_add(a, b).v;
  if (softfloat_exceptionFlags & softfloat_flag_overflow)
    r = (r & 0x80000000u) | 0x7F800000u;
  if ((r & 0x7FFFFFFFu) == 0)
    r = 0;
  return {r, uint8_t(softfloat_exceptionFlags)};
}

} // namespace rvbna
