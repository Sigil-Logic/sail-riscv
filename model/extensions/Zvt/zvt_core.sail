// =======================================================================================
// This Sail RISC-V architecture model, comprising all files and
// directories except where otherwise noted is subject the BSD
// two-clause license in the LICENSE file.
//
// SPDX-License-Identifier: BSD-2-Clause
// =======================================================================================

// Zvt (the RISC-V Vector Matrix Extension, VME): the pure core.
//
// The functions here read no register and write none: the execute clauses
// (zvt_insts.sail) read the state, call these, and write the results.  That
// split keeps the semantics of each instruction separate from the register
// plumbing, so that each core function can be read, tested and reasoned
// about on its own.
//
// The specification is src/unpriv/zvt.adoc in riscv/riscv-isa-manual at commit
// 98b0e96fb240a0b6a54bf814fc2f6a139a409e10 (CC-BY-4.0), cited as [VME]; [V] is
// src/unpriv/vector-common.adoc at the same commit.  Where [VME] leaves a
// choice open, or is silent, the comments say which choice this model makes.
//
// Numbers are Sail ints.  zvt_div and zvt_mod are total, with x / 0 = 0 and
// x mod 0 = x, so that no definition needs a separate nonzero-divisor case.

val zvt_div : forall 'b. (int, int('b)) -> int
function zvt_div(a, b) = if b == 0 then 0 else quot_round_zero(a, b)

val zvt_mod : forall 'b. (int, int('b)) -> int
function zvt_mod(a, b) = if b == 0 then a else rem_round_zero(a, b)

// ---- Derived quantities -----------------------------------------------------

// TWIDEN = 1 << (mtwiden - 1), with mtwiden = 0 giving 1.
function zvt_twiden(mtwiden : bits(2)) -> int =
  match mtwiden { 0b10 => 2, 0b11 => 4, _ => 1 }

// ETE = TEW < 64 ? TE : TE / 2 ([VME] Matrix Tile State).
function zvt_ete(tew : int) -> int = if tew < 64 then te else zvt_div(te, 2)

// EVE = VLEN / SEW.
function zvt_eve(sew : int) -> int = zvt_div(vlen, sew)

// The number of tiles at width tew; 0 for a width with no tiles.
function zvt_ntiles(tew : int) -> int =
  if tew == 8 then 16
  else if tew == 16 then 8
  else if tew == 32 then 4
  else if tew == 64 then 8
  else 0

// The spacing of valid tile specifiers: 16 / NTiles.
function zvt_tile_stride(tew : int) -> int = zvt_div(16, zvt_ntiles(tew))

// A tile specifier t is valid at width tew when t < 16 and
// t mod (16 / NTiles) = 0.
function zvt_valid_tile(tew : int, t : int) -> bool =
  t < 16 & 0 < zvt_ntiles(tew) & zvt_mod(t, zvt_tile_stride(tew)) == 0

// The KMAX table ([VME] Supported SEW, TWIDEN, and KMAX values), total over
// the pairs it lists.
function zvt_kmax(sew : int, tw : int) -> option(int) =
  if sew == 8 & tw == 1 then Some(4)
  else if sew == 8 & tw == 2 then Some(4)
  else if sew == 16 & tw == 1 then Some(2)
  else if sew == 8 & tw == 4 then Some(4)
  else if sew == 16 & tw == 2 then Some(2)
  else if sew == 32 & tw == 1 then Some(1)
  else if sew == 16 & tw == 4 then Some(2)
  else if sew == 32 & tw == 2 then Some(1)
  else if sew == 64 & tw == 1 then Some(1)
  else None()

// KMAX, with 0 for a pair outside the table.
function zvt_kmax_d(sew : int, tw : int) -> int =
  match zvt_kmax(sew, tw) { Some(k) => k, None() => 0 }

// The TEWs available to moves, loads and stores (this model's reading of
// [VME]): 32 always, and 8, 16 and 64 with Zvt8e, Zvt16e and Zvt64e.
function zvt_supported_tew(tew : int) -> bool =
    tew == 32
  | (tew == 8 & hartSupports(Ext_Zvt8e))
  | (tew == 16 & hartSupports(Ext_Zvt16e))
  | (tew == 64 & hartSupports(Ext_Zvt64e))

// Zvt<N>t means TE >= N; derived from TE, not a parameter.
function zvt_nt(n : int) -> bool = n <= te

// The bytes of tile storage, and of the vector register file.
function zvt_tile_bytes() -> int = 16 * te * te
function zvt_vreg_bytes() -> int = 32 * zvt_div(vlen, 8)

// ---- vtype ------------------------------------------------------------------

// SEW = 8 * 2^vsew.
function zvt_sew(vt : Vtype) -> int = 8 * pow2(unsigned(vt[vsew]))

// LMUL = 2^vlmul with vlmul signed, as lmul_num / lmul_den.
function zvt_lmul_num(vt : Vtype) -> int =
  if unsigned(vt[vlmul]) < 4 then pow2(unsigned(vt[vlmul])) else 1

function zvt_lmul_den(vt : Vtype) -> int =
  if unsigned(vt[vlmul]) < 5 then 1 else pow2(8 - unsigned(vt[vlmul]))

// VLMAX = LMUL * VLEN / SEW.
function zvt_vlmax(vt : Vtype) -> int =
  zvt_div(zvt_lmul_num(vt) * vlen, zvt_lmul_den(vt) * zvt_sew(vt))

// The registers in a group: max(1, LMUL).
function zvt_emul_regs(vt : Vtype) -> int = zvt_lmul_num(vt)

// ---- Punning ([VME] Tile Punning) -------------------------------------------

// The byte offset of element (row, col) of tile at width tew in tile storage
// viewed as 16 * TE * TE bytes ([VME] Tile Punning, case by case).  The
// default case returns 0 and is unreachable under legality.
function zvt_punning(tile : int, row : int, col : int, tew : int) -> int = {
  let m = zvt_div(te, 4);
  let (ptile, minor, major) : (int, int, int) =
    if tew == 8 then
      (tile,
       zvt_mod(row, 4) * 4 + zvt_mod(col, 4),
       zvt_div(row, 4) * m + zvt_div(col, 4))
    else if tew == 16 then
      (tile + zvt_mod(zvt_div(row, 2), 2),
       zvt_mod(row, 2) * 4 + zvt_mod(col, 2) * 2 + zvt_mod(zvt_div(col, 2), 2) * 8,
       zvt_div(row, 4) * m + zvt_div(col, 4))
    else if tew == 32 then
      (tile + 2 * zvt_div(row, zvt_div(te, 2)) + zvt_mod(zvt_div(col, 2), 2),
       zvt_mod(row, 2) * 8 + zvt_mod(col, 2) * 4,
       zvt_mod(zvt_div(row, 2), m) * m + zvt_div(col, 4))
    else if tew == 64 then
      (tile + zvt_div(row, m),
       zvt_mod(col, 2) * 8,
       zvt_mod(row, m) * m + zvt_div(col, 2))
    else (0, 0, 0);
  if tew == 8 | tew == 16 | tew == 32 | tew == 64
  then ptile * te * te + major * 16 + minor
  else 0
}

// Where a byte of tile storage, viewed as 16 * TE * TE bytes, lives in this
// model: physical tile register off / TE^2, byte off mod TE^2 of it.
function zvt_phys_tile(off : int) -> int = zvt_div(off, te * te)
function zvt_phys_byte(off : int) -> int = zvt_mod(off, te * te)

// ---- The tile subset specifier ([VME] Tile Subset Specifier) ----------------

function zvt_tss_tile(tss : xlenbits) -> int = unsigned(tss[30 .. 27])
function zvt_tss_pattern(tss : xlenbits) -> int = unsigned(tss[26 .. 24])
function zvt_tss_index(tss : xlenbits) -> int = unsigned(tss[23 .. 0])
function zvt_tss_reserved(tss : xlenbits) -> bool = tss[xlen - 1 .. 31] != zeros()

// The tile a TSS selects at width tew: its tile field with the
// log2(16 / NTiles) low bits ignored.
function zvt_tss_eff_tile(tew : int, tss : xlenbits) -> int = {
  let t = zvt_tss_tile(tss);
  t - zvt_mod(t, zvt_tile_stride(tew))
}

// Element j of the subset, as (row, col): pattern 0 selects row index,
// pattern 1 column index.
function zvt_tss_elem(tss : xlenbits, j : int) -> (int, int) =
  if zvt_tss_pattern(tss) == 0 then (zvt_tss_index(tss), j) else (j, zvt_tss_index(tss))

// A TSS is not reserved: bits XLEN-1:31 are zero, the pattern is 0 or 1, and
// the index is below ETE.  This model makes every reserved case of [VME] an
// illegal-instruction exception.
function zvt_tss_ok(tew : int, tss : xlenbits) -> bool =
  not(zvt_tss_reserved(tss)) & zvt_tss_pattern(tss) <= 1 & zvt_tss_index(tss) < zvt_ete(tew)

// ---- Arithmetic formats -----------------------------------------------------

enum zvt_fmt = {
  ZVT_UInt8, ZVT_Int8, ZVT_Int32, ZVT_E4M3, ZVT_E5M2,
  ZVT_FP16, ZVT_BF16, ZVT_FP32, ZVT_FP64,
}

function zvt_is_fp_op(op : zvt_arith_op) -> bool =
  match op { ZVT_VTFMM => true, ZVT_VTFMM_ALT => true, _ => false }

// The arithmetic type table ([VME] Matrix Arithmetic Instructions): the
// formats of A, B and C, and the extension that provides the row.  Any
// other combination is reserved, and so illegal in this model.
function zvt_type_row(op : zvt_arith_op, sw : int, tw : int, alt : bool)
    -> option((zvt_fmt, zvt_fmt, zvt_fmt, extension)) =
  match op {
    ZVT_VTFMM =>
      if sw == 8 & tw == 4 then Some(ZVT_E4M3, if alt then ZVT_E5M2 else ZVT_E4M3, ZVT_FP32, Ext_Zvtofp8fmm)
      else if sw == 16 & tw == 2 & not(alt) then Some(ZVT_FP16, ZVT_FP16, ZVT_FP32, Ext_Zvtfp16fmm)
      else if sw == 32 & tw == 1 & not(alt) then Some(ZVT_FP32, ZVT_FP32, ZVT_FP32, Ext_Zvtfmm)
      else if sw == 64 & tw == 1 & not(alt) then Some(ZVT_FP64, ZVT_FP64, ZVT_FP64, Ext_Zvtdmm)
      else None(),
    ZVT_VTFMM_ALT =>
      if sw == 8 & tw == 4 then Some(ZVT_E5M2, if alt then ZVT_E5M2 else ZVT_E4M3, ZVT_FP32, Ext_Zvtofp8fmm)
      else if sw == 16 & tw == 2 & alt then Some(ZVT_BF16, ZVT_BF16, ZVT_FP32, Ext_Zvtbf16fmm)
      else None(),
    ZVT_VTMMU =>
      if sw == 8 & tw == 4 then Some(ZVT_UInt8, if alt then ZVT_Int8 else ZVT_UInt8, ZVT_Int32, Ext_Zvti8i32mm)
      else None(),
    ZVT_VTMMS =>
      if sw == 8 & tw == 4 then Some(ZVT_Int8, if alt then ZVT_Int8 else ZVT_UInt8, ZVT_Int32, Ext_Zvti8i32mm)
      else None(),
  }

// ---- Legality ---------------------------------------------------------------

// TEW of vtzero: SEW * TWIDEN (this model's reading of [VME]).
function zvt_vtzero_tew(vt : Vtype, mt : Mtype) -> int = zvt_sew(vt) * zvt_twiden(mt[mtwiden])

// TEW of the moves: SEW, whatever TWIDEN is (this model's reading of [VME]).
function zvt_move_tew(vt : Vtype) -> int = zvt_sew(vt)

// The matrix unit is configured: mtwiden != 0.
function zvt_configured(mt : Mtype) -> bool = mt[mtwiden] != 0b00

// A vector register group specifier: in range and a multiple of the group
// size ([V] Vector Register Grouping).
function zvt_vgroup_ok(vt : Vtype, v : int) -> bool =
  v < 32 & zvt_mod(v, zvt_emul_regs(vt)) == 0

// An A or B operand specifier: a group specifier whose value modulo 8 is
// less than 8 / KMAX ([VME] Matrix-Multiply Operations).
function zvt_operand_ok(vt : Vtype, km : int, v : int) -> bool =
  zvt_vgroup_ok(vt, v) & zvt_mod(v, 8) < zvt_div(8, km)

// vtzero (this model's reading of [VME]).
function zvt_legal_vtzero(vt : Vtype, mt : Mtype, vst : int, mtd : int) -> bool =
  vt[vill] == 0b0 & zvt_configured(mt) & vst == 0
  & zvt_valid_tile(zvt_vtzero_tew(vt, mt), mtd)

// The four arithmetic instructions.  frm is checked at SEW >= 32, which
// rounds with it; an invalid frm is then illegal (this model's reading of [VME]).
function zvt_legal_arith(vt : Vtype, mt : Mtype, vst : int, frm : bits(3),
                         op : zvt_arith_op, mtd : int, vs2 : int, vs1 : int) -> bool = {
  let sw = zvt_sew(vt);
  let tw = zvt_twiden(mt[mtwiden]);
  let km = zvt_kmax_d(sw, tw);
  let row_ok : bool = match zvt_type_row(op, sw, tw, vt[altfmt] == 0b1) {
    Some(_, _, _, e) => hartSupports(e),
    None()           => false,
  };
  vt[vill] == 0b0 & zvt_configured(mt) & row_ok & vst == 0
  & zvt_mod(mtd, 2) == 0 & zvt_valid_tile(sw * tw, mtd)
  & zvt_operand_ok(vt, km, vs2) & zvt_operand_ok(vt, km, vs1)
  & (not(zvt_is_fp_op(op)) | sw < 32 | unsigned(frm) <= 4)
}

// vtmv.v.t and vtmv.t.v.
function zvt_legal_move(vt : Vtype, mt : Mtype, v : int, tss : xlenbits) -> bool = {
  let tew = zvt_move_tew(vt);
  vt[vill] == 0b0 & zvt_configured(mt) & zvt_supported_tew(tew) & zvt_tss_ok(tew, tss)
  & zvt_vgroup_ok(vt, v)
}

// vtle and vtse.
function zvt_legal_ldst(vt : Vtype, mt : Mtype, eew : int, tss : xlenbits) -> bool =
  vt[vill] == 0b0 & zvt_configured(mt) & zvt_supported_tew(eew) & zvt_tss_ok(eew, tss)

// msettm and msettk need a configured matrix unit; msetmtype[i] and msettn
// are always legal.
function zvt_legal_msettm_k(mt : Mtype) -> bool = zvt_configured(mt)

// vtdiscard.
function zvt_legal_vtdiscard(vt : Vtype) -> bool = vt[vill] == 0b0

// ---- Tile storage and vector registers as values ---------------------------

// The core takes tile storage and the vector register file as values and
// returns new tile storage; the execute clauses read mt0..mt15 and vr0..vr31
// into these and write back the result.
type zvt_tiles = vector(16, tilebits)
type zvt_vrf = vector(32, vlenbits)

// The low w bits set, for 0 <= w <= 64.
function zvt_low_mask(w : int) -> bits(64) =
  if w >= 64 then ones() else if w <= 0 then zeros() else sail_shiftright(ones() : bits(64), 64 - w)

// The w-bit field of x at bit lo, zero-extended to 64 bits.
val zvt_extract : forall 'n, 'n >= 0. (bits('n), int, int) -> bits(64)
function zvt_extract(x, lo, w) = {
  assert(lo >= 0, "zvt_extract: negative offset");
  sail_mask(64, sail_shiftright(x, lo)) & zvt_low_mask(w)
}

// x with its w-bit field at bit lo replaced by the low w bits of v.
val zvt_insert : forall 'n, 'n >= 0. (bits('n), int, int, bits(64)) -> bits('n)
function zvt_insert(x, lo, w, v) = {
  assert(lo >= 0, "zvt_insert: negative offset");
  let m = sail_shiftleft(sail_mask(length(x), zvt_low_mask(w)), lo);
  (x & ~(m)) | sail_shiftleft(sail_mask(length(x), v & zvt_low_mask(w)), lo)
}

// tile_read: element (row, col) of logical tile `tile` at width tew.  Its
// tew / 8 bytes lie in one physical tile, since punning's minor offset plus
// tew / 8 is at most 16.
function zvt_tile_read(tl : zvt_tiles, tew : int, tile : int, row : int, col : int) -> bits(64) = {
  let off = zvt_punning(tile, row, col, tew);
  let p = zvt_phys_tile(off);
  assert(0 <= p & p < 16, "zvt_tile_read: tile out of range");
  zvt_extract(tl[p], 8 * zvt_phys_byte(off), tew)
}

// tile_write: tile storage with element (row, col) set to the low tew bits
// of v; no other byte changes.
function zvt_tile_write(tl : zvt_tiles, tew : int, tile : int, row : int, col : int,
                        v : bits(64)) -> zvt_tiles = {
  let off = zvt_punning(tile, row, col, tew);
  let p = zvt_phys_tile(off);
  assert(0 <= p & p < 16, "zvt_tile_write: tile out of range");
  [tl with p = zvt_insert(tl[p], 8 * zvt_phys_byte(off), tew, v)]
}

// vreg_off: the byte offset of element i of the eew-wide view of the group
// based at register v ([V] Mapping of Vector Elements to Vector Register
// State).
function zvt_vreg_off(eew : int, v : int, i : int) -> int =
  v * zvt_div(vlen, 8) + i * zvt_div(eew, 8)

// vreg_read: that element, zero-extended to 64 bits.  An element never
// straddles two registers, since VLEN is a multiple of EEW.
function zvt_vreg_read(vr : zvt_vrf, w : int, v : int, i : int) -> bits(64) = {
  let vb = zvt_div(vlen, 8);
  let off = zvt_vreg_off(w, v, i);
  let r = zvt_div(off, vb);
  assert(0 <= r & r < 32, "zvt_vreg_read: register out of range");
  zvt_extract(vr[r], 8 * zvt_mod(off, vb), w)
}

// operand_row_reg: the base register of row k of an A or B operand
// ([VME] Matrix-Multiply Operations).
function zvt_operand_row_reg(km : int, vs : int, k : int) -> int = vs + k * zvt_div(8, km)

// Element (k, i) of gather's classical matrix: row k, column i.  Column i of
// the matrix is k |-> zvt_gather_elem(..., k, i) for k < tk.
function zvt_gather_elem(vr : zvt_vrf, w : int, km : int, vs : int, k : int, i : int) -> bits(64) =
  zvt_vreg_read(vr, w, zvt_operand_row_reg(km, vs, k), i)

// ---- Kernels ----------------------------------------------------------------
//
// Each kernel computes one accumulator element from C[i, j], column i of A
// and column j of B, reading the columns from the register file through
// zvt_gather_elem, and folds over k with a loop.

// Whether an integer format is signed.
function zvt_fmt_signed(f : zvt_fmt) -> bool =
  match f { ZVT_Int8 => true, ZVT_Int32 => true, _ => false }

// ext32, as an integer: the operand sign- or zero-extended.
function zvt_ext_int(sgn : bool, x : bits(8)) -> int =
  if sgn then signed(x) else unsigned(x)

// int_dot_acc: c + sum_k a_k * b_k modulo 2^32; no flags.
function zvt_int_dot_acc(sa : bool, sb : bool, c : bits(32), vr : zvt_vrf, km : int,
                         vs2 : int, vs1 : int, tk : int, i : int, j : int) -> bits(32) = {
  var acc : bits(32) = c;
  foreach (k from 0 to (tk - 1)) {
    let a : bits(8) = zvt_gather_elem(vr, 8, km, vs2, k, i)[7 .. 0];
    let b : bits(8) = zvt_gather_elem(vr, 8, km, vs1, k, j)[7 .. 0];
    acc = acc + to_bits_truncate(32, zvt_ext_int(sa, a) * zvt_ext_int(sb, b));
  };
  acc
}

// nv_of: only the invalid (NV) and overflow (OF) flags.  Flags are
// NV, DZ, OF, UF, NX from bit 4 down, as sail-riscv's SoftFloat interface
// returns them.
function zvt_nv_of(f : bits(5)) -> bits(5) = f & 0b10100

// prod_acc at FP32: each product rounded with frm, then added to the
// accumulator with frm; no fused multiply-add.  Products are added in
// ascending k.
function zvt_prod_acc32(frm : bits(3), c : bits(32), vr : zvt_vrf, km : int,
                        vs2 : int, vs1 : int, tk : int, i : int, j : int) -> (bits(32), bits(5)) = {
  var acc : bits(32) = c;
  var fl : bits(5) = zeros();
  foreach (k from 0 to (tk - 1)) {
    let a : bits(32) = zvt_gather_elem(vr, 32, km, vs2, k, i)[31 .. 0];
    let b : bits(32) = zvt_gather_elem(vr, 32, km, vs1, k, j)[31 .. 0];
    let (fl1, m) = riscv_f32Mul(frm, a, b);
    let (fl2, y) = riscv_f32Add(frm, acc, m);
    acc = y;
    fl = fl | fl1 | fl2;
  };
  (acc, zvt_nv_of(fl))
}

// prod_acc at FP64, as at FP32.
function zvt_prod_acc64(frm : bits(3), c : bits(64), vr : zvt_vrf, km : int,
                        vs2 : int, vs1 : int, tk : int, i : int, j : int) -> (bits(64), bits(5)) = {
  var acc : bits(64) = c;
  var fl : bits(5) = zeros();
  foreach (k from 0 to (tk - 1)) {
    let a = zvt_gather_elem(vr, 64, km, vs2, k, i);
    let b = zvt_gather_elem(vr, 64, km, vs1, k, j);
    let (fl1, m) = riscv_f64Mul(frm, a, b);
    let (fl2, y) = riscv_f64Add(frm, acc, m);
    acc = y;
    fl = fl | fl1 | fl2;
  };
  (acc, zvt_nv_of(fl))
}

// The kernel the type table selects, applied to one element.  The SEW <= 16
// floating-point kernels (OFP8, FP16 and BF16, whose dot products [VME]
// specifies by reference to bulk-normalized accumulation) are not yet
// modelled, and assert.
function zvt_kernel(fa : zvt_fmt, fb : zvt_fmt, frm : bits(3), c : bits(64), vr : zvt_vrf,
                    km : int, vs2 : int, vs1 : int, tk : int, i : int, j : int)
    -> (bits(64), bits(5)) =
  match fa {
    ZVT_UInt8 => (zero_extend(64, zvt_int_dot_acc(false, zvt_fmt_signed(fb), c[31 .. 0], vr, km, vs2, vs1, tk, i, j)), zeros()),
    ZVT_Int8  => (zero_extend(64, zvt_int_dot_acc(true, zvt_fmt_signed(fb), c[31 .. 0], vr, km, vs2, vs1, tk, i, j)), zeros()),
    ZVT_FP32  => {
      let (y, f) = zvt_prod_acc32(frm, c[31 .. 0], vr, km, vs2, vs1, tk, i, j);
      (zero_extend(64, y), f)
    },
    ZVT_FP64  => zvt_prod_acc64(frm, c, vr, km, vs2, vs1, tk, i, j),
    _ => {
      assert(false, "Zvt: the SEW <= 16 floating-point kernels are not yet modelled");
      (c, zeros())
    },
  }

// ---- Computation ([VME] Matrix Arithmetic Instructions) ---------------------

// dim_tn: tn = min(vl, ETE).
function zvt_dim_tn(cur_vl : int, tew : int) -> int = min(cur_vl, zvt_ete(tew))

// The tile tail rule, fixed by configuration (zvt_params.sail).
function zvt_tail_ones() -> bool = zvt_tile_tail == ZvtTileTail_Ones

// b1_run: one arithmetic instruction for the formats fa and fb.  If tm, tn
// or tk is 0, tile storage is unchanged and no flags are raised.
// Otherwise every element (i, j) of the destination with i, j < ETE is body
// (i < tm and j < tn), and receives the kernel's result computed from the
// original tile storage; or tail, and becomes all 1s under the ones tail
// policy.
function zvt_b1_run(tl : zvt_tiles, vr : zvt_vrf, vt : Vtype, mt : Mtype, cur_vl : int,
                    frm : bits(3), fa : zvt_fmt, fb : zvt_fmt,
                    mtd : int, vs2 : int, vs1 : int) -> (zvt_tiles, bits(5)) = {
  let sw = zvt_sew(vt);
  let tw = zvt_twiden(mt[mtwiden]);
  let tew = sw * tw;
  let km = zvt_kmax_d(sw, tw);
  let m = unsigned(mt[tm]);
  let n = zvt_dim_tn(cur_vl, tew);
  let kk = unsigned(mt[tk]);
  if m == 0 | n == 0 | kk == 0 then (tl, zeros())
  else {
    let e = zvt_ete(tew);
    var out : zvt_tiles = tl;
    var fl : bits(5) = zeros();
    foreach (i from 0 to (e - 1)) {
      foreach (j from 0 to (e - 1)) {
        if i < m & j < n then {
          let (y, f) = zvt_kernel(fa, fb, frm, zvt_tile_read(tl, tew, mtd, i, j),
                                  vr, km, vs2, vs1, kk, i, j);
          out = zvt_tile_write(out, tew, mtd, i, j, y);
          fl = fl | f;
        } else if zvt_tail_ones() then {
          out = zvt_tile_write(out, tew, mtd, i, j, ones());
        }
      }
    };
    (out, fl)
  }
}

// arith: the type table selects the kernel.  A reserved row
// changes nothing; legality excludes it.
function zvt_arith(tl : zvt_tiles, vr : zvt_vrf, vt : Vtype, mt : Mtype, cur_vl : int,
                   frm : bits(3), op : zvt_arith_op,
                   mtd : int, vs2 : int, vs1 : int) -> (zvt_tiles, bits(5)) =
  match zvt_type_row(op, zvt_sew(vt), zvt_twiden(mt[mtwiden]), vt[altfmt] == 0b1) {
    Some(fa, fb, _, _) =>
      if fa == ZVT_Int32 then (tl, zeros())
      else zvt_b1_run(tl, vr, vt, mt, cur_vl, frm, fa, fb, mtd, vs2, vs1),
    None() => (tl, zeros()),
  }

// vtzero: the tile update of the arithmetic instructions, applied to a zero
// matrix at TEW = SEW * TWIDEN.  It does
// not read tk, so with tm or tn 0 every element of the tile is tail.
function zvt_vtzero(tl : zvt_tiles, vt : Vtype, mt : Mtype, cur_vl : int, mtd : int) -> zvt_tiles = {
  let tew = zvt_vtzero_tew(vt, mt);
  let m = unsigned(mt[tm]);
  let n = zvt_dim_tn(cur_vl, tew);
  let e = zvt_ete(tew);
  var out : zvt_tiles = tl;
  foreach (i from 0 to (e - 1)) {
    foreach (j from 0 to (e - 1)) {
      if i < m & j < n then out = zvt_tile_write(out, tew, mtd, i, j, zeros())
      else if zvt_tail_ones() then out = zvt_tile_write(out, tew, mtd, i, j, ones())
    }
  };
  out
}

// ---- Whether a tile byte is written ----------------------------------------
//
// In this model, MS becomes Dirty when an instruction writes any tile byte,
// whether or not its value changes, so the execute clauses cannot infer it by
// comparing tile values.  These predicates say whether an instruction writes
// at least one tile byte.  ETE is at least 2, so element (0, 0) is always in
// range.

// arith writes when the row is a computation and tm, tn and tk are nonzero.
function zvt_arith_wrote(vt : Vtype, mt : Mtype, cur_vl : int, op : zvt_arith_op) -> bool = {
  let sw = zvt_sew(vt);
  let tw = zvt_twiden(mt[mtwiden]);
  let row_ok : bool = match zvt_type_row(op, sw, tw, vt[altfmt] == 0b1) {
    Some(fa, _, _, _) => fa != ZVT_Int32,
    None()            => false,
  };
  row_ok & unsigned(mt[tm]) > 0 & zvt_dim_tn(cur_vl, sw * tw) > 0 & unsigned(mt[tk]) > 0
}

// vtzero writes its body, if non-empty, and its tail under the ones policy.
function zvt_vtzero_wrote(vt : Vtype, mt : Mtype, cur_vl : int) -> bool = {
  let tew = zvt_vtzero_tew(vt, mt);
  let m = unsigned(mt[tm]);
  let n = zvt_dim_tn(cur_vl, tew);
  let e = zvt_ete(tew);
  (m > 0 & n > 0) | (zvt_tail_ones() & (m < e | n < e))
}

// vtmv.t.v, and vtle without a fault, write the body vstart .. body_end - 1
// and, under the ones policy, the tail max(vstart, body_end) .. ETE - 1.
function zvt_subset_wrote(tew : int, cur_vl : int, vst : int) -> bool = {
  let be = min(cur_vl, zvt_ete(tew));  // body_end
  vst < cur_vl & (vst < be | (zvt_tail_ones() & max(vst, be) < zvt_ete(tew)))
}

// ---- Moves ([VME] Move Tile Subset between Tile State and Vector Registers)
//
// The subset has elements j = 0 .. ETE - 1: prestart if j < vstart, body if
// vstart <= j < body_end, tail otherwise.  When vstart >= vl, nothing
// changes.

// body_end: min(vl, ETE).
function zvt_body_end(cur_vl : int, tew : int) -> int = min(cur_vl, zvt_ete(tew))

// vreg_put: the register file with element i of the eew-wide view of the
// group based at v set to the low eew bits of x.
function zvt_vreg_write(vr : zvt_vrf, w : int, v : int, i : int, x : bits(64)) -> zvt_vrf = {
  let vb = zvt_div(vlen, 8);
  let off = zvt_vreg_off(w, v, i);
  let r = zvt_div(off, vb);
  assert(0 <= r & r < 32, "zvt_vreg_write: register out of range");
  [vr with r = zvt_insert(vr[r], 8 * zvt_mod(off, vb), w, x)]
}

// The tile tail of a subset: elements max(vstart, body_end) .. ETE - 1
// become all 1s under the ones policy, and are unchanged otherwise.  Shared
// by vtmv.t.v and vtle; vtle applies it whether or not it traps.
function zvt_subset_tail(tl : zvt_tiles, tew : int, tss : xlenbits, cur_vl : int, vst : int)
    -> zvt_tiles = {
  let t = zvt_tss_eff_tile(tew, tss);
  var out : zvt_tiles = tl;
  if zvt_tail_ones() then {
    foreach (j from max(vst, zvt_body_end(cur_vl, tew)) to (zvt_ete(tew) - 1)) {
      let (r, c) = zvt_tss_elem(tss, j);
      out = zvt_tile_write(out, tew, t, r, c, ones());
    }
  };
  out
}

// vtmv.v.t vd, rs1: tile subset to vector register group, at TEW = SEW.  The
// vector tail is undisturbed, as sail-riscv's V tail policy is.
function zvt_vtmv_v_t(tl : zvt_tiles, vr : zvt_vrf, vt : Vtype, cur_vl : int, vst : int,
                      vd : int, tss : xlenbits) -> zvt_vrf = {
  let tew = zvt_move_tew(vt);
  let t = zvt_tss_eff_tile(tew, tss);
  var out : zvt_vrf = vr;
  if vst < cur_vl then {
    foreach (j from vst to (zvt_body_end(cur_vl, tew) - 1)) {
      let (r, c) = zvt_tss_elem(tss, j);
      out = zvt_vreg_write(out, tew, vd, j, zvt_tile_read(tl, tew, t, r, c));
    }
  };
  out
}

// vtmv.t.v rs1, vs2: vector register group to tile subset, at TEW = SEW.
// The tile tail of the subset follows the tile tail policy, whatever vta is
// (this model's reading of [VME]).
function zvt_vtmv_t_v(tl : zvt_tiles, vr : zvt_vrf, vt : Vtype, cur_vl : int, vst : int,
                      vs2 : int, tss : xlenbits) -> zvt_tiles = {
  let tew = zvt_move_tew(vt);
  let t = zvt_tss_eff_tile(tew, tss);
  if cur_vl <= vst then tl
  else {
    var out : zvt_tiles = tl;
    foreach (j from vst to (zvt_body_end(cur_vl, tew) - 1)) {
      let (r, c) = zvt_tss_elem(tss, j);
      out = zvt_tile_write(out, tew, t, r, c, zvt_vreg_read(vr, tew, vs2, j));
    };
    zvt_subset_tail(out, tew, tss, cur_vl, vst)
  }
}

// ---- Loads and stores ([VME] Load/Store Tile Subset to Memory) -------------
//
// The element loop, with its memory accesses, faults and vstart, is in the
// vtle and vtse execute clauses.  The core
// supplies the address arithmetic, the alignment test, and the tile updates.

// elem_addr: base + j * EEW / 8, modulo 2^XLEN, as every sail-riscv address
// is.
function zvt_elem_addr(base : xlenbits, eew : int, j : int) -> xlenbits =
  base + to_bits_truncate(xlen, j * zvt_div(eew, 8))

// misaligned: base is not a multiple of EEW / 8.  Every element shares
// base's alignment, since elements are EEW / 8 bytes apart.
function zvt_misaligned(base : xlenbits, eew : int) -> bool =
  zvt_mod(unsigned(base), zvt_div(eew, 8)) != 0

// The tile storage after element j of the subset is loaded with x.
function zvt_load_elem(tl : zvt_tiles, eew : int, tss : xlenbits, j : int, x : bits(64))
    -> zvt_tiles = {
  let (r, c) = zvt_tss_elem(tss, j);
  zvt_tile_write(tl, eew, zvt_tss_eff_tile(eew, tss), r, c, x)
}

// The data vtse writes for element j of the subset.
function zvt_store_elem(tl : zvt_tiles, eew : int, tss : xlenbits, j : int) -> bits(64) = {
  let (r, c) = zvt_tss_elem(tss, j);
  zvt_tile_read(tl, eew, zvt_tss_eff_tile(eew, tss), r, c)
}

// ---- Supported vtype values -------------------------------------------------

// vtype_supported: "valid per vsetvl semantics", with the altfmt settings
// [VME] requires.  vtype_optional is constant false: this model supports no
// optional vtype settings.  Since VLEN >= ELEN, this agrees with the
// check in execute_vsetvl_type (vext_vset_insts.sail).
function zvt_vtype_supported(vt : Vtype) -> bool = {
  let m = min(elen, vlen);
  let required : bool =
    8 * zvt_lmul_den(vt) <= m & zvt_sew(vt) * zvt_lmul_den(vt) <= m
    & (vt[altfmt] == 0b0 | zvt_altfmt_supported(vt[vsew]));
  vt[vill] == 0b0 & vt[reserved] == zeros() & unsigned(vt[vsew]) <= 3
  & vt[vlmul] != 0b100 & zvt_sew(vt) <= elen & required
}

// ---- Configuration ([VME] Matrix Configuration Instructions) ---------------

// vtype_ill: vill set, every other field 0.
function zvt_vtype_ill() -> Vtype = [Mk_Vtype(zeros()) with vill = 0b1]

// floor(log2 n), with 0 for n <= 1.
function zvt_log2(n : int) -> int = {
  var k : int = 0;
  foreach (b from 1 to 63) {
    if pow2(b) <= n then k = b
  };
  k
}

// matrix_lmul: min(8 / KMAX, 8 / TWIDEN, ceil(ETE / EVE)).
function zvt_matrix_lmul(sw : int, tw : int) -> int = {
  let e = zvt_ete(sw * tw);
  let v = zvt_eve(sw);
  min(zvt_div(8, zvt_kmax_d(sw, tw)), min(zvt_div(8, tw), zvt_div(e + v - 1, v)))
}

// mset_core: the msetmtype[i] behavior pseudocode of [VME] Matrix
// Configuration Instructions, from the point where mtype and vtype hold the
// new values.  An unsupported TEW sets vill (this model's reading of [VME]).
function zvt_mset_core(mt : Mtype, vt : Vtype) -> (Vtype, Mtype) = {
  let ill0 = not(zvt_vtype_supported(vt));
  let (nvt1, nmt1, ill1) : (Vtype, Mtype, bool) =
    if mt[mtwiden] == 0b00 then (vt, mt, ill0)
    else {
      let sw = zvt_sew(vt);
      let tw = zvt_twiden(mt[mtwiden]);
      let l = zvt_matrix_lmul(sw, tw);
      let vt' = [vt with vlmul = to_bits_truncate(3, zvt_log2(l)), vma = 0b1, vta = 0b1];
      let mt' = [mt with
        tm = to_bits_truncate(14, min(unsigned(mt[tm]), min(l * zvt_eve(sw), zvt_ete(sw * tw)))),
        tk = to_bits_truncate(3, min(unsigned(mt[tk]), zvt_kmax_d(sw, tw)))];
      (vt', mt', ill0 | elen < sw * tw | not(zvt_supported_tew(sw * tw)))
    };
  let nmt2 = if ill1 | mt[mtwiden] == 0b00 then Mk_Mtype(zeros()) else nmt1;
  let nvt2 = if ill1 then zvt_vtype_ill() else nvt1;
  (nvt2, nmt2)
}

// mtype_fields: the mtype value rs1 encodes; the reserved bits are dropped.
function zvt_mtype_fields(x : xlenbits) -> Mtype = {
  let r = Mk_Mtype(x);
  [Mk_Mtype(zeros()) with tm = r[tm], tk = r[tk], mtwiden = r[mtwiden]]
}

// msetmtype: mset_core on rs1 and rs2.  (The execute clause also sets
// vl = 0.)
function zvt_msetmtype(rs1 : xlenbits, rs2 : xlenbits) -> (Vtype, Mtype) =
  zvt_mset_core(zvt_mtype_fields(rs1), Mk_Vtype(rs2))

// msetmtypei: as msetmtype, from mtype = 0 with mtwiden, and vtype = 0 with
// vsew and altfmt, supplied by immediates.
function zvt_msetmtypei(vsew_imm : bits(2), alt : bits(1), mtw : bits(2)) -> (Vtype, Mtype) =
  zvt_mset_core([Mk_Mtype(zeros()) with mtwiden = mtw],
                [Mk_Vtype(zeros()) with vsew = zero_extend(vsew_imm), altfmt = alt])

// avl_cap: an XLEN-bit AVL as a number, capped at 2 * VLMAX.
function zvt_avl_cap(avl : xlenbits, vmax : int) -> int =
  if unsigned(avl) < 2 * vmax then unsigned(avl) else 2 * vmax

// zmin_nat: min(x, n) for an XLEN-bit x.
function zvt_zmin_nat(x : xlenbits, n : int) -> int =
  if unsigned(x) < n then unsigned(x) else n

// The tm that msettm sets, and the vl that msettn sets when mtwiden != 0:
// min(rs1, LMUL * EVE, ETE), where LMUL * EVE = VLMAX.
function zvt_tile_dim(vt : Vtype, mt : Mtype, rs1 : xlenbits) -> int =
  zvt_zmin_nat(rs1, min(zvt_vlmax(vt), zvt_ete(zvt_sew(vt) * zvt_twiden(mt[mtwiden]))))

// The tk that msettk sets: min(rs1, KMAX).
function zvt_msettk_val(vt : Vtype, mt : Mtype, rs1 : xlenbits) -> int =
  zvt_zmin_nat(rs1, zvt_kmax_d(zvt_sew(vt), zvt_twiden(mt[mtwiden])))

// ---- Context status ([VME] Context Status) ----------------------------------

// MS after an instruction: vtdiscard makes MS Initial; a write of
// tile state makes it Dirty; otherwise it is unchanged.
function zvt_ms_after(wrote : bool, discard : bool, x : ExtStatus) -> ExtStatus =
  if discard then Initial else if wrote then Dirty else x
