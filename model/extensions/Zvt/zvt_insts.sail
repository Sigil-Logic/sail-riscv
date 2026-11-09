// =======================================================================================
// This Sail RISC-V architecture model, comprising all files and
// directories except where otherwise noted is subject the BSD
// two-clause license in the LICENSE file.
//
// SPDX-License-Identifier: BSD-2-Clause
// =======================================================================================

// Zvt (the RISC-V Vector Matrix Extension, VME): instructions.
//
// Each execute clause composes, in this order:
//   1. the context-status traps: VS Off through the decode
//      guard, which requires currentlyEnabled(Ext_Zvtbase); MS Off for an
//      instruction that accesses tile state; FS Off for one that accesses
//      fcsr;
//   2. legality, by the core's legal_<class> (zvt_core.sail);
//   3. the semantic function, by the core, with the shell reading and writing
//      the architectural state;
//   4. the status updates: MS Dirty if a tile byte was written, Initial
//      after vtdiscard; VS Dirty and vstart = 0 by set_vstart; FS Dirty for
//      the floating-point arithmetic instructions.
// Encodings and assembly syntax are from src/unpriv/zvt.adoc in
// riscv/riscv-isa-manual at commit 98b0e96fb240a0b6a54bf814fc2f6a139a409e10
// (CC-BY-4.0), cited as [VME].  [VME] gives no assembly syntax for some
// instructions; the comments say where this model chooses one.

// ---- Shell helpers ----------------------------------------------------------

// Tile storage as a value: element p is mtp.
function zvt_read_tiles() -> zvt_tiles = {
  var tl : zvt_tiles = vector_init(zeros());
  foreach (p from 0 to 15) { tl[p] = read_tile_reg(p) };
  tl
}

// Write back the physical tiles that changed.  write_tile_reg sets MS Dirty,
// which is correct: a changed tile implies a tile byte was written.
function zvt_write_tiles(old : zvt_tiles, new : zvt_tiles) -> unit =
  foreach (p from 0 to 15) {
    if new[p] != old[p] then write_tile_reg(p, new[p])
  }

// The vector register file as a value: element r is vr.
function zvt_read_vregs() -> zvt_vrf = {
  var vr : zvt_vrf = vector_init(zeros());
  foreach (r from 0 to 31) { vr[r] = rV(Vregno(r)) };
  vr
}

function zvt_write_vregs(old : zvt_vrf, new : zvt_vrf) -> unit =
  foreach (r from 0 to 31) {
    if new[r] != old[r] then wV(Vregno(r), new[r])
  }

// FS is Off, in mstatus or, when V=1, in vsstatus.
function zvt_fs_off() -> bool =
  extStatus_map(mstatus[FS]) == Off
  | (inVirtualPrivilege() & extStatus_map(vsstatus[FS]) == Off)

// SD as the OR of the dirty context-status fields.
function zvt_sd(s : Mstatus) -> bits(1) =
  bool_to_bit(extStatus_map(s[FS]) == Dirty | extStatus_map(s[XS]) == Dirty
              | extStatus_map(s[VS]) == Dirty | extStatus_map(s[MS]) == Dirty)

// The end of every Zvt instruction that completes: MS Dirty if a tile byte
// was written (MS never becomes Dirty otherwise), then vstart = 0, which also
// sets VS Dirty.
function zvt_finish(wrote : bool) -> unit = {
  if wrote then dirty_m_context();
  set_vstart(zeros())
}

function zvt_write_mtype(mt : Mtype) -> unit = {
  mtype = mt;
  csr_write_callback("mtype", mtype.bits)
}

// ---- Assembly helpers -------------------------------------------------------

mapping zvt_tile_name : bits(4) <-> string = {
  0b0000 <-> "mt0",  0b0001 <-> "mt1",  0b0010 <-> "mt2",  0b0011 <-> "mt3",
  0b0100 <-> "mt4",  0b0101 <-> "mt5",  0b0110 <-> "mt6",  0b0111 <-> "mt7",
  0b1000 <-> "mt8",  0b1001 <-> "mt9",  0b1010 <-> "mt10", 0b1011 <-> "mt11",
  0b1100 <-> "mt12", 0b1101 <-> "mt13", 0b1110 <-> "mt14", 0b1111 <-> "mt15",
}

mapping zvt_sew_name : bits(2) <-> string = {
  0b00 <-> "e8", 0b01 <-> "e16", 0b10 <-> "e32", 0b11 <-> "e64",
}

mapping zvt_alt_name : bits(1) <-> string = {
  0b0 <-> "", 0b1 <-> "alt",
}

// wN gives TWIDEN; w0 leaves the matrix unit unconfigured.
mapping zvt_twiden_name : bits(2) <-> string = {
  0b00 <-> "w0", 0b01 <-> "w1", 0b10 <-> "w2", 0b11 <-> "w4",
}

mapping zvt_eew_name : bits(2) <-> string = {
  0b00 <-> "8", 0b01 <-> "16", 0b10 <-> "32", 0b11 <-> "64",
}

// ---- msettn, msettm, msettk ([VME] msettn, msettm and msettk) ---------------

mapping encdec_zvt_msett_op : zvt_msett_op <-> bits(3) = {
  ZVT_MSETTN <-> 0b000,
  ZVT_MSETTM <-> 0b001,
  ZVT_MSETTK <-> 0b010,
}

mapping zvt_msett_op_name : zvt_msett_op <-> string = {
  ZVT_MSETTN <-> "msettn",
  ZVT_MSETTM <-> "msettm",
  ZVT_MSETTK <-> "msettk",
}

union clause instruction = ZVT_MSETT : (zvt_msett_op, regidx, regidx)

mapping clause encdec = ZVT_MSETT(op, rs1, rd)
  <-> 0b100 @ 0b0 @ 0b100 @ 0b00 @ encdec_zvt_msett_op(op) @ encdec_reg(rs1) @ 0b111 @ encdec_reg(rd) @ 0b1010111
  when currentlyEnabled(Ext_Zvtbase)

// msettn: with the matrix unit unconfigured, vl is set as by vsetvl with
// the current vtype, through sail-riscv's calculate_new_vl, and
// 0 if vill is set; otherwise vl = min(rs1, LMUL * EVE, ETE).  rd and rs1
// are not special-cased when x0.
function clause execute ZVT_MSETT(ZVT_MSETTN, rs1, rd) = {
  let vt = vtype;
  let mt = mtype;
  let avl = X(rs1);
  let v : int =
    if mt[mtwiden] != 0b00 then zvt_tile_dim(vt, mt, avl)
    else if vt[vill] == 0b1 then 0
    else {
      let vmax = zvt_vlmax(vt);
      assert(1 <= vmax & vmax <= vlen, "msettn: VLMAX out of range for a legal vtype");
      calculate_new_vl(avl, vmax)
    };
  vl = to_bits_truncate(v);
  X(rd) = to_bits_truncate(xlen, v);
  csr_write_callback("vl", zero_extend(vl));
  zvt_finish(false);
  RETIRE_SUCCESS
}

// msettm: tm = min(rs1, LMUL * EVE, ETE).  Illegal when mtwiden = 0.
function clause execute ZVT_MSETT(ZVT_MSETTM, rs1, rd) = {
  let vt = vtype;
  let mt = mtype;
  if not(zvt_legal_msettm_k(mt)) then return Illegal_Instruction();
  let v = zvt_tile_dim(vt, mt, X(rs1));
  zvt_write_mtype([mt with tm = to_bits_truncate(14, v)]);
  X(rd) = to_bits_truncate(xlen, v);
  zvt_finish(false);
  RETIRE_SUCCESS
}

// msettk: tk = min(rs1, KMAX).  Illegal when mtwiden = 0.
function clause execute ZVT_MSETT(ZVT_MSETTK, rs1, rd) = {
  let vt = vtype;
  let mt = mtype;
  if not(zvt_legal_msettm_k(mt)) then return Illegal_Instruction();
  let v = zvt_msettk_val(vt, mt, X(rs1));
  zvt_write_mtype([mt with tk = to_bits_truncate(3, v)]);
  X(rd) = to_bits_truncate(xlen, v);
  zvt_finish(false);
  RETIRE_SUCCESS
}

mapping clause assembly = ZVT_MSETT(op, rs1, rd)
  <-> zvt_msett_op_name(op) ^ spc() ^ reg_name(rd) ^ sep() ^ reg_name(rs1)

// ---- msetmtype, msetmtypei ([VME] Matrix Configuration Instructions) --------

// The new vtype and mtype; vl = 0.  An unsupported vtype sets vill whatever
// illegal_vtype_reserved_behavior is: [VME] specifies vill here.
function zvt_execute_msetmtype(nvt : Vtype, nmt : Mtype) -> ExecutionResult = {
  vl = zeros();
  vtype = nvt;
  zvt_write_mtype(nmt);
  csr_write_callback("vtype", vtype.bits);
  csr_write_callback("vl", zero_extend(vl));
  zvt_finish(false);
  RETIRE_SUCCESS
}

union clause instruction = ZVT_MSETMTYPEI : (bits(1), bits(2), bits(2))

mapping clause encdec = ZVT_MSETMTYPEI(alt, vsew, mtw)
  <-> 0b100 @ alt @ 0b100 @ vsew @ 0b100 @ 0b000 @ mtw @ 0b111 @ 0b00000 @ 0b1010111
  when currentlyEnabled(Ext_Zvtbase)

function clause execute ZVT_MSETMTYPEI(alt, vsew, mtw) = {
  let (nvt, nmt) = zvt_msetmtypei(vsew, alt, mtw);
  zvt_execute_msetmtype(nvt, nmt)
}

// [VME]: "msetmtypei eM[alt], wN".
mapping clause assembly = ZVT_MSETMTYPEI(alt, vsew, mtw)
  <-> "msetmtypei" ^ spc() ^ zvt_sew_name(vsew) ^ zvt_alt_name(alt) ^ sep() ^ zvt_twiden_name(mtw)

union clause instruction = ZVT_MSETMTYPE : (regidx, regidx)

mapping clause encdec = ZVT_MSETMTYPE(rs2, rs1)
  <-> 0b100 @ 0b0 @ 0b010 @ encdec_reg(rs2) @ encdec_reg(rs1) @ 0b111 @ 0b00000 @ 0b1010111
  when currentlyEnabled(Ext_Zvtbase)

function clause execute ZVT_MSETMTYPE(rs2, rs1) = {
  let (nvt, nmt) = zvt_msetmtype(X(rs1), X(rs2));
  zvt_execute_msetmtype(nvt, nmt)
}

// [VME] gives no syntax; the operands are in encoding order, mtype first.
mapping clause assembly = ZVT_MSETMTYPE(rs2, rs1)
  <-> "msetmtype" ^ spc() ^ reg_name(rs1) ^ sep() ^ reg_name(rs2)

// ---- vtle, vtse ([VME] Load/Store Tile Subset to Memory) --------------------

// EEW in bytes from the eee field (1xx is reserved and does not decode).
function zvt_eew_bytes(e : bits(2)) -> {1, 2, 4, 8} =
  match e { 0b00 => 1, 0b01 => 2, 0b10 => 4, 0b11 => 8 }

// The misaligned-element exception (zvt_misaligned_exception), raised before
// any access.  Every element shares base's alignment, so it is raised at
// element vstart.  vstart is written (unchanged) through set_vstart, as on
// the other fault paths, so that VS becomes Dirty: this model marks VS Dirty
// for every Zvt instruction that is not illegal, trapping ones included.
function zvt_misaligned_trap(base : xlenbits, eew : int, vst : int, is_load : bool) -> ExecutionResult = {
  set_vstart(to_bits_truncate(vst));
  let access : MemoryAccessType(mem_payload) = if is_load then Load(Vector) else Store(Vector);
  let e : ExceptionType = match (zvt_misaligned_exception, is_load) {
    (ZvtMisaligned_AlignmentException, true)  => E_Load_Addr_Align(),
    (ZvtMisaligned_AlignmentException, false) => E_SAMO_Addr_Align(),
    (ZvtMisaligned_AccessFault, true)         => E_Load_Access_Fault(),
    (ZvtMisaligned_AccessFault, false)        => E_SAMO_Access_Fault(),
  };
  memory_exception(e, zvt_elem_addr(base, eew, vst), default_access_info(access))
}

union clause instruction = ZVT_VTLE : (bits(2), regidx, regidx)

mapping clause encdec = ZVT_VTLE(eew, rs2, rs1)
  <-> 0b0 @ eew @ 0b1 @ 0b00 @ 0b1 @ encdec_reg(rs2) @ encdec_reg(rs1) @ 0b111 @ 0b00000 @ 0b0000111
  when currentlyEnabled(Ext_Zvtbase)

// vtle: elements vstart .. body_end - 1 are read in ascending order.  At the
// first fault, vstart is set to its index and the exception is returned; the
// elements after it are unchanged.  The tile tail of the subset is written
// under the ones tail policy both on completion and on a fault, before the
// exception is returned.
function clause execute ZVT_VTLE(eewb, rs2, rs1) = {
  if zvt_tile_state_off() then return Illegal_Instruction();
  let w = zvt_eew_bytes(eewb);
  let eew = 8 * w;
  let tss = X(rs2);
  let vt = vtype;
  if not(zvt_legal_ldst(vt, mtype, eew, tss)) then return Illegal_Instruction();
  let base = X(rs1);
  let cur_vl = unsigned(vl);
  let vst = unsigned(vstart);
  if cur_vl <= vst then { zvt_finish(false); return RETIRE_SUCCESS };
  let tl = zvt_read_tiles();
  let tail = zvt_subset_tail(tl, eew, tss, cur_vl, vst);
  let tail_wrote = zvt_tail_ones() & max(vst, zvt_body_end(cur_vl, eew)) < zvt_ete(eew);
  if zvt_misaligned(base, eew) then {
    zvt_write_tiles(tl, tail);
    if tail_wrote then dirty_m_context();
    return zvt_misaligned_trap(base, eew, vst, true)
  };
  var out : zvt_tiles = tl;
  var wrote : bool = false;
  foreach (j from vst to (zvt_body_end(cur_vl, eew) - 1)) {
    set_vstart(to_bits_truncate(j));
    match vmem_read(rs1, to_bits_truncate(xlen, j * w), w, Load(Vector), false, false, false) {
      Ok(x)  => {
        out = zvt_load_elem(out, eew, tss, j, zero_extend(64, x));
        wrote = true;
      },
      Err(e) => {
        zvt_write_tiles(tl, zvt_subset_tail(out, eew, tss, cur_vl, vst));
        if wrote | tail_wrote then dirty_m_context();
        return e
      },
    }
  };
  zvt_write_tiles(tl, zvt_subset_tail(out, eew, tss, cur_vl, vst));
  zvt_finish(wrote | tail_wrote);
  RETIRE_SUCCESS
}

mapping clause assembly = ZVT_VTLE(eew, rs2, rs1)
  <-> "vtle" ^ zvt_eew_name(eew) ^ spc() ^ reg_name(rs2) ^ sep() ^ "(" ^ reg_name(rs1) ^ ")"

union clause instruction = ZVT_VTSE : (bits(2), regidx, regidx)

mapping clause encdec = ZVT_VTSE(eew, rs2, rs1)
  <-> 0b0 @ eew @ 0b1 @ 0b00 @ 0b1 @ encdec_reg(rs2) @ encdec_reg(rs1) @ 0b111 @ 0b00000 @ 0b0100111
  when currentlyEnabled(Ext_Zvtbase)

// vtse: elements vstart .. body_end - 1 are written in ascending order; at
// the first fault, vstart is set to its index and the exception returned.
// Memory has no tail, and tile storage is not written.
function clause execute ZVT_VTSE(eewb, rs2, rs1) = {
  if zvt_tile_state_off() then return Illegal_Instruction();
  let w = zvt_eew_bytes(eewb);
  let eew = 8 * w;
  let tss = X(rs2);
  if not(zvt_legal_ldst(vtype, mtype, eew, tss)) then return Illegal_Instruction();
  let base = X(rs1);
  let cur_vl = unsigned(vl);
  let vst = unsigned(vstart);
  if cur_vl <= vst then { zvt_finish(false); return RETIRE_SUCCESS };
  if zvt_misaligned(base, eew) then return zvt_misaligned_trap(base, eew, vst, false);
  let tl = zvt_read_tiles();
  foreach (j from vst to (zvt_body_end(cur_vl, eew) - 1)) {
    set_vstart(to_bits_truncate(j));
    let data = zvt_store_elem(tl, eew, tss, j);
    match vmem_write(rs1, to_bits_truncate(xlen, j * w), w, data[(8 * w - 1) .. 0], Store(Vector), false, false, false) {
      Ok(true)  => (),
      Ok(false) => internal_error(__FILE__, __LINE__, "store got false from vmem_write"),
      Err(e)    => return e,
    }
  };
  zvt_finish(false);
  RETIRE_SUCCESS
}

mapping clause assembly = ZVT_VTSE(eew, rs2, rs1)
  <-> "vtse" ^ zvt_eew_name(eew) ^ spc() ^ reg_name(rs2) ^ sep() ^ "(" ^ reg_name(rs1) ^ ")"

// ---- vtmv.v.t, vtmv.t.v ([VME] Move Tile Subset between Tile State and
//      Vector Registers) ---------------------------------------------------

union clause instruction = ZVT_VTMV_V_T : (regidx, vregidx)

mapping clause encdec = ZVT_VTMV_V_T(rs1, vd)
  <-> 0b010000 @ 0b1 @ 0b11111 @ encdec_reg(rs1) @ 0b110 @ encdec_vreg(vd) @ 0b1010111
  when currentlyEnabled(Ext_Zvtbase)

// vtmv.v.t reads tile state, so MS Off makes it illegal; it writes none.
function clause execute ZVT_VTMV_V_T(rs1, vd) = {
  if zvt_tile_state_off() then return Illegal_Instruction();
  let vt = vtype;
  let tss = X(rs1);
  let vdn = unsigned(vregidx_bits(vd));
  if not(zvt_legal_move(vt, mtype, vdn, tss)) then return Illegal_Instruction();
  let vr = zvt_read_vregs();
  zvt_write_vregs(vr, zvt_vtmv_v_t(zvt_read_tiles(), vr, vt, unsigned(vl), unsigned(vstart), vdn, tss));
  zvt_finish(false);
  RETIRE_SUCCESS
}

mapping clause assembly = ZVT_VTMV_V_T(rs1, vd)
  <-> "vtmv.v.t" ^ spc() ^ vreg_name(vd) ^ sep() ^ reg_name(rs1)

union clause instruction = ZVT_VTMV_T_V : (vregidx, regidx)

mapping clause encdec = ZVT_VTMV_T_V(vs2, rs1)
  <-> 0b010111 @ 0b1 @ encdec_vreg(vs2) @ encdec_reg(rs1) @ 0b110 @ 0b00000 @ 0b1010111
  when currentlyEnabled(Ext_Zvtbase)

function clause execute ZVT_VTMV_T_V(vs2, rs1) = {
  if zvt_tile_state_off() then return Illegal_Instruction();
  let vt = vtype;
  let tss = X(rs1);
  let vs2n = unsigned(vregidx_bits(vs2));
  if not(zvt_legal_move(vt, mtype, vs2n, tss)) then return Illegal_Instruction();
  let cur_vl = unsigned(vl);
  let vst = unsigned(vstart);
  let tl = zvt_read_tiles();
  zvt_write_tiles(tl, zvt_vtmv_t_v(tl, zvt_read_vregs(), vt, cur_vl, vst, vs2n, tss));
  zvt_finish(zvt_subset_wrote(zvt_move_tew(vt), cur_vl, vst));
  RETIRE_SUCCESS
}

mapping clause assembly = ZVT_VTMV_T_V(vs2, rs1)
  <-> "vtmv.t.v" ^ spc() ^ reg_name(rs1) ^ sep() ^ vreg_name(vs2)

// ---- Matrix arithmetic ([VME] Matrix Arithmetic Instructions) ---------------

// funct3 and rd[0] select the operation.  rd[4:2] holds t[3:1]; t[0], not
// encoded, is 0.
mapping zvt_arith_op_name : zvt_arith_op <-> string = {
  ZVT_VTFMM     <-> "vtfmm.tvv",
  ZVT_VTFMM_ALT <-> "vtfmm.alt.tvv",
  ZVT_VTMMU     <-> "vtmmu.tvv",
  ZVT_VTMMS     <-> "vtmms.tvv",
}

union clause instruction = ZVT_ARITH : (zvt_arith_op, bits(3), vregidx, vregidx)

mapping clause encdec = ZVT_ARITH(ZVT_VTFMM, t, vs2, vs1)
  <-> 0b111100 @ 0b1 @ encdec_vreg(vs2) @ encdec_vreg(vs1) @ 0b001 @ t @ 0b0 @ 0b0 @ 0b1110111
  when currentlyEnabled(Ext_Zvtbase)
mapping clause encdec = ZVT_ARITH(ZVT_VTFMM_ALT, t, vs2, vs1)
  <-> 0b111100 @ 0b1 @ encdec_vreg(vs2) @ encdec_vreg(vs1) @ 0b001 @ t @ 0b0 @ 0b1 @ 0b1110111
  when currentlyEnabled(Ext_Zvtbase)
mapping clause encdec = ZVT_ARITH(ZVT_VTMMU, t, vs2, vs1)
  <-> 0b111100 @ 0b1 @ encdec_vreg(vs2) @ encdec_vreg(vs1) @ 0b000 @ t @ 0b0 @ 0b0 @ 0b1110111
  when currentlyEnabled(Ext_Zvtbase)
mapping clause encdec = ZVT_ARITH(ZVT_VTMMS, t, vs2, vs1)
  <-> 0b111100 @ 0b1 @ encdec_vreg(vs2) @ encdec_vreg(vs1) @ 0b000 @ t @ 0b0 @ 0b1 @ 0b1110111
  when currentlyEnabled(Ext_Zvtbase)

function clause execute ZVT_ARITH(op, t, vs2, vs1) = {
  let fp = zvt_is_fp_op(op);
  if zvt_tile_state_off() | (fp & zvt_fs_off()) then return Illegal_Instruction();
  let vt = vtype;
  let mt = mtype;
  let frm = fcsr[FRM];
  let mtd = unsigned(t @ 0b0);
  let vs2n = unsigned(vregidx_bits(vs2));
  let vs1n = unsigned(vregidx_bits(vs1));
  if not(zvt_legal_arith(vt, mt, unsigned(vstart), frm, op, mtd, vs2n, vs1n))
  then return Illegal_Instruction();
  let cur_vl = unsigned(vl);
  let tl = zvt_read_tiles();
  let (tl', fl) = zvt_arith(tl, zvt_read_vregs(), vt, mt, cur_vl, frm, op, mtd, vs2n, vs1n);
  zvt_write_tiles(tl, tl');
  // This model marks FS Dirty for every instruction that accesses fcsr;
  // accrue_fflags alone follows fflags_dirty_policy.
  if fp then {
    accrue_fflags(fl);
    dirty_fd_context();
  };
  zvt_finish(zvt_arith_wrote(vt, mt, cur_vl, op));
  RETIRE_SUCCESS
}

mapping clause assembly = ZVT_ARITH(op, t, vs2, vs1)
  <-> zvt_arith_op_name(op) ^ spc() ^ zvt_tile_name(t @ 0b0) ^ sep() ^ vreg_name(vs2) ^ sep() ^ vreg_name(vs1)

// ---- vtzero ([VME] Matrix Tile-Zero Instruction) ----------------------------

union clause instruction = ZVT_VTZERO : bits(4)

mapping clause encdec = ZVT_VTZERO(t)
  <-> 0b010000 @ 0b1 @ 0b11110 @ 0b00000 @ 0b110 @ t @ 0b0 @ 0b1010111
  when currentlyEnabled(Ext_Zvtbase)

function clause execute ZVT_VTZERO(t) = {
  if zvt_tile_state_off() then return Illegal_Instruction();
  let vt = vtype;
  let mt = mtype;
  let mtd = unsigned(t);
  if not(zvt_legal_vtzero(vt, mt, unsigned(vstart), mtd)) then return Illegal_Instruction();
  let cur_vl = unsigned(vl);
  let tl = zvt_read_tiles();
  zvt_write_tiles(tl, zvt_vtzero(tl, vt, mt, cur_vl, mtd));
  zvt_finish(zvt_vtzero_wrote(vt, mt, cur_vl));
  RETIRE_SUCCESS
}

mapping clause assembly = ZVT_VTZERO(t)
  <-> "vtzero" ^ spc() ^ zvt_tile_name(t)

// ---- vtdiscard ([VME] Zvt Context Discard Instruction) ----------------------

union clause instruction = ZVT_VTDISCARD : unit

mapping clause encdec = ZVT_VTDISCARD()
  <-> 0b010000 @ 0b1 @ 0b11100 @ 0b00000 @ 0b110 @ 0b00000 @ 0b1010111
  when currentlyEnabled(Ext_Zvtbase)

// vtdiscard sets MS, and vsstatus.MS when V=1, to Initial (ms_after with
// discard), and changes no tile.  It is illegal if MS is Off or vill is set,
// but not because mtwiden is 0.
function clause execute ZVT_VTDISCARD() = {
  if zvt_tile_state_off() then return Illegal_Instruction();
  if not(zvt_legal_vtdiscard(vtype)) then return Illegal_Instruction();
  mstatus[MS] = extStatus_map(zvt_ms_after(false, true, extStatus_map(mstatus[MS])));
  mstatus[SD] = zvt_sd(mstatus);
  long_csr_write_callback("mstatus", "mstatush", read_mstatus().bits);
  if inVirtualPrivilege() then {
    vsstatus[MS] = extStatus_map(zvt_ms_after(false, true, extStatus_map(vsstatus[MS])));
    vsstatus[SD] = zvt_sd(vsstatus);
    csr_write_callback("vsstatus", vsstatus.bits[xlen - 1 .. 0]);
  };
  zvt_finish(false);
  RETIRE_SUCCESS
}

mapping clause assembly = ZVT_VTDISCARD() <-> "vtdiscard"
