// =======================================================================================
// This Sail RISC-V architecture model, comprising all files and
// directories except where otherwise noted is subject the BSD
// two-clause license in the LICENSE file.
//
// SPDX-License-Identifier: BSD-2-Clause
// =======================================================================================

// Zvt (the RISC-V Vector Matrix Extension, VME): extension enabling, the mtype
// CSR, the MS context status, and tile storage.
//
// The specification is src/unpriv/zvt.adoc in riscv/riscv-isa-manual at commit
// 98b0e96fb240a0b6a54bf814fc2f6a139a409e10 (CC-BY-4.0), cited as [VME]; [V] is
// src/unpriv/vector-common.adoc at the same commit.  Where [VME] leaves a
// choice open, or is silent, the comments say which choice this model makes.

// ---- Extensions ([VME] Overview) --------------------------------------------

// Zvtbase needs a vector unit (Zve32x).  The MS context status does not gate
// the extension: only instructions that access tile state check it ([VME]
// Context Status), and vset* and mset* do not.
function clause currentlyEnabled(Ext_Zvtbase) = hartSupports(Ext_Zvtbase) & currentlyEnabled(Ext_Zve32x)
function clause currentlyEnabled(Ext_Zvt8e) = hartSupports(Ext_Zvt8e) & currentlyEnabled(Ext_Zvtbase)
function clause currentlyEnabled(Ext_Zvt16e) = hartSupports(Ext_Zvt16e) & currentlyEnabled(Ext_Zvtbase)
function clause currentlyEnabled(Ext_Zvt64e) = hartSupports(Ext_Zvt64e) & currentlyEnabled(Ext_Zvtbase) & currentlyEnabled(Ext_Zve64x)
function clause currentlyEnabled(Ext_Zvti8i32mm) = hartSupports(Ext_Zvti8i32mm) & currentlyEnabled(Ext_Zvt8e)
function clause currentlyEnabled(Ext_Zvtofp8fmm) = hartSupports(Ext_Zvtofp8fmm) & currentlyEnabled(Ext_Zvt8e) & currentlyEnabled(Ext_Zve32f)
function clause currentlyEnabled(Ext_Zvtfp16fmm) = hartSupports(Ext_Zvtfp16fmm) & currentlyEnabled(Ext_Zvt16e) & currentlyEnabled(Ext_Zve32f)
function clause currentlyEnabled(Ext_Zvtbf16fmm) = hartSupports(Ext_Zvtbf16fmm) & currentlyEnabled(Ext_Zvt16e) & currentlyEnabled(Ext_Zve32f)
function clause currentlyEnabled(Ext_Zvtfmm) = hartSupports(Ext_Zvtfmm) & currentlyEnabled(Ext_Zvtbase) & currentlyEnabled(Ext_Zve32f)
function clause currentlyEnabled(Ext_Zvtdmm) = hartSupports(Ext_Zvtdmm) & currentlyEnabled(Ext_Zvt64e) & currentlyEnabled(Ext_Zve64d)

// altfmt = 1 in vtype is supported only where [VME] requires it (Matrix
// Arithmetic Instructions): at SEW=16 with Zvtbf16fmm, and at SEW=8 with
// Zvti8i32mm or Zvtofp8fmm.  This model supports no other altfmt settings.
function zvt_altfmt_supported(sew : bits(3)) -> bool =
    (sew == 0b001 & hartSupports(Ext_Zvtbf16fmm))
  | (sew == 0b000 & (hartSupports(Ext_Zvti8i32mm) | hartSupports(Ext_Zvtofp8fmm)))

// ---- The mtype CSR ([VME] Matrix Type (mtype) CSR) -------------------------

// mtype is read-only (address 0xC23).  Bits not named below are reserved and
// read as zero; they are never written non-zero.
bitfield Mtype : xlenbits = {
  tm      : 23 .. 10,
  tk      : 7 .. 5,
  mtwiden : 1 .. 0,
}
register mtype : Mtype = Mk_Mtype(zeros())

mapping clause csr_name_map = 0xC23 <-> "mtype"

// mtype belongs to the vector context: VS, not MS, controls access to it
// ([VME] Context Status).
function clause is_CSR_accessible(0xC23, _, _) = currentlyEnabled(Ext_Zvtbase)
function clause is_CSR_exception_virtual(0xC23, _, _) = false
function clause read_CSR(0xC23) = mtype.bits

// vset[i]vl[i] write zero to mtype ([VME] Matrix Configuration Instructions).
function zvt_clear_mtype() -> unit =
  if hartSupports(Ext_Zvtbase) then {
    mtype = Mk_Mtype(zeros());
    csr_write_callback("mtype", mtype.bits);
  }

// ---- The MS context status ([VME] Context Status) ---------------------------

// Any write of tile state sets MS, and vsstatus.MS when V=1, to Dirty, as
// dirty_v_context does for VS.  This model counts a write that leaves the
// bytes unchanged as a write.
function dirty_m_context() -> unit = {
  assert(hartSupports(Ext_Zvtbase));
  mstatus[MS] = extStatus_map(Dirty);
  mstatus[SD] = 0b1;
  long_csr_write_callback("mstatus", "mstatush", read_mstatus().bits);
  if inVirtualPrivilege() then {
    vsstatus[MS] = extStatus_map(Dirty);
    vsstatus[SD] = 0b1;
    csr_write_callback("vsstatus", vsstatus.bits[xlen - 1 .. 0]);
  }
}

// An instruction that accesses tile state is illegal when MS is Off, or, when
// V=1, when vsstatus.MS is Off ([VME] Context Status).
function zvt_tile_state_off() -> bool =
  extStatus_map(mstatus[MS]) == Off
  | (inVirtualPrivilege() & extStatus_map(vsstatus[MS]) == Off)

// ---- Tile storage ([VME] Matrix Tile State) ---------------------------------

// Sixteen physical TEW=8 tiles.  Viewed as one flat byte array of 16 * TE^2
// bytes, byte p * TE^2 + q of tile storage is byte q of register mtp.
register mt0  : tilebits
register mt1  : tilebits
register mt2  : tilebits
register mt3  : tilebits
register mt4  : tilebits
register mt5  : tilebits
register mt6  : tilebits
register mt7  : tilebits
register mt8  : tilebits
register mt9  : tilebits
register mt10 : tilebits
register mt11 : tilebits
register mt12 : tilebits
register mt13 : tilebits
register mt14 : tilebits
register mt15 : tilebits

function read_tile_reg(p : range(0, 15)) -> tilebits =
  match p {
    0  => mt0,
    1  => mt1,
    2  => mt2,
    3  => mt3,
    4  => mt4,
    5  => mt5,
    6  => mt6,
    7  => mt7,
    8  => mt8,
    9  => mt9,
    10 => mt10,
    11 => mt11,
    12 => mt12,
    13 => mt13,
    14 => mt14,
    15 => mt15,
  }

// Writing a physical tile sets MS to Dirty.
function write_tile_reg(p : range(0, 15), v : tilebits) -> unit = {
  match p {
    0  => mt0 = v,
    1  => mt1 = v,
    2  => mt2 = v,
    3  => mt3 = v,
    4  => mt4 = v,
    5  => mt5 = v,
    6  => mt6 = v,
    7  => mt7 = v,
    8  => mt8 = v,
    9  => mt9 = v,
    10 => mt10 = v,
    11 => mt11 = v,
    12 => mt12 = v,
    13 => mt13 = v,
    14 => mt14 = v,
    15 => mt15 = v,
  };
  dirty_m_context()
}
