// =======================================================================================
// This Sail RISC-V architecture model, comprising all files and
// directories except where otherwise noted is subject the BSD
// two-clause license in the LICENSE file.
//
// SPDX-License-Identifier: BSD-2-Clause
// =======================================================================================

// Zvt (the RISC-V Vector Matrix Extension, VME): the tile edge TE.
//
// The Zvt model is in extensions/Zvt/.  TE is here, in core, for the
// same reason VLEN is (vlen.sail): a configured type-level parameter with a
// global constraint must be in scope in every module.  The prose
// specification is src/unpriv/zvt.adoc in riscv/riscv-isa-manual at commit
// 98b0e96fb240a0b6a54bf814fc2f6a139a409e10 (CC-BY-4.0).

// TE, the tile edge in elements at TEW <= 32 ([VME] Matrix Tile State).  [VME]
// requires 4 <= TE <= VLEN/4, and the 14-bit mtype.tm field bounds TE by 8192.
type te_exp : Int = config extensions.Zvt.te_exp
constraint 2 <= te_exp <= 13 & te_exp <= vlen_exp - 2
let te_exp = sizeof(te_exp)
type te : Int = 2 ^ te_exp
let te = sizeof(te)

// One physical tile, a TEW=8 tile of TE x TE bytes.  Tile storage is sixteen
// of them (extensions/Zvt/zvt_regs.sail).
type tilebits = bits(8 * te * te)
