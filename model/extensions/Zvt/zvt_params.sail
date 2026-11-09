// =======================================================================================
// This Sail RISC-V architecture model, comprising all files and
// directories except where otherwise noted is subject the BSD
// two-clause license in the LICENSE file.
//
// SPDX-License-Identifier: BSD-2-Clause
// =======================================================================================

// Zvt (the RISC-V Vector Matrix Extension, VME): implementation parameters.
//
// The specification is src/unpriv/zvt.adoc in riscv/riscv-isa-manual at commit
// 98b0e96fb240a0b6a54bf814fc2f6a139a409e10 (CC-BY-4.0), cited as [VME]; [V] is
// src/unpriv/vector-common.adoc at the same commit.  Where [VME] leaves a
// choice open, or is silent, the comments say which choice this model makes.

// TE, the tile edge, and tilebits, one physical tile, are declared in
// core/zvt_te.sail: like VLEN, a configured type-level parameter with a global
// constraint must be in scope in every module.

// The tile tail policy, fixed per configuration.  [VME] lets a tile tail
// element be either left undisturbed or overwritten with all 1s.  The choice
// applies to the elements of the instruction's TEW, both when a tile load
// completes and when it traps.
enum ZvtTileTail = {
  ZvtTileTail_Undisturbed,
  ZvtTileTail_Ones,
}
let zvt_tile_tail : ZvtTileTail = config extensions.Zvt.tile_tail

// The exception a misaligned tile load or store raises, fixed per
// configuration.  [VME] (Load/Store Tile Subset to Memory): "Misaligned
// element addresses raise either a load/store address-misaligned exception or
// a load/store access-fault exception."  Unlike V, a misaligned Zvt access
// never proceeds, so this does not use the global vector misaligned policy.
enum ZvtMisalignedException = {
  ZvtMisaligned_AlignmentException,
  ZvtMisaligned_AccessFault,
}
let zvt_misaligned_exception : ZvtMisalignedException = config extensions.Zvt.misaligned_exception
