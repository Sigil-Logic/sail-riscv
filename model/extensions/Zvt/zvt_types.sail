// =======================================================================================
// This Sail RISC-V architecture model, comprising all files and
// directories except where otherwise noted is subject the BSD
// two-clause license in the LICENSE file.
//
// SPDX-License-Identifier: BSD-2-Clause
// =======================================================================================

// Zvt (the RISC-V Vector Matrix Extension, VME): the operand types of its
// instruction clauses, which must precede the scattered instruction union.

// The four matrix arithmetic instructions.
enum zvt_arith_op = { ZVT_VTFMM, ZVT_VTFMM_ALT, ZVT_VTMMU, ZVT_VTMMS }

// msettn, msettm and msettk.
enum zvt_msett_op = { ZVT_MSETTN, ZVT_MSETTM, ZVT_MSETTK }
