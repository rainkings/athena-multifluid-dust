#ifndef GLOBALS_HPP_
#define GLOBALS_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file globals.hpp
//! \brief namespace containing external global variables

namespace Globals {
extern int my_rank, nranks;
}

// CPU-time (std::clock(), summed across OpenMP threads) profiling of the main
// per-step work, switchable via the input parameter "problem/time_profile".
// Only accumulates when TimeProfile::enabled is true; the accumulators are reset
// (and the report printed) every ncycle_out cycles in main.cpp.
namespace TimeProfile {
enum Cat {
  HYDRO = 0,        // gas hydro task-list tasks
  DUSTFLUIDS,       // dust fluid task-list tasks
  SOURCE,           // MySource (SRC_TERM task)
  USERWORKINLOOP,   // Mesh::UserWorkInLoop (mesh-level RT), timed in main.cpp
  CONS2PRIM,        // conserved->primitive (gas + dust fluids)
  BVAL,             // physical boundary conditions (PHY_BVAL, CLEAR_ALLBND)
  USERWORK,         // per-block USERWORK task (MeshBlock::UserWorkInLoop)
  OTHER,            // everything else (NEW_DT, PROLONG, FLAG_AMR, ...)
  NCAT
};
extern double accum[NCAT];
extern bool enabled;
void Reset();
}

#endif // GLOBALS_HPP_
