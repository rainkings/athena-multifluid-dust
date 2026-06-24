#ifndef RELAXATION_RELAXATION_HPP_ 
#define RELAXATION_RELAXATION_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file phase_change.hpp
//! \brief definitions for PhaseChange class
//! 
//! Phase change module for multi-composition pebbles with N_p pebble sizes
//! and N_z=2 compositions (ice + refractory) per pebble size.

// C headers

// C++ headers
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"

// Forward declarations
class MeshBlock;
class ParameterInput;
class Units;

class Relaxation {
  friend class Hydro;
  friend class DustFluids;
  friend class EquationOfState;

 public: 
  Relaxation(MeshBlock *pmb, ParameterInput *pin);

  void RelaxationSource(MeshBlock *pmb, const Real time, const Real dt, const Real gm0, const Real alpha_vis,
      const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
      AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s, AthenaArray<Real> &v_frag);

  void GetDivisionMasses(Real mmin, Real mmax, AthenaArray<Real> &m_div, std::string mode="log"); 
  AthenaArray<Real> mmax_array;    // maximum mass allowed by fragmentation [code mass]
  AthenaArray<Real> m_p_array;     // the evolved pebble mass array, added for relaxation source term [code mass]
  AthenaArray<Real> m_p0_array;    // initial pebble mass array [g] (used to initialize rho_Np)
  Real Tem_gas(const Real rad); 

        
 private:
  MeshBlock *pmy_block;  // ptr to MeshBlock containing this Relaxation class

  Real Get_mu(Real fv); 
  
  Real Tem0, r0, Tslope; // do this for the moment [26.04.29]Zhixuan
  Real rho_sil_inter_, rho_ice_inter_, mmin; 
  Real T_relax_prefactor;
    
};
#endif // RELAXATION_RELAXATION_HPP_
