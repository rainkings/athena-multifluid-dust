#ifndef PHASE_CHANGE_PHASE_CHANGE_HPP_
#define PHASE_CHANGE_PHASE_CHANGE_HPP_
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

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"

// Forward declarations
class MeshBlock;
class ParameterInput;
class Units;

//! \class PhaseChange
//! \brief Phase change data and functions for multi-composition pebbles
//!
//! Handles phase transitions between ice and vapor for pebbles with multiple compositions.
//! Supports N_p pebble sizes and N_z=2 compositions (ice + refractory) per pebble size.
//! Dust fluid layout: For each pebble size p (0..N_p-1):
//!   - dust_id = 2*p: ice composition
//!   - dust_id = 2*p + 1: refractory composition
//!   - dust_id = 2*N_p: vapor (tracer)

class PhaseChange {
  friend class Hydro;
  friend class DustFluids;
  friend class EquationOfState;

 public:
  PhaseChange(MeshBlock *pmb, ParameterInput *pin);

  // Public data arrays
  AthenaArray<Real> rho_Np_array;  // pebble number density array [1/cm^3] (Yu, 2025-11-18: replaced m_p_array and s_p_array)
  AthenaArray<Real> q_latent;      // latent heat absorption/release rate [code unit]
  AthenaArray<Real> m_p0_array;    // initial pebble mass array [g] (used to initialize rho_Np)

  // this is for heat conduction
  AthenaArray<Real> q_diff;      // heat conduction rate [code unit]

  Real L_heat, Cd_water, P_eq0;

  // Public functions
  void PhaseChangeSource(MeshBlock *pmb, const Real time, const Real dt,
      const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
      AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
  
  // Helper functions to derive m_p and s_p from rho_Np (Yu, 2025-11-18)
  Real Get_m_p_from_rho_Np(Real rho_I, Real rho_sil, Real rho_Np);
  Real Get_s_p_from_m_p(Real m_p, Real rho_I, Real rho_sil);
  
  // Calculate stopping time for a pebble (Yu, 2025-11-18)
  Real Get_stopping_time(Units *punit, AthenaArray<Real> rho_d, Real T, Real rho_g, Real rho_v, Real rho_Np);
  
  // Get temperature from gas internal energy (public utility function)
  Real Get_T_rhoe_g(Real rhoe_g, Real rho_g, Real fv);

 private:
  MeshBlock *pmy_block;  // ptr to MeshBlock containing this PhaseChange
  
  // Problem-specific constants (initialized from ParameterInput)
  Real min_tol_;
  Real f_ICE_inter0_;
  Real rho_sil_inter_;
  Real rho_ice_inter_;
  Real dfloor_, dffloor_;  // density floors (from problem)
  
  // Helper functions for phase change calculation
  Real Get_T_rhoe(Real rhoe, Real rho_g, const AthenaArray<Real> &rho_d_array, Real fv);
  Real Get_rhomu_d(const AthenaArray<Real> &rho_d_array);
  Real Get_rhoe(Real rhoE_total, Real rho_g, Real E_kg,
                const AthenaArray<Real> &rho_d_array, const AthenaArray<Real> &E_kd_array);
  Real Get_Z(Real rho_g, Real T);
  Real Get_P_eq(Real T, Real T_a);
  Real Get_mu(Real fv);
  void phase_trans(Real rhoe, Real rho_g, const AthenaArray<Real> &rho_I,
                   Real rho_v, Real &drho);
  Real GetSublimationRate(Real Tem, Real rho_v, 
                          Real s_p, Real rho_Np_p);
};

#endif // PHASE_CHANGE_PHASE_CHANGE_HPP_
