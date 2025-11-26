//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file phase_change.cpp
//! \brief implementation of PhaseChange class

// C headers

// C++ headers
#include <cmath>
#include <sstream>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../defs.hpp"
#include "../dustfluids/dustfluids.hpp"
#include "../eos/eos.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../units/units.hpp"
#include "phase_change.hpp"
#include "phase_change_constants.hpp"

//----------------------------------------------------------------------------------------
//! \fn PhaseChange::PhaseChange(MeshBlock *pmb, ParameterInput *pin)
//! \brief PhaseChange constructor

PhaseChange::PhaseChange(MeshBlock *pmb, ParameterInput *pin): 
    pmy_block(pmb),
    q_latent(pmb->ncells3, pmb->ncells2, pmb->ncells1),
    q_diff(pmb->ncells3, pmb->ncells2, pmb->ncells1) {
  
  // Dust layout configuration (Yu, 2025-11-16) use compile-time N_P and N_Z
  if(NVapor != 1) {
    std::stringstream msg;
    msg << "### FATAL ERROR in PhaseChange::PhaseChange" << std::endl
        << "Only one vapor is allowed now." << std::endl;
    ATHENA_ERROR(msg);
  }
  
  // Verify consistency: NDUSTFLUIDS should equal N_P * N_Z + NVapor
  if(NDUSTFLUIDS != N_P * N_Z + NVapor) {
    std::stringstream msg;
    msg << "### FATAL ERROR in PhaseChange::PhaseChange" << std::endl
        << "NDUSTFLUIDS (" << NDUSTFLUIDS << ") != N_P * N_Z + NVapor ("
        << N_P << " * " << N_Z << " + " << NVapor << " = " 
        << N_P * N_Z + NVapor << ")" << std::endl;
    ATHENA_ERROR(msg);
  }
  
  // (Yu, 2025-11-18) Store rho_Np_array
  rho_Np_array.NewAthenaArray(N_P, pmb->ncells3, pmb->ncells2, pmb->ncells1);
  m_p0_array.NewAthenaArray(N_P);
  
  // Register arrays for restart file I/O (Yu, 2025-11-18)
  pmb->RegisterMeshBlockData(rho_Np_array);

  // Initialize problem-specific constants from ParameterInput
  Units *punit = pmb->pmy_mesh->punit;
  
  min_tol_ = pin->GetOrAddReal("problem", "min_tol", 1.e-7);
  f_ICE_inter0_ = pin->GetOrAddReal("problem", "f_ICE_inter0", 0.5);
  
  // (Yu, 2025-11-18) Convert input parameters from CGS to code units
  for (int p = 0; p < N_P; ++p) {
    //[25.11.25]lzx: here we should multiply by N_Z to get the correct dust_id 
    Real m_p0_cgs = pin->GetReal("dust", "m_p0_" + std::to_string(p*N_Z+1)); // [g]
    m_p0_array(p) = m_p0_cgs / punit->code_mass_cgs; // Convert to code units
  }
  
  Real rho_sil_inter_cgs = pin->GetOrAddReal("problem", "rho_sil_inter", 3.0); // [g/cm^3]
  Real rho_ice_inter_cgs = pin->GetOrAddReal("problem", "rho_ice_inter", 1.0); // [g/cm^3]
  rho_sil_inter_ = rho_sil_inter_cgs / punit->code_density_cgs; // Convert to code units
  rho_ice_inter_ = rho_ice_inter_cgs / punit->code_density_cgs; // Convert to code units
  
  dfloor_ = pin->GetOrAddReal("hydro", "dfloor", 1.e-10);
  dffloor_ = pin->GetOrAddReal("dust", "dffloor", 1.e-10);

  L_heat = L_heat_cgs / SQR(punit->code_velocity_cgs); // erg/g
  P_eq0 = P_eq0_cgs / punit->code_pressure_cgs;
  Cd_water = Cd_water_cgs / (SQR(punit->code_velocity_cgs)); // (Yu) erg/g/K, need to check.
  // KELVIN = SQR(punit->code_velocity_cgs) / (Constants::k_boltzmann_cgs / Constants::hydrogen_mass_cgs);
}

//----------------------------------------------------------------------------------------
//! \fn void PhaseChange::PhaseChangeSource(...)
//! \brief Main phase change source term function
//! Generalized for N_p pebble sizes

void PhaseChange::PhaseChangeSource(MeshBlock *pmb, const Real time, const Real dt,
const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_df,
    const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        // Allocate arrays for all pebble sizes
        AthenaArray<Real> E_kd_array, rho_d_array0, rho_d_array, rho_d_array1, drho_d_ratio_array,drho_d_max_array;
        E_kd_array.NewAthenaArray(N_P);
        rho_d_array0.NewAthenaArray(N_P);
        rho_d_array.NewAthenaArray(N_P);
        rho_d_array1.NewAthenaArray(N_P);
        drho_d_ratio_array.NewAthenaArray(N_P);
        drho_d_max_array.NewAthenaArray(N_P);

        // Velocity arrays (assuming 3D, access dimension from mesh if needed)
        const int ndim = (pmb->pmy_mesh->f3 ? 3 : (pmb->pmy_mesh->f2 ? 2 : 1));
        AthenaArray<Real> gas_vel_array0, gas_vel_array, gas_vel_array1, dust_vel_array;
        gas_vel_array0.NewAthenaArray(ndim);
        gas_vel_array.NewAthenaArray(ndim);
        gas_vel_array1.NewAthenaArray(ndim);
        dust_vel_array.NewAthenaArray(N_P, ndim);

        // Get gas properties
        Real &rho_g  = cons(IDN, k, j, i);
        Real &gas_mom1 = cons(IM1, k, j, i);
        Real &gas_mom2 = cons(IM2, k, j, i);
        Real &gas_mom3 = cons(IM3, k, j, i);
        Real &gas_erg  = cons(IEN, k, j, i);
        
        // Gas kinetic energy
        Real gas_vel1 = gas_mom1/rho_g;
        Real gas_vel2 = gas_mom2/rho_g;
        Real gas_vel3 = gas_mom3/rho_g;
        Real E_kg = 0.5*(SQR(gas_vel1) + SQR(gas_vel2) + SQR(gas_vel3));
        gas_vel_array(0) = gas_vel1;
        gas_vel_array(1) = gas_vel2;
        gas_vel_array(2) = gas_vel3;
        gas_vel_array1 = gas_vel_array;
        gas_vel_array0 = gas_vel_array;

        // Loop over all pebble sizes to collect ice densities and kinetic energies
        for (int p = 0; p < N_P; ++p) {
          int ice_id = N_Z * p;      // ice composition (z=0)
          // int refrac_id = N_Z * p + 1; // refractory composition (z=1)
          int rho_id = 4*ice_id;
          int v1_id = rho_id + 1;
          int v2_id = rho_id + 2;
          int v3_id = rho_id + 3;

          Real &rho_I_p = cons_df(rho_id, k, j, i);
          Real d_vel1 = cons_df(v1_id, k, j, i) / rho_I_p;
          Real d_vel2 = cons_df(v2_id, k, j, i) / rho_I_p;
          Real d_vel3 = cons_df(v3_id, k, j, i) / rho_I_p;
          
          // Store ice density for pebble p
          rho_d_array(p) = rho_I_p;
          
          // Store kinetic energy for pebble p
          E_kd_array(p) = 0.5*(SQR(d_vel1) + SQR(d_vel2) + SQR(d_vel3));
          dust_vel_array(p, 0) = d_vel1;
          dust_vel_array(p, 1) = d_vel2;
          dust_vel_array(p, 2) = d_vel3;
        }

        // Get vapor properties
        int vapor_rho_id = 4*vapor_id;
        int vapor_v1_id = vapor_rho_id + 1;
        int vapor_v2_id = vapor_rho_id + 2;
        int vapor_v3_id = vapor_rho_id + 3;

        Real &rho_v = cons_df(vapor_rho_id, k, j, i);
        Real &v1_mom1 = cons_df(vapor_v1_id, k, j, i);
        Real &v1_mom2 = cons_df(vapor_v2_id, k, j, i);
        Real &v1_mom3 = cons_df(vapor_v3_id, k, j, i);

        Real fv = rho_v/rho_g;
        Real fv0 = fv;

        // if (i==2 && j==2) {
        //   // Debug output for the first cell
        //   std::cout << "Initial: rho_g=" << rho_g << ", rho_v=" << rho_v << ", fv=" << fv << std::endl;
        // }
        //
        // Check for NaN
        if(std::isnan(fv) || (fv < 0.0) || std::isnan(gas_mom1) || std::isnan(gas_erg)){
          std::stringstream msg;
          msg << "### FATAL ERROR in PhaseChange::PhaseChangeSource" << std::endl
              << "NaN detected: fv = " << fv << ", gas_mom1 = " << gas_mom1
              << ", gas_erg = " << gas_erg << std::endl;
          ATHENA_ERROR(msg);
        }

        // Initialize arrays
        rho_d_array1 = rho_d_array;
        rho_d_array0 = rho_d_array;

        // Get total energy (conserved quantity)
        Real rhoe_g = gas_erg - E_kg*rho_g;
        Real Tem = Get_T_rhoe_g(rhoe_g, rho_g, fv);
        Real rhomu_d = Get_rhomu_d(rho_d_array); // chemical potential of volatiles. chemical potential of gas is set to be 0.
        Real rhoE_total = gas_erg + rhomu_d;
        
        for (int p = 0; p < N_P; ++p) {
          rhoE_total += E_kd_array(p)*rho_d_array(p) + Cd_water*rho_d_array(p)*Tem;
        }

        // Define variables for iterative solver
        Real drho, rho_g1, rho_v1;
        Real fx0, fx1, fx2, x0, x1, x2; // secant points
        Real rhoe, rhoe1;
        rho_g1 = rho_g;

        // First calculation of phase change
        rhoe = Get_rhoe(rhoE_total, rho_g, E_kg, rho_d_array, E_kd_array);
        phase_trans(rhoe, rho_g, rho_d_array, rho_v, drho);
        fx0 = drho;
        x0 = rho_v;
        
        Real sign = (drho > 0. ? 1.0 : -1.0); // positive: sublimation, negative: condensation
        
        // Calculate analytical evaporation rate (Schoonenberg+ 2017)
        // Per-pebble sublimation/condensation rates and supplies
        Units *punit = pmb->pmy_mesh->punit;
        AthenaArray<Real> drhodt_ice_arr;
        drhodt_ice_arr.NewAthenaArray(N_P);

        ////////////////////////////////////////////////////////////////
        // Compute per-pebble rates
        AthenaArray<Real> rho_comps;
        rho_comps.NewAthenaArray(N_Z); // +1 for vapor
        for (int p = 0; p < N_P; ++p) {
          //
          for (int z = 0; z < N_Z; ++z) {
            Real dust_rho_id = N_Z * p + z;
            rho_comps(z) = cons_df(4*dust_rho_id, k, j, i);
          }

          // update rho_Np for pebble p
          Real &rho_Np_p = rho_Np_array(p, k, j, i);
          Real rho_refrac = rho_comps(rho_comps.GetDim1() - 1); // refractory (always the last one)
          rho_Np_p = rho_refrac/(1.0 - f_ICE_inter0_)/m_p0_array(p); // [code_number_density]
          
          // Derive m_p and s_p from rho_Np and current densities (Yu, 2025-11-18)
          Real m_p = Get_m_p_from_rho_Np(rho_comps, rho_Np_p); // [code_mass]
          Real s_p = Get_s_p_from_m_p(m_p, rho_comps); // [code_length]
          
          drhodt_ice_arr(p) = GetSublimationRate(Tem, rho_v, s_p, rho_Np_p);
        }
        // if (drhodt_ice_arr(1) < drhodt_ice_arr(0)){
        //   std::cout << "drhodt_ice_arr(0)=" << drhodt_ice_arr(0) << std::endl;
        //   std::cout << "drhodt_ice_arr(1)=" << drhodt_ice_arr(1) << std::endl;
        //   std::cout << "s_p(0)=" << Get_s_p_from_m_p(Get_m_p_from_rho_Np(rho_comps, rho_Np_array(0,k,j,i)), rho_comps) << std::endl;
        //   std::cout << "s_p(1)=" << Get_s_p_from_m_p(Get_m_p_from_rho_Np(rho_comps, rho_Np_array(1,k,j,i)), rho_comps) << std::endl;
        //   std::stringstream msg;
        // }

        // calculate drho limited by phase change and material amount
        Real drho_limit = 0.0;
        Real avail_p = 0.0;

        if (sign > 0.0) {
          for (int p = 0; p < N_P; ++p) {
            Real cap_p = 0.0;
            Real rate_p = std::fabs(drhodt_ice_arr(p)) * dt; // sublimation limit
            avail_p = rho_d_array(p) - dffloor_; // material limit
            // if (avail_p < 0.0){
            //   return;
            // }
            avail_p = (avail_p < 1e-100) ? 1e-100 : avail_p;

            cap_p = (rate_p > avail_p) ? avail_p : rate_p;
            drho_d_max_array(p) = cap_p; // how much sublimation could happen for pebble p
            drho_limit += cap_p;
          }
        } else {
          for (int p = 0; p < N_P; ++p) {
            Real cap_p = 0.0;
            Real rate_p = std::fabs(drhodt_ice_arr(p)) * dt; // condensation limit
            cap_p = rate_p;
            drho_limit += cap_p;
            drho_d_max_array(p) = cap_p; // how much condensation could happen for pebble p
          }

          Real avail_v = rho_v - dffloor_;
          avail_v = (avail_v < 1.e-100) ? 1.e-100 : avail_v;

          Real norm_factor = (drho_limit > avail_v) ? (avail_v / drho_limit) : 1.0;
          for (int p = 0; p < N_P; ++p) {
            drho_d_max_array(p) *= norm_factor; // normalized by vapor supply
          }
          drho_limit *= norm_factor;
        }

        // fix the ratio when conducting root finding
        // This could be improved by updating the ratio in the root finding process, but keep it flexible for now.
        for (int p = 0; p < N_P; ++p) {
          drho_d_ratio_array(p) = drho_d_max_array(p) / drho_limit;
        }

        // update gas, vapor, and dust with total supply
        rho_g1 = rho_g + drho_limit * sign;
        rho_v1 = rho_v + drho_limit * sign;
        for (int p = 0; p < N_P; ++p) {
          rho_d_array1(p) = rho_d_array(p) - drho_limit*sign*drho_d_ratio_array(p);
          // if (i==2 && j==2) {
          //   // Debug output for the first cell
          //   std::cout << "Pebble " << p << ": drho_d_ratio_array = " << drho_d_ratio_array(p) << std::endl;
          // }
        }
        ////////////////////////////////////////////////////////////////
        
        // Update rhoe
        rhoe1 = Get_rhoe(rhoE_total, rho_g1, E_kg, rho_d_array1, E_kd_array);
        phase_trans(rhoe1, rho_g1, rho_d_array1, rho_v1, drho);
        fx1 = drho;
        x1 = rho_v1;

        // Store initial values for bisection
        Real rho_left = rho_v;
        Real rho_right = rho_v1;
        Real rho_g0 = rho_g;
        Real rho_v0 = rho_v;
        gas_vel_array0 = gas_vel_array;
        rho_d_array0 = rho_d_array;
        
        // Secant method iteration
        rho_g = rho_g1;
        rho_v = rho_v1;
        rho_d_array = rho_d_array1;
        gas_vel_array = gas_vel_array1;

        if((drho*sign) < 0.) {
          // Secant method
          Real drho_adp;
          Real f_err = 1.0;
          int nite = 0;
          bool bisect = false;

          while(f_err > min_tol_) {
            nite += 1;
            if(nite > 100) {
              bisect = true;
              break;
            }
            
            drho_adp = -fx1/(fx1-fx0)*(x1-x0); // exchange the order of multiplication
            if(std::isnan(drho_adp) || std::isinf(drho_adp)) {
              drho_adp = -(x1-x0)/(fx1-fx0)*fx1;
            }

            if(std::isnan(drho_adp) || std::isinf(drho_adp)) {
              std::cout << "drho_adp = " << drho_adp<< std::endl;
              std::cout << "switch to bisection" << std::endl;
              bisect = true;
              break;
            }

            x2 = x1 + drho_adp;
            rho_g1 = rho_g + drho_adp;
            rho_v1 = rho_v + drho_adp;
            
            for (int p = 0; p < N_P; ++p) {
              rho_d_array1(p) = rho_d_array(p) - drho_adp*drho_d_ratio_array(p);
            }
            // update rhoe
            rhoe1 = Get_rhoe(rhoE_total, rho_g1, E_kg, rho_d_array1, E_kd_array);
            phase_trans(rhoe1, rho_g1, rho_d_array1, rho_v1, drho);
            // update secant point:
            fx2 = drho;
            x0 = x1;
            fx0 = fx1;
            x1 = x2;
            fx1 = fx2;
            // update physical value:
            rho_g = rho_g1;
            rho_v = rho_v1;
            rho_d_array = rho_d_array1;
            gas_vel_array = gas_vel_array1;
            // secant method root error fraction:
            f_err = std::fabs(drho)/rho_v;
          }
          
          if(bisect) {
            // Bisection method fallback
            Real rho_mid;
            rho_g = rho_g0;
            rho_v = rho_v0;
            rho_d_array = rho_d_array0;
            gas_vel_array = gas_vel_array0;
            
            f_err = 1.0;
            nite = 0;

            while(f_err > min_tol_) {
              nite += 1;
              if(nite > 1000) {
                std::stringstream msg;
                msg << "### FATAL ERROR in PhaseChange::PhaseChangeSource" << std::endl
                    << "Bisection iteration > 1000" << std::endl;
                ATHENA_ERROR(msg);
              }
              
              rho_mid = (rho_left + rho_right)/2.0;
              drho_adp = rho_mid - rho_v;

              rho_g1 = rho_g + drho_adp;
              rho_v1 = rho_v + drho_adp;
              
              for (int p = 0; p < N_P; ++p) {
                rho_d_array1(p) = rho_d_array(p) - drho_adp*drho_d_ratio_array(p);
              }
              
              rhoe1 = Get_rhoe(rhoE_total, rho_g1, E_kg, rho_d_array1, E_kd_array);
              phase_trans(rhoe1, rho_g1, rho_d_array1, rho_v1, drho);
              
              if((drho*sign) > 0.) {
                rho_left = rho_v1;
              } else {
                rho_right = rho_v1;
              }

              rho_g = rho_g1;
              rho_v = rho_v1;
              rho_d_array = rho_d_array1;
              gas_vel_array = gas_vel_array1;
              // bisection root error fraction
              f_err = std::fabs(drho)/rho_v;
            }
          }
        }

        // Calculate latent heat absorption/release rate
        q_latent(k,j,i) += -(rho_v - rho_v0)/(pmb->pmy_mesh->dt) * L_heat;

        // Update ice densities for all pebble sizes
        for (int p = 0; p < N_P; ++p) {
          int ice_id = N_Z * p;      // ice composition (z=0)
          int rho_id = 4*ice_id;
          cons_df(rho_id, k, j, i) = rho_d_array(p);
        }

        // Calculate pressure
        fv = rho_v/rho_g;
        Real mu1 = Get_mu(fv);
        Real prs = rho_g*Tem/(mu1*KELVIN);
        // Use calc_gamma for general EOS, GetGamma for adiabatic EOS (Yu, 2025-11-18)
        Real gamma;
        if (std::isnan(fv)){
          std::stringstream msg;
          msg << "### FATAL ERROR in PhaseChange::PhaseChangeSource" << std::endl
              << "fv= nan" << std::endl;
        }
        gamma = pmb->peos->calc_gamma(fv);

        // Update gas energy and momentum
        gas_mom1 = rho_g*gas_vel_array(0);
        gas_mom2 = rho_g*gas_vel_array(1);
        gas_mom3 = rho_g*gas_vel_array(2);
        gas_erg = (prs/(gamma - 1.0) + rho_g*E_kg);
        
        // Update ice momenta for all pebble sizes
        for (int p = 0; p < N_P; ++p) {
          int ice_id = N_Z * p;      // ice composition (z=0)
          int rho_id = 4*ice_id;
          int v1_id = rho_id + 1;
          int v2_id = rho_id + 2;
          int v3_id = rho_id + 3;
          
          cons_df(v1_id, k, j, i) = rho_d_array(p)*dust_vel_array(p, 0);
          cons_df(v2_id, k, j, i) = rho_d_array(p)*dust_vel_array(p, 1);
          cons_df(v3_id, k, j, i) = rho_d_array(p)*dust_vel_array(p, 2);
        }
        
        // Update vapor momentum
        v1_mom1 = rho_v*gas_vel_array(0);
        v1_mom2 = rho_v*gas_vel_array(1);
        v1_mom3 = rho_v*gas_vel_array(2);
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! Helper function: Get temperature from gas internal energy

Real PhaseChange::Get_T_rhoe_g(Real rhoe_g, Real rho_g, Real fv) {
  Real fx = 1.0 - fv; // He/H fraction
  Real e = rhoe_g/rho_g;
  Real T = e*KELVIN/(fx/mu_H2*0.71*2.5 + fx/mu_He*0.29*1.5 + fv/mu_z*3.0);
  return T;
}

//----------------------------------------------------------------------------------------
//! Helper function: Get temperature from total internal energy

Real PhaseChange::Get_T_rhoe(Real rhoe, Real rho_g, const AthenaArray<Real> &rho_d_array, Real fv) {
  Real fx = 1.0 - fv; // He/H fraction
  Real bottom = rho_g/KELVIN*(fx/mu_H2*0.71*2.5 + fx/mu_He*0.29*1.5 + fv/mu_z*3.0);
  
  // Loop over all pebble sizes
  for (int p = 0; p < N_P; ++p) {
    bottom += rho_d_array(p)*Cd_water;
  }
  Real T = rhoe/bottom;
  return T;
}

//----------------------------------------------------------------------------------------
//! Helper function: Get chemical potential of volatiles

Real PhaseChange::Get_rhomu_d(const AthenaArray<Real> &rho_d_array) {
  Real rhomu_d = 0.0;
  for (int p = 0; p < N_P; ++p) {
    rhomu_d += rho_d_array(p)*(-L_heat);
  }
  return rhomu_d;
}

//----------------------------------------------------------------------------------------
//! Helper function: Get internal energy from total energy

Real PhaseChange::Get_rhoe(Real rhoE_total, Real rho_g, Real E_kg,
                          const AthenaArray<Real> &rho_d_array, const AthenaArray<Real> &E_kd_array) {
  Real rhoe = 0.0;
  Real rhomu_d = Get_rhomu_d(rho_d_array);
  rhoe = rhoE_total - rho_g*E_kg - rhomu_d;

  for (int p = 0; p < N_P; ++p) {
    rhoe -= E_kd_array(p)*rho_d_array(p);
  }
  return rhoe;
}

//----------------------------------------------------------------------------------------
//! Helper function: Get equilibrium vapor density

Real PhaseChange::Get_P_eq(Real T, Real T_a) {
  return P_eq0*std::exp(-T_a/T);
}

Real PhaseChange::Get_Z(Real rho_g, Real T) {
  Real P_eq = Get_P_eq(T, T_a);
  Real kB_mp = 1.0/KELVIN;
  Real rhoz = P_eq * mu_z /(T*kB_mp);
  return rhoz;
}

//----------------------------------------------------------------------------------------
//! Helper function: Get mean molecular weight

Real PhaseChange::Get_mu(Real fv) {
  return 1.0/((1.0 - fv)/mu_xy + fv/mu_z);
}

//----------------------------------------------------------------------------------------
//! Helper function: Calculate phase transition rate

void PhaseChange::phase_trans(Real rhoe, Real rho_g, const AthenaArray<Real> &rho_I,
                              Real rho_v, Real &drho) {
  Real T = Get_T_rhoe(rhoe, rho_g, rho_I, rho_v/rho_g);
  Real rhoz = Get_Z(rho_g, T);
  drho = rhoz - rho_v;
  if (std::isnan(drho)) {
    std::stringstream msg;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! Helper function: Calculate sublimation/condensation rate for a single pebble

Real PhaseChange::GetSublimationRate(Real Tem, Real rho_v, 
                                     Real s_p, Real rho_Np_p) {
  
  Real P_eq = Get_P_eq(Tem, T_a);
  Real P_z = rho_v/(KELVIN*mu_z)*Tem;
  // Kinetic prefactor
  Real pref = std::sqrt(8.0*PI)*std::sqrt(Tem/KELVIN/mu_z);
  // Rate for this pebble size (Schoonenberg+ 2017)
  // Note: P_eq > P_z (sublimation), P_eq < P_z (condensation)
  Real drhodt_ice = pref * SQR(s_p) * rho_Np_p * rho_v * (1.0 - P_eq/P_z);
  return drhodt_ice;
}

//----------------------------------------------------------------------------------------
//! Helper function: Derive pebble mass from number density (Yu, 2025-11-18)
//! m_p = (rho_I + rho_sil) / rho_Np

Real PhaseChange::Get_m_p_from_rho_Np(AthenaArray<Real> rho_comps, Real rho_Np) {
  if (rho_Np <= 0.0) {
    return 0.0; // Avoid division by zero
  }
  Real rho_total = 0.0; 
  for (int z = 0; z < rho_comps.GetDim1(); ++z) {
    rho_total += rho_comps(z);
  }
  // Real rho_I = rho_comps(0);      // ice (z=0) 
  // Real rho_sil = rho_comps(1);    // refractory (z=1)
  return  rho_total/ rho_Np;
}

//----------------------------------------------------------------------------------------
//! Helper function: Derive pebble size from mass (Yu, 2025-11-18)
//! s_p = (m_p / (4/3 * pi * rho_p_inter))^(1/3)
//! where rho_p_inter is the internal density of the pebble (mixture of ice and silicate)

Real PhaseChange::Get_s_p_from_m_p(Real m_p, AthenaArray<Real> rho_comps) {
  if (m_p <= 0.0) {
    return 0.0; // Avoid division by zero
  }

  Real rho_I = rho_comps(0);      // ice (z=0) 
  Real rho_sil = rho_comps(1);    // refractory (z=1)
  
  // Calculate mass fractions
  Real rho_total = rho_I + rho_sil;
  if (rho_total <= 0.0) {
    return 0.0;
  }
  
  Real f_ice = rho_I / rho_total;
  Real f_sil = rho_sil / rho_total;
  
  // Calculate effective internal density (harmonic mean of ice and silicate densities)
  Real rho_p_inter = rho_ice_inter_ * rho_sil_inter_ / 
                     (f_ice * rho_sil_inter_ + f_sil * rho_ice_inter_);
  
  // Calculate size from mass assuming spherical pebble
  Real volume = m_p / rho_p_inter;
  Real s_p = std::pow(volume / (FOUR_3RD * PI), ONE_3RD);
  
  return s_p;
}

//----------------------------------------------------------------------------------------
//! Calculate stopping time for a pebble (Yu, 2025-11-18)
//! All quantities in code units

Real PhaseChange::Get_stopping_time(Units *punit, AthenaArray<Real> rho_d, Real T, Real rho_g, Real rho_v, Real rho_Np) {
  if(punit == nullptr) {
    std::stringstream msg;
    msg << "### FATAL ERROR in PhaseChange::Get_stopping_time" << std::endl
        << "punit is nullptr. Units not initialized." << std::endl;
    ATHENA_ERROR(msg);
  }
  
  // rho_d: code_density, rho_g: code_density, rho_v: code_density
  // rho_Np: code_number_density
  Real rho_I = rho_d(0);      // ice (z=0) [code_density]
  Real rho_sil = rho_d(1);    // refractory (z=1) [code_density]

  Real fv = rho_v/rho_g;
  Real mu1 = 1.0/((1.0-fv)/mu_xy + fv/mu_z);  // Mean molecular weight

  // Convert molecular properties to code units for mean free path calculation
  // mu_cgs = mu1 * m_H [g], convert to code_mass
  Real mu_code = mu1 * Constants::hydrogen_mass_cgs / punit->code_mass_cgs; // [code_mass]
  Real sigma_mol_code = sigma_mol_cgs / SQR(punit->code_length_cgs); // [code_length^2]
  Real l_mfp = mu_code/(std::sqrt(2)*rho_g*sigma_mol_code); // [code_length]
  
  // Derive m_p and s_p from rho_Np (all in code units)
  Real m_p = 0.0;
  Real s_p = 0.0;
  Real rho_p_inter = 0.0;
  if (rho_Np > 0.0) {
    m_p = (rho_I + rho_sil) / rho_Np; // [code_mass]
    
    Real f_ice = rho_I/(rho_sil+rho_I);
    Real f_sil = rho_sil/(rho_sil+rho_I);
    rho_p_inter = rho_ice_inter_ * rho_sil_inter_ / 
                  (f_ice * rho_sil_inter_ + f_sil * rho_ice_inter_); // [code_density]
    s_p = m_p/(FOUR_3RD*PI*rho_p_inter); // [code_length^3]
    s_p = std::pow(s_p,ONE_3RD); // [code_length]
  }

  Real t_stop = 0.0;
  if (s_p > 0.0 && rho_p_inter > 0.0) {
    Real cs = std::sqrt(T/(mu1*KELVIN)); // [code_velocity]
    Real vth = std::sqrt(8.0/PI)*cs; // [code_velocity]
    if(s_p < (9.0/4.0*l_mfp)){
      // Epstein regime:
      t_stop = rho_p_inter*s_p/(vth*rho_g);
    }else{
      // Stokes regime:
      t_stop = 4.0*rho_p_inter*SQR(s_p)/(9.0*vth*rho_g*l_mfp);
    }
  }

  return t_stop; // Already in code_time
}


// void Get_vel_new_fromMC(AthenaArray<Real> gas_vel_array,
//   Real rho_g, Real rho_g1, AthenaArray<Real> rho_d_array, AthenaArray<Real> rho_d_array1, Real drho,
//   AthenaArray<Real> rho_ratio_array, AthenaArray<Real> dust_vel_array, bool istracer[NDUSTFLUIDS],
//   AthenaArray<Real> &gas_vel_array1, Real &E_kg, AthenaArray<Real> &E_kd_array){
// // momentum conservation, update E_kg, E_kd_array:

// for (int n = 0; n<= NDIM-1; ++n){
//   gas_vel_array1(n) = (rho_g)*gas_vel_array(n);
// }

// Real denom_gas_vel_new = rho_g1;

// for (int m = 0; m< NDUSTFLUIDS-NRefrac-NVapor; ++m){
//   if(istracer[m]){
//     for (int n = 0; n<= NDIM-1; ++n){
//       gas_vel_array1(n) += rho_d_array(m)*gas_vel_array(n);
//     }
//     denom_gas_vel_new += rho_d_array1(m);
//   }else{
//     for (int n = 0; n<= NDIM-1; ++n){
//       gas_vel_array1(n) += drho*rho_ratio_array(m)*dust_vel_array(m,n);
//     }
//   }
// }

// for (int n = 0; n<= NDIM-1; ++n){
//   gas_vel_array1(n) /= denom_gas_vel_new;
// }

// E_kg = 0.0;
// for (int n = 0; n<= NDIM-1; ++n){
//   E_kg += 0.5*SQR(gas_vel_array1(n));
// }

// for (int m = 0; m< NDUSTFLUIDS-NRefrac-NVapor; ++m){
//   if(istracer[m]){
//     E_kd_array(m) = E_kg;
//   }
// }

// // Real mom1 = gas_vel_array(0)*(rho_d_array(1)+rho_g) + rho_d_array(0)*dust_vel_array(0,0);
// // Real mom2 = gas_vel_array(1)*(rho_d_array(1)+rho_g) + rho_d_array(0)*dust_vel_array(0,1);

// // Real mom1_after = gas_vel_array1(0)*(rho_d_array1(1)+rho_g1) + rho_d_array1(0)*dust_vel_array(0,0);
// // Real mom2_after = gas_vel_array1(1)*(rho_d_array1(1)+rho_g1) + rho_d_array1(0)*dust_vel_array(0,1);

// // std::cout << "dif_mom1=" << mom1-mom1_after <<std::endl;
// // std::cout << "dif_mom2=" << mom2-mom2_after <<std::endl;

// return;
// }
