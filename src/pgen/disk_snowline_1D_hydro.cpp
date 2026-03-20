//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file disk.cpp
//! \brief Initializes stratified Keplerian accretion disk in both cylindrical and
//! spherical polar coordinates.  Initial conditions are in vertical hydrostatic eqm.

// C headers

// C++ headers
#include <algorithm>  // min
#include <cmath>      // sqrt
#include <cstdlib>    // srand
#include <cstring>    // strcmp()
#include <fstream>
#include <iostream>   // endl
#include <limits>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <vector>
#include <utility>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../dustfluids/dustfluids.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/hydro_diffusion/hydro_diffusion.hpp"
#include "../mesh/mesh.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../parameter_input.hpp"
#include "../phase_change/phase_change.hpp"  // (Yu, 2025-11-16)
#include "../phase_change/phase_change_constants.hpp"  // (Yu, 2025-11-16) For constants: KELVIN, P_eq0, L_heat, etc.
#include "../units/units.hpp"  // (Yu, 2025-11-16) For Constants namespace

namespace {
void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k);
Real PoverRho(const Real rad, const Real phi, const Real z);
Real DenProfileCyl_gas(const Real rad, const Real phi, const Real z);
Real VelProfileCyl_gas(const Real rad, const Real phi, const Real z);
Real DenProfileCyl_dust(const Real rad, const Real phi, const Real z,
                        const Real den_ratio, const Real H_ratio);
Real VelProfileCyl_dust(const Real rad, const Real phi, const Real z);                                             
void Vr_interpolate_outer_nomatter(const Real r_active, const Real r_ghost, const Real sigma_active,
    const Real sigma_ghost, const Real vr_active, Real &vr_ghost);
void sigma_interpolate_inner_nomatter(const Real r_active, const Real r_ghost, const Real sigma_active,
    Real &sigma_ghost, const Real vr_active, const Real vr_ghost);
void sigma_interpolate_inner_log(const Real r_active, const Real r_ghost, const Real sigma_active,
    Real &sigma_ghost);

// phase change
void Get_vel_new_fromMC(AthenaArray<Real> gas_vel_array,
    Real rho_g, Real rho_g1, AthenaArray<Real> rho_d_array, AthenaArray<Real> rho_d_array1, Real drho,
    AthenaArray<Real> rho_ratio_array, AthenaArray<Real> dust_vel_array, 
    AthenaArray<Real> &gas_vel_array1, Real &E_kg, AthenaArray<Real> &E_kd_array);
void Get_E_kg(AthenaArray<Real> gas_vel_array, Real &E_kg);
Real Get_T_rhoe_g(Real rhoe_g, Real rho_g, Real fv);
Real Get_T_rhoe(Real rhoe, Real rho_g, AthenaArray<Real> rho_d_array, Real fv);
Real Get_rhomu_d(AthenaArray<Real> rho_d_array);
Real Get_rhoe(Real rhoE_total, Real rho_g, Real E_kg, AthenaArray<Real> rho_d_array, AthenaArray<Real> E_kd_array);
Real Get_Z(Real rho_g, Real T);
void phase_trans(Real rad, Real rho_g, Real rho_v, Real &drho);
void phase_trans_2D(Real rad, Real rho_g, Real rho_v, Real taus, Real &drho);
Real Get_mu(Real fv);
Real Get_stopping_time(AthenaArray<Real> rho_d, Real T, Real rho_g, Real rad, Real &m_p_out);
Real TemProfile(const Real rad);
Real Get_Cs_gas(const Real rad, Real fv);
Real Get_H_gas(const Real rad, Real fv);
Real Get_nu_gas(const Real rad, Real fv);
Real Get_eta_vk(AthenaArray<Real> rad, Real k, Real j, Real i, const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_df);
Real Get_v_gas(Real k, Real j, Real i, MeshBlock *pmb, const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_df);
Real Get_eta_vk_old(Real rad, Real k, Real j, Real i, const AthenaArray<Real> &prim, Real fv, Real dr);
// problem parameters which are useful to make global to this file
Real gm0, r0, rho0, T0, dslope, Tslope, p0_over_r0, p_over_r_slope, gamma_gas, Omega0, nu_alpha;
Real dfloor, dffloor;
  //snowline
Real f_ICE_inter0, m_p0, taus0, rho_sil_inter, rho_ice_inter, alpha;
Real M_dot_g; // M_sun/yr
Real p2g_flux;
int Nrad;

Real initial_D2G[NDUSTFLUIDS], Stokes_number[NDUSTFLUIDS], H_ratio[NDUSTFLUIDS];
bool mom_correct_Flag, Isothermal_Flag, Damping_Flag;

Real x1min, x1max;
Real damping_rate, radius_inner_damping, radius_outer_damping, inner_ratio_region, outer_ratio_region, inner_width_damping, outer_width_damping;
Real min_tol, max_dfvdt, dust_start_injection, injection_Tsoft;


// User Sources
void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void LocalIsothermalEOS(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void ThermalRelaxation(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void RadiativeCondution(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void drift_vel(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
// void phase_change(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
//     const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
//     AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);

// dustfluid settings
void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
  const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time,
  const int il, const int iu, const int jl, const int ju, const int kl, const int ku);

void MyDustDiffusivity(DustFluids *pdf, MeshBlock *pmb,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &stopping_time,
      AthenaArray<Real> &nu_dust, AthenaArray<Real> &cs_dust,
      int is, int ie, int js, int je, int ks, int ke);
} // namespace

// User-defined boundary conditions for disk simulations
void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh);

// void LocalIsothermalEOS(MeshBlock *pmb, int il, int iu, int jl,
    // int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons);
void InnerWaveDampingGas(MeshBlock *pmb, const Real time, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons);
void OuterWaveDampingGas(MeshBlock *pmb, const Real time, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in mesh class.  Can also be used
//! to initialize variables which are global to (and therefore can be passed to) other
//! functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Get parameters for gravitatonal potential of central point mass
  gm0 = pin->GetOrAddReal("problem", "gm0", 0.0);
  r0  = pin->GetOrAddReal("problem", "r0", 1.0);
  mom_correct_Flag = pin->GetBoolean("problem",   "mom_correct_Flag");
  Isothermal_Flag  = pin->GetBoolean("problem", "Isothermal_Flag");
  Damping_Flag             = pin->GetBoolean("problem", "Damping_Flag");
  //Relaxation_Flag = pin->GetBoolean("problem",   "Relaxation_Flag");

  // Get parameters for initial density and velocity
  rho0 = pin->GetReal("problem", "rho0");
  dslope = pin->GetOrAddReal("problem", "dslope", -1.0);
  T0 = pin->GetOrAddReal("problem", "T0", 150.0);

  // Get parameters of initial pressure and cooling parameters
  if (NON_BAROTROPIC_EOS) {
    p0_over_r0 = pin->GetOrAddReal("problem", "p0_over_r0", 0.0025);
    Tslope     = pin->GetReal("problem", "Tslope");
    p_over_r_slope = Tslope;
    gamma_gas  = pin->GetReal("hydro", "gamma");
  } else {
    p0_over_r0 = SQR(pin->GetReal("hydro", "iso_sound_speed"));
  }

  Real float_min = std::numeric_limits<float>::min();
  dfloor  = pin->GetOrAddReal("hydro", "dfloor",  (1024*(float_min)));
  dffloor = pin->GetOrAddReal("dust",  "dffloor", (1024*(float_min)));
  Omega0    = pin->GetOrAddReal("orbital_advection", "Omega0", 0.0);
  nu_alpha  = pin->GetOrAddReal("problem", "nu_alpha", 0.0);
  
  // Dust to gas ratio && dust stopping time
  if (NDUSTFLUIDS > 0) {
    for (int n=0; n<NDUSTFLUIDS; n++) {
      initial_D2G[n]   = pin->GetReal("dust", "initial_D2G_" + std::to_string(n+1));
      Stokes_number[n] = pin->GetReal("dust", "Stokes_number_" + std::to_string(n+1));
      H_ratio[n]        = pin->GetReal("dust", "Hratio_" + std::to_string(n+1));
    }
  }

  // phase_change
  min_tol = pin->GetOrAddReal("problem", "min_tol", 1.e-7);
  max_dfvdt = pin->GetOrAddReal("problem", "max_dfvdt", 10.0);
  dust_start_injection = pin->GetReal("problem", "dust_start_injection");
  injection_Tsoft = pin->GetReal("problem", "injection_Tsoft");
  f_ICE_inter0 = pin->GetOrAddReal("problem", "f_ICE_inter0", 0.5);
  rho_sil_inter = pin->GetOrAddReal("problem", "rho_sil_inter", 3.0); // cgs, internal density of silicate
  rho_ice_inter = pin->GetOrAddReal("problem", "rho_ice_inter", 1.0); // cgs, internal density of ice
  taus0 = pin->GetOrAddReal("problem", "taus0", 0.03); // cgs, initial stokes number at r0
  M_dot_g = pin->GetOrAddReal("problem", "M_dot_g", 1.e-8); // M_sun/ Yr, gas accretion rate
  p2g_flux = pin->GetOrAddReal("problem", "p2g_flux", 0.8); // pebble to gas accretion rate
  // alpha = pin->GetOrAddReal("problem", "alpha", 3.e-3); // pebble to gas accretion rate
  m_p0 = -1.0;
  // The parameters of damping zones
  x1min = pin->GetReal("mesh", "x1min");
  x1max = pin->GetReal("mesh", "x1max");

  //ratio of the orbital periods between the edge of the wave-killing zone and the corresponding edge of the mesh
  inner_ratio_region = pin->GetOrAddReal("problem", "inner_dampingregion_ratio", 1.5);
  outer_ratio_region = pin->GetOrAddReal("problem", "outer_dampingregion_ratio", 1.2);

  radius_inner_damping = x1min*std::pow(inner_ratio_region, TWO_3RD);
  radius_outer_damping = x1max*std::pow(outer_ratio_region, -TWO_3RD);

  inner_width_damping = radius_inner_damping - x1min;
  outer_width_damping = x1max - radius_outer_damping;

  // The normalized wave damping timescale, in unit of dynamical timescale.
  damping_rate = pin->GetOrAddReal("problem", "damping_rate", 1.0);

  // enroll user-defined boundary condition
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiskOuterX1);
  }

  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, DiskInnerX1);
  }

  // Enroll local isothermal equation of state
  EnrollUserExplicitSourceFunction(MySource);
  // Enroll dust settings
  if (NDUSTFLUIDS > 0) {
    // Enroll user-defined dust stopping time
    EnrollUserDustStoppingTime(MyStoppingTime);
    // Enroll user-defined dust diffusivity
    EnrollDustDiffusivity(MyDustDiffusivity);
  }

    // print examination
  std::cout << "Parameters in this simulation"<< std::endl;
  std::cout << "Tslope ="<< Tslope<< std::endl;
  std::cout << "dslope ="<< dslope<< std::endl;
  std::cout << "p_over_r_slope ="<< p_over_r_slope<< std::endl;
  // std::cout << "UNIT_PRS ="<< UNIT_PRS<< std::endl;
  std::cout << "KELVIN ="<< KELVIN<< std::endl;
  std::cout << "T_a=" << T_a << "K" << std::endl;
  std::cout << "inner_damping_radius =" << radius_inner_damping/r0*3.0 <<std::endl;

  return;
}

// enroll user defined output variables
void MeshBlock::InitUserMeshBlockData(ParameterInput *pin){
  AllocateUserOutputVariables(4);
  SetUserOutputVariableName(0,"Tem");
  // SetUserOutputVariableName(1,"gamma");
  SetUserOutputVariableName(1,"st");
  // SetUserOutputVariableName(3,"dfvdt");
  SetUserOutputVariableName(2,"dif");
  SetUserOutputVariableName(3,"m_p");
  // SetUserOutputVariableName(6,"flx");
  // SetUserOutputVariableName(7,"ice");
  
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Initializes Keplerian accretion disk.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real x1, x2, x3;
	Real igm1 = 1.0/(gamma_gas - 1.0);
  OrbitalVelocityFunc &vK = porb->OrbitalVelocity;
  Nrad = pcoord->pmy_block->ncells1;
  
  //  Initialize density and momenta
  for (int k=ks; k<=ke; ++k) {
    x3 = pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
      x2 = pcoord->x2v(j);
      for (int i=is-NGHOST; i<=ie+NGHOST; ++i) {
        x1 = pcoord->x1v(i);
        GetCylCoord(pcoord, rad, phi, z, i, j, k); // convert to cylindrical coordinates
        Real vel_K     = vK(porb, x1, x2, x3);

        // compute initial conditions in cylindrical coordinates
        Real den_gas = DenProfileCyl_gas(rad, phi, z);
        Real h_gas = Get_H_gas(rad,0.0);
        Real sigma_gas = den_gas*std::sqrt(2.0*PI)*h_gas;
        Real gas_nu = Get_nu_gas(rad,0.0);
        Real v_acc_gas = -1.5*gas_nu/rad; // -1.5*nu_gas/rad

        Real vel_gas_phi = VelProfileCyl_gas(rad, phi, z);
        if (porb->orbital_advection_defined)
          vel_gas_phi -= vel_K;

        Real &gas_dens = phydro->u(IDN, k, j, i);
        Real &gas_mom1 = phydro->u(IM1, k, j, i);
        Real &gas_mom2 = phydro->u(IM2, k, j, i);
        Real &gas_mom3 = phydro->u(IM3, k, j, i);

        gas_dens = sigma_gas;
        gas_mom1 = sigma_gas*v_acc_gas;
        if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
          gas_mom2 = sigma_gas*vel_gas_phi;
          gas_mom3 = 0.0;
        } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
          gas_mom2 = 0.0;
          gas_mom3 = sigma_gas*vel_gas_phi;
        }

        Real Tem = TemProfile(rad);
        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN, k, j, i)  = gas_dens*Tem/(KELVIN*mu_xy)*igm1;
          phydro->u(IEN, k, j, i) += 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3))/gas_dens;
          phydro->Tem(k, j, i) = Tem;
        }

        // store all dustfluid volume density.
        AthenaArray<Real> rho_dustfluid_array;
        rho_dustfluid_array.NewAthenaArray(NDUSTFLUIDS);
        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            // compute initial conditions in cylindrical coordinates
            Real den_dust = DenProfileCyl_dust(rad, phi, z, initial_D2G[dust_id], H_ratio[dust_id]);
            Real vel_dust_phi = VelProfileCyl_dust(rad, phi, z);
            if (porb->orbital_advection_defined)
              vel_dust_phi -= vel_K;
            rho_dustfluid_array(dust_id) = den_dust;

            Real sigma_dust = den_dust*std::sqrt(2.*PI)*h_gas;
            pdustfluids->df_u(rho_id, k, j, i) = sigma_dust;

            if(dust_id != NDUSTFLUIDS-1){
              pdustfluids->df_u(v1_id,  k, j, i) = 0.0;
              if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
                pdustfluids->df_u(v2_id, k, j, i) = sigma_dust*vel_dust_phi;
                pdustfluids->df_u(v3_id, k, j, i) = 0.0;
              } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
                pdustfluids->df_u(v2_id, k, j, i) = 0.0;
                pdustfluids->df_u(v3_id, k, j, i) = sigma_dust*vel_dust_phi;
              }
            }else{
              pdustfluids->df_u(v1_id,  k, j, i) = sigma_dust*v_acc_gas;
              if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
                pdustfluids->df_u(v2_id, k, j, i) = sigma_dust*vel_gas_phi;
                pdustfluids->df_u(v3_id, k, j, i) = 0.0;
              } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
                pdustfluids->df_u(v2_id, k, j, i) = 0.0;
                pdustfluids->df_u(v3_id, k, j, i) = sigma_dust*vel_gas_phi;
              }
            }
					}
				}
        std::cout << "At (i,j,k)=(" << i << "," << j << "," << k << "):" << std::endl;
        std::cout << "gas_sigma = " << sigma_gas << std::endl;
        std::cout << "gas_vel1 = " << v_acc_gas<< std::endl;
        std::cout << "gas_vel2 = " << vel_gas_phi<< std::endl;
        std::cout << "gas_Tem = " << Tem << std::endl;
        std::cout << "gas_erg = " << phydro->u(IEN, k, j, i) << std::endl;
        std::cout << "=========================================" << std::endl;

        // initialize stopping time change
        // initialize dust/vapor diffusivity
        if (pphase_change != nullptr) {
          for (int p = 0; p < N_P; ++p) {
              int refrac_id = N_Z * (p+1) - 1;
              int rho_id = 4*refrac_id;
              Real sigma_sil_p = pdustfluids->df_u(rho_id, k, j, i);
              Real &rho_Np = pphase_change->rho_Np_array(p, k, j, i);
              rho_Np = sigma_sil_p /(1.0 - 0.0)/(pphase_change->m_p0_array(p)); // [code_number_density]
          }
        }

        Real omega_dyn = std::sqrt(gm0/std::pow(rad,3.0));
        
        if (pphase_change != nullptr) {
          // vapor diffusivity (with artificial decay for outer boundary)
          Real &vapor_diffusivity = pdustfluids->nu_dustfluids_array(vapor_id, k, j, i);
          vapor_diffusivity = gas_nu;

          // deal with vapor separately
          Real &st_time_vapor = pdustfluids->stopping_time_array(vapor_id, k, j, i);
          st_time_vapor = Stokes_number[vapor_id]/omega_dyn;

          // Loop over pebbles to calculate stopping time per pebble
          AthenaArray<Real> rho_dustfluid_array_pebble;
          rho_dustfluid_array_pebble.NewAthenaArray(N_Z);
          for (int p = 0; p < N_P; ++p) {

            // collect dustfluid of the same pebble
            for (int z = 0; z < N_Z; ++z) {
              int dust_id = N_Z * p + z;
              int rho_id = 4*dust_id;
              Real rhod= pdustfluids->df_u(rho_id, k, j, i)/std::sqrt(2.0*PI)/h_gas/H_ratio[p]; // convert back to volume density for stopping time calculation
              rho_dustfluid_array_pebble(z) = rhod;
            }

            // calculate stopping time for the pebble
            // (Yu, 2025-11-18) Use rho_Np_array instead of m_p_array/s_p_array
            Real rho_Np = pphase_change->rho_Np_array(p, k, j, i);
            // (Yu, 2025-11-16) Pass vapor density separately since it's not in per-pebble array
            Real rho_v = pdustfluids->df_u(4*vapor_id, k, j, i);
            Real gas_dens = phydro->u(IDN, k, j, i)/std::sqrt(2.0*PI)/h_gas; // convert back to volume density for stopping time calculation
                
              // std::cout << "rho_dustfluid_array_pebble(" << z << ") = " << rho_dustfluid_array_pebble(z) << " at (k,j,i)=("<<k<<","<<j<<","<<i<<")!" << std::endl;
            Real t_stop = pphase_change->Get_stopping_time(pmy_mesh->punit, rho_dustfluid_array_pebble, Tem, gas_dens, rho_v, rho_Np);
            // ===== upper limit, St can be extremely high for upper layer =====
            t_stop = (t_stop > 0.5) ? 0.5 : t_stop;

            // apply stopping time to all compositions of this pebble
            for (int z = 0; z < N_Z; ++z) {
              int dust_id = N_Z * p + z;
              Real &st_time = pdustfluids->stopping_time_array(dust_id, k, j, i);
              
              if(pmy_mesh->time > dust_start_injection){
                st_time = t_stop;
              }else{
                st_time = 1.e-8;
              }
              // apply st floor 
              st_time = (st_time > 1.e-8) ? st_time : 1.e-8;
              
              // calculate diffusivity
              Real taus_peb = t_stop*omega_dyn;
              Real &diffusivity = pdustfluids->nu_dustfluids_array(dust_id, k, j, i);
              diffusivity = gas_nu/(1.+SQR(taus_peb));
            }
          }
        }

      }
    }
  }
  
  return;
}

void MeshBlock::UserWorkInLoop(){

  int dk     = NGHOST;
  int dj     = NGHOST;
  if (block_size.nx3 == 1) dk = 0;
  if (block_size.nx2 == 1) dj = 0;
  int kl = ks - dk;     int ku = ke + dk;
  int jl = js - dj;     int ju = je + dj;
  int il = is - NGHOST; int iu = ie + NGHOST;

  Real &dt   = pmy_mesh->dt;
  Real &time = pmy_mesh->time;

    // wave damping
  if (Damping_Flag) {
    InnerWaveDampingGas(this, time, dt, il, iu, jl, ju, kl, ku, phydro->w, phydro->u);
  }

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is-NGHOST; i<=ie+NGHOST; ++i) {
        // pre-calculation
        const Real &gas_den = phydro->w(IDN,k ,j ,i);
        const Real &gas_vel1 = phydro->w(IVX, k, j, i);
        const Real &gas_vel2 = phydro->w(IVY, k, j, i);
        const Real &gas_vel3 = phydro->w(IVZ, k, j, i);

        // copy gas velocity to tracer;
        int dust_id = NDUSTFLUIDS-1;
        int rho_id  = 4*dust_id;
        int v1_id   = rho_id + 1;
        int v2_id   = rho_id + 2;
        int v3_id   = rho_id + 3;
        
        const Real &dust_rho  = pdustfluids->df_w(rho_id, k, j, i);
        Real &dust_vel1 = pdustfluids->df_w(v1_id,  k, j, i);
        Real &dust_vel2 = pdustfluids->df_w(v2_id,  k, j, i);
        Real &dust_vel3 = pdustfluids->df_w(v3_id,  k, j, i);

        const Real &dust_den  = pdustfluids->df_w(rho_id, k, j, i);
        Real &dust_mom1 = pdustfluids->df_w(v1_id,  k, j, i);
        Real &dust_mom2 = pdustfluids->df_w(v2_id,  k, j, i);
        Real &dust_mom3 = pdustfluids->df_w(v3_id,  k, j, i);

        dust_vel1 = gas_vel1;
        dust_vel2 = gas_vel2;
        dust_vel3 = gas_vel3;
        dust_mom1 = dust_den*gas_vel1;
        dust_mom2 = dust_den*gas_vel2;
        dust_mom3 = dust_den*gas_vel3;

        // copy refractory velocity to ice
        // refractory
        // dust_id = 1;
        // rho_id  = 4*dust_id;
        // v1_id   = rho_id + 1;
        // v2_id   = rho_id + 2;
        // v3_id   = rho_id + 3;
        //
        // const Real &d1_vel1 = pdustfluids->df_w(v1_id,  k, j, i);
        // const Real &d1_vel2 = pdustfluids->df_w(v2_id,  k, j, i);
        // const Real &d1_vel3 = pdustfluids->df_w(v3_id,  k, j, i);
        
        // ice
        // dust_id = 0;
        // rho_id  = 4*dust_id;
        // v1_id   = rho_id + 1;
        // v2_id   = rho_id + 2;
        // v3_id   = rho_id + 3;
        //
        // Real &d0_vel1 = pdustfluids->df_w(v1_id,  k, j, i);
        // Real &d0_vel2 = pdustfluids->df_w(v2_id,  k, j, i);
        // Real &d0_vel3 = pdustfluids->df_w(v3_id,  k, j, i);
        //
        // const Real &d0_den  = pdustfluids->df_u(rho_id, k, j, i);
        // Real &d0_mom1 = pdustfluids->df_u(v1_id,  k, j, i);
        // Real &d0_mom2 = pdustfluids->df_u(v2_id,  k, j, i);
        // Real &d0_mom3 = pdustfluids->df_u(v3_id,  k, j, i);
        //
        // d0_vel1 = d1_vel1;
        // d0_vel2 = d1_vel2;
        // d0_vel3 = d1_vel3;
        // d0_mom1 = d1_vel1*d0_den;
        // d0_mom2 = d1_vel2*d0_den;
        // d0_mom3 = d1_vel3*d0_den;
      }
    }
  }

  // update self-defined data:
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is-NGHOST; i<is; ++i) {
        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real &m_p_ghost = pphase_change->m_p0_array(k,j,i);
            Real &m_p_active = pphase_change->m_p0_array(k,j,2*is-i-1);
            m_p_ghost = m_p_active;

            // if(!pdustfluids->istracer[dust_id]){  
            //   Real &stopping_time_ghost = pdustfluids->stopping_time_array(dust_id,k,j,i);
            //   Real &stopping_time_active = pdustfluids->stopping_time_array(dust_id,k,j,2*is-i-1);
            //   stopping_time_ghost = stopping_time_active;
            // }
          }
        }
      }
    }
  }

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=ie+1; i<=ie+NGHOST; ++i) {
        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real &m_p_ghost = pphase_change->m_p0_array(k,j,i);
            Real &m_p_active = pphase_change->m_p0_array(k,j,2*ie-i+1);
            m_p_ghost = m_p_active;

            // if(!pdustfluids->istracer[dust_id]){  
            //   Real &stopping_time_ghost = pdustfluids->stopping_time_array(dust_id,k,j,i);
            //   Real &stopping_time_active = pdustfluids->stopping_time_array(dust_id,k,j,2*ie-i+1);
            //   stopping_time_ghost = stopping_time_active;
            // }
          }
        }
      }
    }
  }

  // locally isothermal
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is-NGHOST; i<=ie+NGHOST; ++i) {
        Real rad, phi, z;
        GetCylCoord(pcoord, rad, phi, z, i, j, k);

        Real &gas_dens = phydro->u(IDN, k, j, i);
        Real &gas_mom1 = phydro->u(IM1, k, j, i);
        Real &gas_mom2 = phydro->u(IM2, k, j, i);
        Real &gas_mom3 = phydro->u(IM3, k, j, i);
        Real &gas_erg  = phydro->u(IEN, k, j, i);

        const Real &rho_v = pdustfluids->df_u(4*(NDUSTFLUIDS-1), k, j, i);

        // temperature profile: fixed
        Real Tem = TemProfile(rad);
        Real gas_vel1 = gas_mom1/gas_dens;
        Real gas_vel2 = gas_mom2/gas_dens;
        Real gas_vel3 = gas_mom3/gas_dens;

        //! \fn gen density (do not allow H/He diffusion, fix it to initial profile):
        // simple model
        // Real den_gas_0 = DenProfileCyl_gas(rad, phi, z);
        // Real h_gas_0 = Get_H_gas(rad,0.0);
        // Real sigma_gas_0 = den_gas_0*std::sqrt(2.0*PI)*h_gas_0;        
        // gas_dens += rho_v;
        // gas_mom1 = gas_dens*gas_vel1;
        // gas_mom2 = gas_dens*gas_vel2;
        // gas_mom3 = gas_dens*gas_vel3;
        
        // get pressure of local temperature
        Real fv = rho_v/gas_dens;
        if(std::isnan(fv)){
          std::cout <<"fv= nan in LocalIsothermalEOS" << std::endl;
          std::cout << "rhov =" << rho_v << std::endl;
          quick_exit(1);
          // fv = 0.0;
        }
        Real mu = Get_mu(fv);
        Real press = gas_dens * Tem /(KELVIN*mu);
        Real gamma = peos->calc_gamma(fv);

        gas_erg    = press/(gamma-1.0) + 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3))/gas_dens;

        // update prim
        phydro->w(IDN, k, j, i) = gas_dens;
        phydro->w(IPR, k, j, i) = press;
        phydro->Tem(k, j, i) = Tem;
      }
    }
  }


}


void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin){
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is-NGHOST; i<=ie+NGHOST; ++i) {   
        const Real &rho_g = phydro->w(IDN,k,j,i);
        const Real &press = phydro->w(IPR,k,j,i);
        const Real &rho_v = pdustfluids->df_w(4*(NDUSTFLUIDS-1),k,j,i);
        Real E_kg = 0.5*(SQR(phydro->u(IM1,k,j,i)) + SQR(phydro->u(IM2,k,j,i)) + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
        Real rhoe_g = phydro->u(IEN,k,j,i) - E_kg;
        Real fv;
        
        // std::cout << "E_kg =  " << E_kg<< std::endl;
        // std::cout << "rhoe =  " << rhoe<< std::endl;

        fv = rho_v/rho_g;
        Real T = Get_T_rhoe_g(rhoe_g,rho_g,fv);
        Real mu = Get_mu(fv);
        Real prs = rho_g*T/(mu*KELVIN);
        // output
        user_out_var(0,k,j,i) = T;
        // user_out_var(1,k,j,i) = prs/rhoe_g + 1.0;
        user_out_var(1,k,j,i) = pdustfluids->stopping_time_array(0,k,j,i);
        // user_out_var(3,k,j,i) = pdustfluids->dfv_dt(k,j,i);
        // user_out_var(2,k,j,i) = pdustfluids->nu_dustfluids_array(2,k,j,i);
        user_out_var(2,k,j,i) = phydro->hdif.nu(HydroDiffusion::DiffProcess::alpha, k, j, i);
        // user_out_var(3,k,j,i) = pphase_change->m_p0_array(k,j,i);
        // user_out_var(6,k,j,i) = phydro->flux[IDN](k,j,i);
        // user_out_var(7,k,j,i) = pdustfluids->df_flux[X1DIR](0,k,j,i);
      }
    }
  }
  return;
}

namespace {
//----------------------------------------------------------------------------------------
//! transform to cylindrical coordinate

Real Get_T_rhoe_g(Real rhoe_g, Real rho_g, Real fv){
  Real fx = 1.0-fv; // He/H fraction
  Real e = rhoe_g/rho_g;
  Real T = e*KELVIN/(fx/mu_H2*0.71*2.5+fx/mu_He*0.29*1.5+fv/mu_z*3.0);
  return T;
}

Real Get_T_rhoe(Real rhoe, Real rho_g, AthenaArray<Real> rho_d_array, Real fv){
  Real fx = 1.0-fv; // He/H fraction
  Real bottom = rho_g/KELVIN*(fx/mu_H2*0.71*2.5+fx/mu_He*0.29*1.5+fv/mu_z*3.0);
  for (int m = 0; m<= NDUSTFLUIDS-N_P-2; ++m){
    bottom += rho_d_array(m)*Cd_water_cgs;
  }
  Real T = rhoe/bottom;

  return T;
}

Real Get_rhomu_d(AthenaArray<Real> rho_d_array){
  Real rhomu_d = 0.0;
  for (int m = 0; m<= NDUSTFLUIDS-N_P-2; ++m){
    rhomu_d += rho_d_array(m)*(-L_heat_cgs);
  }
  return rhomu_d;
}

Real Get_rhoe(Real rhoE_total, Real rho_g, Real E_kg, AthenaArray<Real> rho_d_array, AthenaArray<Real> E_kd_array){
  Real rhoe = 0.0;
  Real rhomu_d = Get_rhomu_d(rho_d_array);
  rhoe = rhoE_total- rho_g*E_kg- rhomu_d;

  for (int m = 0; m<= NDUSTFLUIDS-N_P-2; ++m){
    rhoe -= E_kd_array(m)*rho_d_array(m);
  }

  return rhoe;
}

Real Get_Z(Real rho_g, Real T){
  Real z,P_eq,rhoz,kB_mp;
  
  P_eq = P_eq0_cgs*exp(-T_a/T);
  kB_mp = 1.0/KELVIN;
  rhoz = P_eq * mu_z /(T*kB_mp);

  // z = rhoz/rho_g;
  // z = 1/(1+(1/(z+1e-8)));  // softening of z profile
  
  return rhoz;
}

Real Get_mu(Real fv){
  return 1./((1.-fv)/mu_xy + fv / mu_z);
}

void Get_E_kg(AthenaArray<Real> gas_vel_array, Real &E_kg){
  E_kg = 0.0;
  for (int n = 0; n<= 1-1; ++n){
    E_kg += 0.5*SQR(gas_vel_array(n));
  }
  return;
}

void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s){
  
  // drift_vel(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);
  // if(PHASE_CHANGE){
  //   phase_change(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);
  // } 
  // drift_vel(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);
  // LocalIsothermalEOS(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);

  return;
}

void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
  const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time,
  const int il, const int iu, const int jl, const int ju, const int kl, const int ku){
//   int nc1 = pmb->ncells1;
//   AthenaArray<Real> rad_arr, phi_arr, z_arr;
//   rad_arr.NewAthenaArray(nc1);
//   phi_arr.NewAthenaArray(nc1);
//   z_arr.NewAthenaArray(nc1);

//   // update stopping time change
//   AthenaArray<Real> rho_dustfluid_array;
//   rho_dustfluid_array.NewAthenaArray(NDUSTFLUIDS);
//   Real rho_g, rho_v, fv;
//   Real inv_sqrt_gm0 = 1.0/std::sqrt(gm0);

//   for (int n=0; n<NDUSTFLUIDS; ++n) {
//     int dust_id = n;
//     for (int k=pmb->ks; k<=pmb->ke; ++k) {
//       for (int j=pmb->js; j<=pmb->je; ++j) {
// #pragma omp simd
//         for (int i=pmb->is; i<=pmb->ie; ++i) {
//           GetCylCoord(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k);
//           rho_g = prim(IDN, k, j, i);
//           rho_v = prim_df(4*(NDUSTFLUIDS-1), k, j ,i);
//           fv = rho_g/rho_v;
//           Real h_gas = Get_H_gas(rad_arr(i),fv);
//           Real Tem = TemProfile(rad_arr(i));
//           Real omega_dyn = std::sqrt(gm0/std::pow(rad_arr(i),3.0));
//           rho_dustfluid_array(0) = prim_df(0, k, j ,i)/(std::sqrt(2.*PI)*h_gas);
//           rho_dustfluid_array(1) = prim_df(4*1, k, j ,i)/(std::sqrt(2.*PI)*h_gas);
//           rho_dustfluid_array(2) = rho_v/(std::sqrt(2.*PI)*h_gas);

//           Real rho_g_3D = rho_g/(std::sqrt(2.*PI)*h_gas);
//           Real &m_p = pmb->pdustfluids->m_p_array(k,j,i);
//           Real t_stop = Get_stopping_time(rho_dustfluid_array,Tem,rho_g_3D,rad_arr(i),m_p);
//           Real gas_nu = Get_nu_gas(rad_arr(i),fv);

//           Real &diffusivity = pmb->pdustfluids->nu_dustfluids_array(dust_id, k, j, i);
//           Real &st_time = pmb->pdustfluids->stopping_time_array(dust_id, k,j,i);

//           if(!pmb->pdustfluids->istracer[dust_id]){  
//             if(time > dust_start_injection){
//               st_time = t_stop;
//             }else{
//               st_time= 1.e-8;
//             }
//             // apply st floor 
//             st_time = (st_time > 1.e-8) ? st_time : 1.e-8;
//           }
//           //calculate diffusivity
//           Real taus_peb= t_stop*omega_dyn;
//           if(!pmb->pdustfluids->istracer[dust_id]){
//             diffusivity = gas_nu/(1.+SQR(taus_peb));
//             // diffusivity = 0.0;
//           }else{
//             diffusivity = gas_nu;
//           }
//         }
//       }
//     }
//   }
  return;
}

void MyDustDiffusivity(DustFluids *pdf, MeshBlock *pmb,
    const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
    const AthenaArray<Real> &stopping_time, AthenaArray<Real> &nu_dust,
    AthenaArray<Real> &cs_dust, int is, int ie, int js, int je, int ks, int ke) {

  // int nc1 = pmb->ncells1;
  // AthenaArray<Real> rad_arr, phi_arr, z_arr;
  // rad_arr.NewAthenaArray(nc1);
  // phi_arr.NewAthenaArray(nc1);
  // z_arr.NewAthenaArray(nc1);

  // Real inv_sqrt_gm0 = 1.0/std::sqrt(gm0);

//   for (int n=0; n<NDUSTFLUIDS; n++) {
//     int dust_id = n;
//     for (int k=ks; k<=ke; ++k) {
//       for (int j=js; j<=je; ++j) {
// #pragma omp simd
//         for (int i=is; i<=ie; ++i) {
//           GetCylCoord(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k);
//           //rad_arr(i) = pmb->pcoord->x1v(i);
//           Real inv_Omega_K = std::pow(rad_arr(i), 1.5)*inv_sqrt_gm0;
//           Real fv = prim_df(4*(NDUSTFLUIDS-1), k, j, i)/w(IDN, k, j, i);
//           Real nu_gas      = Get_nu_gas(rad_arr(i),fv);

//           Real &diffusivity = nu_dust(dust_id, k, j, i);
//           diffusivity       = nu_gas/(1.0 + SQR(Stokes_number[dust_id]));

//           Real &soundspeed  = cs_dust(dust_id, k, j, i);
//           soundspeed        = std::sqrt(diffusivity*inv_Omega_K);
//         }
//       }
//     }
//   }
  return;
}



void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k) {

  rad = pco->x1v(i);
  phi = pco->x2v(j);
  z   = pco->x3v(k);
  return;
}

//----------------------------------------------------------------------------------------
//! 1D case for tem/cs/scale height calculation
Real TemProfile(const Real rad) {
  Real tmp = T0* std::pow((rad/r0),p_over_r_slope);
  return tmp;
}

Real Get_Cs_gas(const Real rad, Real fv) {
  Real tem  = TemProfile(rad);
  // Real mu = Get_mu(fv);
  Real mu = mu_xy; // simple model, sound speed indep. of mu
  return std::sqrt(tem/(KELVIN*mu));
}

Real Get_H_gas(const Real rad, Real fv) {
  Real tem  = TemProfile(rad);
  // Real mu = Get_mu(fv);
  Real mu = mu_xy; // simple model, scale height indep. of mu
  Real cs = std::sqrt(tem/(KELVIN*mu));
  Real rad3 = SQR(rad)*rad;
  Real omega = std::sqrt(gm0/rad3);

  return cs/omega;
  // return 1.0/std::sqrt(2.0*PI);
}

Real Get_nu_gas(const Real rad, Real fv) {
  // Real tem  = TemProfile(rad);
  // // Real mu = Get_mu(fv);
  // Real mu = mu_xy; // simple model, scale height indep. of mu
  // Real cs = std::sqrt(tem/(KELVIN*mu));
  // Real rad3 = SQR(rad)*rad;
  // Real omega = std::sqrt(gm0/rad3);
  // Real nu_gas = nu_alpha*SQR(cs)/omega;

  Real h_gas = Get_H_gas(rad,0.0);
  Real cs = Get_Cs_gas(rad,0.0);
  Real nu_gas = nu_alpha*cs*h_gas;

  return nu_gas;
}

Real Get_eta_vk(AthenaArray<Real> rad, Real k, Real j, Real i, const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_df){
  AthenaArray<Real> prs_sigma, prs, fv, h_gas, sigma_vapor, sigma_gas;
  prs_sigma.NewAthenaArray(5);
  h_gas.NewAthenaArray(5);
  fv.NewAthenaArray(5);
  sigma_vapor.NewAthenaArray(5);
  sigma_gas.NewAthenaArray(5);
  prs.NewAthenaArray(5);

  Real rho_g_3D, mu1, tem;

  for (int n=0; n<5; ++n) {
    prs_sigma(n) = prim(IPR, k, j, i+n-2);
    sigma_gas(n) = prim(IDN, k, j, i+n-2);
    sigma_vapor(n) = prim_df(4*(NDUSTFLUIDS-1), k, j, i+n-2);
    fv(n) = sigma_vapor(n)/sigma_gas(n);
    h_gas(n) = Get_H_gas(rad(n),fv(n));

    // full model-- taking into account the change of mu
    // prs(n) = prs_sigma(n)/(std::sqrt(2.0*PI)*h_gas(n));

    // keep eta vk unchanged.-- simple model
    tem = TemProfile(rad(n));
    rho_g_3D = DenProfileCyl_gas(rad(n),0.0,0.0);
    prs(n) = rho_g_3D*tem/(mu_xy*KELVIN);
  }

  Real dr = fabs(rad(1)-rad(0));
  // dfdx[i] = (-y[i+2]+8*y[i+1]-8*y[i-1]+y[i-2])/(12*dx);
  Real dprsdr = (-prs(4) + 8.0*prs(3)-8.0*prs(1)+prs(0))/(12.0*dr);
  // Real dprsdr = (prs(3)-prs(2))/(dr);
  Real cs = Get_Cs_gas(rad(2),fv(2));
  Real vk = std::sqrt(gm0/rad(2));
  Real eta_vk = 0.5*SQR(cs)/vk*(rad(2)/prs(2))*dprsdr;

  return eta_vk;
}

Real Get_v_gas(Real k, Real j, Real i, MeshBlock *pmb, const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_df){
  AthenaArray<Real> fv, nu_gas, sigma_vapor, sigma_gas, tmp, rad;
  nu_gas.NewAthenaArray(5);
  fv.NewAthenaArray(5);
  sigma_vapor.NewAthenaArray(5);
  sigma_gas.NewAthenaArray(5);
  tmp.NewAthenaArray(5);
  rad.NewAthenaArray(5);

  Real dr = fabs(pmb->pcoord->dx1v(i));

  for (int n=0; n<5; ++n) {
    if(i>=2 and (Nrad -i)>=3){
      rad(n) = pmb->pcoord->x1v(i+n-2);
      sigma_gas(n) = prim(IDN, k, j, i+n-2);
      sigma_vapor(n) = prim_df(4*(NDUSTFLUIDS-1), k, j, i+n-2);
      fv(n) = sigma_vapor(n)/sigma_gas(n);
    }else if(i<2){
      rad(n) = pmb->pcoord->x1v(i+n);
      sigma_gas(n) = prim(IDN, k, j, i+n);
      sigma_vapor(n) = prim_df(4*(NDUSTFLUIDS-1), k, j, i+n);
      fv(n) = sigma_vapor(n)/sigma_gas(n);
    }else{
      // std::cout << "i="<< i << std::endl;
      rad(n) = pmb->pcoord->x1v(i-n);
      sigma_gas(n) = prim(IDN, k, j, i-n);
      sigma_vapor(n) = prim_df(4*(NDUSTFLUIDS-1), k, j, i-n);
      fv(n) = sigma_vapor(n)/sigma_gas(n);
    }
    
    nu_gas(n) = Get_nu_gas(rad(n),fv(n));
    tmp(n) = nu_gas(n)*sigma_gas(n)*std::sqrt(rad(n));
  }

  Real dtmpdr;
  Real v_gas_acc;
  if(i>=2 and (Nrad -i)>=3){
    dtmpdr = (-tmp(4) + 8.0*tmp(3)-8.0*tmp(1)+tmp(0))/(12.0*dr);
    v_gas_acc = -3.0/(std::sqrt(rad(2))*sigma_gas(2))*dtmpdr;
  }else if(i<2){
    dtmpdr = (-25.0/12.0*tmp(0) + 4.0*tmp(1)-3.0*tmp(2) +4.0/3.0*tmp(3)-1.0/4.0*tmp(4))/dr;
    v_gas_acc = -3.0/(std::sqrt(rad(0))*sigma_gas(0))*dtmpdr;
  }else{
    dtmpdr = -(-25.0/12.0*tmp(0) + 4.0*tmp(1)-3.0*tmp(2) +4.0/3.0*tmp(3)-1.0/4.0*tmp(4))/dr;
    v_gas_acc = -3.0/(std::sqrt(rad(0))*sigma_gas(0))*dtmpdr;
  }

  return v_gas_acc;
}

Real Get_eta_vk_old(Real rad, Real k, Real j, Real i, const AthenaArray<Real> &prim, Real fv, Real dr){
  const Real &prs1 = prim(IPR, k, j, i-2);
  const Real &prs2 = prim(IPR, k, j, i-1);
  const Real &prs3 = prim(IPR, k, j, i);
  const Real &prs4 = prim(IPR, k, j, i+1);
  const Real &prs5 = prim(IPR, k, j, i+2);

  // dfdx[i] = (-y[i+2]+8*y[i+1]-8*y[i-1]+y[i-2])/(12*dx);
  Real dprsdr = (-prs5 + 8.0*prs4-8.0*prs2+prs1)/(12.0*dr);
  Real cs = Get_Cs_gas(rad,fv);
  Real vk = std::sqrt(gm0/rad);
  Real eta_vk = 0.5*SQR(cs)/vk*(rad/prs3)*dprsdr;

  return eta_vk;
}

//----------------------------------------------------------------------------------------
//! computes density in cylindrical coordinates

Real DenProfileCyl_gas(const Real rad, const Real phi, const Real z) {
  Real den;
  Real p_over_r = p0_over_r0;
  if (NON_BAROTROPIC_EOS) p_over_r = PoverRho(rad, phi, z);
  Real denmid = rho0*std::pow(rad/r0,dslope);
  Real dentem = denmid*std::exp(gm0/p_over_r*(1./std::sqrt(SQR(rad)+SQR(z))-1./rad));
  den = dentem;
  return std::max(den,dfloor);
}

//----------------------------------------------------------------------------------------
//! computes rotational velocity in cylindrical coordinates
Real VelProfileCyl_gas(const Real rad, const Real phi, const Real z) {
  Real p_over_r = PoverRho(rad, phi, z);
  // Real vel = (dslope+p_over_r_slope)*p_over_r/(gm0/rad) + (1.0+p_over_r_slope)
            //  - p_over_r_slope*rad/std::sqrt(rad*rad+z*z);
  Real vel = (dslope + p_over_r_slope + Tslope/2.0 + 1.5)*p_over_r/(gm0/rad) + (1.0+p_over_r_slope)
             - p_over_r_slope*rad/std::sqrt(rad*rad+z*z);
  vel = std::sqrt(gm0/rad)*std::sqrt(vel) - rad*Omega0;
  return vel;
}

Real DenProfileCyl_dust(const Real rad, const Real phi, const Real z, const Real den_ratio, const Real H_ratio) {
  Real den;
  Real p_over_r = p0_over_r0;
  if (NON_BAROTROPIC_EOS) p_over_r = PoverRho(rad, phi, z);
  Real denmid = den_ratio*rho0*std::pow(rad/r0,dslope);
  Real dentem = denmid*std::exp(gm0/(SQR(H_ratio)*p_over_r)*(1./std::sqrt(SQR(rad)+SQR(z))-1./rad));
  den         = dentem;
  return std::max(den,dffloor);
}

Real VelProfileCyl_dust(const Real rad, const Real phi, const Real z) {
  Real dis = std::sqrt(SQR(rad) + SQR(z));
  Real vel = std::sqrt(gm0/dis) - rad*Omega0;
  return vel;
}

void Vr_interpolate_outer_nomatter(const Real r_active, const Real r_ghost, const Real sigma_active,
    const Real sigma_ghost, const Real vr_active, Real &vr_ghost) {
      
  vr_ghost = (sigma_active*r_active*vr_active)/(sigma_ghost*r_ghost);
  return;
}

void sigma_interpolate_inner_nomatter(const Real r_active, const Real r_ghost, const Real sigma_active,
    Real &sigma_ghost, const Real vr_active, const Real vr_ghost) {

  sigma_ghost = (sigma_active*r_active*vr_active)/(vr_ghost*r_ghost);
  return;
}

void sigma_interpolate_inner_log(const Real r_active, const Real r_ghost, const Real sigma_active,
    Real &sigma_ghost) {
  
  Real sigma_slope = dslope + Tslope/2.0 + 1.5;
  Real logdr = std::log(r_active/r_ghost);
  sigma_ghost = std::exp(std::log(sigma_active) - logdr*sigma_slope);

  return;
}

//----------------------------------------------------------------------------------------
//! computes pressure/density in cylindrical coordinates

Real PoverRho(const Real rad, const Real phi, const Real z) {
  Real poverr;
  poverr = p0_over_r0*std::pow(rad/r0, p_over_r_slope);
  return poverr;
}

} // namespace

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values
void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  Real rad_active(0.0), phi_active(0.0), z_active(0.0);
  Real rad_ghost(0.0),  phi_ghost(0.0),  z_ghost(0.0);
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        GetCylCoord(pco, rad_active, phi_active, z_active, il,   j, k);
        GetCylCoord(pco, rad_ghost,  phi_ghost,  z_ghost,  il-i, j, k);
        Real vel_K     = vK(pmb->porb, pco->x1v(il-i), pco->x2v(j), pco->x3v(k));

        Real &gas_sigma_ghost  = prim(IDN, k, j, il-i);
        Real &gas_vel1_ghost = prim(IM1, k, j, il-i);
        Real &gas_vel2_ghost = prim(IM2, k, j, il-i);
        Real &gas_vel3_ghost = prim(IM3, k, j, il-i);
        Real &gas_pres_ghost = prim(IEN, k, j, il-i);
        
        Real &gas_sigma_active  = prim(IDN, k, j, il);
        Real &gas_vel1_active = prim(IM1, k, j, il);
        Real &gas_vel2_active = prim(IM2, k, j, il);

        Real &gas_pres_active = prim(IPR, k, j, il);

        Real gas_nu = Get_nu_gas(rad_ghost,0.0);
        Real v_acc_gas = -1.5*gas_nu/rad_ghost; // negative accretion vel
        Real vel_gas_phi = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
        if (pmb->porb->orbital_advection_defined)
          vel_gas_phi -= vel_K;
        
        // const mass flux bc
        // gas_vel1_ghost = gas_vel1_active;
        gas_vel1_ghost = v_acc_gas;
        gas_vel2_ghost = vel_gas_phi;
        gas_vel3_ghost = 0.0;
        // gas_sigma_ghost = gas_sigma_active*gas_vel1_active*rad_active/(rad_ghost*gas_vel1_ghost);

        // bc that keeps the slope of density the same
        // sigma_interpolate_inner_log(rad_active,rad_ghost,gas_sigma_active,gas_sigma_ghost);
        
        Real fv = prim_df(4*(NDUSTFLUIDS-1), k, j, il)/gas_sigma_active;
        // keep H/He uneffected at far boundary
        Real gas_rho_xy = DenProfileCyl_gas(rad_ghost,phi_ghost,z_ghost);
        Real h_gas = Get_H_gas(rad_ghost,0.0);
        gas_sigma_ghost = gas_rho_xy*std::sqrt(2.0*PI)*h_gas/(1.0-fv);
        
        // std::cout << "gas_sigma_ghost = " << gas_sigma_ghost << std::endl;
        // std::cout << "v_phi_gas = " << vel_gas_phi << std::endl;
        // std::cout << "gas_vel1_ghost = " << gas_vel1_ghost << std::endl;
        // relaxing bc given mas flux
        // Real Mdot_xy = -M_dot_g*CONST_Msun/CONST_yr; // negative velocity
        // Mdot_xy /= (SQR(UNIT_LENGTH)*UNIT_LENGTH*UNIT_DENSITY/UNIT_T);
        // gas_sigma_ghost = (1.0+p2g_flux*f_ICE_inter0)*Mdot_xy/(2.0*PI*rad_ghost*v_acc_gas);

        if (NON_BAROTROPIC_EOS){
          // Real fv = 0.0;
          Real T1 = TemProfile(rad_ghost);
          Real mu1 = Get_mu(fv);
          gas_pres_ghost = gas_sigma_ghost*T1/(mu1*KELVIN);
        }

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;
            
            if(dust_id != vapor_id){
              // free outflow for dust
              prim_df(rho_id,k,j,il-i) = prim_df(rho_id,k,j,il+i-1);
              prim_df(v1_id,k,j,il-i) = prim_df(v1_id,k,j,il+i-1);
              prim_df(v2_id,k,j,il-i) = prim_df(v2_id,k,j,il+i-1);
              prim_df(v3_id,k,j,il-i) = prim_df(v3_id,k,j,il+i-1);
            }else{
              Real &vapor_sigma_ghost  = prim_df(rho_id, k, j, il-i);
              Real &vapor_vel1_ghost = prim_df(v1_id,  k, j, il-i);
              Real &vapor_vel2_ghost = prim_df(v2_id,  k, j, il-i);
              Real &vapor_vel3_ghost = prim_df(v3_id,  k, j, il-i);
              
              // keep the consistency between gas and vapor.
              vapor_vel1_ghost = gas_vel1_ghost;
              vapor_vel2_ghost = vel_gas_phi;
              vapor_vel3_ghost = 0.0;
              vapor_sigma_ghost = fv*gas_sigma_ghost;
            }
          }
        }
      }
    }
  }
}

void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad_ghost(0.0),  phi_ghost(0.0),  z_ghost(0.0);
  Real rad_active(0.0), phi_active(0.0), z_active(0.0);
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        GetCylCoord(pco, rad_active, phi_active, z_active, iu, j, k);
        Real &gas_rho_active  = prim(IDN, k, j, iu);
        Real &gas_vel1_active = prim(IM1, k, j, iu);
        Real &gas_pres_active = prim(IPR, k, j, iu);
        
        GetCylCoord(pco, rad_ghost, phi_ghost, z_ghost, iu+i, j, k);
        Real &gas_sigma_ghost  = prim(IDN, k, j, iu+i);
        Real &gas_vel1_ghost = prim(IM1, k, j, iu+i);
        Real &gas_vel2_ghost = prim(IM2, k, j, iu+i);
        Real &gas_vel3_ghost = prim(IM3, k, j, iu+i);
        Real &gas_pres_ghost = prim(IEN, k, j, iu+i);
        Real vel_K     = vK(pmb->porb, pco->x1v(iu+i), pco->x2v(j), pco->x3v(k));
        
        Real gas_rho = DenProfileCyl_gas(rad_ghost,phi_ghost,z_ghost);
        Real h_gas = Get_H_gas(rad_ghost,0.0);
        Real omega_dyn = std::sqrt(gm0/std::pow(rad_ghost,3.0));
        gas_sigma_ghost = gas_rho*std::sqrt(2.0*PI)*h_gas;
        
        Real cs = Get_Cs_gas(rad_ghost,0.0);
        Real vk_0 = std::sqrt(gm0/rad_ghost);
        Real eta_vk = 0.5*SQR(cs)/vk_0*(Tslope+dslope);

        // viscous solution
        Real v_acc_gas;
        Real gas_nu = Get_nu_gas(rad_ghost,0.0);
        v_acc_gas = -1.5*gas_nu/rad_ghost; // negative accretion vel

        Real vel_gas_phi = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
        if (pmb->porb->orbital_advection_defined)
          vel_gas_phi -= vel_K;

        Real Mdot_xy = -M_dot_g*Constants::solar_mass_cgs/Constants::yr_cgs;
        Mdot_xy /= pmb->pmy_mesh->punit->code_mass_cgs/ pmb->pmy_mesh->punit->code_time_cgs; // in code unit      

        // gas_sigma_ghost = gas_rho_active*rad_active/rad_ghost;
        // gas_vel1_ghost = Mdot_xy/(2.0*PI*rad_ghost*gas_sigma_ghost);
        // gas_vel1_ghost = gas_vel1_active;
        // gas_sigma_ghost = Mdot_xy/(2.0*PI*rad_ghost*gas_vel1_ghost);

        gas_vel1_ghost = v_acc_gas;
        gas_vel2_ghost = vel_gas_phi;
        gas_vel3_ghost = 0.0;
        v_acc_gas = gas_vel1_ghost;

        if(std::isnan(gas_vel1_ghost)){
          std::cout <<"gas_vel1_ghost = nan" << std::endl;
          quick_exit(1);
        }
        if(std::isnan(gas_sigma_ghost)){
          std::cout <<"gas_sigma_ghost = nan" << std::endl;
          quick_exit(1);
        }
        
        if (NON_BAROTROPIC_EOS)
          gas_pres_ghost = gas_sigma_ghost*SQR(cs);

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real &dust_sigma_ghost  = prim_df(rho_id, k, j, iu+i);
            Real &dust_vel1_ghost = prim_df(v1_id,  k, j, iu+i);
            Real &dust_vel2_ghost = prim_df(v2_id,  k, j, iu+i);
            Real &dust_vel3_ghost = prim_df(v3_id,  k, j, iu+i);

            if(dust_id != vapor_id){
              Real t_stop = pmb->pdustfluids->stopping_time_array(dust_id, k,j,iu-i+1);
              Real taus = t_stop*omega_dyn;
              Real v_drift_peb = (v_acc_gas + 2.0*eta_vk*taus)/(1.+SQR(taus));
              
              dust_vel1_ghost = v_drift_peb;
              // Real H_peb = h_gas*std::sqrt(nu_alpha/(nu_alpha+taus));
              Real Mdot_peb = p2g_flux*M_dot_g*Constants::solar_mass_cgs/Constants::yr_cgs;
              // Mdot_peb /= (SQR(UNIT_LENGTH)*std::sqrt(2.0*PI)*H_peb*UNIT_LENGTH*UNIT_DENSITY/UNIT_T); // in code unit
              Mdot_peb /= (pmb->pmy_mesh->punit->code_mass_cgs/ pmb->pmy_mesh->punit->code_time_cgs); // in code unit 
              Real sigma_peb = Mdot_peb/(2.0*PI*rad_ghost*fabs(v_drift_peb));

              dust_sigma_ghost = sigma_peb*f_ICE_inter0*(1.0-std::exp(-0.5*SQR((time-dust_start_injection)/injection_Tsoft)));
            }else{
              dust_sigma_ghost = DenProfileCyl_dust(rad_ghost, phi_ghost, z_ghost, initial_D2G[dust_id], H_ratio[dust_id])*std::sqrt(2.*PI)*h_gas;
              dust_vel1_ghost = gas_vel1_ghost;
            }

            Real vel_dust_phi = VelProfileCyl_dust(rad_ghost, phi_ghost, z_ghost);
            if (pmb->porb->orbital_advection_defined)
              vel_dust_phi -= vel_K;
            dust_vel2_ghost = vel_dust_phi;
            dust_vel3_ghost = 0.0;
          }
        }
      }
    }
  }
}

void InnerWaveDampingGas(MeshBlock *pmb, const Real time, const Real dt, int il, int iu, int jl, int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons) {

  Real inv_inner_damp = 1.0/inner_width_damping;
  Real inv_outer_damp = 1.0/outer_width_damping;

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;

  int nc1 = pmb->ncells1;

  AthenaArray<Real> omega_dyn, R_func, inv_damping_tau;
  AthenaArray<Real> rad_arr, phi_arr, z_arr;

  omega_dyn.NewAthenaArray(nc1);
  R_func.NewAthenaArray(nc1);
  inv_damping_tau.NewAthenaArray(nc1);
  rad_arr.NewAthenaArray(nc1);
  phi_arr.NewAthenaArray(nc1);
  z_arr.NewAthenaArray(nc1);

  for (int k=kl; k<=ku; ++k) {
    Real x3 = pmb->pcoord->x3v(k);
    for (int j=jl; j<=ju; ++j) {
      Real x2 = pmb->pcoord->x2v(j);
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        Real x1 = pmb->pcoord->x1v(i);
        GetCylCoord(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k); // convert to cylindrical coordinates

        if (rad_arr(i) <= radius_inner_damping) {
          // See de Val-Borro et al. 2006 & 2007
          omega_dyn(i)       = std::sqrt(gm0/(rad_arr(i)*rad_arr(i)*rad_arr(i)));
          R_func(i)          = SQR((rad_arr(i) - radius_inner_damping)*inv_inner_damp);
          inv_damping_tau(i) = (damping_rate*omega_dyn(i));

          Real vis_vel_r      = -1.5*Get_nu_gas(rad_arr(i),0.0)/rad_arr(i);
          Real gas_vel1_0 = vis_vel_r;

          Real &gas_rho  = prim(IDN, k, j, i);
          Real &gas_vel1 = prim(IM1, k, j, i);
          Real &gas_vel2 = prim(IM2, k, j, i);
          Real &gas_vel3 = prim(IM3, k, j, i);
          Real &gas_pre  = prim(IPR, k, j, i);

          Real &gas_dens = cons(IDN, k, j, i);
          Real &gas_mom1 = cons(IM1, k, j, i);
          Real &gas_mom2 = cons(IM2, k, j, i);
          Real &gas_mom3 = cons(IM3, k, j, i);
          Real &gas_erg  = cons(IEN, k, j, i);

          Real delta_gas_rho  = 0.0;
          // Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*R_func(i);
          Real delta_gas_vel2 = 0.0;
          Real delta_gas_vel3 = 0.0;
          Real delta_gas_pre  = 0.0;

          // std::cout <<"inv_damping_tau"<< inv_damping_tau(i) << std::endl;
          // std::cout <<"delta_gas_vel1" << delta_gas_vel1 << std::endl;
          // quick_exit(1);
          
          // initial internal energy
          Real eth0 = gas_erg - 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3))/gas_dens;

          gas_rho  += delta_gas_rho;
          gas_vel1 += delta_gas_vel1;
          gas_vel2 += delta_gas_vel2;
          gas_vel3 += delta_gas_vel3;
          gas_pre  += delta_gas_pre;

          gas_dens = gas_rho;
          gas_mom1 = gas_dens*gas_vel1;
          gas_mom2 = gas_dens*gas_vel2;
          gas_mom3 = gas_dens*gas_vel3;

          Real Ek = 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3));
          gas_erg = eth0 + Ek/gas_dens;
        }
      }
    }
  }
  return;
}

