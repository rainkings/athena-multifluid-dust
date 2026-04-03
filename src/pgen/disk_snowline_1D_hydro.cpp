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
#include <src/defs.hpp>                                      // (Yu) for N_P, N_Z macros
#include "../phase_change/phase_change_constants.hpp"        // (Yu) KELVIN, mu_xy, P_eq0, etc.
#include "../units/units.hpp"                                // (Yu) punit->code_*

namespace {
void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k);
Real PoverRho(const Real rad, const Real phi, const Real z);
Real DenProfileCyl_gas(const Real rad, const Real phi, const Real z);
Real VelProfileCyl_gas(const Real rad, const Real phi, const Real z);
// Real DenProfileCyl_dust(const Real rad, const Real phi, const Real z,     // (Yu) dust disabled
//                         const Real den_ratio, const Real H_ratio);
// Real VelProfileCyl_dust(const Real rad, const Real phi, const Real z);    // (Yu) dust disabled
void Vr_interpolate_outer_nomatter(const Real r_active, const Real r_ghost, const Real sigma_active,
    const Real sigma_ghost, const Real vr_active, Real &vr_ghost);
void sigma_interpolate_inner_nomatter(const Real r_active, const Real r_ghost, const Real sigma_active,
    Real &sigma_ghost, const Real vr_active, const Real vr_ghost);
void sigma_interpolate_inner_log(const Real r_active, const Real r_ghost, const Real sigma_active,
    Real &sigma_ghost);

// (Yu) Phase-change and dust functions — commented out for first migration step
// void Get_vel_new_fromMC(...);
// void Get_E_kg(...);
// Real Get_T_rhoe_g(...);
// Real Get_T_rhoe(...);
// Real Get_rhomu_d(...);
// Real Get_rhoe(...);
// Real Get_Z(...);
// void phase_trans(...);
// void phase_trans_2D(...);
// Real Get_stopping_time(...);
// Real Get_eta_vk(...);
// Real Get_v_gas(...);
// Real Get_eta_vk_old(...);

// Gas helper functions (kept active)
Real Get_mu(Real fv);
Real TemProfile(const Real rad, const Real phi, const Real z);  // (Yu) updated to 3D signature
Real Get_Cs_gas(const Real rad, Real fv);
Real Get_H_gas(const Real rad, Real fv);
Real Get_nu_gas(const Real Tem, const Real rad, Real fv);       // (Yu) added Tem argument

// problem parameters which are useful to make global to this file
// (Yu) renamed: dslope→pvalue, Tslope/p_over_r_slope→qvalue, p0_over_r0→cs2_0, nu_alpha→alpha_vis
Real gm0, r0, rho0, T0, gamma_gas, Omega0, alpha_vis, nu_slope, cs2_0, qvalue, pvalue;
Real dfloor, dffloor, pfloor;
// snowline
Real M_dot_g, L_star;                   // M_sun/yr
Real KELVIN;                            // Temperature conversion factor from PhaseChange module
// Real p2g_flux[NDUSTFLUIDS];          // (Yu) dust disabled
int Nrad;

Real initial_D2G[NDUSTFLUIDS], Stokes_number[NDUSTFLUIDS], Hratio[NDUSTFLUIDS];
bool mom_correct_Flag, Isothermal_Flag, Damping_Flag, Theta_Gas_Damping_Flag, Allow_T_change_Flag;

Real x1min, x1max;
Real damping_rate, radius_inner_damping, radius_outer_damping, inner_ratio_region, outer_ratio_region, inner_width_damping, outer_width_damping;
Real min_tol, max_dfvdt, dust_start_injection, injection_Tsoft, t_restart;
Real kappa0, t_iterate, beta, f_vi;


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
// void drift_vel(...)  // (Yu) dust disabled
// void phase_change(...)  // (Yu) phase change disabled

// dustfluid settings
// (Yu) updated MyStoppingTime signature to match DustStoppingTimeFunc in athena.hpp
void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time,
    const int il, const int iu, const int jl, const int ju, const int kl, const int ku);
void MyDustDiffusivity(DustFluids *pdf, MeshBlock *pmb,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &stopping_time,
      AthenaArray<Real> &nu_dust, AthenaArray<Real> &cs_dust,
      int is, int ie, int js, int je, int ks, int ke);
// (Yu) User-defined thermal conductivity (new in updated code)
void MyConductivity(HydroDiffusion *phdif, MeshBlock *pmb,
    const AthenaArray<Real> &w, const AthenaArray<Real> &bc,
    int is, int ie, int js, int je, int ks, int ke);
} // namespace

// User-defined boundary conditions for disk simulations
void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh);

// InnerWaveDampingGas / OuterWaveDampingGas are called directly from UserWorkInLoop
// (not enrolled as SrcTermFunc), so they keep the direct-call signature.
// void InnerWaveDampingGas(MeshBlock *pmb, const Real time, const Real dt, int il, int iu,
//     int jl, int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons);
// void OuterWaveDampingGas(MeshBlock *pmb, const Real time, const Real dt, int il, int iu,
//     int jl, int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons);

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
  mom_correct_Flag       = pin->GetBoolean("problem", "mom_correct_Flag");
  Isothermal_Flag        = pin->GetBoolean("problem", "Isothermal_Flag");
  Damping_Flag           = pin->GetBoolean("problem", "Damping_Flag");
  Theta_Gas_Damping_Flag = pin->GetOrAddBoolean("problem", "Theta_Gas_Damping_Flag", false);
  Allow_T_change_Flag    = pin->GetBoolean("problem", "Allow_T_change_Flag");

  // Get parameters for initial density and velocity
  rho0   = pin->GetReal("problem", "rho0");
  pvalue = pin->GetOrAddReal("problem", "pvalue", -1.0);   // (Yu) was dslope
  T0     = pin->GetOrAddReal("problem", "T0", 150.0);
  kappa0     = pin->GetOrAddReal("problem", "kappa0", 1.0);  // cgs
  kappa0    *= punit->code_density_cgs * punit->code_length_cgs;  // convert to code units
  t_iterate  = pin->GetOrAddReal("problem", "t_iterate", 0.0);
  // Compute KELVIN from code units (same calculation as in PhaseChange module)
  KELVIN = SQR(punit->code_velocity_cgs) / (Constants::k_boltzmann_cgs / Constants::hydrogen_mass_cgs);
  beta       = pin->GetOrAddReal("problem", "beta", 1.0);
  f_vi       = pin->GetOrAddReal("problem", "f_vi", 1.0);

  // Get parameters of initial pressure and cooling parameters
  if (NON_BAROTROPIC_EOS) {
    cs2_0     = pin->GetOrAddReal("problem", "cs2_0", 0.0025);  // (Yu) was p0_over_r0
    qvalue    = pin->GetReal("problem", "qvalue");               // (Yu) was Tslope
    gamma_gas = pin->GetReal("hydro", "gamma");
  } else {
    cs2_0 = SQR(pin->GetReal("hydro", "iso_sound_speed"));
  }

  Real float_min = std::numeric_limits<float>::min();
  dfloor  = pin->GetOrAddReal("hydro", "dfloor",  (1024*(float_min)));
  pfloor  = pin->GetOrAddReal("hydro", "pfloor",  (1024*(float_min)));
  dffloor = pin->GetOrAddReal("dust",  "dffloor", (1024*(float_min)));
  Omega0    = pin->GetOrAddReal("orbital_advection", "Omega0", 0.0);
  alpha_vis = pin->GetOrAddReal("problem", "alpha_vis", 0.0);   // (Yu) was nu_alpha
  nu_slope  = pin->GetOrAddReal("problem", "nu_slope", qvalue+1.5);

  // Dust to gas ratio && dust stopping time  (Yu: dust disabled — values read but not used)
  if (NDUSTFLUIDS > 0) {
    for (int n=0; n<NDUSTFLUIDS; n++) {
      initial_D2G[n]   = pin->GetOrAddReal("dust", "initial_D2G_"   + std::to_string(n+1), 0.0);
      Stokes_number[n] = pin->GetOrAddReal("dust", "Stokes_number_" + std::to_string(n+1), 0.01);
      Hratio[n]        = pin->GetOrAddReal("dust", "Hratio_"        + std::to_string(n+1), 1.0);
    }
  }

  // Injection / phase-change parameters (kept as optional; dust/phase-change disabled for now)
  min_tol              = pin->GetOrAddReal("problem", "min_tol", 1.e-7);
  max_dfvdt            = pin->GetOrAddReal("problem", "max_dfvdt", 10.0);
  dust_start_injection = pin->GetOrAddReal("problem", "dust_start_injection", 0.0);
  injection_Tsoft      = pin->GetOrAddReal("problem", "injection_Tsoft", 0.1);
  M_dot_g  = pin->GetOrAddReal("problem", "M_dot_g",  1.e-8);  // M_sun/yr
  L_star   = pin->GetOrAddReal("problem", "L_star",   1.0);    // L_sun

  // The parameters of damping zones
  x1min = pin->GetReal("mesh", "x1min");
  x1max = pin->GetReal("mesh", "x1max");

  //ratio of the orbital periods between the edge of the wave-killing zone and the mesh edge
  inner_ratio_region = pin->GetOrAddReal("problem", "inner_dampingregion_ratio", 1.5);
  outer_ratio_region = pin->GetOrAddReal("problem", "outer_dampingregion_ratio", 1.2);

  radius_inner_damping = x1min*std::pow(inner_ratio_region, TWO_3RD);
  radius_outer_damping = x1max*std::pow(outer_ratio_region, -TWO_3RD);

  inner_width_damping = radius_inner_damping - x1min;
  outer_width_damping = x1max - radius_outer_damping;

  // The normalized wave damping timescale, in unit of dynamical timescale.
  damping_rate = pin->GetOrAddReal("problem", "damping_rate", 1.0);

  // enroll user-defined boundary conditions
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiskOuterX1);
  }
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, DiskInnerX1);
  }

  // Enroll source function
  EnrollUserExplicitSourceFunction(MySource);
  // Enroll dust callbacks  (Yu: dust disabled for now, but keep enroll in case NDUSTFLUIDS > 0)
  if (NDUSTFLUIDS > 0) {
    EnrollUserDustStoppingTime(MyStoppingTime);
    EnrollDustDiffusivity(MyDustDiffusivity);
  }
  // Enroll thermal conduction
  EnrollConductionCoefficient(MyConductivity);

  // print examination
  std::cout << "Parameters in this simulation" << std::endl;
  std::cout << "qvalue ="    << qvalue    << std::endl;
  std::cout << "pvalue ="    << pvalue    << std::endl;
  std::cout << "alpha_vis =" << alpha_vis << std::endl;
  std::cout << "inner_damping_radius =" << radius_inner_damping/r0*3.0 << std::endl;

  return;
}

// enroll user defined output variables
void MeshBlock::InitUserMeshBlockData(ParameterInput *pin){
  // (Yu) record restart time (same pattern as 2D reference)
  t_restart = pmy_mesh->time;

  AllocateUserOutputVariables(3);
  SetUserOutputVariableName(0, "Tem");
  SetUserOutputVariableName(1, "st");
  SetUserOutputVariableName(2, "dif");
  
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

  int dk = NGHOST; int dj = NGHOST;
  if (block_size.nx3 == 1) dk = 0;
  if (block_size.nx2 == 1) dj = 0;
  int kl = ks - dk;     int ku = ke + dk;
  int jl = js - dj;     int ju = je + dj;
  int il = is - NGHOST; int iu = ie + NGHOST;

  //  Initialize density and momenta
  for (int k=kl; k<=ku; ++k) {
    x3 = pcoord->x3v(k);
    for (int j=jl; j<=ju; ++j) {
      x2 = pcoord->x2v(j);
      for (int i=il; i<=iu; ++i) {
        x1 = pcoord->x1v(i);
        GetCylCoord(pcoord, rad, phi, z, i, j, k);
        Real vel_K = vK(porb, x1, x2, x3);

        Real Tem = TemProfile(rad, phi, z);  // (Yu) updated to 3D signature

        // compute initial conditions in cylindrical coordinates
        Real den_gas   = DenProfileCyl_gas(rad, phi, z);
        Real h_gas     = Get_H_gas(rad, 0.0);
        Real sigma_gas = den_gas*std::sqrt(2.0*PI)*h_gas;
        Real gas_nu    = Get_nu_gas(Tem, rad, 0.0);   // (Yu) added Tem argument
        Real v_acc_gas = -1.5*gas_nu/rad;

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

        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN, k, j, i)  = gas_dens*Tem/(KELVIN*mu_xy)*igm1;
          phydro->u(IEN, k, j, i) += 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3))/gas_dens;
          phydro->Tem(k, j, i) = Tem;
        }

        // (Yu) Dust fluid initialization — disabled for now
        // if (NDUSTFLUIDS > 0) {
        //   for (int n=0; n<NDUSTFLUIDS; ++n) {
        //     int dust_id = n;
        //     int rho_id  = 4*dust_id;
        //     int v1_id   = rho_id + 1;
        //     int v2_id   = rho_id + 2;
        //     int v3_id   = rho_id + 3;
        //     Real den_dust     = DenProfileCyl_dust(rad, phi, z, initial_D2G[dust_id], Hratio[dust_id]);
        //     Real vel_dust_phi = VelProfileCyl_dust(rad, phi, z);
        //     if (porb->orbital_advection_defined)
        //       vel_dust_phi -= vel_K;
        //     Real sigma_dust = den_dust*std::sqrt(2.*PI)*h_gas;
        //     pdustfluids->df_u(rho_id, k, j, i) = sigma_dust;  // (Yu) df_cons → df_u
        //     if (!pdustfluids->istracer[dust_id]) {
        //       pdustfluids->df_u(v1_id, k, j, i) = 0.0;
        //       pdustfluids->df_u(v2_id, k, j, i) = sigma_dust*vel_dust_phi;
        //       pdustfluids->df_u(v3_id, k, j, i) = 0.0;
        //     } else {
        //       pdustfluids->df_u(v1_id, k, j, i) = sigma_dust*v_acc_gas;
        //       pdustfluids->df_u(v2_id, k, j, i) = sigma_dust*vel_gas_phi;
        //       pdustfluids->df_u(v3_id, k, j, i) = 0.0;
        //     }
        //   }
        // }

        // Initialize gas viscosity (alpha prescription)
        Real &gas_nu_arr = phydro->hdif.nu(HydroDiffusion::DiffProcess::alpha, k, j, i);
        gas_nu_arr = alpha_vis * std::pow(rad/r0, nu_slope);

        // (Yu) Dust stopping time / diffusivity — disabled
        // Real omega_dyn = std::sqrt(gm0/std::pow(rad,3.0));
        // for (int n=0; n<NDUSTFLUIDS; ++n) { ... }

        Real Mdot_gas = -M_dot_g* Constants::solar_mass_cgs/ Constants::yr_cgs; // in cgs
        Mdot_gas /= (pmy_mesh->punit->code_mass_cgs /pmy_mesh->punit->code_time_cgs);
  
        if (pbval->block_bcs[BoundaryFace::outer_x1] == BoundaryFlag::user) {
          phydro->inflx_x1(0,0,0) = Mdot_gas/(2*PI*pcoord->x1v(ie+1));
        };
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
  // if (Damping_Flag) {
  //   InnerWaveDampingGas(this, time, dt, il, iu, jl, ju, kl, ku, phydro->w, phydro->u);
  // }

  // (Yu) Dust tracer velocity copy — disabled (dust disabled for now)
  // for (int k=ks; k<=ke; ++k) {
  //   for (int j=js; j<=je; ++j) {
  //     for (int i=is-NGHOST; i<=ie+NGHOST; ++i) {
  //       // copy gas velocity to vapor tracer (dust_id = NDUSTFLUIDS-1)
  //       // copy refractory velocity to ice (dust_id = 0)
  //       // using df_w / df_u  (Yu: renamed from df_prim / df_cons)
  //     }
  //   }
  // }

  // (Yu) m_p_array ghost zone update — disabled (phase change disabled)
  // for (int k=ks; k<=ke; ++k) { ... pphase_change->rho_Np_array(p,k,j,i) ... }

  // locally isothermal: reset pressure to T-profile value each step
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=il; i<=iu; ++i) {
        Real rad, phi, z;
        GetCylCoord(pcoord, rad, phi, z, i, j, k);

        Real &gas_dens = phydro->u(IDN, k, j, i);
        Real &gas_mom1 = phydro->u(IM1, k, j, i);
        Real &gas_mom2 = phydro->u(IM2, k, j, i);
        Real &gas_mom3 = phydro->u(IM3, k, j, i);
        Real &gas_erg  = phydro->u(IEN, k, j, i);

        // (Yu) fv=0 since vapor/dust disabled
        Real fv  = 0.0;
        Real Tem = TemProfile(rad, phi, z);  // (Yu) updated to 3D signature
        Real mu  = Get_mu(fv);
        Real press = gas_dens * Tem / (KELVIN*mu);
        Real gamma = peos->calc_gamma(fv);

        gas_erg = press/(gamma-1.0)
                + 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3))/gas_dens;

        phydro->w(IDN, k, j, i) = gas_dens;
        phydro->w(IPR, k, j, i) = press;
        phydro->Tem(k, j, i)    = Tem;
      }
    }
  }

  // (Yu) Phase change block — disabled
  // if (pphase_change != nullptr && N_Z > 0) { ... }

}


void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin){
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real rad, phi, z;
        GetCylCoord(pcoord, rad, phi, z, i, j, k);

        // (Yu) fv = 0 since vapor/dust disabled
        Real fv  = 0.0;
        Real Tem = phydro->Tem(k, j, i);

        user_out_var(0, k, j, i) = Tem;
        // (Yu) stopping time — dust disabled, output zero placeholder
        user_out_var(1, k, j, i) = 0.0;
        user_out_var(2, k, j, i) = phydro->hdif.nu(HydroDiffusion::DiffProcess::alpha, k, j, i);
      }
    }
  }
  return;
}

namespace {
//----------------------------------------------------------------------------------------
//! transform to cylindrical coordinate

// (Yu) Phase-change helper functions — disabled for now
// Real Get_T_rhoe_g(Real rhoe_g, Real rho_g, Real fv) { ... }
// Real Get_T_rhoe(Real rhoe, Real rho_g, AthenaArray<Real> rho_d_array, Real fv) { ... }
// Real Get_rhomu_d(AthenaArray<Real> rho_d_array) { ... }
// Real Get_rhoe(...) { ... }
// Real Get_Z(Real rho_g, Real T) { ... }
// void Get_E_kg(...) { ... }
// void Get_vel_new_fromMC(...) { ... }
// void phase_trans(Real rad, Real rho_g, Real rho_v, Real &drho) { ... }
// void phase_trans_2D(Real rad, Real rho_g, Real rho_v, Real taus, Real &drho) { ... }
// Real Get_stopping_time(...) { ... }

Real Get_mu(Real fv){
  return 1./((1.-fv)/mu_xy + fv/mu_z);
}

void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s){
  // (Yu) Phase change source — disabled for now
  // (Yu) drift_vel source   — disabled for now
  return;
}

// (Yu) Updated signature to match DustStoppingTimeFunc in athena.hpp
void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time,
    const int il, const int iu, const int jl, const int ju, const int kl, const int ku) {
  // (Yu) Dust stopping time — disabled for now
  // Implement using pphase_change->Get_stopping_time(pmb->pmy_mesh->punit, ...)
  // when dust is re-enabled.
  return;
}

void MyDustDiffusivity(DustFluids *pdf, MeshBlock *pmb,
    const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
    const AthenaArray<Real> &stopping_time, AthenaArray<Real> &nu_dust,
    AthenaArray<Real> &cs_dust, int is, int ie, int js, int je, int ks, int ke) {
  // (Yu) Dust diffusivity — disabled for now
  return;
}

// (Yu) User-defined thermal conductivity — stub, implement when radiative conduction is needed
void MyConductivity(HydroDiffusion *phdif, MeshBlock *pmb,
    const AthenaArray<Real> &w, const AthenaArray<Real> &bc,
    int is, int ie, int js, int je, int ks, int ke) {
  return;
}


// (Yu) drift_vel — disabled (uses old dust API and Get_eta_vk/Get_v_gas helpers)
// void drift_vel(MeshBlock *pmb, ...) { ... }


// void phase_change(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
//     const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
//     AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

//   for (int k=pmb->ks; k<=pmb->ke; ++k) {
//     for (int j=pmb->js; j<=pmb->je; ++j) {
// #pragma omp simd
//       for (int i=pmb->is; i<=pmb->ie; ++i) {
//         Real rad, phi, z;
//         GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
//         // store some constants of pebbles. (No vapor included here to keep the code safe)
//         AthenaArray<Real> E_kd_array, rho_d_array, rho_d_array1, rho_ratio_array;
//         E_kd_array.NewAthenaArray(NDUSTFLUIDS-NRefrac-1);
//         rho_d_array.NewAthenaArray(NDUSTFLUIDS-NRefrac-1);
//         rho_d_array1.NewAthenaArray(NDUSTFLUIDS-NRefrac-1);
//         rho_ratio_array.NewAthenaArray(NDUSTFLUIDS-NRefrac-1);
//         // store velocity
//         AthenaArray<Real> gas_vel_array, gas_vel_array1, dust_vel_array;
//         gas_vel_array.NewAthenaArray(NDIM);
//         gas_vel_array1.NewAthenaArray(NDIM);
//         dust_vel_array.NewAthenaArray(NDUSTFLUIDS-NRefrac-1,NDIM);

//         // pre-calculation
//         Real &rho_g  = cons(IDN, k, j, i);
//         Real &gas_mom1 = cons(IM1, k, j, i);
//         Real &gas_mom2 = cons(IM2, k, j, i);
//         Real &gas_mom3 = cons(IM3, k, j, i);
//         Real &gas_erg  = cons(IEN, k, j, i);
//           // gas kinetic energy
//         Real gas_vel1 = gas_mom1/rho_g;
//         Real gas_vel2 = gas_mom2/rho_g;
//         Real gas_vel3 = gas_mom3/rho_g;
//         Real E_kg = 0.5*(SQR(gas_vel1) + SQR(gas_vel2) + SQR(gas_vel3));
//         gas_vel_array(0) = gas_vel1;
//         gas_vel_array(1) = gas_vel2;
//         gas_vel_array(2) = gas_vel3;
//         gas_vel_array1 = gas_vel_array;

//         // pebble 1
//         int dust_id = 0;
//         int rho_id  = 4*dust_id;
//         int v1_id   = rho_id + 1;
//         int v2_id   = rho_id + 2;
//         int v3_id   = rho_id + 3;

//         Real &rho_I  = cons_df(rho_id, k, j, i);
//         Real &d1_mom1 = cons_df(v1_id,  k, j, i);
//         Real &d1_mom2 = cons_df(v2_id,  k, j, i);
//         Real &d1_mom3 = cons_df(v3_id,  k, j, i);
//         // ICE kinetic energy
//         Real d1_vel1 = d1_mom1/rho_I;
//         Real d1_vel2 = d1_mom2/rho_I;
//         Real d1_vel3 = d1_mom3/rho_I;
//         E_kd_array(0) = 0.5*(SQR(d1_vel1) + SQR(d1_vel2) + SQR(d1_vel3));
//         dust_vel_array(dust_id,0) = d1_vel1;
//         dust_vel_array(dust_id,1) = d1_vel2;
//         dust_vel_array(dust_id,2) = d1_vel3;
        
//         // tracer particle &  vapor fraction:
//         dust_id = NDUSTFLUIDS-1;
//         rho_id  = 4*dust_id;
//         v1_id   = rho_id + 1;
//         v2_id   = rho_id + 2;
//         v3_id   = rho_id + 3;

//         Real &rho_v  = cons_df(rho_id, k, j, i);
//         Real &v1_mom1 = cons_df(v1_id, k, j, i);
//         Real &v1_mom2 = cons_df(v2_id, k, j, i);
//         Real &v1_mom3 = cons_df(v3_id, k, j, i);

//         Real fv = rho_v/rho_g;
//         Real fv0 = fv;

//         if(std::isnan(rho_g)){
//           std::cout << "rho_g is nan \n" << std::endl;
//           std::cout << "i=" << i << std::endl;
//           std::cout << "rad=" << rad << std::endl;
//           quick_exit(1);
//         }
//         if(std::isnan(gas_mom1)){
//             std::cout << "gas_mom1 is nan \n" << std::endl;
//             quick_exit(1);
//         }
//         if(std::isnan(gas_erg)){
//             std::cout << "gas_erg is nan \n" << std::endl;
//             quick_exit(1);
//         }

//         //////////////////////////////////////////////
//         // assign initial values

//         Real rho_g0 = rho_g;
//         rho_d_array(0) = rho_I;
//         Real rho_sil = cons_df(4*1, k, j, i);
//         // dust array initiation
//         rho_d_array1 = rho_d_array;
//         /////////////////////////////////////////////

//         // define variables used in intermediate steps
//         Real drho, rho_g1, rho_v1;
//         Real fx0, fx1, fx2, x0, x1, x2; // secant points
//         rho_g1 = rho_g;

//         // first calculation of phase change quantity and save their initial value
//         phase_trans(rad,rho_g,rho_v,drho);
//         fx0 = drho;
//         x0 = rho_v;

//         Real sign = (drho > 0. ? 1.0:-1.0);
        
//         // supplement
//         Real rho_d_supply = 0.0;
//         AthenaArray<Real> rho_d_supply_array;
//         rho_d_supply_array.NewAthenaArray(NDUSTFLUIDS-NRefrac-1);

//         for (int m = 0; m<= NDUSTFLUIDS-NRefrac-2; ++m){
//           rho_d_supply_array(m) = rho_d_array(m) - dfloor;
//           // still need a floor, which should be much smaller than density floor
//           rho_d_supply_array(m) = (rho_d_supply_array(m) < 1.e-21) ? 1.e-21 : rho_d_supply_array(m);
//           rho_d_supply += rho_d_supply_array(m);
//         }

//         Real rho_v_supply = rho_v - dfloor;
//         rho_v_supply = (rho_v_supply < 1.e-21) ? 1.e-21 : rho_v_supply;
        
//         for (int m = 0; m<= NDUSTFLUIDS-NRefrac-2; ++m){
//           rho_ratio_array(m) = rho_d_supply_array(m)/rho_d_supply;
//         }

//         Real drho_supply = (sign > 0.0 ? rho_d_supply : -rho_v_supply);
//         rho_g1 = rho_g + drho_supply;
//         rho_v1 = rho_v + drho_supply;
//         for (int m = 0; m<= NDUSTFLUIDS-NRefrac-2; ++m){
//           rho_d_array1(m) = rho_d_array(m) - drho_supply*rho_ratio_array(m);
//         }
//         // update drho
//         phase_trans(rad, rho_g1,rho_v1,drho);
//         fx1 = drho;
//         x1 = rho_v1;

//         //***** 2nd step of secant: start from [rhov,rhov1]  *******//
//         // (drho*sign) > 0.
//         rho_g = rho_g1;
//         rho_v = rho_v1;
//         rho_d_array = rho_d_array1;
//         gas_vel_array = gas_vel_array1;

//         if( (drho*sign) < 0.){
//           // secant
//           Real drho_adp;
//           Real f_err = 1.0;
//           int nite = 0;

//           while(f_err>min_tol){
//             nite += 1;
//             if(nite > 1000){
//               std::cout << "nite > 1000, break" <<std::endl;
//               quick_exit(1);
//             }
//             drho_adp = -fx1*(x1-x0)/(fx1-fx0);

//             if(std::isnan(drho_adp)){
//               std::cout << "drho_adp = Nan" <<std::endl;
//               break;
//             }

//             x2 = x1 + drho_adp;

//             rho_g1 = rho_g + drho_adp;
//             rho_v1 = rho_v + drho_adp;
//             // only condense to become st = 0 dusts (already implemented in [phase_trans]).
//             for (int m = 0; m<= NDUSTFLUIDS-NRefrac-2; ++m){
//               rho_d_array1(m) = rho_d_array(m) - drho_adp*rho_ratio_array(m);
//             }
            
//             // update drho
//             phase_trans(rad, rho_g1, rho_v1, drho);
            
//             // update secant point:
//             fx2 = drho;
//             x0 = x1;
//             fx0 = fx1;
//             x1 = x2;
//             fx1 = fx2;

//             // update physical value:
//             rho_g = rho_g1;
//             rho_v = rho_v1;
//             rho_d_array = rho_d_array1;
//             gas_vel_array = gas_vel_array1;

//             // secant method root error fraction:
//             f_err = fabs(drho)/rho_v;
//           }
//         }

//         // calculate dfvdt
//         pmb->pdustfluids->dfv_dt(k,j,i) = (rho_v/rho_g-fv0)/dt;
//         // cons update
//         rho_I = rho_d_array(0);

//         // calculate pressure
//         fv = rho_v/rho_g;
//         Real Tem = TemProfile(rad);
//         Real mu1 = Get_mu(fv);
//         Real prs = rho_g*Tem/(mu1*KELVIN);
//         Real gamma = pmb->peos->calc_gamma(fv);

//         // update stopping time change
//         // update dust diffusivity
//         //////////////////////////////////////////////
//         // tranfer to physical quantity
//         AthenaArray<Real> rho_dustfluid_array;
//         Real h_gas = Get_H_gas(rad,fv);
//         Real omega_dyn = std::sqrt(gm0/std::pow(rad,3.0));

//         rho_dustfluid_array.NewAthenaArray(NDUSTFLUIDS);
//         rho_dustfluid_array(0) = rho_I/(std::sqrt(2.*PI)*h_gas);
//         rho_dustfluid_array(1) = rho_sil/(std::sqrt(2.*PI)*h_gas);
//         rho_dustfluid_array(2) = rho_v/(std::sqrt(2.*PI)*h_gas);

//         Real rho_g_3D = rho_g/(std::sqrt(2.*PI)*h_gas);

//         Real &m_p = pmb->pdustfluids->m_p_array(k,j,i);
//         Real t_stop = Get_stopping_time(rho_dustfluid_array,Tem,rho_g_3D,rad,m_p);
//         Real gas_nu = Get_nu_gas(rad,fv);
//         Real time = pmb->pmy_mesh->time;
//         for (int n=0; n<NDUSTFLUIDS; ++n) {
//           int dust_id = n;
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

//         // Cons update: gas energy, momentum. dust1, 2 momentum.
//         gas_mom1 = rho_g*gas_vel_array(0);
//         gas_mom2 = rho_g*gas_vel_array(1);
//         gas_mom3 = rho_g*gas_vel_array(2);
//         gas_erg = (prs/(gamma - 1.0) + rho_g*E_kg);
        
//         d1_mom1 = rho_I*dust_vel_array(0,0);
//         d1_mom2 = rho_I*dust_vel_array(0,1);
//         d1_mom3 = rho_I*dust_vel_array(0,2);
//         v1_mom1 = rho_v*gas_vel_array(0);
//         v1_mom2 = rho_v*gas_vel_array(1);
//         v1_mom3 = rho_v*gas_vel_array(2);
//       }
//     }
//   }
//   return;
// }

// void LocalIsothermalEOS(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
//     const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
//     AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

//   // Local Isothermal equation of state
//   Real rad, phi, z;
//   int is = pmb->is; int ie = pmb->ie;
//   int js = pmb->js; int je = pmb->je;
//   int ks = pmb->ks; int ke = pmb->ke;

//   // Real igm1 = 1.0/(gamma_gas - 1.0);
//   for (int k=ks; k<=ke; ++k) { // include ghost zone
//     for (int j=js; j<=je; ++j) { // prim, cons
// #pragma omp simd
//       for (int i=is; i<=ie; ++i) {
//         GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);

//         Real &gas_dens = cons(IDN, k, j, i);
//         Real &gas_mom1 = cons(IM1, k, j, i);
//         Real &gas_mom2 = cons(IM2, k, j, i);
//         Real &gas_mom3 = cons(IM3, k, j, i);
//         Real &gas_erg  = cons(IEN, k, j, i);

//         const Real &rho_v = cons_df(4*(NDUSTFLUIDS-1), k, j, i);

//         // temperature profile: fixed
//         Real Tem = TemProfile(rad);
//         Real gas_vel1 = gas_mom1/gas_dens;
//         Real gas_vel2 = gas_mom2/gas_dens;
//         Real gas_vel3 = gas_mom3/gas_dens;

//         //! \fn gen density (do not allow H/He diffusion, fix it to initial profile):
//         // simple model
//         // Real den_gas_0 = DenProfileCyl_gas(rad, phi, z);
//         // Real h_gas_0 = Get_H_gas(rad,0.0);
//         // Real sigma_gas_0 = den_gas_0*std::sqrt(2.0*PI)*h_gas_0;        
//         // gas_dens = rho_v + sigma_gas_0;
//         // gas_mom1 = gas_dens*gas_vel1;
//         // gas_mom2 = gas_dens*gas_vel2;
//         // gas_mom3 = gas_dens*gas_vel3;
//         //
        
//         // get pressure of local temperature
//         Real fv = rho_v/gas_dens;
//         if(std::isnan(fv)){
//           std::cout <<"fv= nan in LocalIsothermalEOS" << std::endl;
//           std::cout << "rhov =" << rho_v << std::endl;
//           quick_exit(1);
//           // fv = 0.0;
//         }
//         Real mu = Get_mu(fv);
//         Real press = gas_dens * Tem /(KELVIN*mu);
//         Real gamma = pmb->peos->calc_gamma(fv);

//         gas_erg    = press/(gamma-1.0) + 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3))/gas_dens;
//       }
//     }
//   } 
  
//   return;
// }

void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k) {

  rad = pco->x1v(i);
  phi = pco->x2v(j);
  z   = pco->x3v(k);
  return;
}

//----------------------------------------------------------------------------------------
//! Temperature profile in cylindrical coordinates  (Yu: updated to 3D signature)
Real TemProfile(const Real rad, const Real phi, const Real z) {
  Real tmp = T0 * std::pow((rad/r0), qvalue);  // (Yu) p_over_r_slope → qvalue
  return tmp;
}

Real Get_Cs_gas(const Real rad, Real fv) {
  Real tem = TemProfile(rad, 0.0, 0.0);
  Real mu  = Get_mu(fv);
  return std::sqrt(tem/(KELVIN*mu));
}

Real Get_H_gas(const Real rad, Real fv) {
  Real tem   = TemProfile(rad, 0.0, 0.0);
  Real mu    = Get_mu(fv);
  Real cs    = std::sqrt(tem/(KELVIN*mu));
  Real omega = std::sqrt(gm0/(SQR(rad)*rad));
  return cs/omega;
}

// (Yu) added Tem as first argument (same as 2D reference)
Real Get_nu_gas(const Real Tem, const Real rad, Real fv) {
  Real mu  = mu_xy;  // simple model, scale height independent of mu
  Real cs2 = Tem/(mu*KELVIN);
  Real omega = std::sqrt(gm0/(rad*rad*rad));
  Real nu_gas = alpha_vis * cs2 / omega;  // (Yu) nu_alpha → alpha_vis
  return nu_gas;
}

// (Yu) Get_eta_vk, Get_v_gas, Get_eta_vk_old — disabled (rely on vapor tracer)
// Real Get_eta_vk(...) { ... }
// Real Get_v_gas(...) { ... }
// Real Get_eta_vk_old(...) { ... }

//----------------------------------------------------------------------------------------
//! computes density in cylindrical coordinates  (Yu: dslope→pvalue, p0_over_r0→cs2_0)

Real DenProfileCyl_gas(const Real rad, const Real phi, const Real z) {
  Real cs2    = PoverRho(rad, phi, z);
  Real denmid = rho0 * std::pow(rad/r0, pvalue);   // (Yu) dslope → pvalue
  Real dentem = denmid * std::exp(gm0/cs2*(1./std::sqrt(SQR(rad)+SQR(z))-1./rad));
  return std::max(dentem, dfloor);
}

//----------------------------------------------------------------------------------------
//! computes rotational velocity in cylindrical coordinates
//! For the 1D (surface-density) disk, the pressure gradient term uses the *surface*
//! density power-law: sigma_slope = pvalue + qvalue/2 + 1.5, not the midplane pvalue.
//! This restores: old (dslope + p_over_r_slope + Tslope/2 + 1.5) → (sigma_slope + qvalue).
Real VelProfileCyl_gas(const Real rad, const Real phi, const Real z) {
  Real cs2 = PoverRho(rad, phi, z);
  Real sigma_slope = pvalue + qvalue/2.0 + 1.5;  // surface density power-law
  Real vel = (sigma_slope + qvalue)*cs2/(gm0/rad) + (1.0+qvalue)
             - qvalue*rad/std::sqrt(rad*rad+z*z);
  vel = std::sqrt(gm0/rad)*std::sqrt(vel) - rad*Omega0;
  return vel;
}

// (Yu) DenProfileCyl_dust and VelProfileCyl_dust — disabled for now
// Real DenProfileCyl_dust(const Real rad, ...) { ... }
// Real VelProfileCyl_dust(const Real rad, ...) { ... }

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
  Real sigma_slope = pvalue + qvalue/2.0 + 1.5;  // (Yu) dslope+Tslope → pvalue+qvalue
  Real logdr = std::log(r_active/r_ghost);
  sigma_ghost = std::exp(std::log(sigma_active) - logdr*sigma_slope);
  return;
}

//----------------------------------------------------------------------------------------
//! computes pressure/density in cylindrical coordinates  (Yu: p0_over_r0→cs2_0, p_over_r_slope→qvalue)

Real PoverRho(const Real rad, const Real phi, const Real z) {
  Real poverr = cs2_0 * std::pow(rad/r0, qvalue);   // (Yu) renamed variables
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

        Real Tem_ghost = TemProfile(rad_ghost, phi_ghost, z_ghost);  // (Yu) 3D signature
        Real gas_nu    = Get_nu_gas(Tem_ghost, rad_ghost, 0.0);     // (Yu) added Tem arg
        Real v_acc_gas = -1.5*gas_nu/rad_ghost;
        Real vel_gas_phi = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
        if (pmb->porb->orbital_advection_defined)
          vel_gas_phi -= vel_K;

        gas_vel1_ghost = v_acc_gas;
        gas_vel2_ghost = vel_gas_phi;
        gas_vel3_ghost = 0.0;

        // (Yu) fv=0 since vapor disabled
        Real fv = 0.0;
        Real gas_rho_xy = DenProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
        Real h_gas = Get_H_gas(rad_ghost, 0.0);
        gas_sigma_ghost = gas_rho_xy * std::sqrt(2.0*PI) * h_gas;

        if (NON_BAROTROPIC_EOS){
          Real mu1 = Get_mu(fv);
          gas_pres_ghost = gas_sigma_ghost * Tem_ghost / (mu1*KELVIN);
        }

        // (Yu) Dust boundary condition — disabled for now
        // if (NDUSTFLUIDS > 0) { ... use prim_df ... }
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
        
        Real Tem_ghost   = TemProfile(rad_ghost, phi_ghost, z_ghost);
        Real cs          = Get_Cs_gas(rad_ghost, 0.0);
        Real gas_nu      = Get_nu_gas(Tem_ghost, rad_ghost, 0.0);
        Real v_acc_gas   = -1.5*gas_nu/rad_ghost;         // viscous inflow (negative)

        Real vel_gas_phi = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
        if (pmb->porb->orbital_advection_defined)
          vel_gas_phi -= vel_K;

        gas_vel1_ghost = v_acc_gas;
        gas_vel2_ghost = vel_gas_phi;
        gas_vel3_ghost = 0.0;

        // Outer boundary influx: Sigma = Mdot / (2*pi*r * v_r)
        // Convert M_dot_g [M_sun/yr] → code units [code_mass / code_time]
        // using static constants solar_mass_cgs and yr_cgs from units.hpp

        Real gas_rho = DenProfileCyl_gas(rad_ghost,phi_ghost,z_ghost);
        Real h_gas = Get_H_gas(rad_ghost,0.0);
        Real omega_dyn = std::sqrt(gm0/std::pow(rad_ghost,3.0));
        gas_sigma_ghost = gas_rho*std::sqrt(2.0*PI)*h_gas;

        if (NON_BAROTROPIC_EOS)
          gas_pres_ghost = gas_sigma_ghost * SQR(cs);

        // (Yu) Dust boundary condition — disabled for now
        // if (NDUSTFLUIDS > 0) { ... }
      }
    }
  }
}

// void InnerWaveDampingGas(MeshBlock *pmb, const Real time, const Real dt, int il, int iu, int jl, int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons) {

//   Real inv_inner_damp = 1.0/inner_width_damping;
//   Real inv_outer_damp = 1.0/outer_width_damping;

//   OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;

//   int nc1 = pmb->ncells1;

//   AthenaArray<Real> omega_dyn, R_func, inv_damping_tau;
//   AthenaArray<Real> rad_arr, phi_arr, z_arr;

//   omega_dyn.NewAthenaArray(nc1);
//   R_func.NewAthenaArray(nc1);
//   inv_damping_tau.NewAthenaArray(nc1);
//   rad_arr.NewAthenaArray(nc1);
//   phi_arr.NewAthenaArray(nc1);
//   z_arr.NewAthenaArray(nc1);

//   for (int k=kl; k<=ku; ++k) {
//     Real x3 = pmb->pcoord->x3v(k);
//     for (int j=jl; j<=ju; ++j) {
//       Real x2 = pmb->pcoord->x2v(j);
// #pragma omp simd
//       for (int i=il; i<=iu; ++i) {
//         Real x1 = pmb->pcoord->x1v(i);
//         GetCylCoord(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k); // convert to cylindrical coordinates

//         if (rad_arr(i) <= radius_inner_damping) {
//           // See de Val-Borro et al. 2006 & 2007
//           omega_dyn(i)       = std::sqrt(gm0/(rad_arr(i)*rad_arr(i)*rad_arr(i)));
//           R_func(i)          = SQR((rad_arr(i) - radius_inner_damping)*inv_inner_damp);
//           inv_damping_tau(i) = (damping_rate*omega_dyn(i));

//           Real Tem_i     = TemProfile(rad_arr(i), 0.0, 0.0);  // (Yu) 3D signature
//           Real vis_vel_r = -1.5*Get_nu_gas(Tem_i, rad_arr(i), 0.0)/rad_arr(i);  // (Yu) added Tem
//           Real gas_vel1_0 = vis_vel_r;

//           Real &gas_rho  = prim(IDN, k, j, i);
//           Real &gas_vel1 = prim(IM1, k, j, i);
//           Real &gas_vel2 = prim(IM2, k, j, i);
//           Real &gas_vel3 = prim(IM3, k, j, i);
//           Real &gas_pre  = prim(IPR, k, j, i);

//           Real &gas_dens = cons(IDN, k, j, i);
//           Real &gas_mom1 = cons(IM1, k, j, i);
//           Real &gas_mom2 = cons(IM2, k, j, i);
//           Real &gas_mom3 = cons(IM3, k, j, i);
//           Real &gas_erg  = cons(IEN, k, j, i);

//           Real delta_gas_rho  = 0.0;
//           // Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*R_func(i)*inv_damping_tau(i)*dt;
//           Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*R_func(i);
//           Real delta_gas_vel2 = 0.0;
//           Real delta_gas_vel3 = 0.0;
//           Real delta_gas_pre  = 0.0;

//           // std::cout <<"inv_damping_tau"<< inv_damping_tau(i) << std::endl;
//           // std::cout <<"delta_gas_vel1" << delta_gas_vel1 << std::endl;
//           // quick_exit(1);
          
//           // initial internal energy
//           Real eth0 = gas_erg - 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3))/gas_dens;

//           gas_rho  += delta_gas_rho;
//           gas_vel1 += delta_gas_vel1;
//           gas_vel2 += delta_gas_vel2;
//           gas_vel3 += delta_gas_vel3;
//           gas_pre  += delta_gas_pre;

//           gas_dens = gas_rho;
//           gas_mom1 = gas_dens*gas_vel1;
//           gas_mom2 = gas_dens*gas_vel2;
//           gas_mom3 = gas_dens*gas_vel3;

//           Real Ek = 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3));
//           gas_erg = eth0 + Ek/gas_dens;
//         }
//       }
//     }
//   }
//   return;
// }

