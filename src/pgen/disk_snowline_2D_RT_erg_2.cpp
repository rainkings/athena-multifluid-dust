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
#include "../phase_change/phase_change.hpp"  // (Yu, 2025-11-16)
#include "../phase_change/phase_change_constants.hpp"  // (Yu, 2025-11-16) For constants: KELVIN, P_eq0, L_heat, etc.
#include "../units/units.hpp"  // (Yu, 2025-11-16) For Constants namespace

namespace {
void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k);
Real PoverRho(const Real rad, const Real phi, const Real z);
Real DenProfileCyl_gas(const Real rad, const Real phi, const Real z);
Real DenProfileCyl_gas_fv(const Real rad, const Real phi, const Real z, Real fv);
Real DenProfileCyl_gas_fv_T(const Real rad, const Real phi, const Real z, Real fv, Real Tem);
Real VelProfileCyl_gas(const Real rad, const Real phi, const Real z);
Real VelProfileCyl_gas_fv(const Real rad, const Real phi, const Real z, Real fv);
Real VelProfileCyl_gas_fv_T(const Real rad, const Real phi, const Real z, Real fv, Real Tem);
Real VelProfileCyl_gas_fv_T_pvalue(const Real rad, const Real phi, const Real z, Real fv, Real Tem);
Real Vr_ProfileCyl_gas(const Real rad, const Real phi, const Real z);
Real Vr_ProfileCyl_gas_fv(const Real rad, const Real phi, const Real z, Real fv);
Real Vr_ProfileCyl_gas_fv_T(const Real rad, const Real phi, const Real z, Real fv, Real Tem);
Real Vr_ProfileCyl_gas_fv_T_pvalue(const Real rad, const Real phi, const Real z, Real fv, Real Tem, Real pnew, Real qnew);
Real five_point(AthenaArray<Real> f, Real x1, Real dx1, int k, int j, int i);
Real five_point_ed(AthenaArray<Real> f, Real x1, Real dx1, int k, int j, int i);
Real five_point_bg(AthenaArray<Real> f, Real x1, Real dx1, int k, int j, int i);
Real NewtonIntpl(const Real f0, const Real f1, const Real f2, const Real x0, const Real x1, const Real x2, const Real x_exp);
void LagIntpl(AthenaArray<Real> x, AthenaArray<Real> y, AthenaArray<Real> x_exp, AthenaArray<Real> &y_exp);
Real DenProfileCyl_dust(const Real rad, const Real phi, const Real z,
                        const Real den_ratio, const Real H_ratio);
Real VelProfileCyl_dust(const Real rad, const Real phi, const Real z);
Real UserOrbitalVelocity(OrbitalAdvection * porb, Real x1, Real x2, Real x3);                                           
void Vr_interpolate_outer_nomatter(const Real r_active, const Real r_ghost, const Real sigma_active,
    const Real sigma_ghost, const Real vr_active, Real &vr_ghost);
void sigma_interpolate_inner_nomatter(const Real r_active, const Real r_ghost, const Real sigma_active,
    Real &sigma_ghost, const Real vr_active, const Real vr_ghost);
void sigma_interpolate_inner_log(const Real r_active, const Real r_ghost, const Real sigma_active,
    Real &sigma_ghost);

// phase change helper functions (some moved to PhaseChange module) (Yu, 2025-11-16)
Real Get_mu(Real fv);  // Keep this as it's used in many places outside phase change
Real TemProfile(const Real rad, const Real theta, const Real phi);
Real Get_kappa(const Real d2g, const Real fv);
Real Get_nu_gas(const Real Tem, const Real rad, Real fv);
void Vr_outflow(const Real r_active, const Real r_ghost, const Real rho_active,
                    const Real rho_ghost, const Real vr_active, Real &vr_ghost);
void Vr_Mdot(const Real r_active, const Real r_ghost, const Real rho_active,
                    const Real rho_ghost, const Real vr_active, Real &vr_ghost);
// problem parameters which are useful to make global to this file
Real gm0, r0, rho0, T0, gamma_gas, Omega0, alpha_vis, nu_slope, cs2_0, qvalue, pvalue;
Real dfloor, dffloor, pfloor;
  //snowline
Real f_ICE_inter0, rho_sil_inter, rho_ice_inter;
Real M_dot_g, L_star; // M_sun/yr
Real p2g_flux;

Real initial_D2G[NDUSTFLUIDS], Stokes_number[NDUSTFLUIDS], Hratio[NDUSTFLUIDS], weight_dust[NDUSTFLUIDS];
bool mom_correct_Flag, Isothermal_Flag, Damping_Flag, Theta_Gas_Damping_Flag ,Allow_T_change_Flag;

Real x1min, x1max, x2min, x2max;
Real damping_rate, radius_inner_damping, radius_outer_damping, inner_ratio_region, outer_ratio_region, inner_width_damping, outer_width_damping, theta_upper_damping, theta_lower_damping, upper_altitude_damping, lower_altitude_damping;
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
void InnerWaveDampingGas(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void OuterWaveDampingGas(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);

// dustfluid settings
void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
  const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time,
  const int il, const int iu, const int jl, const int ju, const int kl, const int ku);
void MyDustDiffusivity(DustFluids *pdf, MeshBlock *pmb,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &stopping_time,
      AthenaArray<Real> &nu_dust, AthenaArray<Real> &cs_dust,
      int is, int ie, int js, int je, int ks, int ke);
// User-defined conductivity
void MyConductivity(HydroDiffusion *phdif, MeshBlock *pmb,
    const AthenaArray<Real> &w, const AthenaArray<Real> &bc,
    int is, int ie, int js, int je, int ks, int ke);

Real CompressedX2(Real x, RegionSize rs);
} // namespace

// User-defined boundary conditions for disk simulations
void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh);

void UpperWaveDampingGas(MeshBlock *pmb, const Real time, const Real dt, int il, int iu,
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
  Theta_Gas_Damping_Flag   = pin->GetOrAddBoolean("problem", "Theta_Gas_Damping_Flag",   false);
  Allow_T_change_Flag = pin->GetBoolean("problem",   "Allow_T_change_Flag");

  // Get parameters for initial density and velocity
  rho0 = pin->GetReal("problem", "rho0");
  pvalue = pin->GetOrAddReal("problem", "pvalue", -1.0);
  T0 = pin->GetOrAddReal("problem", "T0", 150.0);
  kappa0 = pin->GetReal("problem", "kappa0"); // cgs
  kappa0 *= punit->code_density_cgs * punit->code_length_cgs; // code unit
  t_iterate = pin->GetReal("problem", "t_iterate");
  beta = pin->GetReal("problem", "beta");
  f_vi = pin->GetOrAddReal("problem", "f_vi", 1.0);

  // Get parameters of initial pressure and cooling parameters
  if (NON_BAROTROPIC_EOS) {
    cs2_0 = pin->GetOrAddReal("problem", "cs2_0", 0.0025);
    qvalue     = pin->GetReal("problem", "qvalue");
    gamma_gas  = pin->GetReal("hydro", "gamma");
  } else {
    cs2_0 = SQR(pin->GetReal("hydro", "iso_sound_speed"));
  }

  Real float_min = std::numeric_limits<float>::min();
  dfloor  = pin->GetOrAddReal("hydro", "dfloor",  (1024*(float_min)));
  pfloor  = pin->GetOrAddReal("hydro", "pfloor",  (1024*(float_min)));
  dffloor = pin->GetOrAddReal("dust",  "dffloor", (1024*(float_min)));
  Omega0    = pin->GetOrAddReal("orbital_advection", "Omega0", 0.0);
  alpha_vis  = pin->GetOrAddReal("problem", "alpha_vis", 0.0);
  nu_slope = pin->GetOrAddReal("problem", "nu_slope", qvalue+1.5);
  
  // Dust to gas ratio && dust stopping time
  if (NDUSTFLUIDS > 0) {
    for (int n=0; n<NDUSTFLUIDS; n++) {
      initial_D2G[n]   = pin->GetReal("dust", "initial_D2G_" + std::to_string(n+1));
      Stokes_number[n] = pin->GetReal("dust", "Stokes_number_" + std::to_string(n+1));
      Hratio[n]        = pin->GetReal("dust", "Hratio_" + std::to_string(n+1));
      weight_dust[n]   = 2.0/(Stokes_number[n] + SQR(1.0+initial_D2G[n])/Stokes_number[n]);
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
  M_dot_g = pin->GetOrAddReal("problem", "M_dot_g", 1.e-8); // M_sun/ Yr, gas accretion rate
  L_star = pin->GetOrAddReal("problem", "L_star", 1.0); // luminosity of central star [L_sun]
  p2g_flux = pin->GetOrAddReal("problem", "p2g_flux", 0.8); // pebble to gas accretion rate
  // The parameters of damping zones
  x1min = pin->GetReal("mesh", "x1min");
  x1max = pin->GetReal("mesh", "x1max");
  x2min = pin->GetReal("mesh", "x2min");
  x2max = pin->GetReal("mesh", "x2max");

  //ratio of the orbital periods between the edge of the wave-killing zone and the corresponding edge of the mesh
  inner_ratio_region = pin->GetOrAddReal("problem", "inner_dampingregion_ratio", 1.5);
  outer_ratio_region = pin->GetOrAddReal("problem", "outer_dampingregion_ratio", 1.2);

  radius_inner_damping = x1min*std::pow(inner_ratio_region, TWO_3RD);
  radius_outer_damping = x1max*std::pow(outer_ratio_region, -TWO_3RD);

  inner_width_damping = radius_inner_damping - x1min;
  outer_width_damping = x1max - radius_outer_damping;

  // upper_altitude_damping = 0.04*std::sqrt(cs2_0);
  // lower_altitude_damping = 0.02*std::sqrt(cs2_0);
  upper_altitude_damping = (x2max-x2min)*0.4;
  lower_altitude_damping = (x2max-x2min)*0.1;

  theta_upper_damping = x2min + upper_altitude_damping;
  theta_lower_damping = x2max - upper_altitude_damping;

  // The normalized wave damping timescale, in unit of dynamical timescale.
  damping_rate = pin->GetOrAddReal("problem", "damping_rate", 1.0);

  // enroll user-defined boundary condition
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiskOuterX1);
  }

  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, DiskInnerX1);
  }

  if (mesh_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x2, DiskInnerX2);
  }
  // if (mesh_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("user")) {
  //   EnrollUserBoundaryFunction(BoundaryFace::outer_x2, DiskOuterX2);
  // }

  // Enroll local isothermal equation of state
  EnrollUserExplicitSourceFunction(MySource);
  // Enroll dust settings
  if (NDUSTFLUIDS > 0) {
    // Enroll user-defined dust stopping time
    EnrollUserDustStoppingTime(MyStoppingTime);
    // Enroll user-defined dust diffusivity
    EnrollDustDiffusivity(MyDustDiffusivity);
  }
  // Enroll user-defined thermal conduction
  EnrollConductionCoefficient(MyConductivity);

  // Enroll userdef mesh, x2
  if (pin->GetReal("mesh","x2rat") < 0.0){
    EnrollUserMeshGenerator(X2DIR, CompressedX2);
  }

  // global optical depth
  if(true){
    AllocateRealUserMeshDataField(8);
    // tau_vi, tau_eff, q_total, F_z, Tem_RT, int_1 & int_2 to get rho(z)
    ruser_mesh_data[0].NewAthenaArray(mesh_size.nx3, mesh_size.nx2+NGHOST, mesh_size.nx1 + 2*NGHOST);
    ruser_mesh_data[1].NewAthenaArray(mesh_size.nx3, mesh_size.nx2+NGHOST, mesh_size.nx1 + 2*NGHOST);
    ruser_mesh_data[2].NewAthenaArray(mesh_size.nx3, mesh_size.nx2+NGHOST, mesh_size.nx1 + 2*NGHOST);
    ruser_mesh_data[3].NewAthenaArray(mesh_size.nx3, mesh_size.nx2+NGHOST, mesh_size.nx1 + 2*NGHOST);
    ruser_mesh_data[4].NewAthenaArray(mesh_size.nx3, mesh_size.nx2+NGHOST, mesh_size.nx1 + 2*NGHOST);
    ruser_mesh_data[5].NewAthenaArray(mesh_size.nx3, mesh_size.nx2+NGHOST, mesh_size.nx1 + 2*NGHOST);
    ruser_mesh_data[6].NewAthenaArray(mesh_size.nx3, mesh_size.nx2+NGHOST, mesh_size.nx1 + 2*NGHOST);
    ruser_mesh_data[7].NewAthenaArray(mesh_size.nx3, mesh_size.nx2+NGHOST, mesh_size.nx1 + 2*NGHOST);
  }

  // print parameters
  if(Globals::my_rank == 0){
    std::cout << "==============================================================" << std::endl;
    std::cout << "Key parameters for this simulation:" << std::endl;
    // std::cout << "Latent_heat_flag = " << Latent_heat_flag << std::endl;
    std::cout << "kappa0 = " << kappa0 << "cm^2/g" << std::endl;
    std::cout << "L_star = " << L_star << "L_sun" << std::endl;
    std::cout << "==============================================================" << std::endl;
    std::cout << "Parameters in this simulation"<< std::endl;
    std::cout << "qvalue ="<< qvalue<< std::endl;
    std::cout << "pvalue ="<< pvalue<< std::endl;
    std::cout << "KELVIN ="<< KELVIN<< std::endl;
    // std::cout << "p_eq0 ="<< P_eq0<< std::endl;
    // std::cout << "L_heat ="<< L_heat<< std::endl;
    std::cout << "T_a=" << T_a << "K" << std::endl;
    // std::cout << "a_semi =" << a_semi << "AU" <<std::endl;
    // std::cout << "Cd_water =" << Cd_water <<std::endl;
    std::cout << "inner_damping_radius =" << radius_inner_damping/r0*3.0 <<std::endl;
    std::cout << "outer_damping_radius =" << radius_outer_damping/r0*3.0 <<std::endl;
    std::cout << "theta_upper_damping = " << theta_upper_damping << std::endl;
    std::cout << "theta_lower_damping = " << theta_lower_damping << std::endl;

    std::cout << "M_dot_g = " << M_dot_g << "M_sun/yr" << std::endl;
    std::cout << "alpha_vis = " << alpha_vis << std::endl;

    std::cout << "dust_start_injection = " << dust_start_injection << std::endl;

    // print some informations for multi-population dust
    if (N_P > 1) {
      std::cout << "==============================================================" << std::endl;
      std::cout << "Multi-population dust is enabled." << std::endl;
      std::cout << "There are " << N_P << " dust bins defined." << std::endl;
      // for (int n=0; n<N_P; ++n) {
      //   std::cout << "m_p0_" << (n+1) << " = " << pphase_change->m_p0_array(n) << std::endl;
      //   std::cout << "p2g_flux_" << (n+1) << " = " << p2g_flux[n] << std::endl;
      // }
    }

    // check whether the small particles parameters are set 
    // std::cout << NDUSTFLUIDS << " dust fluids are defined." << std::endl;
    // for (int n=0; n<NDUSTFLUIDS; n++) {
    //   std::cout << *(Stokes_number + n) << std::endl;
    //   std::cout << *(Hratio + n) << std::endl;
    // }
    return;
  }
  // // print examination (Yu, 2025-11-16)
  // std::cout << "Parameters in this simulation"<< std::endl;
  // std::cout << "qvalue ="<< qvalue<< std::endl;
  // std::cout << "pvalue ="<< pvalue<< std::endl;
  //
  // // Calculate phase change constants (same as PhaseChange constructor) (Yu, 2025-11-16)
  // std::cout << "inner_damping_radius =" << radius_inner_damping/r0*3.0 <<std::endl;
  // std::cout << "outer_damping_radius =" << radius_outer_damping/r0*3.0 <<std::endl;
  // std::cout << "theta_upper_damping = " << theta_upper_damping << std::endl;
  // std::cout << "theta_lower_damping = " << theta_lower_damping << std::endl;
  // std::cout << "M_dot_g = " << M_dot_g << "M_sun/yr" << std::endl;
  // std::cout << "alpha_vis = " << alpha_vis << std::endl;

  return;
}

// enroll user defined output variables
void MeshBlock::InitUserMeshBlockData(ParameterInput *pin){
  // record the restart time
  t_restart = pmy_mesh->time;

  if(true){
    // copy data from mesh to meshblock
    // restart procedure
    int dk     = NGHOST;
    int dj     = NGHOST;
    if (block_size.nx3 == 1) dk = 0;
    if (block_size.nx2 == 1) dj = 0;
    int kl = ks - dk;     int ku = ke + dk;
    int jl = js - dj;     int ju = je + dj;
    int il = is - NGHOST; int iu = ie + NGHOST;

    for (int k=kl; k<=ku; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu; i++) {
          int ti = static_cast<int>(loc.lx1)*block_size.nx1+(i-il);
          int tj = static_cast<int>(loc.lx2)*block_size.nx2+(j-jl);
          int tk = static_cast<int>(loc.lx3)*block_size.nx3+(k-ks);

          phydro->Tem(k, j, i) = pmy_mesh->ruser_mesh_data[4](tk,tj,ti);
        }
      }
    }
  }

  AllocateUserOutputVariables(14);
  SetUserOutputVariableName(0,"Tem");
  SetUserOutputVariableName(1,"st");
  SetUserOutputVariableName(2,"m_p");
  SetUserOutputVariableName(3,"s_p");
  SetUserOutputVariableName(4,"q_latent");
  SetUserOutputVariableName(5,"q_z");
  SetUserOutputVariableName(6,"flx_ice_x1");
  SetUserOutputVariableName(7,"flx_ice_x2");
  SetUserOutputVariableName(8,"flx_vap_x1");
  SetUserOutputVariableName(9,"flx_vap_x2");
  SetUserOutputVariableName(10,"q_diff");
  SetUserOutputVariableName(11,"flx_x1");
  SetUserOutputVariableName(12,"flx_x2");
  SetUserOutputVariableName(13,"dif");
  
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Initializes Keplerian accretion disk.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real rad(0.0), phi(0.0), z(0.0); // cylindrical coordinate
  Real x1, x2, x3; // sphercial coordinate
	Real igm1 = 1.0/(gamma_gas - 1.0);
  OrbitalVelocityFunc &vK = porb->OrbitalVelocity;

  int dk     = NGHOST;
  int dj     = NGHOST;
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
        GetCylCoord(pcoord, rad, phi, z, i, j, k); // convert to cylindrical coordinates
        Real vel_K     = vK(porb, x1, x2, x3);
        Real Tem = TemProfile(rad,phi,z);

        // compute initial conditions in cylindrical coordinates
        Real den_gas = DenProfileCyl_gas_fv_T(rad, phi, z, 0.0, Tem);
        Real v_acc_gas = Vr_ProfileCyl_gas_fv_T(rad, phi, z, 0.0, Tem);
        Real vel_gas_phi = VelProfileCyl_gas_fv_T(rad, phi, z, 0.0, Tem);
        if (porb->orbital_advection_defined)
          vel_gas_phi -= vel_K;

        Real &gas_dens = phydro->u(IDN, k, j, i);
        Real &gas_mom1 = phydro->u(IM1, k, j, i);
        Real &gas_mom2 = phydro->u(IM2, k, j, i);
        Real &gas_mom3 = phydro->u(IM3, k, j, i);

        gas_dens = den_gas;
        gas_mom1 = den_gas*v_acc_gas*std::sin(x2);
        gas_mom2 = den_gas*v_acc_gas*std::cos(x2);
        gas_mom3 = den_gas*vel_gas_phi;

        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN, k, j, i)  = gas_dens*Tem/(KELVIN*mu_xy)*igm1;
          phydro->u(IEN, k, j, i) += 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3))/gas_dens;
          phydro->Tem(k, j, i) = Tem;
        }

        // Step 1: Store all dustfluid volume density to calculate stopping time.
        AthenaArray<Real> rho_dustfluid_array;
        rho_dustfluid_array.NewAthenaArray(NDUSTFLUIDS);
        Real omega_dyn = std::sqrt(gm0/std::pow(rad,3.0));
        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            // compute initial conditions in cylindrical coordinates
            Real den_dust = DenProfileCyl_dust(rad, phi, z, initial_D2G[dust_id], Hratio[dust_id]);
            Real vel_dust_phi = VelProfileCyl_dust(rad, phi, z);
            if (porb->orbital_advection_defined)
              vel_dust_phi -= vel_K;
            rho_dustfluid_array(dust_id) = den_dust;

            pdustfluids->df_u(rho_id, k, j, i) = den_dust;
            if(dust_id == vapor_id){
              pdustfluids->df_u(v1_id,  k, j, i) = den_dust*v_acc_gas*std::sin(x2);
              pdustfluids->df_u(v2_id,  k, j, i) = den_dust*v_acc_gas*std::cos(x2);
              pdustfluids->df_u(v3_id,  k, j, i) = den_dust*vel_dust_phi;
            }else{
              pdustfluids->df_u(v1_id,  k, j, i) = 0.0;
              pdustfluids->df_u(v2_id,  k, j, i) = 0.0;
              pdustfluids->df_u(v3_id,  k, j, i) = den_dust*vel_dust_phi;
            }
					}
        }

        // (Yu, 2025-11-18) Initialize rho_Np_array from initial refractory densities
        // Step 2: Initialize rho_Np_array after dust densities are set
        if (pphase_change != nullptr && N_Z > 0) {
          for (int p = 0; p < N_P; ++p) {
            int refrac_id = N_Z * p + 1; // refractory composition (z=1)
            int rho_id = 4*refrac_id;
            Real rho_sil_p = pdustfluids->df_u(rho_id, k, j, i); // [code_density]
            // Calculate initial rho_Np from refractory density (all in code units)
            Real &rho_Np = pphase_change->rho_Np_array(p, k, j, i);
            rho_Np = rho_sil_p/(1.0 - f_ICE_inter0)/pphase_change->m_p0_array(p); // [code_number_density]
          }
        }

        // Step 3: Calculate gas viscosity, stopping time and diffusivity
        // (Yu, 2025-11-16) Calculate stopping time per pebble following new pattern
        Real &gas_nu = phydro->hdif.nu(HydroDiffusion::DiffProcess::alpha, k, j, i);
        gas_nu = alpha_vis* std::pow(rad/r0, nu_slope);
        
        if (pphase_change != nullptr) {
          // vapor diffusivity (with artificial decay for outer boundary)
          Real w_damp = 0.05*(x1max-x1min);
          Real f_decay_art = std::tanh(std::pow((rad-x1max)/w_damp ,2.0)); // outer bc decay
          Real &vapor_diffusivity = pdustfluids->nu_dustfluids_array(vapor_id, k, j, i);
          vapor_diffusivity = gas_nu * f_decay_art;

          Real &st_time_vapor = pdustfluids->stopping_time_array(vapor_id, k, j, i);
          st_time_vapor = Stokes_number[vapor_id]/omega_dyn;

          // Loop over pebbles to calculate stopping time per pebble
          AthenaArray<Real> rho_dustfluid_array_pebble;
          for (int p = 0; p < N_P; ++p) {
            rho_dustfluid_array_pebble.NewAthenaArray(N_Z);

            // collect dustfluid of the same pebble
            for (int z = 0; z < N_Z; ++z) {
              int dust_id = N_Z * p + z;
              int rho_id = 4*dust_id;
              rho_dustfluid_array_pebble(z) = pdustfluids->df_u(rho_id, k, j, i);
            }

            // calculate stopping time for the pebble
            // (Yu, 2025-11-18) Use rho_Np_array instead of m_p_array/s_p_array
            Real rho_Np = pphase_change->rho_Np_array(p, k, j, i);
            // (Yu, 2025-11-16) Pass vapor density separately since it's not in per-pebble array
            Real rho_v = pdustfluids->df_u(4*vapor_id, k, j, i);
            Real t_stop = pphase_change->Get_stopping_time(pmy_mesh->punit, rho_dustfluid_array_pebble, Tem, gas_dens, rho_v, rho_Np);
            // ===== upper limit, St can be extremely high for upper layer =====
            t_stop = (t_stop > 0.5) ? 0.5 : t_stop;
            if(std::fabs(z) > 2.0 *std::pow((rad/r0), 1.5 + qvalue/2)){
              t_stop = 1.e-4;
            }

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
        }  // end if (pphase_change != nullptr)
      }
    }
  }

  int istart = is; int iend = ie; int NGHOST_ti = NGHOST;
  if (pbval->block_bcs[BoundaryFace::inner_x1] == BoundaryFlag::user){
    istart = is-NGHOST;
    NGHOST_ti = 0;
    // std::cout << "in" << std::endl;  
  }else if (pbval->block_bcs[BoundaryFace::outer_x1] == BoundaryFlag::user){
    iend = ie+NGHOST;
  }
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=istart; i<=iend; ++i) {
        int ti = static_cast<int>(loc.lx1)*block_size.nx1+(i-istart) + NGHOST_ti ;
        int tj = static_cast<int>(loc.lx2)*block_size.nx2+(j-js)+ NGHOST;
        int tk = static_cast<int>(loc.lx3)*block_size.nx3+(k-ks);
        
        Real rad, phi, z;
        GetCylCoord(pcoord, rad, phi, z, i, j, k);
        pmy_mesh->ruser_mesh_data[4](tk,tj,ti) = TemProfile(rad,phi,z);
      }
    }
  }

  return;
}

void Mesh::UserWorkInLoop() {
  // some constants
  Real Mdot_gas = -M_dot_g* Constants::solar_mass_cgs/ Constants::yr_cgs / 2.0; // in cgs
  Mdot_gas /= (punit->code_mass_cgs /punit->code_time_cgs);

  #ifdef NDUSTFLUIDS
    AthenaArray<Real> &tau_vi = ruser_mesh_data[0];
    AthenaArray<Real> &tau_eff = ruser_mesh_data[1];
    AthenaArray<Real> &q_z = ruser_mesh_data[2];
    AthenaArray<Real> &q_int = ruser_mesh_data[3];
    AthenaArray<Real> &Tem_active = ruser_mesh_data[4];
    AthenaArray<Real> &int_1 = ruser_mesh_data[5];
    AthenaArray<Real> &int_2 = ruser_mesh_data[6];
    AthenaArray<Real> &rho_z = ruser_mesh_data[7];
  #endif

  AthenaArray<Real> tau_vi_f, q_int_f, tau_eff_f;
  tau_vi_f.NewAthenaArray(mesh_size.nx3, mesh_size.nx2+NGHOST+1, mesh_size.nx1 + 2*NGHOST);
  q_int_f.NewAthenaArray(mesh_size.nx3, mesh_size.nx2+NGHOST+1, mesh_size.nx1 + 2*NGHOST);
  tau_eff_f.NewAthenaArray(mesh_size.nx3, mesh_size.nx2+NGHOST+1, mesh_size.nx1 + 2*NGHOST);
  AthenaArray<Real> rhokappa;
  rhokappa.NewAthenaArray(mesh_size.nx3, mesh_size.nx2+NGHOST, mesh_size.nx1 + 2*NGHOST);

  /////////////////////////////////////////////////////////////////////////
  if(Allow_T_change_Flag){
    // step 0, calculate tau_vi
    for (int bn=0; bn<nblocal; ++bn) {
      MeshBlock *pmb = my_blocks(bn);
      LogicalLocation &loc = pmb->loc;
      if (loc.level == root_level) { // root level
        int istart = pmb->is; int iend = pmb->ie; int NGHOST_ti = NGHOST;
        if (pmb->pbval->block_bcs[BoundaryFace::inner_x1] == BoundaryFlag::user){
          istart = pmb->is-NGHOST;
          NGHOST_ti = 0;
        }else if (pmb->pbval->block_bcs[BoundaryFace::outer_x1] == BoundaryFlag::user){
          iend = pmb->ie+NGHOST;
        }
        for (int k=pmb->ks; k<=pmb->ke; k++) {
          for (int j=pmb->js; j<=pmb->je; j++) {
            for (int i=istart; i<=iend; i++) {            
              int ti = static_cast<int>(loc.lx1)*pmb->block_size.nx1+(i-istart)+ NGHOST_ti;
              int tj = static_cast<int>(loc.lx2)*pmb->block_size.nx2+(j-pmb->js)+ NGHOST;
              int tk = static_cast<int>(loc.lx3)*pmb->block_size.nx3+(k-pmb->ks);

              Real dx2 = pmb->pcoord->dx2f(j)* pmb->pcoord->x1v(i);
              Real rho_gas = pmb->phydro->w(IDN,k,j,i);
              Real rho_peb = pmb->pdustfluids->df_w(0,k,j,i) + pmb->pdustfluids->df_w(4,k,j,i);
              Real fv = pmb->pdustfluids->df_w(4*(vapor_id),k,j,i) / rho_gas;
              Real kappa = Get_kappa(rho_peb/rho_gas, fv);

              // store rhokappa
              rhokappa(tk,tj,ti) = rho_gas*kappa;
              rhokappa(tk,0,ti) = 0.0;
              rhokappa(tk,1,ti) = 0.0;
              // direct summation
              tau_vi_f(tk,tj+1,ti) = rhokappa(tk,tj,ti)*dx2 + tau_vi_f(tk,tj,ti);
              tau_vi(tk,tj,ti) = 0.5*(tau_vi_f(tk,tj,ti) + tau_vi_f(tk,tj+1,ti));
              // Simpson's 1/3 rule:
              // tau_vi(tk,tj,ti) = tau_vi(tk,tj-2,ti) + dx2/3.0*(rhokappa(tk,tj-2,ti) + 4.0*rhokappa(tk,tj-1,ti) + rhokappa(tk,tj,ti));
              // Trapezoidal rule:
              // tau_vi(tk,tj,ti) = 0.5*dx2*(rhokappa(tk,tj-1,ti) + rhokappa(tk,tj,ti)) + tau_vi(tk,tj-1,ti);
            }
          }
        }
      } else {
        std::stringstream msg;
        msg << "### FATAL ERROR in disk_snowline_2D_RT.cpp ProblemGenerator"   << std::endl
            << "This problem prohibits level > 0 currently"
            << "with error_output=true."  << std::endl;
          ATHENA_ERROR(msg);
      }
    }
  
    // do some iteration here:
    int N_iter = 0;
    int N_iter_max = 1;

    while(N_iter < N_iter_max){
      // step 1, calculate q_diff
      N_iter += 1;

      //step 2: calc q_int
      Real UNIT_ERG = punit->code_energydensity_cgs * punit->code_volume_cgs;
      Real UNIT_FLX = punit->code_energydensity_cgs * punit->code_velocity_cgs;
      
      for (int bn=0; bn<nblocal; ++bn) {
        MeshBlock *pmb = my_blocks(bn);
        LogicalLocation &loc = pmb->loc;
        if (loc.level == root_level) { // root level
          int istart = pmb->is; int iend = pmb->ie; int NGHOST_ti = NGHOST;
          if (pmb->pbval->block_bcs[BoundaryFace::inner_x1] == BoundaryFlag::user){
            istart = pmb->is-NGHOST;
            NGHOST_ti = 0;
          }else if (pmb->pbval->block_bcs[BoundaryFace::outer_x1] == BoundaryFlag::user){
            iend = pmb->ie+NGHOST;
          }
          for (int k=pmb->ks; k<=pmb->ke; k++) {
            for (int j=pmb->js; j<=pmb->je; j++) {
              for (int i=istart; i<=iend; i++) {
                int ti = static_cast<int>(loc.lx1)*pmb->block_size.nx1+(i-istart)+ NGHOST_ti;
                int tj = static_cast<int>(loc.lx2)*pmb->block_size.nx2+(j-pmb->js)+ NGHOST;
                int tk = static_cast<int>(loc.lx3)*pmb->block_size.nx3+(k-pmb->ks);

                Real rad, phi, z;
                GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
                Real dx2 = pmb->pcoord->dx2f(j)* pmb->pcoord->x1v(i);
                Real rho_gas = pmb->phydro->w(IDN,k,j,i);
                Real rho_peb = pmb->pdustfluids->df_w(0,k,j,i) + pmb->pdustfluids->df_w(4,k,j,i);
                Real fv = pmb->pdustfluids->df_w(4*(vapor_id),k,j,i) / rho_gas;             
                Real H_gas = std::pow(rad/r0, qvalue/2.0 + 1.5);
                Real cos_inc = (qvalue/2.0 + 0.5) * (4.0*H_gas/rad); // cosine of the incident angle to disk surface

                // accquire tau_(-inf):
                // Real tau_inf = 2.0 * tau_vi(tk, mesh_size.nx2+NGHOST-1, ti);
                Real tau_inf = 2.0 * tau_vi_f(tk, mesh_size.nx2+NGHOST, ti);
                Real E0 = L_star*Constants::solar_lum_cgs / (UNIT_ERG/punit->code_time_cgs) / (8.0*PI*SQR(rad));
                Real omega = std::sqrt(gm0/std::pow(rad,3.0));
                Real q_irr = E0* f_vi*rhokappa(tk,tj,ti)*(std::exp(-f_vi*tau_vi(tk,tj,ti)/cos_inc) + std::exp(-f_vi*(tau_inf - tau_vi(tk,tj,ti))/cos_inc));
                Real nu_gas = pmb->phydro->hdif.nu(HydroDiffusion::DiffProcess::alpha, k, j, i);
                Real q_vis = 9.0/4.0 *rho_gas* (1.0-fv)* SQR(omega) *nu_gas; // only include viscous heating by H-He
                ////////////////////
                // including latent heat
                // (Yu, 2025-11-16) Updated to use PhaseChange module arrays
                // q_z(tk,tj,ti) = q_irr + q_vis + pmb->pphase_change->q_latent(k,j,i) + pmb->pphase_change->q_diff(k,j,i);
                q_z(tk,tj,ti) = q_irr + q_vis;
                // q_z(tk,tj,ti) = q_irr + q_vis + pmb->pphase_change->q_diff(k,j,i);
                // linear interpolation
                q_int_f(tk,tj+1,ti) = q_z(tk,tj,ti)*dx2 + q_int_f(tk,tj,ti);
                q_int(tk,tj,ti) = 0.5*(q_int_f(tk,tj+1,ti) + q_int_f(tk,tj,ti));
                // Simpson's 1/3 rule:
                // q_int(tk,tj,ti) = q_int(tk,tj-2,ti) +  dx2/3.0*(q_z(tk,tj-2,ti) + 4.0*q_z(tk,tj-1,ti) + q_z(tk,tj,ti));
                // Trapezoidal rule:
                // q_int(tk,tj,ti) = 0.5*dx2*(q_z(tk,tj-1,ti) + q_z(tk,tj,ti)) + q_int(tk,tj-1,ti);
              }
            }
          }
        } else {
          std::stringstream msg;
          msg << "### FATAL ERROR in disk_snowline_2D_RT.cpp ProblemGenerator"   << std::endl
              << "This problem prohibits level > 0 currently"
              << "with error_output=true."  << std::endl;
            ATHENA_ERROR(msg);
        }
      }

      // step 3: calc tau_eff
      for (int bn=0; bn<nblocal; ++bn) {
        MeshBlock *pmb = my_blocks(bn);
        LogicalLocation &loc = pmb->loc;
        if (loc.level == root_level) { // root level
          int istart = pmb->is; int iend = pmb->ie; int NGHOST_ti = NGHOST;
          if (pmb->pbval->block_bcs[BoundaryFace::inner_x1] == BoundaryFlag::user){
            istart = pmb->is-NGHOST;
            NGHOST_ti = 0;
          }else if (pmb->pbval->block_bcs[BoundaryFace::outer_x1] == BoundaryFlag::user){
            iend = pmb->ie+NGHOST;
          }
          for (int k=pmb->ks; k<=pmb->ke; k++) {
            for (int j=pmb->js; j<=pmb->je; j++) {
              for (int i=istart; i<=iend; i++) {
                int ti = static_cast<int>(loc.lx1)*pmb->block_size.nx1+(i-istart)+ NGHOST_ti;
                int tj = static_cast<int>(loc.lx2)*pmb->block_size.nx2+(j-pmb->js)+ NGHOST;
                int tk = static_cast<int>(loc.lx3)*pmb->block_size.nx3+(k-pmb->ks);

                Real dx2 = pmb->pcoord->dx2f(j)* pmb->pcoord->x1v(i);
                Real F_inf = q_int_f(tk, mesh_size.nx2+NGHOST, ti);
                
                // linear interpolation
                tau_eff_f(tk,tj+1,ti) = rhokappa(tk,tj,ti)*(1.0 - q_int(tk,tj,ti)/F_inf) * dx2 + tau_eff_f(tk,tj,ti);
                tau_eff(tk,tj,ti) = 0.5*(tau_eff_f(tk,tj+1,ti) + tau_eff_f(tk,tj,ti));
                // Simpson's 1/3 rule:
                // tau_eff(tk,tj,ti) = tau_eff(tk,tj-2,ti) + dx2/3.0*(rhokappa(tk,tj-2,ti)*(1.0 - q_int(tk,tj-2,ti)/F_inf) + 4.0*rhokappa(tk,tj-1,ti)*(1.0 - q_int(tk,tj-1,ti)/F_inf) + rhokappa(tk,tj,ti)*(1.0 - q_int(tk,tj,ti)/F_inf));
                // Trapezoidal rule:
                // tau_eff(tk,tj,ti) = dx2/2.0* (rhokappa(tk,tj,ti)*(1.0 - q_int(tk,tj,ti)/F_inf) + rhokappa(tk,tj-1,ti)*(1.0 - q_int(tk,tj-1,ti)/F_inf)) + tau_eff(tk,tj-1,ti);
              }
            }
          }
        } else {
          std::stringstream msg;
          msg << "### FATAL ERROR in disk_snowline_2D_RT.cpp ProblemGenerator"   << std::endl
              << "This problem prohibits level > 0 currently"
              << "with error_output=true."  << std::endl;
            ATHENA_ERROR(msg);
        }
      }

      // step 4: Get temperature
      for (int bn=0; bn<nblocal; ++bn) {
        MeshBlock *pmb = my_blocks(bn);
        LogicalLocation &loc = pmb->loc;
        if (loc.level == root_level) { // root level
          int istart = pmb->is; int iend = pmb->ie; int NGHOST_ti = NGHOST;
          if (pmb->pbval->block_bcs[BoundaryFace::inner_x1] == BoundaryFlag::user){
            istart = pmb->is-NGHOST;
            NGHOST_ti = 0;
          }else if (pmb->pbval->block_bcs[BoundaryFace::outer_x1] == BoundaryFlag::user){
            iend = pmb->ie+NGHOST;
          }
          for (int k=pmb->ks; k<=pmb->ke; k++) {
            for (int j=pmb->js; j<=pmb->je; j++) {
              for (int i=istart; i<=iend; i++) {
                int ti = static_cast<int>(loc.lx1)*pmb->block_size.nx1+(i-istart)+ NGHOST_ti;
                int tj = static_cast<int>(loc.lx2)*pmb->block_size.nx2+(j-pmb->js)+ NGHOST;
                int tk = static_cast<int>(loc.lx3)*pmb->block_size.nx3+(k-pmb->ks);

                Real rad(0.0), phi(0.0), z(0.0); // cylindrical coordinate
                GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
                
                Real rho_gas = pmb->phydro->w(IDN,k,j,i);
                Real rho_peb = pmb->pdustfluids->df_w(0,k,j,i) + pmb->pdustfluids->df_w(4,k,j,i);
                Real fv = pmb->pdustfluids->df_w(4*(vapor_id),k,j,i) / rho_gas;
                Real kappa = Get_kappa(rho_peb/rho_gas, fv);
                Real F_inf = q_int_f(tk, mesh_size.nx2+NGHOST, ti);
                Real T_eff = std::pow(F_inf/(Constants::sigma_cgs/UNIT_FLX), 0.25);
                
                Real &Tem = pmb->phydro->Tem(k, j, i);
                Real Tem0 = Tem;
                Tem = 0.75*tau_eff(tk,tj,ti) + std::sqrt(3.0)/4.0 +  q_z(tk,tj,ti)/(4.0*rhokappa(tk,tj,ti)*F_inf);
                Tem = std::pow(Tem, 0.25)*T_eff;

                if(std::isnan(Tem)){
                  std::cout << "Tem = " << Tem << std::endl;
                  std::cout << "F_inf = " << F_inf << std::endl;
                  std::cout << "ti, tj = " << ti << " " << tj << std::endl;
                  std::cout << "tau_eff = " << tau_eff(tk,tj,ti) << std::endl;
                  std::cout << "q_z = " << q_z(tk,tj,ti) << std::endl;
                  quick_exit(1);
                }

                // beta-cooling //////////////////////
                Real omega_dyn = std::sqrt(gm0/(SQR(rad)*rad));
                Real t_cool = beta / omega_dyn;
                if (time < t_iterate){
                  // when do iteration for T, rho, don't need long relaxing timescale
                  t_cool = 50.0/ omega_dyn;
                }
                Real dT = (Tem - Tem0)/t_cool*dt;
                Tem = Tem0 + dT;
                //////////////////////////////////////

                // store the whole temperature array.  
                Tem_active(tk,tj,ti) = Tem;
              }
            }
          }
        } else {
          std::stringstream msg;
          msg << "### FATAL ERROR in disk_snowline_2D_RT.cpp ProblemGenerator"   << std::endl
              << "This problem prohibits level > 0 currently"
              << "with error_output=true."  << std::endl;
            ATHENA_ERROR(msg);
        }
      }
    }
  }

  // reset q_latent, q_diff;
  for (int bn=0; bn<nblocal; ++bn) {
    MeshBlock *pmb = my_blocks(bn);
    LogicalLocation &loc = pmb->loc;
    if (loc.level == root_level) { // root level
      for (int k=pmb->ks; k<=pmb->ke; k++) {
        for (int j=pmb->js; j<=pmb->je; j++) {
          for (int i=pmb->is; i<=pmb->ie; i++) {
            // (Yu, 2025-11-16) Updated to use PhaseChange module arrays
            // reset q_latent and copy its value to dfv_dt:
            pmb->user_out_var(4,k,j,i) = pmb->pphase_change->q_latent(k,j,i);
            pmb->user_out_var(10,k,j,i) = pmb->pphase_change->q_diff(k,j,i);
            pmb->pphase_change->q_latent(k,j,i) = 0.0;
            pmb->pphase_change->q_diff(k,j,i) = 0.0;
          }
        }
      }
    }
  }
  ////////////////////////////////////////////////////////////////

  // step 5: integrate over (mu m_p g)/(k_B/T) (z = infty -> 0)
  for (int bn=0; bn<nblocal; ++bn) {
    MeshBlock *pmb = my_blocks(bn);
    LogicalLocation &loc = pmb->loc;
    if (loc.level == root_level) { // root level
      int istart = pmb->is; int iend = pmb->ie; int NGHOST_ti = NGHOST;
      if (pmb->pbval->block_bcs[BoundaryFace::inner_x1] == BoundaryFlag::user){
        istart = pmb->is-NGHOST;
        NGHOST_ti = 0;
      }else if (pmb->pbval->block_bcs[BoundaryFace::outer_x1] == BoundaryFlag::user){
        iend = pmb->ie+NGHOST;
      }
      for (int k=pmb->ks; k<=pmb->ke; k++) {
        for (int j=pmb->js; j<=pmb->je; j++) {
          for (int i=istart; i<=iend; i++) {
            int ti = static_cast<int>(loc.lx1)*pmb->block_size.nx1+(i-istart)+ NGHOST_ti;
            int tj = static_cast<int>(loc.lx2)*pmb->block_size.nx2+(j-pmb->js)+ NGHOST;
            int tk = static_cast<int>(loc.lx3)*pmb->block_size.nx3+(k-pmb->ks);
            
            Real rad, phi, z;
            GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
            Real dx2 = pmb->pcoord->dx2f(j)* pmb->pcoord->x1v(i);
            Real grav_z = -gm0*z/std::pow(rad,3.0);
            Real fv = pmb->pdustfluids->df_w(4*(vapor_id),k,j,i) / pmb->phydro->w(IDN,k,j,i);
            Real mu = Get_mu(fv);

            int_1(tk,tj,ti) = mu*KELVIN*grav_z/Tem_active(tk,tj,ti) *dx2 + int_1(tk,tj-1,ti);
          }
        }
      }
    } else {
      std::stringstream msg;
      msg << "### FATAL ERROR in disk_snowline_2D_RT.cpp ProblemGenerator"   << std::endl
          << "This problem prohibits level > 0 currently"
          << "with error_output=true."  << std::endl;
        ATHENA_ERROR(msg);
    }
  }

  // step 6: integrate over T_mid/T(z) * exp(int_2)
  for (int bn=0; bn<nblocal; ++bn) {
    MeshBlock *pmb = my_blocks(bn);
    LogicalLocation &loc = pmb->loc;
    if (loc.level == root_level) { // root level
      int istart = pmb->is; int iend = pmb->ie; int NGHOST_ti = NGHOST;
      if (pmb->pbval->block_bcs[BoundaryFace::inner_x1] == BoundaryFlag::user){
        istart = pmb->is-NGHOST;
        NGHOST_ti = 0;
      }else if (pmb->pbval->block_bcs[BoundaryFace::outer_x1] == BoundaryFlag::user){
        iend = pmb->ie+NGHOST;
      }
      for (int k=pmb->ks; k<=pmb->ke; k++) {
        for (int j=pmb->js; j<=pmb->je; j++) {
          for (int i=istart; i<=iend; i++) {
            int ti = static_cast<int>(loc.lx1)*pmb->block_size.nx1+(i-istart)+ NGHOST_ti;
            int tj = static_cast<int>(loc.lx2)*pmb->block_size.nx2+(j-pmb->js)+ NGHOST;
            int tk = static_cast<int>(loc.lx3)*pmb->block_size.nx3+(k-pmb->ks);
            
            Real Tem_mid = Tem_active(tk,mesh_size.nx2+NGHOST-1,ti);
            Real int_1_mid = int_1(tk,mesh_size.nx2+NGHOST-1,ti);
            Real dx2 = pmb->pcoord->dx2f(j)* pmb->pcoord->x1v(i);

            int_2(tk,tj,ti) = Tem_mid/Tem_active(tk,tj,ti) * std::exp(int_1_mid - int_1(tk,tj,ti))*dx2 + int_2(tk,tj-1,ti);
          }
        }
      }
    } else {
      std::stringstream msg;
      msg << "### FATAL ERROR in disk_snowline_2D_RT.cpp ProblemGenerator"   << std::endl
          << "This problem prohibits level > 0 currently"
          << "with error_output=true."  << std::endl;
        ATHENA_ERROR(msg);
    }
  }

  // step 7: assign rho_z

  for (int bn=0; bn<nblocal; ++bn) {
    MeshBlock *pmb = my_blocks(bn);
    LogicalLocation &loc = pmb->loc;
    if (loc.level == root_level) { // root level
      int istart = pmb->is; int iend = pmb->ie; int NGHOST_ti = NGHOST;
      if (pmb->pbval->block_bcs[BoundaryFace::inner_x1] == BoundaryFlag::user){
        istart = pmb->is-NGHOST;
        NGHOST_ti = 0;
      }else if (pmb->pbval->block_bcs[BoundaryFace::outer_x1] == BoundaryFlag::user){
        iend = pmb->ie+NGHOST;
      }
      for (int k=pmb->ks; k<=pmb->ke; k++) {
        for (int j=pmb->js; j<=pmb->je; j++) {
          for (int i=istart; i<=iend; i++) {
            int ti = static_cast<int>(loc.lx1)*pmb->block_size.nx1+(i-istart)+ NGHOST_ti;
            int tj = static_cast<int>(loc.lx2)*pmb->block_size.nx2+(j-pmb->js)+ NGHOST;
            int tk = static_cast<int>(loc.lx3)*pmb->block_size.nx3+(k-pmb->ks);

            Real rad, phi, z;
            GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
            Real x1 = pmb->pcoord->x1v(i);
            Real nu_gas = alpha_vis*std::pow(rad/r0, nu_slope);
            
            Real Tem_mid = Tem_active(tk,mesh_size.nx2+NGHOST-1,ti);
            Real int_1_mid = int_1(tk,mesh_size.nx2+NGHOST-1,ti);
            Real sigma = std::fabs(Mdot_gas)/(3.0*PI*nu_gas);
            Real rho_mid = sigma/ (int_2(tk,mesh_size.nx2+NGHOST-1,ti) - int_2(tk,NGHOST,ti));

            rho_z(tk,tj,ti) = rho_mid*Tem_mid/Tem_active(tk,tj,ti)* std::exp(int_1_mid - int_1(tk,tj,ti));
            rho_z(tk,tj,ti) = std::fmax(rho_z(tk,tj,ti), dfloor+dffloor);

            // inner bc
            if(time > t_iterate and pmb->pbval->block_bcs[BoundaryFace::inner_x1] == BoundaryFlag::user and x1 < x1min){
              Real x1_active = pmb->pcoord->x1v(pmb->is);
              Real x1_ghost = pmb->pcoord->x1v(i);
              rho_z(tk,tj,ti) = (pmb->phydro->w(IDN,k,j,pmb->is) -pmb->pdustfluids->df_w(4*(vapor_id),k,j,pmb->is)) * SQR(x1_active)/SQR(x1_ghost);
              // rho_z(tk,tj,ti) = std::exp(2.0*std::log(pmb->phydro->w(IDN, k, j, i+1)) - std::log(pmb->phydro->w(IDN, k, j, i+2)));
            }
            // outer bc
            // if(time > t_iterate and pmb->pbval->block_bcs[BoundaryFace::outer_x1] == BoundaryFlag::user and x1 > x1max){
            //   // Real x1_active = pmb->pcoord->x1v(pmb->ie);
            //   // Real x1_ghost = pmb->pcoord->x1v(i);
            //   // rho_z(tk,tj,ti) = pmb->phydro->w(IDN,k,j,pmb->ie) * SQR(x1_active)/SQR(x1_ghost);
            //   rho_z(tk,tj,ti) = std::exp(2.0*std::log(pmb->phydro->w(IDN, k, j, i-1)) - std::log(pmb->phydro->w(IDN, k, j, i-2)));
            // }
          }
        }
      }
    } else {
      std::stringstream msg;
      msg << "### FATAL ERROR in disk_snowline_2D_RT.cpp ProblemGenerator"   << std::endl
          << "This problem prohibits level > 0 currently"
          << "with error_output=true."  << std::endl;
        ATHENA_ERROR(msg);
    }
  }

  // Get the surface density
  AthenaArray<Real> sigma_gas_est;
  sigma_gas_est.NewAthenaArray(NGHOST);
  for (int bn=0; bn<nblocal; ++bn) {
    MeshBlock *pmb = my_blocks(bn);
    if (pmb->pbval->block_bcs[BoundaryFace::inner_x1] == BoundaryFlag::user){
      int tk = 0;

      for (int k=pmb->ks; k<=pmb->ke; k++) {
        for (int j=pmb->js; j<=pmb->je; j++) {
          for (int i = 0; i < pmb->is; i++){
            int ti = i;
            int tj = static_cast<int>(pmb->loc.lx2)*pmb->block_size.nx2+(j-pmb->js)+ NGHOST;
            Real dx2 = pmb->pcoord->dx2f(j)* pmb->pcoord->x1v(i);
            sigma_gas_est(i) += rho_z(tk,tj,ti)*dx2;
          }
        }
      }
    }
  }

  for (int bn=0; bn<nblocal; ++bn) { 
    MeshBlock *pmb = my_blocks(bn);
    if (pmb->pbval->block_bcs[BoundaryFace::inner_x1] == BoundaryFlag::user){
      int tk = 0;

      for (int k=pmb->ks; k<=pmb->ke; k++) {
        for (int j=pmb->js; j<=pmb->je; j++) {
          for (int i = 0; i < pmb->is; i++){
            int ti = i;
            int tj = static_cast<int>(pmb->loc.lx2)*pmb->block_size.nx2+(j-pmb->js)+ NGHOST;
            Real rad, phi, z;
            GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
            Real nu_gas = alpha_vis*std::pow(rad/r0, nu_slope);
            Real sigma0 = std::fabs(Mdot_gas)/(3.0*PI*nu_gas);
            rho_z(tk,tj,ti) *= sigma0/sigma_gas_est(i);
          }
        }
      }
    }
  }


  // Get the total mass flux;
  Real Mdot_est = 0.0;
  AthenaArray<Real> Mdot_peb_est;
  Mdot_peb_est.NewAthenaArray(NDUSTFLUIDS);
  // Initialize to zero (only pebble compositions will be set, vapor is ignored)
  for (int n=0; n<NDUSTFLUIDS; n++) {
    Mdot_peb_est(n) = 0.0;
  }
  Real Mdot_peb = f_ICE_inter0*p2g_flux*Mdot_gas;

  for (int bn=0; bn<nblocal; ++bn) {
    MeshBlock *pmb = my_blocks(bn);
    if (pmb->pbval->block_bcs[BoundaryFace::outer_x1] == BoundaryFlag::user){
      int tk = 0;
      int ti = pmb->pmy_mesh->mesh_size.nx1 + NGHOST;
      int tj_mid = pmb->pmy_mesh->mesh_size.nx2 + NGHOST-1;
      Real Tem_mid = pmb->pmy_mesh->ruser_mesh_data[4](tk,tj_mid,ti);
      Real Tem_mid_ghost = pmb->pmy_mesh->ruser_mesh_data[4](tk,tj_mid,ti+1);

      for (int k=pmb->ks; k<=pmb->ke; k++) {
        for (int j=pmb->js; j<=pmb->je; j++) {
          int tj = static_cast<int>(pmb->loc.lx2)*pmb->block_size.nx2+(j-pmb->js)+ NGHOST;
          Real x2 = pmb->pcoord->x2v(j);
          Real dx2 = pmb->pcoord->dx2v(j);
          Real x1_active = pmb->pcoord->x1v(pmb->ie);
          Real ds_ghost = 2.0*pmb->pcoord->GetFace1Area(k,j,pmb->ie+1);
          
          Real rad_active(0.0), phi_active(0.0), z_active(0.0);
          Real rad_ghost(0.0),  phi_ghost(0.0),  z_ghost(0.0);
          GetCylCoord(pmb->pcoord, rad_active, phi_active, z_active, pmb->ie, j, k);
          GetCylCoord(pmb->pcoord, rad_ghost, phi_ghost, z_ghost, pmb->ie+1, j, k);

          Real Tem_active = pmb->phydro->Tem(k, j, pmb->ie);
          Real Tem_ghost = pmb->phydro->Tem(k, j, pmb->ie+1);
          Real gas_rho0 = pmb->pmy_mesh->ruser_mesh_data[7](tk, tj, ti);
          Real vr0 = Vr_ProfileCyl_gas_fv_T(rad_ghost, phi_ghost, z_ghost, 0.0, Tem_mid_ghost);

          // gas flux
          pmb->phydro->inflx_x1(k,j,0) = gas_rho0*(vr0*std::sin(x2));
          Mdot_est += pmb->phydro->inflx_x1(k,j,0)*ds_ghost;

          // pebble flux:
          Real v_drift_peb = -0.004; // random drift velocity
          Real dust_vel1 = v_drift_peb*std::sin(x2);
          
          Real sigma_peb = 2.0*Mdot_peb /(v_drift_peb * 2.0*PI*rad_active);
          Real cs0 = std::sqrt(Tem_mid/(mu_xy*KELVIN));
          Real h_peb = Hratio[0]* cs0/std::sqrt(gm0/(rad_active*SQR(rad_active)));
          Real rho_peb_mid = sigma_peb / (sqrt(2.0*PI)*h_peb);
          Real dust_rho = rho_peb_mid* std::exp(-0.5*SQR(z_active/h_peb));
          dust_rho = (dust_rho > dffloor) ? dust_rho : dffloor;
          
          for (int p=0; p<N_P; p++) {
            for (int n=0; n<N_Z; n++) {
              int dust_id = N_Z * p + n;
              pmb->pdustfluids->inflx_dust_x1(dust_id,k,j,0) = dust_rho*dust_vel1;
              Mdot_peb_est(dust_id) += pmb->pdustfluids->inflx_dust_x1(dust_id,k,j,0)*ds_ghost;
            }
          }
        }
      }
    }
  }

  Real flx_ratio = Mdot_gas / Mdot_est;
  AthenaArray<Real> flx_peb_ratio;
  flx_peb_ratio.NewAthenaArray(NDUSTFLUIDS);
  // Calculate flux ratio for each pebble composition (vapor entries remain uninitialized but unused)
  for (int p=0; p<N_P; p++) {
    for (int n=0; n<N_Z; n++) {
    int dust_id = N_Z * p + n;
      flx_peb_ratio(dust_id) = Mdot_peb / Mdot_peb_est(dust_id);
    }
  }

  for (int bn=0; bn<nblocal; ++bn) {
    MeshBlock *pmb = my_blocks(bn);
    if (pmb->pbval->block_bcs[BoundaryFace::outer_x1] == BoundaryFlag::user){
      for (int k=pmb->ks; k<=pmb->ke; k++) {
        for (int j=pmb->js; j<=pmb->je; j++) {
          pmb->phydro->inflx_x1(k,j,0) *= flx_ratio;
          for (int p=0; p<N_P; p++) {
            for (int n=0; n<N_Z; n++) {
              int dust_id = N_Z * p + n;
              if(time < dust_start_injection){
                pmb->pdustfluids->inflx_dust_x1(dust_id,k,j,0) = 0.0;
              } else {
                pmb->pdustfluids->inflx_dust_x1(dust_id,k,j,0) *= flx_peb_ratio(dust_id);
              }
            }
          }
        }
      }
    }
  }
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
  if (Theta_Gas_Damping_Flag) {
    UpperWaveDampingGas(this, time, dt, il, iu, jl, ju, kl, ku, phydro->w, phydro->u);
    // LowerWaveDampingGas(this, time, dt, il, iu, jl, ju, kl, ku, phydro->w, phydro->u);
  }

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        // pre-calculation
        const Real &gas_den = phydro->w(IDN,k ,j ,i);
        const Real &gas_vel1 = phydro->w(IVX, k, j, i);
        const Real &gas_vel2 = phydro->w(IVY, k, j, i);
        const Real &gas_vel3 = phydro->w(IVZ, k, j, i);

        // copy gas velocity to tracer;
        int dust_id = vapor_id;
        int rho_id  = 4*dust_id;
        int v1_id   = rho_id + 1;
        int v2_id   = rho_id + 2;
        int v3_id   = rho_id + 3;
        
        Real &dust_rho  = pdustfluids->df_w(rho_id, k, j, i);
        Real &dust_vel1 = pdustfluids->df_w(v1_id,  k, j, i);
        Real &dust_vel2 = pdustfluids->df_w(v2_id,  k, j, i);
        Real &dust_vel3 = pdustfluids->df_w(v3_id,  k, j, i);

        Real &dust_den  = pdustfluids->df_u(rho_id, k, j, i);
        Real &dust_mom1 = pdustfluids->df_u(v1_id,  k, j, i);
        Real &dust_mom2 = pdustfluids->df_u(v2_id,  k, j, i);
        Real &dust_mom3 = pdustfluids->df_u(v3_id,  k, j, i);

        dust_vel1 = gas_vel1;
        dust_vel2 = gas_vel2;
        dust_vel3 = gas_vel3;
        dust_mom1 = dust_den*gas_vel1;
        dust_mom2 = dust_den*gas_vel2;
        dust_mom3 = dust_den*gas_vel3;

        // copy refractory velocity to ice
        // refractory
        dust_id = 1;
        rho_id  = 4*dust_id;
        v1_id   = rho_id + 1;
        v2_id   = rho_id + 2;
        v3_id   = rho_id + 3;
        
        const Real &d1_vel1 = pdustfluids->df_w(v1_id,  k, j, i);
        const Real &d1_vel2 = pdustfluids->df_w(v2_id,  k, j, i);
        const Real &d1_vel3 = pdustfluids->df_w(v3_id,  k, j, i);
        
        // ice
        dust_id = 0;
        rho_id  = 4*dust_id;
        v1_id   = rho_id + 1;
        v2_id   = rho_id + 2;
        v3_id   = rho_id + 3;
        
        Real &d0_vel1 = pdustfluids->df_w(v1_id,  k, j, i);
        Real &d0_vel2 = pdustfluids->df_w(v2_id,  k, j, i);
        Real &d0_vel3 = pdustfluids->df_w(v3_id,  k, j, i);

        const Real &d0_den  = pdustfluids->df_u(rho_id, k, j, i);
        Real &d0_mom1 = pdustfluids->df_u(v1_id,  k, j, i);
        Real &d0_mom2 = pdustfluids->df_u(v2_id,  k, j, i);
        Real &d0_mom3 = pdustfluids->df_u(v3_id,  k, j, i);

        d0_vel1 = d1_vel1;
        d0_vel2 = d1_vel2;
        d0_vel3 = d1_vel3;
        d0_mom1 = d1_vel1*d0_den;
        d0_mom2 = d1_vel2*d0_den;
        d0_mom3 = d1_vel3*d0_den;

        // this is to calculate the temperature for the ghost cells defining boundary conditions
        if(i < is or i > ie or j < js or j > je){
          Real E_kg = 0.5*(SQR(gas_vel1) + SQR(gas_vel2) + SQR(gas_vel3))*gas_den;
          Real rhoe_g = phydro->u(IEN,k,j,i) - E_kg;
          Real fv = pdustfluids->df_w(4*(vapor_id), k, j, i)/gas_den;
          Real T_from_erg = pphase_change->Get_T_rhoe_g(rhoe_g, gas_den,fv);
          
          phydro->Tem(k, j, i) = T_from_erg;          
        }

      }
    }
  }
}


void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin){

  int dk     = NGHOST;
  int dj     = NGHOST;
  if (block_size.nx3 == 1) dk = 0;
  if (block_size.nx2 == 1) dj = 0;
  int kl = ks - dk;     int ku = ke + dk;
  int jl = js - dj;     int ju = je + dj;
  int il = is - NGHOST; int iu = ie + NGHOST;

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        const Real &rho_g = phydro->w(IDN,k,j,i);
        const Real &press = phydro->w(IPR,k,j,i);
        const Real &rho_v = pdustfluids->df_w(4*(vapor_id),k,j,i);
        Real E_kg = 0.5*(SQR(phydro->u(IM1,k,j,i)) + SQR(phydro->u(IM2,k,j,i)) + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
        Real rhoe_g = phydro->u(IEN,k,j,i) - E_kg;
        Real fv;

        fv = rho_v/rho_g;
        Real T;
        if (pphase_change != nullptr) {
          T = pphase_change->Get_T_rhoe_g(rhoe_g, rho_g, fv);
        } else {
          // Fallback if phase change module not available (shouldn't happen if N_Z > 0)
          Real fx = 1.0 - fv;
          Real e = rhoe_g/rho_g;
          T = e*KELVIN/(fx/mu_H2*0.71*2.5 + fx/mu_He*0.29*1.5 + fv/mu_z*3.0);
        }
        Real mu = Get_mu(fv);
        Real prs = rho_g*T/(mu*KELVIN);
        // output
        user_out_var(0,k,j,i) = phydro->Tem(k,j,i);
        // user_out_var(1,k,j,i) = prs/rhoe_g + 1.0;
        user_out_var(1,k,j,i) = pdustfluids->stopping_time_array(0,k,j,i);
        // user_out_var(1,k,j,i) = phydro->hdif.kappa(HydroDiffusion::DiffProcess::aniso, k, j, i);
        if (pphase_change != nullptr && N_P > 0) {
          Real rho_Np = pphase_change->rho_Np_array(0, k, j, i); // [code_number_density]
          Real rho_I = pdustfluids->df_w(4*0, k, j, i); // ice for pebble 0 [code_density]
          Real rho_sil = pdustfluids->df_w(4*1, k, j, i); // refractory for pebble 0 [code_density]
          Real m_p = pphase_change->Get_m_p_from_rho_Np(rho_I, rho_sil, rho_Np); // [code_mass]
          Real s_p = pphase_change->Get_s_p_from_m_p(m_p, rho_I, rho_sil); // [code_length]
          Units *punit = pmy_mesh->punit;
          user_out_var(3,k,j,i) = s_p * punit->code_length_cgs; // Convert to CGS for output
        }
        // user_out_var(4,k,j,i) = pdustfluids->dfv_dt(k,j,i);
        
        user_out_var(6,k,j,i) = pdustfluids->df_flux[X1DIR](0,k,j,i);
        user_out_var(7,k,j,i) = pdustfluids->df_flux[X2DIR](0,k,j,i);
        user_out_var(8,k,j,i) = pdustfluids->df_flux[X1DIR](4*(vapor_id),k,j,i);
        user_out_var(9,k,j,i) = pdustfluids->df_flux[X2DIR](4*(vapor_id),k,j,i);
        user_out_var(11,k,j,i) = phydro->flux[X1DIR](IDN,k,j,i);
        user_out_var(12,k,j,i) = phydro->flux[X2DIR](IDN,k,j,i);
        user_out_var(13,k,j,i) = pdustfluids->nu_dustfluids_array(vapor_id,k,j,i);
      }
    }
  }

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        int ti = static_cast<int>(loc.lx1)*block_size.nx1+(i-il);
        int tj = static_cast<int>(loc.lx2)*block_size.nx2+(j-jl);
        int tk = static_cast<int>(loc.lx3)*block_size.nx3+(k-kl);

        // radiative heating rate, q_z
        user_out_var(2,k,j,i) = pmy_mesh->ruser_mesh_data[1](tk, tj, ti);
        user_out_var(5,k,j,i) = pmy_mesh->ruser_mesh_data[2](tk, tj, ti);
      }
    }
  }

  return;
}

namespace {
//----------------------------------------------------------------------------------------
//! transform to cylindrical coordinate

Real Get_mu(Real fv){
  return 1./((1.-fv)/mu_xy + fv / mu_z);
}

void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s){
  
  RadiativeCondution(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);
  LocalIsothermalEOS(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);
  if(N_Z > 0 and time > t_iterate){  // (Yu, 2025-11-16)
    pmb->pphase_change->PhaseChangeSource(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);
  }

  return;
}

void LocalIsothermalEOS(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s){

    OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      Real x3 = pmb->pcoord->x3v(k);
      for (int j=pmb->js; j<=pmb->je; ++j) {
        Real x2 = pmb->pcoord->x2v(j);
#pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          Real x1 = pmb->pcoord->x1v(i);
          Real rad, phi, z;
          GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
          Real vel_K     = vK(pmb->porb, x1, x2, x3);

          Real &gas_dens = cons(IDN, k, j, i);
          Real &gas_mom1 = cons(IM1, k, j, i);
          Real &gas_mom2 = cons(IM2, k, j, i);
          Real &gas_mom3 = cons(IM3, k, j, i);
          Real &gas_erg  = cons(IEN, k, j, i);

          Real &rho_v = cons_df(4*(vapor_id), k, j, i);
          Real fv = rho_v/gas_dens;
          Real gas_vel1_0 = gas_mom1/gas_dens;
          Real gas_vel2_0 = gas_mom2/gas_dens;
          Real gas_vel3_0 = gas_mom3/gas_dens;
          Real Ek0 = 0.5*(SQR(gas_vel1_0) + SQR(gas_vel2_0) + SQR(gas_vel3_0));

          if(std::isnan(fv) or std::isnan(Ek0) or (fv < 0.0) or gas_dens < 0.0){
            std::cout <<"fv= nan in LocalIsothermalEOS" << std::endl;
            std::cout << "rhov =" << rho_v << std::endl;
            std::cout << "gas_dens =" << gas_dens << std::endl;
            std::cout << "rad, phi, z = " << rad << ", " << phi << ", " << z << std::endl;
            std::cout << "i, j, k = " << i << ", " << j << ", " << k << std::endl;
            quick_exit(1);
          }

          // reset the gas vertical profile according to midplane temperature.
          Real gas_vel1_1 = gas_vel1_0;
          Real gas_vel2_1 = gas_vel2_0;
          Real gas_vel3_1 = gas_vel3_0;

          // temperature profile: fixed
          Real Tem;
          Tem = pmb->phydro->Tem(k, j, i);

          if(time < t_iterate){
            LogicalLocation &loc = pmb->loc;
            int ti = static_cast<int>(loc.lx1)*pmb->block_size.nx1+(i-pmb->is) + NGHOST;
            int tj = static_cast<int>(loc.lx2)*pmb->block_size.nx2+(j-pmb->js)+ NGHOST;
            int tj_mid = pmb->pmy_mesh->mesh_size.nx2 + NGHOST -1;
            int tk = static_cast<int>(loc.lx3)*pmb->block_size.nx3+(k-pmb->ks);
            Real Tem_mid = pmb->pmy_mesh->ruser_mesh_data[4](tk,tj_mid,ti);

            gas_dens = pmb->pmy_mesh->ruser_mesh_data[7](tk,tj,ti) + rho_v;
            gas_dens = (gas_dens > dfloor +dffloor) ? gas_dens : (dfloor+dffloor);
            if(time < 0.195){
              gas_dens = DenProfileCyl_gas_fv_T(rad, phi, z, fv, Tem_mid) + rho_v;
            }

            Real v_acc_gas = -1.5*pmb->phydro->hdif.nu(HydroDiffusion::DiffProcess::alpha, k, j, i) / rad;
            
            gas_vel1_1 = v_acc_gas*std::sin(x2);
            gas_vel2_1 = v_acc_gas*std::cos(x2);
            gas_vel3_1 = VelProfileCyl_gas_fv_T(rad, phi, z, fv, Tem_mid);
            if (pmb->porb->orbital_advection_defined)
              gas_vel3_1 -= vel_K;
          }
          ///////////////////////////////////////////////////////////////////
          
          // get pressure of local temperature
          Real mu = Get_mu(fv);
          Real press = gas_dens * Tem /(KELVIN*mu);
          Real gamma = pmb->peos->calc_gamma(fv);
          gas_erg    = press/(gamma-1.0) + gas_dens*0.5*(SQR(gas_vel1_1) + SQR(gas_vel2_1) + SQR(gas_vel3_1));
          gas_mom1   = gas_dens*gas_vel1_1;
          gas_mom2   = gas_dens*gas_vel2_1;
          gas_mom3   = gas_dens*gas_vel3_1;
        }
      }
    } 
}


void RadiativeCondution(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s){
      
    // radiative diffusion flx at x1, x2 direction
    AthenaArray<Real> &x1flux = pmb->phydro->hdif.cndflx[X1DIR];
    AthenaArray<Real> &x2flux = pmb->phydro->hdif.cndflx[X2DIR];

    Real kappaf, dTdx;
    HydroDiffusion phdif = pmb->phydro->hdif;
    
    AthenaArray<Real> x1area, x2area, x2area_p1, vol;
    x1area.NewAthenaArray(pmb->ncells1+1);
    x2area.NewAthenaArray(pmb->ncells1);
    x2area_p1.NewAthenaArray(pmb->ncells1);
    vol.NewAthenaArray(pmb->ncells1);

    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
        // calculate x1-flux divergence
        pmb->pcoord->Face1Area(k, j, pmb->is, pmb->ie+1, x1area);
        pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, vol);

#pragma omp simd private(kappaf, dTdx)
        for (int i=pmb->is; i<=pmb->ie+1; ++i) {
          kappaf = 0.5*(phdif.kappa(HydroDiffusion::DiffProcess::aniso, k, j, i) + phdif.kappa(HydroDiffusion::DiffProcess::aniso, k, j, i-1));
          dTdx = (pmb->phydro->Tem(k,j,i) - pmb->phydro->Tem(k,j,i-1))/pmb->pcoord->dx1v(i-1);                                                                                                                   
          x1flux(k,j,i) = -kappaf*dTdx;
        }
#pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          LogicalLocation &loc = pmb->loc;
          int ti = static_cast<int>(loc.lx1)*pmb->block_size.nx1+(i-pmb->is)+ NGHOST;
          int tj = static_cast<int>(loc.lx2)*pmb->block_size.nx2+(j-pmb->js)+ NGHOST;
          int tk = static_cast<int>(loc.lx3)*pmb->block_size.nx3+(k-pmb->ks);
          // reflecting boundary at disc midplane
          Real tau_vi = pmb->pmy_mesh->ruser_mesh_data[0](tk,tj,ti);

          // set boundary to be not influenced by diffusion
          Real rad = pmb->pcoord->x1v(i);
          Real w_damp = 0.05*(x1max-x1min);
          Real f_decay_art = std::tanh(std::pow((rad-x1max)/w_damp ,2.0)); // outer bc decay
          Real f_decay_art2 = std::tanh(std::pow((rad-1.1*x1min)/w_damp,2.0));
          if(rad <= 1.1*x1min){
            f_decay_art2 = 0.0;
          }
          Real f_decay_art3 = std::exp(-1.0/(tau_vi));
          // pmb->pphase_change->q_diff(k, j, i) += dt/pmb->pmy_mesh->dt*(x1area(i+1)*x1flux(k,j,i+1) - x1area(i)*x1flux(k,j,i))/(-vol(i)) *f_decay_art*f_decay_art2*f_decay_art3;
          if (pmb->pphase_change != nullptr) {
            pmb->pphase_change->q_diff(k, j, i) = 0.0;
          }
        }
        
        // do not include heat conduction in Athena++ steps
        for (int i=pmb->is-NGHOST; i<=pmb->ie+1+NGHOST; ++i) {
          x1flux(k,j,i) = 0.0;
        }

      }
    } 
}

void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
  const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time,
  const int il, const int iu, const int jl, const int ju, const int kl, const int ku){
  // (Yu, 2025-11-18) Calculate stopping time per pebble using PhaseChange module
  
  if (pmb->pphase_change == nullptr || N_Z == 0) {
    return; // No phase change module or no pebble compositions
  }
  
  PhaseChange *pphase_change = pmb->pphase_change;
  DustFluids *pdustfluids = pmb->pdustfluids;
  Hydro *phydro = pmb->phydro;
  Coordinates *pcoord = pmb->pcoord;
  Mesh *pmesh = pmb->pmy_mesh;
  
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        Real rad, phi, z;
        GetCylCoord(pcoord, rad, phi, z, i, j, k);
        
        Real gas_den = prim(IDN, k, j, i);
        Real Tem = phydro->Tem(k, j, i);

        // vapor stopping time: set to a big value.
        Real omega_dyn = std::sqrt(gm0/std::pow(rad,3.0));
        Real &vapor_stopping_time = stopping_time(vapor_id, k, j, i);
        vapor_stopping_time = Stokes_number[vapor_id]/omega_dyn;
        
        // Loop over pebbles to calculate stopping time per pebble
        for (int p = 0; p < N_P; ++p) {
          AthenaArray<Real> rho_dustfluid_array_pebble;
          rho_dustfluid_array_pebble.NewAthenaArray(N_Z);
          
          // collect dustfluid of the same pebble (primitive variables)
          for (int z = 0; z < N_Z; ++z) {
            int dust_id = N_Z * p + z;
            int rho_id = 4*dust_id;
            rho_dustfluid_array_pebble(z) = prim_df(rho_id, k, j, i);
          }
          
          // calculate stopping time for the pebble
          Real rho_Np = pphase_change->rho_Np_array(p, k, j, i);
          Real rho_v = prim_df(4*vapor_id, k, j, i);
          Real t_stop = pphase_change->Get_stopping_time(pmesh->punit, rho_dustfluid_array_pebble, Tem, gas_den, rho_v, rho_Np);
          
          // ===== upper limit, St can be extremely high for upper layer =====
          t_stop = (t_stop > 0.5) ? 0.5 : t_stop;
          if(std::fabs(z) > 2.0 *std::pow((rad/r0), 1.5 + qvalue/2)){
            t_stop = 1.e-4;
          }
          
          // apply stopping time to all compositions of this pebble
          for (int z = 0; z < N_Z; ++z) {
            int dust_id = N_Z * p + z;
            Real &st_time = stopping_time(dust_id, k, j, i);
            
            if(pmesh->time > dust_start_injection){
              st_time = t_stop;
            }else{
              st_time = 1.e-8;
            }
            // apply st floor 
            st_time = (st_time > 1.e-8) ? st_time : 1.e-8;
          }
        }
      }
    }
  }
  
  return;
}

void MyDustDiffusivity(DustFluids *pdf, MeshBlock *pmb,
    const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
    const AthenaArray<Real> &stopping_time, AthenaArray<Real> &nu_dust,
    AthenaArray<Real> &cs_dust, int is, int ie, int js, int je, int ks, int ke) {
  // (Yu, 2025-11-18) Calculate dust diffusivity per pebble using PhaseChange module
  
  if (pmb->pphase_change == nullptr || N_Z == 0) {
    return; // No phase change module or no pebble compositions
  }
  
  PhaseChange *pphase_change = pmb->pphase_change;
  Coordinates *pcoord = pmb->pcoord;
  Hydro *phydro = pmb->phydro;
  
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        Real rad, phi, z;
        GetCylCoord(pcoord, rad, phi, z, i, j, k);
        Real omega_dyn = std::sqrt(gm0/std::pow(rad,3.0));
        
        // vapor diffusivity
        Real &vapor_diffusivity = nu_dust(vapor_id, k, j, i);
        Real &gas_nu = phydro->hdif.nu(HydroDiffusion::DiffProcess::alpha, k, j, i);
        gas_nu = alpha_vis* std::pow(rad/r0, nu_slope); // fix the gas viscosity.
        // artifical decay for outer boundary
        Real w_damp = 0.05*(x1max-x1min);
        Real f_decay_art = std::tanh(std::pow((rad-x1max)/w_damp ,2.0)); // outer bc decay
        vapor_diffusivity = gas_nu * f_decay_art;
        
        // Loop over pebbles to calculate diffusivity per pebble
        for (int p = 0; p < N_P; ++p) {
          // Get stopping time for this pebble (all compositions have same stopping time)
          int dust_id_first = N_Z * p; // first composition of pebble p
          Real t_stop = stopping_time(dust_id_first, k, j, i);
          
          // Calculate diffusivity for all compositions of this pebble
          for (int z = 0; z < N_Z; ++z) {
            int dust_id = N_Z * p + z;
            Real taus_peb = t_stop*omega_dyn;
            Real &diffusivity = nu_dust(dust_id, k, j, i);
            diffusivity = gas_nu/(1.+SQR(taus_peb));

            Real &soundspeed  = cs_dust(dust_id, k, j, i);
            soundspeed        = std::sqrt(diffusivity/omega_dyn);
          }
        }
      }
    }
  }
  
  return;
}

void MyConductivity(HydroDiffusion *phdif, MeshBlock *pmb,
    const AthenaArray<Real> &w, const AthenaArray<Real> &bc,
    int is, int ie, int js, int je, int ks, int ke) {
  
  for (int k=ks; k<=ke; ++k) { // include ghost zone
    for (int j=js; j<=je; ++j) { // prim, cons
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        Real Tem = pmb->phydro->Tem(k, j, i);

        const Real &gas_rho = w(IDN, k, j, i);
        Real fv = pmb->pdustfluids->df_w(4*vapor_id, k, j, i) / gas_rho;
        if(std::isnan(fv) or (fv < 0.0)){
          std::cout << "Conductivity, fv = " << fv << std::endl;
          std::cout << "gas_rho = " << gas_rho << std::endl;
        }
        Real kappa_R = Get_kappa(0.0, fv);
        Real UNIT_SB = pmb->pmy_mesh->punit->code_energydensity_cgs * pmb->pmy_mesh->punit->code_velocity_cgs; // flx/(K^4)

        Real &kappa_heat = phdif->kappa(HydroDiffusion::DiffProcess::aniso, k, j, i);
        Real f_decay = 1.0;
        // kappa_heat = 1.0/(3.0*kappa_R*gas_rho)* 16.0* (Constants::sigma_cgs/UNIT_SB) *SQR(Tem)*Tem*f_decay;
        kappa_heat = 0.0;
      }
    }
  }
  return;
}


void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k) {
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    rad=pco->x1v(i);
    phi=pco->x2v(j);
    z=pco->x3v(k);
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    rad=std::abs(pco->x1v(i)*std::sin(pco->x2v(j)));
    phi=pco->x3v(k);
    z=pco->x1v(i)*std::cos(pco->x2v(j));
  }
  return;
}

//----------------------------------------------------------------------------------------
// tempertature defined in cylindrical coordinate
Real TemProfile(const Real rad, const Real phi, const Real z) {
  
  Real tmp = T0* std::pow((rad/r0),qvalue);
  return tmp;
}

Real Get_kappa(const Real d2g, const Real fv) {
  Real kappa_R = kappa0*(1.0-fv); // const dust-to-gas ratio w.r.t. H/He
  return kappa_R;
}

Real Get_nu_gas(const Real Tem, const Real rad, Real fv) {
  Real mu = mu_xy; // simple model, scale height indep. of mu
  Real cs2 = Tem/(mu*KELVIN);
  Real omega = std::sqrt(gm0/(rad*rad*rad));
  Real nu_gas = alpha_vis*cs2/omega;

  return nu_gas;
}

//----------------------------------------------------------------------------------------
//! computes density in cylindrical coordinates

Real DenProfileCyl_gas(const Real rad, const Real phi, const Real z) {
  Real den;
  Real cs2    = PoverRho(rad, phi, z);
  Real denmid = rho0*std::pow(rad/r0, pvalue);
  Real dentem = denmid*std::exp(gm0/cs2*(1./std::sqrt(SQR(rad)+SQR(z))-1./rad));
  den         = dentem;
  return std::max(den, dfloor+dffloor);
}

Real DenProfileCyl_gas_fv(const Real rad, const Real phi, const Real z, Real fv) {
  Real den;
  // Real cs2    = PoverRho(rad, phi, z);
  Real mu = Get_mu(fv);
  Real cs2 = TemProfile(rad, phi, z)/(KELVIN*mu);
  Real denmid = std::sqrt(mu/mu_xy)* rho0*std::pow(rad/r0, pvalue);
  Real dentem = denmid*std::exp(gm0/cs2*(1./std::sqrt(SQR(rad)+SQR(z))-1./rad));
  den         = dentem;
  return std::max(den, dfloor+dffloor);
}

Real DenProfileCyl_gas_fv_T(const Real rad, const Real phi, const Real z, Real fv, Real Tem) {
  Real den;
  Real mu = Get_mu(fv);
  Real cs2 = Tem/(KELVIN*mu);
  Real H_gas = std::sqrt(cs2)/std::sqrt(gm0/(rad*rad*rad));
  Real sigma_slope = pvalue + 1.5 + qvalue/2.0;
  Real denmid = 1.0 * std::pow(rad/r0,sigma_slope) / H_gas;
  Real dentem = denmid*std::exp(gm0/cs2*(1./std::sqrt(SQR(rad)+SQR(z))-1./rad));
  den         = dentem;
  return std::max(den, dfloor+dffloor);
}


//----------------------------------------------------------------------------------------
//! computes rotational velocity in cylindrical coordinates
Real VelProfileCyl_gas(const Real rad, const Real phi, const Real z) {
  Real cs2 = PoverRho(rad, phi, z);
  Real vel = (pvalue+qvalue)*cs2/(gm0/rad) + (1.0+qvalue) - qvalue*rad/std::sqrt(rad*rad+z*z);
  vel      = std::sqrt(gm0/rad)*std::sqrt(vel) - rad*Omega0;
  return vel;
}

//! computes rotational velocity in cylindrical coordinates
Real VelProfileCyl_gas_fv(const Real rad, const Real phi, const Real z, Real fv) {
  // Real cs2 = PoverRho(rad, phi, z);
  Real mu = Get_mu(fv);
  Real cs2 = TemProfile(rad, phi, z)/(KELVIN*mu);
  Real vel = (pvalue+qvalue)*cs2/(gm0/rad) + (1.0+qvalue) - qvalue*rad/std::sqrt(rad*rad+z*z);
  vel      = std::sqrt(gm0/rad)*std::sqrt(vel) - rad*Omega0;
  return vel;
}

Real VelProfileCyl_gas_fv_T(const Real rad, const Real phi, const Real z, Real fv, Real Tem) {
  // Real cs2 = PoverRho(rad, phi, z);
  Real mu = Get_mu(fv);
  Real cs2 = Tem/(KELVIN*mu);
  Real vel = (pvalue+qvalue)*cs2/(gm0/rad) + (1.0+qvalue) - qvalue*rad/std::sqrt(rad*rad+z*z);
  vel      = std::sqrt(gm0/rad)*std::sqrt(vel) - rad*Omega0;
  return vel;
}

Real VelProfileCyl_gas_fv_T_pvalue(const Real rad, const Real phi, const Real z, Real fv, Real Tem, Real pnew, Real qnew) {
  // Real cs2 = PoverRho(rad, phi, z);
  Real mu = Get_mu(fv);
  Real cs2 = Tem/(KELVIN*mu);
  Real vel = (pnew+qnew)*cs2/(gm0/rad) + (1.0+qnew) - qnew*rad/std::sqrt(rad*rad+z*z);
  if(vel < 0){
    std :: cout << "Negative velocity" << std::endl;
    quick_exit(1);
  }
  vel      = std::sqrt(gm0/rad)*std::sqrt(vel) - rad*Omega0;
  return vel;
}

Real Vr_ProfileCyl_gas(const Real rad, const Real phi, const Real z) {
  Real H_gas = std::pow(rad/r0, qvalue/2.0 + 1.5);
  Real vel_r = -alpha_vis *(1.0/r0)*std::pow(rad/r0, qvalue + 0.5)* (3.0*pvalue + 2.0*qvalue + 6.0 + (5.0*qvalue + 9.0)/2.0*SQR(z/H_gas));
  return vel_r;
}

Real Vr_ProfileCyl_gas_fv(const Real rad, const Real phi, const Real z, Real fv) {
  Real H_gas = std::pow(rad/r0, qvalue/2.0 + 1.5);
  Real mu = Get_mu(fv);
  H_gas *= std::sqrt(mu_xy/mu); // scale height decreases due to vapor injection
  Real vel_r = -alpha_vis *(1.0/r0)*std::pow(rad/r0, qvalue + 0.5)* (3.0*pvalue + 2.0*qvalue + 6.0 + (5.0*qvalue + 9.0)/2.0*SQR(z/H_gas));
  return vel_r;
}

Real Vr_ProfileCyl_gas_fv_T(const Real rad, const Real phi, const Real z, Real fv, Real Tem) {
  Real mu = Get_mu(fv);
  Real omega = std::sqrt(gm0/(rad*rad*rad));
  Real H_gas = std::sqrt(Tem/(KELVIN*mu))/omega;
  Real vel_r = -alpha_vis *(1.0/r0)*std::pow(rad/r0, qvalue + 0.5)* (3.0*pvalue + 2.0*qvalue + 6.0 + (5.0*qvalue + 9.0)/2.0*SQR(z/H_gas));
  return vel_r;
}

Real Vr_ProfileCyl_gas_fv_T_pvalue(const Real rad, const Real phi, const Real z, Real fv, Real Tem, Real pnew, Real qnew) {
  Real mu = Get_mu(fv);
  Real omega = std::sqrt(gm0/(rad*rad*rad));
  Real H_gas = std::sqrt(Tem/(KELVIN*mu))/omega;
  Real vel_r = -alpha_vis *(1.0/r0)*std::pow(rad/r0, qnew + 0.5)* (3.0*pnew + 2.0*qnew + 6.0 + (5.0*qnew + 9.0)/2.0*SQR(z/H_gas));
  return vel_r;
}

Real five_point(AthenaArray<Real> f, Real x1, Real dx1, int k, int j, int i){

  Real f0 = f(k,j,i-2);
  Real f1 = f(k,j,i-1);
  Real f2 = f(k,j,i);
  Real f3 = f(k,j,i+1);
  Real f4 = f(k,j,i+2);

  Real dfdx = (-f4 + 8.0*f3 - 8.0*f1 + f0)/(12.0*dx1);
  Real slope = dfdx *x1/ f2;  
  return slope;
}

Real five_point_ed(AthenaArray<Real> f, Real x1, Real dx1, int k, int j, int i){

  Real f0 = f(k,j,i);
  Real f1 = f(k,j,i-1);
  Real f2 = f(k,j,i-2);
  Real f3 = f(k,j,i-3);
  Real f4 = f(k,j,i-4);

  Real dfdx = (3.0*f4 -16.0*f3 + 36.0*f2 - 48.0*f1 + 25.0*f0)/(12.0*dx1);
  Real slope = dfdx *x1/ f2;  
  return slope;
}

Real five_point_bg(AthenaArray<Real> f, Real x1, Real dx1, int k, int j, int i){

  Real f0 = f(k,j,i);
  Real f1 = f(k,j,i+1);
  Real f2 = f(k,j,i+2);
  Real f3 = f(k,j,i+3);
  Real f4 = f(k,j,i+4);

  Real dfdx = -(3.0*f4 -16.0*f3 + 36.0*f2 - 48.0*f1 + 25.0*f0)/(12.0*dx1);
  Real slope = dfdx *x1/ f2;  
  return slope;
}

Real NewtonIntpl(const Real f0, const Real f1, const Real f2, const Real x0, const Real x1, const Real x2, const Real x_exp) {
  Real df01 = (f1-f0)/(x1-x0);
  Real df12 = (f2-f1)/(x2-x1);

  Real a0 = df01;
  Real a1 = (df12-df01)/(x2-x0);
  
  Real f_intpl = f0 + a0*(x_exp-x0) + a1*(x_exp-x0)*(x_exp-x1);
  return f_intpl;

}

void LagIntpl(AthenaArray<Real> x, AthenaArray<Real> y, AthenaArray<Real> x_exp, AthenaArray<Real> &y_exp) {
  Real result;
  int n = x.GetSize();
  Real term;
  for (int k = 0; k < x_exp.GetSize(); ++k){
    result = 0.0;
    for (int i = 0; i < n; ++i) {
      term = y(i);
      for (int j = 0; j < n; ++j) {
        if (j != i) {
          term *= (x_exp(k) - x(j)) / (x(i) - x(j));
        }
      }
      result += term;
    }

    y_exp(k) = result;
  }
  
  return;
}

Real DenProfileCyl_dust(const Real rad, const Real phi, const Real z, const Real den_ratio, const Real H_ratio) {
  Real den;
  Real cs2    = PoverRho(rad, phi, z);
  Real denmid = den_ratio*rho0*std::pow(rad/r0, pvalue);
  Real dentem = denmid*std::exp(gm0/(SQR(H_ratio)*cs2)*(1./std::sqrt(SQR(rad)+SQR(z))-1./rad));
  den         = dentem;
  return std::max(den, dffloor);
}

Real VelProfileCyl_dust(const Real rad, const Real phi, const Real z) {
  Real dis = std::sqrt(SQR(rad) + SQR(z));
  Real vel = std::sqrt(gm0/dis) - rad*Omega0;
  // Real vel = std::sqrt(gm0/dis) - std::sqrt(gm0/rad);
  return vel;
}

Real UserOrbitalVelocity(OrbitalAdvection * porb, Real x1, Real x2, Real x3){
  return std::sqrt(porb->gm/x1)-std::sqrt(porb->gm/x1);
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
  
  Real sigma_slope = pvalue + qvalue/2.0 + 1.5;
  Real logdr = std::log(r_active/r_ghost);
  sigma_ghost = std::exp(std::log(sigma_active) - logdr*sigma_slope);

  return;
}

void Vr_outflow(const Real r_active, const Real r_ghost, const Real rho_active,
               const Real rho_ghost, const Real vr_active, Real &vr_ghost) {

  vr_ghost = (vr_active <= 0.0) ? ((rho_active*SQR(r_active)*vr_active)/(rho_ghost*SQR(r_ghost))) : 0.0;
  return;
}

void Vr_Mdot(const Real r_active, const Real r_ghost, const Real rho_active,
               const Real rho_ghost, const Real vr_active, Real &vr_ghost) {

  vr_ghost = rho_active*SQR(r_active)*vr_active/(rho_ghost*SQR(r_ghost));
  // vr_ghost = (vr_active <= 0.0) ? ((rho_active*SQR(r_active)*vr_active)/(rho_ghost*SQR(r_ghost))) : 0.0;
  return;
}

//----------------------------------------------------------------------------------------
//! computes pressure/density in cylindrical coordinates

Real PoverRho(const Real rad, const Real phi, const Real z) {
  Real poverr;
  poverr = cs2_0*std::pow(rad/r0, qvalue);
  return poverr;
}

Real CompressedX2(Real x, RegionSize rs)
{
  // Real w=std::acos(1.0-x)/(PI/2.0); // cosine
  Real w=std::pow(x,1.0/3.0); // power law

  return w*rs.x2max+(1.0-w) * rs.x2min;
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
      Real x2 = pmb->pcoord->x2v(j);
      for (int i=1; i<=ngh; ++i) {
        GetCylCoord(pco, rad_active, phi_active, z_active, il,   j, k);
        GetCylCoord(pco, rad_ghost,  phi_ghost,  z_ghost,  il-i, j, k);
        Real x1_active = pmb->pcoord->x1v(il);
        Real x1_ghost = pmb->pcoord->x1v(il-i);
        Real vel_K     = vK(pmb->porb, pco->x1v(il-i), pco->x2v(j), pco->x3v(k));

        Real &Tem_ghost = pmb->phydro->Tem(k, j, il-i);
        
        Real &gas_rho_ghost  = prim(IDN, k, j, il-i);
        Real &gas_vel1_ghost = prim(IM1, k, j, il-i);
        Real &gas_vel2_ghost = prim(IM2, k, j, il-i);
        Real &gas_vel3_ghost = prim(IM3, k, j, il-i);
        Real &gas_pres_ghost = prim(IEN, k, j, il-i);
        
        Real &gas_rho_active  = prim(IDN, k, j, il);
        Real &gas_vel1_active = prim(IM1, k, j, il);
        Real &gas_vel2_active = prim(IM2, k, j, il);
        Real &gas_vel3_active = prim(IM3, k, j, il);
        Real &gas_pres_active = prim(IPR, k, j, il); 

        // take fv from active zone
        Real fv = prim_df(4*(vapor_id), k, j, il)/gas_rho_active;
        
        // get midplane temperture
        int tk = 0; int ti = il-i;
        int tj = static_cast<int>(pmb->loc.lx2)*pmb->block_size.nx2+(j-pmb->js)+ NGHOST;
        int tj_mid = pmb->pmy_mesh->mesh_size.nx2 + NGHOST -1;
        Real Tem_mid = pmb->pmy_mesh->ruser_mesh_data[4](tk,tj_mid,ti);

        // keep H/He surface density uneffected at inner boundary
        Real gas_rho_xy = pmb->pmy_mesh->ruser_mesh_data[7](tk,tj,ti);
        if(time < 0.195){
          gas_rho_xy = DenProfileCyl_gas_fv_T(rad_ghost,phi_ghost,z_ghost,fv,Tem_mid);
        }

        // simple and best practice:
        gas_rho_ghost = gas_rho_xy/(1.0-fv);
        gas_vel1_ghost = gas_vel1_active;
        gas_vel2_ghost = gas_vel2_active;
        gas_vel3_ghost = gas_vel3_active*x1_active / (x1_ghost);
        // other bc.
        // gas_rho_ghost = std::exp(2.0*std::log(prim(IDN, k, j, il-i+1)) - std::log(prim(IDN, k, j, il-i+2)));
        // gas_rho_ghost = gas_rho_active*SQR(x1_active)/ SQR(x1_ghost);
        // gas_vel3_ghost = gas_vel3_active;
        // gas_vel3_ghost = gas_vel3_active*SQR(x1_active) / SQR(x1_ghost);

        if (NON_BAROTROPIC_EOS){
          Real mu1 = Get_mu(fv);
          if(time < t_iterate){
            gas_pres_ghost = gas_rho_ghost*Tem_ghost/(mu1*KELVIN);
          }else{
            gas_pres_ghost = std::exp(2.0*std::log(prim(IPR, k, j, il-i+1)) - std::log(prim(IPR, k, j, il-i+2)));
            // gas_pres_ghost = gas_rho_ghost*Tem_ghost/(mu1*KELVIN);
          }
          gas_pres_ghost = (gas_pres_ghost > pfloor) ? gas_pres_ghost : pfloor;
        }

        // copy latent heat (not always necessary... depending on simulation domain):
        if (pmb->pphase_change != nullptr){
          pmb->pphase_change->q_latent(k,j,il-i) = pmb->pphase_change->q_latent(k,j,il);
        }

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;
            
            if(dust_id == vapor_id){
              Real &vapor_rho_ghost  = prim_df(rho_id, k, j, il-i);
              Real &vapor_vel1_ghost = prim_df(v1_id,  k, j, il-i);
              Real &vapor_vel2_ghost = prim_df(v2_id,  k, j, il-i);
              Real &vapor_vel3_ghost = prim_df(v3_id,  k, j, il-i);
              
              // keep the consistency between gas and vapor.
              vapor_vel1_ghost = gas_vel1_ghost;
              vapor_vel2_ghost = gas_vel2_ghost;
              vapor_vel3_ghost = gas_vel3_ghost;
              vapor_rho_ghost = fv*gas_rho_ghost;
            }else{
              // free outflow for dust
              // prim_df(rho_id,k,j,il-i) = prim_df(rho_id,k,j,il);
              prim_df(rho_id,k,j,il-i) = prim_df(rho_id,k,j,il)*SQR(x1_active)/SQR(x1_ghost);
              prim_df(v1_id,k,j,il-i) = prim_df(v1_id,k,j,il);
              prim_df(v2_id,k,j,il-i) = prim_df(v2_id,k,j,il);
              prim_df(v3_id,k,j,il-i) = prim_df(v3_id,k,j,il);
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
      Real x2 = pmb->pcoord->x2v(j);

      for (int i=1; i<=ngh; ++i) {
        // calculate dl in theta direction
        Real x1_ghost = pmb->pcoord->x1v(iu+i);
        Real x1_active = pmb->pcoord->x1v(iu);
        Real dx2 = pmb->pcoord->dx2v(j);
        Real ds_active =  2.0*PI*SQR(x1_active)*std::sin(x2)*dx2;
        Real ds_ghost =  2.0*PI*SQR(x1_ghost)*std::sin(x2)*dx2;

        GetCylCoord(pco, rad_active, phi_active, z_active, iu, j, k);
        Real &gas_rho_active  = prim(IDN, k, j, iu);
        Real &gas_vel1_active = prim(IM1, k, j, iu);
        Real &gas_vel2_active = prim(IM2, k, j, iu);
        Real &gas_vel3_active = prim(IM3, k, j, iu);
        Real &gas_pres_active = prim(IPR, k, j, iu);

        GetCylCoord(pco, rad_ghost, phi_ghost, z_ghost, iu+i, j, k);
        Real &gas_rho_ghost  = prim(IDN, k, j, iu+i);
        Real &gas_vel1_ghost = prim(IM1, k, j, iu+i);
        Real &gas_vel2_ghost = prim(IM2, k, j, iu+i);
        Real &gas_vel3_ghost = prim(IM3, k, j, iu+i);
        Real &gas_pres_ghost = prim(IEN, k, j, iu+i);
        Real vel_K     = vK(pmb->porb, pco->x1v(iu+i), pco->x2v(j), pco->x3v(k));

        int tk = 0; int ti = pmb->pmy_mesh->mesh_size.nx1 + NGHOST + i-1; 
        int tj = static_cast<int>(pmb->loc.lx2)*pmb->block_size.nx2+(j-pmb->js)+ NGHOST;
        int tj_mid = pmb->pmy_mesh->mesh_size.nx2 + NGHOST -1;
        Real Tem_mid = pmb->pmy_mesh->ruser_mesh_data[4](tk,tj_mid,ti);
        
        Real omega_dyn = std::sqrt(gm0/std::pow(rad_ghost,3.0));
        Real &Tem_ghost = pmb->phydro->Tem(k, j, iu+i);

        Real vel_gas_phi = VelProfileCyl_gas_fv_T(rad_ghost, phi_ghost, z_ghost, 0.0, Tem_mid);
        Real vr0 = Vr_ProfileCyl_gas_fv_T(rad_ghost, phi_ghost, z_ghost, 0.0, Tem_mid);

        if (pmb->porb->orbital_advection_defined)
          vel_gas_phi -= vel_K;
        
        // best practice:
        gas_rho_ghost = std::exp(2.0*std::log(prim(IDN, k, j, iu+i-1)) - std::log(prim(IDN, k, j, iu+i-2)));
        gas_vel1_ghost = vr0* std::sin(x2);
        gas_vel2_ghost = 0.0;
        gas_vel3_ghost = vel_gas_phi;
        // other bc.
        // gas_vel2_ghost = vr0* std::cos(x2);
        // gas_vel1_ghost = gas_vel1_active;
        // gas_vel2_ghost = gas_vel2_active;
        // gas_rho_ghost = gas_rho_active*SQR(x1_active)/ SQR(x1_ghost);
        // gas_rho_ghost = pmb->pmy_mesh->ruser_mesh_data[7](tk,tj,ti);

        if(time < 0.195){
          gas_rho_ghost = DenProfileCyl_gas_fv_T(rad_ghost,phi_ghost,z_ghost,0.0,Tem_mid);
        }

        if(std::isnan(gas_vel1_ghost)){
          std::cout <<"gas_vel1_ghost = nan" << std::endl;
          quick_exit(1);
        }
        if(std::isnan(gas_rho_ghost)){
          std::cout <<"gas_rho_ghost = nan" << std::endl;
          quick_exit(1);
        }
        if(std::isnan(gas_vel2_ghost)){
          std::cout <<"gas_vel2_ghost = nan" << std::endl;
          quick_exit(1);
        }
        if(std::isnan(gas_vel3_ghost)){
          std::cout <<"gas_vel3_ghost = nan" << std::endl;
          std::cout << "rad_ghost = " << rad_ghost << std::endl;
          std::cout << "z_ghost = " << z_ghost << std::endl;
          quick_exit(1);
        }
        
        if (NON_BAROTROPIC_EOS)
          // extrapolated pressure
          gas_pres_ghost = std::exp(2.0*std::log(prim(IPR, k, j, iu+i-1)) - std::log(prim(IPR, k, j, iu+i-2)));
          gas_pres_ghost = std::max(gas_pres_ghost, pfloor);

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real &dust_rho_ghost  = prim_df(rho_id, k, j, iu+i);
            Real &dust_vel1_ghost = prim_df(v1_id,  k, j, iu+i);
            Real &dust_vel2_ghost = prim_df(v2_id,  k, j, iu+i);
            Real &dust_vel3_ghost = prim_df(v3_id,  k, j, iu+i);

            Real vel_dust_phi = VelProfileCyl_dust(rad_ghost, phi_ghost, z_ghost);
            if (pmb->porb->orbital_advection_defined)
              vel_dust_phi -= vel_K;

            if(dust_id == vapor_id){
              dust_rho_ghost = initial_D2G[dust_id]*gas_rho_ghost;
              dust_rho_ghost  = (dust_rho_ghost > dffloor) ? dust_rho_ghost : dffloor;
              dust_vel1_ghost = gas_vel1_ghost;
              dust_vel2_ghost = gas_vel2_ghost;
              dust_vel3_ghost = gas_vel3_ghost;
            }else{
              dust_rho_ghost = prim_df(rho_id, k, j, iu);
              dust_vel1_ghost = prim_df(v1_id,  k, j, iu);
              dust_vel2_ghost = prim_df(v2_id,  k, j, iu);
              dust_vel3_ghost = vel_dust_phi;
            }
          }
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;

  Real orb_defined;
  (pmb->porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;

  for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=il; i<=iu; ++i) {
        Real rad_ghost, phi_ghost, z_ghost;
        Real rad_active, phi_active, z_active;
        GetCylCoord(pco, rad_ghost,  phi_ghost,  z_ghost,  i, jl-j, k);
        GetCylCoord(pco, rad_active, phi_active, z_active, i, jl,   k);

        Real cs_square   = pmb->phydro->Tem(k, jl, i)/(mu_xy*KELVIN);

        Real &gas_rho_ghost  = prim(IDN, k, jl-j, i);
        Real &gas_vel1_ghost = prim(IM1, k, jl-j, i);
        Real &gas_vel2_ghost = prim(IM2, k, jl-j, i);
        Real &gas_vel3_ghost = prim(IM3, k, jl-j, i);
        Real &gas_pres_ghost = prim(IEN, k, jl-j, i);

        Real &gas_rho_active  = prim(IDN, k, jl, i);
        Real &gas_vel1_active = prim(IM1, k, jl, i);
        Real &gas_vel2_active = prim(IM2, k, jl, i);
        Real &gas_vel3_active = prim(IM3, k, jl, i);
        Real &gas_pres_active = prim(IEN, k, jl, i);

        // gas_rho_ghost     = DenProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
        Real fv = prim_df(4*(vapor_id), k, jl, i)/gas_rho_active;
        gas_rho_ghost = std::exp(2.0*std::log(prim(IDN, k, jl-j+1, i)) - std::log(prim(IDN, k, jl-j+2, i)));
        // gas_rho_ghost = gas_pres_active;
        gas_rho_ghost = (gas_rho_ghost > dfloor+dffloor) ? gas_rho_ghost : dfloor+dffloor;
        // gas_rho_ghost = NewtonIntpl(prim(IDN, k, jl-j+3, i), prim(IDN, k, jl-j+2, i), prim(IDN, k, jl-j+1, i),pco->x2v(jl-j+3), pco->x2v(jl-j+2), pco->x2v(jl-j+1), pco->x2v(jl-j));
        // Real vel_gas_phi  = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
        // vel_gas_phi      -= orb_defined*vel_K;

        // Vr_outflow(rad_active, rad_ghost, gas_rho_active, gas_rho_ghost,
                // gas_vel1_active, gas_vel1_ghost);
        // gas_vel1_ghost = vis_vel_cyl*std::sin(pco->x2v(jl-j));
        // gas_vel2_ghost = vis_vel_cyl*std::cos(pco->x2v(jl-j));
        gas_vel1_ghost = gas_vel1_active;
        gas_vel2_ghost = gas_vel2_active;
        gas_vel3_ghost = gas_vel3_active;

        gas_pres_ghost = cs_square*gas_rho_ghost;
        // gas_pres_ghost = NewtonIntpl(prim(IPR, k, jl-j+3, i), prim(IPR, k, jl-j+2, i), prim(IPR, k, jl-j+1, i),pco->x2v(jl-j+3), pco->x2v(jl-j+2), pco->x2v(jl-j+1), pco->x2v(jl-j));
        gas_pres_ghost = (gas_pres_ghost > pfloor) ? gas_pres_ghost : pfloor;

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real &dust_rho_ghost  = prim_df(rho_id, k, jl-j, i);
            Real &dust_vel1_ghost = prim_df(v1_id,  k, jl-j, i);
            Real &dust_vel2_ghost = prim_df(v2_id,  k, jl-j, i);
            Real &dust_vel3_ghost = prim_df(v3_id,  k, jl-j, i);

            if(dust_id == vapor_id){
              dust_rho_ghost  = fv*gas_rho_ghost;
              dust_vel1_ghost = gas_vel1_ghost;
              dust_vel2_ghost = gas_vel2_ghost;
              dust_vel3_ghost = gas_vel3_ghost;
            }else{
              dust_rho_ghost = prim_df(rho_id, k, jl, i);
              dust_vel1_ghost = prim_df(v1_id, k, jl, i);
              dust_vel2_ghost = prim_df(v2_id, k, jl, i);
              dust_vel3_ghost = prim_df(v3_id, k, jl, i);
            }
          }
        }
      }
    }
  }
}

void UpperWaveDampingGas(MeshBlock *pmb, const Real time, const Real dt, int il, int iu,
  int jl, int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons) {

Real inv_upper_damp = 1.0/upper_altitude_damping;
Real inv_lower_damp = 1.0/lower_altitude_damping;
Real inv_inner_damp = 1.0/inner_width_damping;
Real inv_outer_damp = 1.0/outer_width_damping;

OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;

Real orb_defined;
(pmb->porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;

int nc1 = pmb->ncells1;

AthenaArray<Real> omega_dyn, Theta_func, inv_damping_tau;
AthenaArray<Real> rad_arr, phi_arr, z_arr;

omega_dyn.NewAthenaArray(nc1);
Theta_func.NewAthenaArray(nc1);
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
      Real rad_min = x1min*std::sin(x2);
      Real rad_max = x1max*std::sin(x2);
      GetCylCoord(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k);

      if (x2 <= theta_upper_damping and x2 > x2min) {
        // See de Val-Borro et al. 2006 & 2007
        omega_dyn(i)       = std::sqrt(gm0/(rad_arr(i)*rad_arr(i)*rad_arr(i)));
        Theta_func(i)      = SQR((x2 - theta_upper_damping)*inv_upper_damp)*2.0;
        Theta_func(i)      = Theta_func(i) > 1.0 ? 1.0 : Theta_func(i);

        // inv_damping_tau(i) = (damping_rate*omega_dyn(i));
        inv_damping_tau(i) = 1.0/dt;

        // hydrostatic balance in theta direction: vtheta = 0
        Real gas_vel2_0 = 0.0;

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

        Real Ek0 = 0.5*(SQR(gas_vel1) + SQR(gas_vel2) + SQR(gas_vel3))*gas_dens;

        // with damping timescale
        // Real delta_gas_rho  = (gas_rho_0  - gas_rho )*Theta_func(i)*inv_damping_tau(i)*dt;
        // Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*Theta_func(i)*inv_damping_tau(i)*dt;
        // Real delta_gas_vel2 = (gas_vel2_0 - gas_vel2)*Theta_func(i)*inv_damping_tau(i)*dt;
        // Real delta_gas_vel3 = (gas_vel3_0 - gas_vel3)*Theta_func(i)*inv_damping_tau(i)*dt;
        // Real delta_gas_pre  = (gas_pre_0  - gas_pre )*Theta_func(i)*inv_damping_tau(i)*dt;

        // immediate damping
        Real delta_gas_rho  = 0.0;
        Real delta_gas_vel1 = 0.0;
        Real delta_gas_vel2 = (gas_vel2_0 - gas_vel2)*Theta_func(i);
        Real delta_gas_vel3 = 0.0;
        Real delta_gas_pre  = 0.0;

        gas_rho  += delta_gas_rho;
        gas_vel1 += delta_gas_vel1;
        gas_vel2 += delta_gas_vel2;
        gas_vel3 += delta_gas_vel3;
        gas_pre  += delta_gas_pre;

        gas_dens = gas_rho;
        gas_mom1 = gas_dens*gas_vel1;
        gas_mom2 = gas_dens*gas_vel2;
        gas_mom3 = gas_dens*gas_vel3;

        Real Ek = 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3))/gas_dens;
        gas_erg = gas_erg - Ek0 + Ek;
      }
    }
  }
}
return;
}
