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
#include <ostream>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

#include <thread> // For std::this_thread::sleep_for
#include <chrono> // For std::chrono::seconds
#include <numeric> // For std::accumulate

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
#include "../units/units.hpp"  // For Constants and unit conversions

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

template<typename T>
void print(const T& value, const char* name);
#define PRINT(x) print(x, #x)

Real Get_thermal_relaxation_time(Real omega_dyn, Real Tem, Real rhokappa, Real rhod, Real rhog, Real Hg, Real z, Real rad);
// phase change
void Get_vel_new_fromMC(AthenaArray<Real> gas_vel_array,
    Real rho_g, Real rho_g1, AthenaArray<Real> rho_d_array, AthenaArray<Real> rho_d_array1, Real drho,
    AthenaArray<Real> rho_ratio_array, AthenaArray<Real> dust_vel_array, bool istracer[NDUSTFLUIDS],
    AthenaArray<Real> &gas_vel_array1, Real &E_kg, AthenaArray<Real> &E_kd_array);
void Get_E_kg(AthenaArray<Real> gas_vel_array, Real &E_kg);
Real Get_T_rhoe_g(Real rhoe_g, Real rho_g, Real fv);
Real Get_T_rhoe(Real rhoe, Real rho_g, AthenaArray<Real> rho_d_array, Real fv);
Real Get_rhomu_d(AthenaArray<Real> rho_d_array);
Real Get_rhoe(Real rhoE_total, Real rho_g, Real E_kg, AthenaArray<Real> rho_d_array, AthenaArray<Real> E_kd_array);
Real Get_Z(Real rho_g, Real T);
void phase_trans(Real rhoe, Real rho_g, AthenaArray<Real> rho_I, Real rho_v, Real &drho);
Real Get_mu(Real fv);
Real Get_stopping_time(AthenaArray<Real> rho_d, Real rho_v, Real T, Real rho_g, Real m_p0, Real rad, Real &m_p_out, Real &s_p_out);
Real TemProfile(const Real rad, const Real theta, const Real phi);
Real Get_kappa(const Real d2g, const Real fv);
Real Get_nu_gas(const Real Tem, const Real rad, Real fv);
void Vr_outflow(const Real r_active, const Real r_ghost, const Real rho_active,
                    const Real rho_ghost, const Real vr_active, Real &vr_ghost);
void Vr_Mdot(const Real r_active, const Real r_ghost, const Real rho_active,
                    const Real rho_ghost, const Real vr_active, Real &vr_ghost);
// problem parameters which are useful to make global to this file
Real gm0, r0, rho0, T0, gamma_gas, Omega0, nu_alpha, nu_slope, cs2_0, qvalue, pvalue;
Real dfloor, dffloor, pfloor;
Real KELVIN;  // Temperature conversion factor from PhaseChange module

//snowline
Real f_ICE_inter0, taus0, rho_sil_inter, rho_ice_inter;
Real M_dot_g, L_star; // M_sun/yr
// Real p2g_flux;

// Check the relation between NDUSTBIN, NCOMPOS, and NDUSTFLUID
static_assert(NDUSTFLUIDS == NDUSTBIN*NCOMPOS + NVapor,
              "NDUSTFLUIDS should be equal to NDUSTBIN*NCOMPOS + 1, "
              "where NDUSTBIN is the number of dust bins and NCOMPOS is the number of compositions.");

// so here we have the correlation: NCOMPOS*NDUSTBIN + NVapor = NDUSTFLUIDS
// and NVapor = NVOLATILE
constexpr int NVOLATILE = NCOMPOS - NRefrac;
static_assert(NVapor == NVOLATILE, 
              "the number of volatile should be equal to the number of vapor");

Real initial_D2G[NDUSTFLUIDS], Stokes_number[NDUSTFLUIDS], Hratio[NDUSTFLUIDS], weight_dust[NDUSTFLUIDS], m_p0[NDUSTBIN], p2g_flux[NDUSTBIN]; 

bool mom_correct_Flag, Isothermal_Flag, Damping_Flag, Theta_Gas_Damping_Flag ,Allow_T_change_Flag, Latent_heat_flag;

Real x1min, x1max, x2min, x2max;
Real damping_rate, radius_inner_damping, radius_outer_damping, inner_ratio_region, outer_ratio_region, inner_width_damping, outer_width_damping, theta_upper_damping, theta_lower_damping, upper_altitude_damping, lower_altitude_damping;
Real min_tol, max_dfvdt, dust_start_injection, injection_Tsoft, t_restart;
Real kappa0, t_iterate, beta;

// User Sources
void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void phase_change(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
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

void tripod_source(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
Real powerlaw_tripod(Real sigma1, Real sigma0, Real amax, Real amin);

// dustfluid settings
void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time_array);
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

void UpperWaveDampingGas(MeshBlock *pmb, const Real time, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons);

//[25.07.04]zxl: just a copy function used in Meshblock::UserWorkInLoop
void copy_velocities(int source_dust_id, int target_dust_id, int k, int j, int i, MeshBlock *pmb, std::string source_type);


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
  Latent_heat_flag = pin->GetOrAddBoolean("problem", "Latent_heat_flag", false);


  // Get parameters for initial density and velocity
  rho0 = pin->GetReal("problem", "rho0");
  pvalue = pin->GetOrAddReal("problem", "pvalue", -1.0);
  T0 = pin->GetOrAddReal("problem", "T0", 150.0);
  kappa0 = pin->GetReal("problem", "kappa0");
  t_iterate = pin->GetReal("problem", "t_iterate");
  beta = pin->GetReal("problem", "beta");

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
  nu_alpha  = pin->GetOrAddReal("problem", "nu_alpha", 0.0);
  nu_slope = pin->GetOrAddReal("problem", "nu_slope", qvalue+1.5);
  
  // Dust to gas ratio && dust stopping time
  if (NDUSTFLUIDS > 0) {
    for (int n=0; n<NDUSTFLUIDS; n++) {
      initial_D2G[n]   = pin->GetReal("dust", "initial_D2G_" + std::to_string(n+1));
      Stokes_number[n] = pin->GetReal("dust", "Stokes_number_" + std::to_string(n+1));
      Hratio[n]        = pin->GetReal("dust", "Hratio_" + std::to_string(n+1));
      //[25.07.03]zxl: is this used??
      weight_dust[n]   = 2.0/(Stokes_number[n] + SQR(1.0+initial_D2G[n])/Stokes_number[n]);
    }
  }

  // get m_p0 and p2g_flux, which can be multi-population
  if (NDUSTBIN > 0) {
    for (int n=0; n<NDUSTBIN; ++n) {
      m_p0[n] = pin->GetReal("problem", "m_p0_" + std::to_string(n+1));
      p2g_flux[n] = pin->GetReal("problem", "p2g_flux_" + std::to_string(n+1));
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
  L_star = pin->GetOrAddReal("problem", "L_star", 1.0); // luminosity of central star [L_sun]


  
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
  upper_altitude_damping = (x2max-x2min)*0.2;
  lower_altitude_damping = (x2max-x2min)*0.1;

  // we only damp the region between x2min and theta_upper_damping
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

  // Enroll user-defined mesh data, which will be stored as restart file.
  // global optical depth
  if(RT_ANA){
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
    std::cout << "Latent_heat_flag = " << Latent_heat_flag << std::endl;
    std::cout << "kappa0 = " << kappa0 << "cm^2/g" << std::endl;
    std::cout << "L_star = " << L_star << "L_sun" << std::endl;
    std::cout << "==============================================================" << std::endl;
    std::cout << "Parameters in this simulation"<< std::endl;
    std::cout << "qvalue ="<< qvalue<< std::endl;
    std::cout << "pvalue ="<< pvalue<< std::endl;
    std::cout << "KELVIN ="<< KELVIN<< std::endl;
    std::cout << "p_eq0 ="<< P_eq0<< std::endl;
    std::cout << "L_heat ="<< L_heat<< std::endl;
    std::cout << "T_a=" << T_a << "K" << std::endl;
    std::cout << "a_semi =" << a_semi << "AU" <<std::endl;
    std::cout << "Cd_water =" << Cd_water <<std::endl;
    std::cout << "inner_damping_radius =" << radius_inner_damping/r0*3.0 <<std::endl;
    std::cout << "outer_damping_radius =" << radius_outer_damping/r0*3.0 <<std::endl;
    std::cout << "theta_upper_damping = " << theta_upper_damping << std::endl;
    std::cout << "theta_lower_damping = " << theta_lower_damping << std::endl;

    std::cout << "M_dot_g = " << M_dot_g << "M_sun/yr" << std::endl;
    std::cout << "nu_alpha = " << nu_alpha << std::endl;

    std::cout << "dust_start_injection" << dust_start_injection << std::endl;

    // print some informations for multi-population dust
    if (NDUSTBIN > 1) {
      std::cout << "==============================================================" << std::endl;
      std::cout << "Multi-population dust is enabled." << std::endl;
      std::cout << "There are " << NDUSTBIN << " dust bins defined." << std::endl;
      for (int n=0; n<NDUSTBIN; ++n) {
        std::cout << "m_p0_" << (n+1) << " = " << m_p0[n] << std::endl;
        std::cout << "p2g_flux_" << (n+1) << " = " << p2g_flux[n] << std::endl;
      }
    }

    // check whether the small particles parameters are set 
    // std::cout << NDUSTFLUIDS << " dust fluids are defined." << std::endl;
    // for (int n=0; n<NDUSTFLUIDS; n++) {
    //   std::cout << *(Stokes_number + n) << std::endl;
    //   std::cout << *(Hratio + n) << std::endl;
    // }
    return;
  }
}

// read user-defined mesh data
void MeshBlock::InitUserMeshBlockData(ParameterInput *pin){
  // record the restart time
  t_restart = pmy_mesh->time;

  if(RT_ANA){
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

  // Allocate user output variables
  // TBD: the output variables need to be assigned
  AllocateUserOutputVariables(24);
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

  //[25.06.29]zxl: allocate the parameters for the other dust 
  SetUserOutputVariableName(14,"st1");
  SetUserOutputVariableName(15,"m_p1");
  SetUserOutputVariableName(16,"s_p1");
  // output the ice flux of the second population
  SetUserOutputVariableName(17,"flx_ice1_x1");
  SetUserOutputVariableName(18,"flx_ice1_x2");

  // output the flux for silicates. 
  SetUserOutputVariableName(19,"flx_sil_x1");
  SetUserOutputVariableName(20,"flx_sil_x2");
  SetUserOutputVariableName(21,"flx_sil1_x1");
  SetUserOutputVariableName(22,"flx_sil1_x2");

  SetUserOutputVariableName(23,"t_relax");
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Initializes Keplerian accretion disk.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // Compute KELVIN from code units (same calculation as in PhaseChange module)
  KELVIN = SQR(pmy_mesh->punit->code_velocity_cgs) / (Constants::k_boltzmann_cgs / Constants::hydrogen_mass_cgs);

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

        // store all dustfluid volume density to calculate stopping time.
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

            // std::cout << "dust_id = " << dust_id << ", den_dust = " << den_dust << ", initial_D2G = " << initial_D2G[dust_id] << std::endl;

            pdustfluids->df_cons(rho_id, k, j, i) = den_dust;
            if(!pdustfluids->istracer[dust_id]){
              pdustfluids->df_cons(v1_id,  k, j, i) = 0.0;
              pdustfluids->df_cons(v2_id,  k, j, i) = 0.0;
              pdustfluids->df_cons(v3_id,  k, j, i) = den_dust*vel_dust_phi;
            }else{
              pdustfluids->df_cons(v1_id,  k, j, i) = den_dust*v_acc_gas*std::sin(x2);
              pdustfluids->df_cons(v2_id,  k, j, i) = den_dust*v_acc_gas*std::cos(x2);
              pdustfluids->df_cons(v3_id,  k, j, i) = den_dust*vel_dust_phi;
            }
					}

          if(pmy_mesh->time > dust_start_injection){

            Real gas_nu = nu_alpha* std::pow(rad/r0,nu_slope);
            for (int pop_id=0; pop_id<NDUSTBIN; ++pop_id) {

              // decide which population is considered here 
              // int pop_id = dust_id/NCOMPOS; // this should be integer

              //[25.07.02]zxl: the m_p_array and s_p_array are changed to be 4D, with the new dimension means the different dust bins.
              Real &m_p = pdustfluids->m_p_array(pop_id,k,j,i);
              Real &s_p = pdustfluids->s_p_array(pop_id,k,j,i);

              AthenaArray<Real> density_slice; 
              density_slice.InitWithShallowSlice(rho_dustfluid_array, 1, pop_id*NCOMPOS, NCOMPOS);

              Real t_stop = Get_stopping_time(density_slice,rho_dustfluid_array(NDUSTFLUIDS-NVapor), Tem,gas_dens,m_p0[pop_id], rad, m_p, s_p);

              // ===== upper limit, St can be extremely high for upper layer =====
              t_stop = (t_stop > 0.5) ? 0.5 : t_stop;
              if(std::fabs(z) > 3.0 *std::pow((rad/r0), 1.5 + qvalue/2)){
                t_stop = 1.e-4;
              }

              // Assign the stopping time and diffusivity to the dustfluids 
              for (int dust_id=pop_id*NCOMPOS; dust_id<(pop_id+1)*NCOMPOS; ++dust_id) {
                Real &diffusivity = pdustfluids->nu_dustfluids_array(dust_id, k, j, i);
                Real &st_time = pdustfluids->stopping_time_array(dust_id, k,j,i);

                if(!pdustfluids->istracer[dust_id]){  
                  st_time = t_stop;
                  // apply st floor 
                  st_time = (st_time > 1.e-8) ? st_time : 1.e-8;
                }
                //calculate diffusivity
                Real taus_peb = t_stop *omega_dyn;

                if(!pdustfluids->istracer[dust_id]){
                  diffusivity = gas_nu/(1.+SQR(taus_peb));
                }else{
                  diffusivity = gas_nu;
                }
              }
            }
          } else {
            //apply the floor value. do not know whether this is needed.
            Real gas_nu = nu_alpha*std::pow(rad/r0,nu_slope);

            for (int pop_id = 0; pop_id <NDUSTBIN; ++pop_id){
              Real &m_p = pdustfluids->m_p_array(pop_id, k,j,i);
              Real &s_p = pdustfluids->s_p_array(pop_id, k,j,i);

              m_p = m_p0[pop_id];
              s_p = m_p/(FOUR_3RD*PI*1.5);
              s_p = std::pow(s_p, ONE_3RD);
            }
            for (int dust_id = 0; dust_id <NDUSTFLUIDS; ++dust_id){

              Real &diffusivity = pdustfluids->nu_dustfluids_array(dust_id, k, j, i);
              Real &st_time = pdustfluids->stopping_time_array(dust_id, k,j,i);

              if(!pdustfluids->istracer[dust_id]){  
                st_time = 1.e-8;
              }
              //calculate diffusivity
              Real taus_peb = 1.e-8 *omega_dyn;

              if(!pdustfluids->istracer[dust_id]){
                diffusivity = gas_nu/(1.+SQR(taus_peb));
              }else{
                diffusivity = gas_nu;
              }

            }
          }

          // Real t_stop = Get_stopping_time(rho_dustfluid_array,Tem,gas_dens,m_p0, rad,m_p,s_p);
            
            // Real taus_peb= t_stop*omega_dyn;
            // if(!pdustfluids->istracer[dust_id]){
            //   diffusivity = gas_nu/(1.+SQR(taus_peb));
            // }else{
            //   diffusivity = gas_nu;
            // }

            // check the size of rho_dustfluid_array
            // std::cout << rho_dustfluid_array.GetDim1() << std::endl; 
            // std::cout << rho_dustfluid_array.GetDim2() << std::endl; 
            // std::cout << rho_dustfluid_array.GetDim3() << std::endl; 
            // check the taus 
            // for (int n=0; n<NDUSTFLUIDS; ++n) {
            //   std::cout << "dust_id = " << n << ", stopping_time = " << 
            //     pdustfluids->stopping_time_array(n, k,j,i) << std::endl;
            // }
          // }

				}
      }
    }
  }

  // initialize tem_active. Random profile, just as initial guess of hydrostatic solution.
  int istart = is; int iend = ie; int NGHOST_ti = NGHOST;
  if (pbval->block_bcs[BoundaryFace::inner_x1] == BoundaryFlag::user){
    istart = is-NGHOST;
    NGHOST_ti = 0;
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
  Real Mdot_gas = -M_dot_g*CONST_Msun/CONST_yr / 2.0; // in cgs
  Mdot_gas /= (SQR(UNIT_LENGTH)*UNIT_LENGTH*UNIT_DENSITY/UNIT_T);

  #ifdef RT_ANA
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
  /// TWO-Stream RT scheme ///////////////////////////////////////////////
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
              // [25.06.30]zxl: change the total rho_peb
              Real rho_peb = 0.0;
              for (int n=0; n<NDUSTFLUIDS - NVapor; ++n) {
                rho_peb += pmb->pdustfluids->df_prim(4*n,k,j,i);
              }
              // Real rho_peb = pmb->pdustfluids->df_prim(0,k,j,i) + pmb->pdustfluids->df_prim(4,k,j,i) + pmb->pdustfluids->df_prim(8,k,j,i) + pmb->pdustfluids->df_prim(12,k,j,i);
              Real fv = pmb->pdustfluids->df_prim(4*(NDUSTFLUIDS - NVapor),k,j,i) / rho_gas;
              Real kappa = Get_kappa(rho_peb/rho_gas, fv);

              // store rhokappa
              rhokappa(tk,tj,ti) = rho_gas*kappa;
              rhokappa(tk,0,ti) = 0.0;
              rhokappa(tk,1,ti) = 0.0;
              // direct summation
              if (tj<=1000){
                tau_vi_f(tk,tj+1,ti) = rhokappa(tk,tj,ti)*dx2 + tau_vi_f(tk,tj,ti);
                tau_vi(tk,tj,ti) = 0.5*(tau_vi_f(tk,tj,ti) + tau_vi_f(tk,tj+1,ti));
              }else{ 
              // Simpson's 1/3 rule:
                tau_vi(tk,tj,ti) = tau_vi(tk,tj-2,ti) + dx2/3.0*(rhokappa(tk,tj-2,ti) + 4.0*rhokappa(tk,tj-1,ti) + rhokappa(tk,tj,ti)); // for some reason, here the dx2 should be devided by 2 rather than 3.
              }
              // std::cout << tau_vi_f(tk,tj,ti) <<std::endl; 
              // std::cout << tau_vi_f(tk,tj+1,ti) <<std::endl;
              // std::cout << tau_vi(tk,tj,ti)<< std::endl;
              // std::cout << tauvi<< std::endl;
              // std::cout << rhokappa(tk,tj,ti) <<std::endl;
              // std::cout << rhokappa(tk,tj-2,ti) <<std::endl;
              // std::cout << rhokappa(tk,tj-1,ti) << std::endl;
              // std::cout <<dx2<<std::endl;

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
      Real UNIT_ERG = UNIT_DENSITY *UNIT_LENGTH *SQR(UNIT_VELOCITY*UNIT_LENGTH);
      Real UNIT_FLX = UNIT_DENSITY * UNIT_VELOCITY * SQR(UNIT_VELOCITY);
      
      AthenaArray<Real> qmax, q_vis_pre, q_irr_pre; 
      qmax.NewAthenaArray(mesh_size.nx1 + 2*NGHOST);
      q_vis_pre.NewAthenaArray(mesh_size.nx1 + 2*NGHOST);
      q_irr_pre.NewAthenaArray(mesh_size.nx1 + 2*NGHOST);
      AthenaArray<Real> delta, dd; 
      delta.NewAthenaArray( mesh_size.nx2 + NGHOST - 1, mesh_size.nx1 + 2*NGHOST);
      dd.NewAthenaArray(mesh_size.nx2 + NGHOST, mesh_size.nx1 + 2*NGHOST);


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
                // TBD: this need to be checked
                Real dx2m1 = pmb->pcoord->dx2f(j-1 )* pmb->pcoord->x1v(i);
                Real dx2m2 = pmb->pcoord->dx2f(j-2 )* pmb->pcoord->x1v(i);
                Real rho_gas = pmb->phydro->w(IDN,k,j,i);
                //[25.06.30]zxl: change the total rho_peb to consider the two-pop
                Real rho_peb = 0.0; // always do the initialization!!
                for (int n=0; n<NDUSTFLUIDS - NVapor; ++n) {
                  rho_peb += pmb->pdustfluids->df_prim(4*n,k,j,i);
                }

                Real fv = pmb->pdustfluids->df_prim(4*(NDUSTFLUIDS - NVapor),k,j,i) / rho_gas;             
                Real H_gas = std::pow(rad/r0, qvalue/2.0 + 1.5);
                Real cos_inc = (qvalue/2.0 + 0.5) * (4.0*H_gas/rad); // cosine of the incident angle to disk surface

                // accquire tau_(-inf):
                Real tau_inf = 2.0 * tau_vi(tk, mesh_size.nx2+NGHOST-1, ti);
                // Real tau_inf = 2.0 * tau_vi_f(tk, mesh_size.nx2+NGHOST, ti);
                Real E0 = L_star*CONST_Lsun/ (UNIT_ERG/UNIT_T) / (8.0*PI*SQR(rad));
                Real omega = std::sqrt(gm0/std::pow(rad,3.0));
                Real q_irr = E0* rhokappa(tk,tj,ti)*(std::exp(-tau_vi(tk,tj,ti)/cos_inc) + std::exp(-(tau_inf - tau_vi(tk,tj,ti))/cos_inc));
                Real nu_gas = pmb->phydro->hdif.nu(HydroDiffusion::DiffProcess::alpha, k, j, i);
                Real q_vis = 9.0/4.0 *rho_gas* (1.0-fv)* SQR(omega) *nu_gas; // only include viscous heating by H-He
                ////////////////////
                // including latent heat
                if (Latent_heat_flag){
                  q_z(tk,tj,ti) = q_irr + q_vis + pmb->pdustfluids->q_latent(k,j,i) + pmb->pdustfluids->q_diff(k,j,i);
                  // std::cout << "latent heat is included in this simulation." << std::endl;
                } else {
                  q_z(tk,tj,ti) = q_irr + q_vis +pmb -> pdustfluids -> q_diff (k,j,i);
                  // std::cout << "latent heat is not included in this simulation." << std::endl;
                }
  
                if (tj > 0) {
                  delta(tj-1,ti) = (q_z(tk, tj, ti) - q_z(tk, tj-1, ti))/dx2;

                  if (tj==1) {
                    dd(tj-1, ti) = delta(tj-1, ti);
                  } else { 
                      
                    if (delta(tj-1, ti) * delta(tj-2, ti) < 0.0){
                      dd(tj-1, ti) = 0.0;
                    } else {
                      Real w1 = 2.0*dx2m1 + dx2m2; 
                      Real w2 = dx2m1 + 2.0*dx2m2;
                      dd(tj-1, ti) = (w1+w2)/(w1/delta(tj-2,ti) + w2/delta(tj-1,ti));
                    }

                  } 
                    
                  if (tj == (mesh_size.nx2 + NGHOST -1)) {
                    dd(tj, ti) = delta(tj-1, ti);
                  }
                }

                // if (pmb->pdustfluids->q_diff(k,j,i)<0){
                //   std::cout << "q_diff < 0" <<std::endl;
                //   std::cout << "q_diff = " << pmb->pdustfluids->q_diff(k,j,i)<<std::endl;
                //   std::cout << "tk,tj,ti=" << tk <<' '<<tj <<' '<<ti <<std::endl;
                // }
                // q_z(tk,tj,ti) = q_irr + q_vis + pmb->pdustfluids->q_latent(k,j,i) + pmb->pdustfluids->q_diff(k,j,i);
                
                // if (q_z(tk,tj,ti)<0){
                //   std::cout << q_z(tk,tj,ti)<< std::endl;
                //   std::cout <<"qdiff="<<pmb -> pdustfluids->q_diff(k,j,i)<<std::endl;
                //   std::cout << "q_vis=" << q_vis <<std::endl;
                //   std::cout << "tk, tj, ti = " <<tk <<','<<tj<< ','<<ti <<std::endl;
                // }
                //
                //
                // if (q_int(tk,tj,ti)<q_int(tk,tj-1,ti)){
                //   std::cout <<"tj="<<tj<<std::endl;
                //   std::cout <<"ti="<<ti<<std::endl;
                //   std::cout <<"tk="<<tk<<std::endl;
                //   std::cout <<"qz="<<q_z(tk,tj,ti)<<std::endl;
                //   std::cout <<"qz-1="<<q_z(tk,tj-1,ti)<<std::endl;
                //   std::cout <<"qz-2="<<q_z(tk,tj-2,ti)<<std::endl;
                //   std::cout <<"qz-3="<<q_z(tk,tj-3,ti)<<std::endl;
                //
                //   std::cout <<"q_diff="<<pmb -> pdustfluids -> q_diff (k,j,i)<<std::endl;
                //   std::cout <<"q_diff-1="<<pmb -> pdustfluids -> q_diff (k,j-1,i)<<std::endl;
                //
                //   std::cout <<"q_vis = "<<q_vis<<std::endl;
                //   std::cout <<"q_vis-1 = "<<q_vis_pre(ti)<<std::endl;
                //
                //   std::cout <<"q_irr = "<<q_irr<<std::endl;
                //   std::cout <<"q_irr-1 = "<<q_irr_pre(ti)<<std::endl;
                //
                //   std::cout <<"qint="<<q_int(tk,tj,ti)<<std::endl;
                //   std::cout <<"qint-1="<<q_int(tk,tj-1,ti)<<std::endl;
                //   std::cout <<"qint-2="<<q_int(tk,tj-2,ti)<<std::endl;
                //   std::cout <<"qint-3="<<q_int(tk,tj-3,ti)<<std::endl;
                //
                //   std::cout <<"dx2 = " << dx2 << std::endl;
                // }
                  //
                // q_vis_pre(ti) = q_vis; 
                // q_irr_pre(ti) = q_irr;
                // if (tj>=mesh_size.nx2+NGHOST-1){
                //   q_int_f(tk,tj+1, ti) = q_z(tk,tj,ti)*dx2 + q_int_f(tk,tj,ti);
                //   // std::cout <<"q_int_f = "<< q_int_f(tk,tj+1, ti) << std::endl;
                //   // std::cout <<"qmax = " << qmax(ti) <<std::endl;
                //   // q_int_f(tk,tj+1,ti) = std::max(q_int_f(tk,tj+1,ti), qmax(ti));
                //   // q_int_f(tk,tj+1,ti) = std::max(q_int_f(tk,tj+1,ti), qmax(ti));
                //   // if (std::isnan(q_int(tk,tj,ti))){
                //   //   std::cout <<"tj="<<tj<<std::endl;
                //   //   std::cout <<"ti="<<ti<<std::endl;
                //   //   std::cout <<"tk="<<tk<<std::endl;
                //   //   std::cout <<"qintf="<<q_int_f(tk,tj+1,ti)<<std::endl;
                //   //   std::cout <<"qz="<<q_z(tk,tj,ti)<<std::endl;
                //   //   std::cout <<"qz-1="<<q_z(tk,tj-1,ti)<<std::endl;
                //   //   std::cout <<"qz-2="<<q_z(tk,tj-2,ti)<<std::endl;
                //   //   std::cout <<"qintf-1="<<qintf<<std::endl;
                //   //   std::cout <<"qint="<<q_int(tk,tj,ti)<<std::endl;
                //   //   std::cout <<"qint-1="<<q_int(tk,tj-1,ti)<<std::endl;
                //   //   std::cout <<"qint-2="<<q_int(tk,tj-2,ti)<<std::endl;
                //   //   std::cout <<"qint-3="<<q_int(tk,tj-3,ti)<<std::endl;
                //   //   std::cout <<"qdiff-1="<<pmb -> pdustfluids->q_diff(k,j-1,i)<<std::endl;
                //   //
                //   // }
                // }
                //
                // std::cout << q_int_f(tk,tj,ti)<<std::endl;
                // std::cout << q_int_f(tk,tj+1,ti)<<std::endl;
                // std::cout<< std::endl;

                // Trapezoidal rule:
                // q_int(tk,tj,ti) = 0.5*dx2*(q_z(tk,tj-1,ti) + q_z(tk,tj,ti)) + q_int(tk,tj-1,ti);
                // std::cout << q_int(tk, tj, ti) << std::endl;
                // std::cout << qq << std::endl;
                // std::cout << q_z(tk, tj, ti) << std::endl;
                // std::cout << q_int(tk,tj,ti) << std::endl;
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
                // TBD: this need to be checked
                Real dx2m1 = pmb->pcoord->dx2f(j-1)* pmb->pcoord->x1v(i);

                // Real aa = (dd(tj, ti) + dd (tj+1, ti) - 2.0*dx2)/SQR(dx2); 
                // Real bb = 3*dx2 - 2*dd(tj,ti) +dd(tj+1,ti); 
                // Real cc = dd(tj,ti);
                //
                // q_int_f(tk,tj+1,ti) = q_int_f(tk,tj,ti) + q_z(tk,tj,ti)*dx2 + cc/2*SQR(dx2) + bb/3*std::pow(dx2, 3) + aa/4*std::pow(dx2, 4);
                //
                q_int_f (tk, tj+1, ti) = q_int_f(tk,tj,ti) + dx2*(q_z(tk,tj,ti) + q_z(tk,tj+1,ti)/2 - dx2/12*(dd(tk,tj+1,ti) - dd(tk,tj,ti)));
                q_int  (tk,tj,ti)   = 0.5*(q_int_f(tk,tj,ti) + q_int_f(tk,tj+1,ti));
                qmax(ti) = std::max(q_int_f(tk,tj+1, ti), q_int_f(tk,tj,ti));

                // if (tj >= mesh_size.nx2 + NGHOST - 1) {
                //   q_int_f(tk,tj+1, ti) = std::max(qmax(ti), q_int_f(tk,tj+1,ti)); 
                // }

                // Real qintfsimp, qintsimp;
                // if (tj<=1000){
                //   // linear interpolation
                //   qintfsimp = q_z(tk,tj,ti)*dx2 + q_int_f(tk,tj,ti);
                // }else {
                //   // Simpson's 1/3 rule:
                //   q_int(tk,tj,ti)= q_int(tk,tj-2,ti) + dx2/3.0*(q_z(tk,tj-2,ti) + 4.0*q_z(tk,tj-1,ti) + q_z(tk,tj,ti));
                //   qintfsimp = 0.5*(q_int(tk,tj-1,ti) + q_int(tk,tj,ti));
                // }
                // // compare the results: 
                // PRINT(qintfsimp);
                // PRINT(q_int_f(tk,tj+1,ti));
                // PRINT(tk);
                // PRINT(tj);
                // PRINT(ti);
                //

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
                
                if (tj<=1000){
                // linear interpolation
                // [25.06.30]zxl: get the tau_eff of the cell by average the value of the faces. 
                // this rhokappa is got at the step 0.  
                  tau_eff_f(tk,tj+1,ti) = rhokappa(tk,tj,ti)*(1.0 - q_int(tk,tj,ti)/F_inf) * dx2 + tau_eff_f(tk,tj,ti);
                  tau_eff(tk,tj,ti) = 0.5*(tau_eff_f(tk,tj+1,ti) + tau_eff_f(tk,tj,ti));
                }else{
                // Simpson's 1/3 rule:
                  tau_eff(tk,tj,ti) = tau_eff(tk,tj-2,ti) + dx2/3.0*(rhokappa(tk,tj-2,ti)*(1.0 - q_int(tk,tj-2,ti)/F_inf) + 4.0*rhokappa(tk,tj-1,ti)*(1.0 - q_int(tk,tj-1,ti)/F_inf) + rhokappa(tk,tj,ti)*(1.0 - q_int(tk,tj,ti)/F_inf));
                }
                // if ((1 - q_int(tk,tj,ti)/F_inf)<0){
                //   std::cout << "F_z<0" <<std::endl;
                //   std::cout << q_int(tk,tj,ti)<<std::endl;
                //   std::cout <<F_inf <<std::endl;
                // }
                //
                // Divide the integration area into 2 parts, since flux is defined in the face, rho, kappa are defined on the cell
                // Real rhokappa_f = (rhokappa(tk,tj,ti) + rhokappa(tk,tj+1,ti)) /2; 
                // tau_eff (tk, tj, ti) =tau_eff(tk, tj-1, ti) + 
                //                       dx2/4*( rhokappa(tk,tj-1,ti)*(1.0 - q_int(tk,tj-1,ti)/F_inf)+ 2*rhokappa_f * (1.0 - q_int_f(tk,tj, ti)/F_inf) + 
                //                       rhokappa(tk,tj,ti)*(1.0 - q_int(tk,tj,ti)/F_inf) );
                
                if (tau_eff(tk,tj,ti)<0){
                  std::cout << "tau_eff <0!" <<std::endl;
                  std::cout << "tk,tj,ti = "<<tk << "," << tj <<"," << ti <<std::endl;
                  std::cout << "F_inf = " << F_inf <<std::endl; 
                  std::cout << "q_int-2 = " << q_int(tk,tj-2,ti) <<std::endl;
                  std::cout << "q_int-1 = " << q_int(tk,tj-1,ti) <<std::endl;
                  std::cout << "q_int = " << q_int(tk,tj,ti) <<std::endl;
                  std::cout << "q_diff = "<< pmb -> pdustfluids -> q_diff (tk,tj,ti)<< std::endl;
                }
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
      Real dTx_old; 
      Real dTx_new;
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
                Real rho_peb = 0.0;
                for (int n=0; n<NDUSTFLUIDS - NVapor; ++n) {
                  rho_peb += pmb->pdustfluids->df_prim(4*n,k,j,i);
                }
                
                Real fv = pmb->pdustfluids->df_prim(4*(NDUSTFLUIDS - NVapor),k,j,i) / rho_gas;
                // Real kappa = Get_kappa(rho_peb/rho_gas, fv);
                Real F_inf = q_int_f(tk, mesh_size.nx2+NGHOST, ti);
                Real T_eff = std::pow(F_inf/(CONST_sigma/UNIT_FLX), 0.25);
                
                Real &Tem = pmb->phydro->Tem(k, j, i);
                // Real Tem0 = Tem;
                Real Tem_eq = 0.75*tau_eff(tk,tj,ti) + std::sqrt(3.0)/4.0 +  q_z(tk,tj,ti)/(4.0*rhokappa(tk,tj,ti)*F_inf);
                Tem_eq = std::pow(Tem_eq, 0.25)*T_eff;

                // debug cell
                if(std::isnan(Tem)){
                  std::cout << "Tem = " << Tem << std::endl;
                  std::cout << "Teff = " <<T_eff << std::endl;
                  std::cout << "rho_kappa = " <<rhokappa(tk,tj,ti) <<std::endl;
                  std::cout << "F_inf = " << F_inf << std::endl;
                  std::cout << "ti, tj = " << ti << " " << tj << std::endl;
                  std::cout << "tau_eff = " << tau_eff(tk,tj,ti) << std::endl;
                  std::cout << "q_z = " << q_z(tk,tj,ti) << std::endl;
                  std::cout << "q_int = " << q_int (tk,tj,ti) <<std::endl;
                  quick_exit(1);
                }

                // beta-cooling //////////////////////
                Real omega_dyn = std::sqrt(gm0/(SQR(rad)*rad));

                Real t_cool;
                // get the relaxation time locally
                if (time < t_iterate){
                  // when do iteration for T, rho, don't need long relaxing timescale
                  t_cool = 50.0/ omega_dyn;
                } else{
                  // Lin & Youdin 2015:
                  Real C_v = ((0.71*2.5 / mu_H2 + 0.29*1.5/ mu_He)*(1.0-fv) + 3.0/mu_z*fv) /KELVIN;
                  // Real l_char = rad * l_cool; // characteristic pertubation scale
                  Real l_char = rad * std::pow(rad/r0, qvalue/2.0 + 1.5);//Get_h(rad);
                  Real kappa_dg_R = Get_kappa(0.0, fv);
                  Real UNIT_SB = UNIT_DENSITY * UNIT_VELOCITY * SQR(UNIT_VELOCITY); // flx/(K^4)
                  Real tau_relax = C_v * omega_dyn / (16.0 * CONST_sigma/UNIT_SB * std::pow(Tem,3.0)) * (1.0 /kappa_dg_R + 3.0*SQR(l_char*rho_gas)*kappa_dg_R);
                  Real H_gas = std::pow(rad/r0, qvalue/2.0 + 1.5);
                  // t_cool = Get_thermal_relaxation_time(omega_dyn, Tem, rhokappa(tk,tj,ti), rho_peb, rho_gas, H_gas, z, rad);
                  // t_cool /= omega_dyn;
                  t_cool = tau_relax/omega_dyn;

                }
                // Real t_cool = beta / omega_dyn;
                Real dT = (Tem_eq - Tem)/t_cool*dt;
                dT = (std::fabs(dT) < std::fabs(Tem_eq-Tem)) ? dT : (Tem_eq-Tem);
                Tem += dT;

                //////////////////////////////////////
                pmb ->user_out_var(23, k,j,i) = t_cool;

                // // calculate the tempeture gradient
                // if (ti < 163){
                //   dTx_old = Tem_active(tk, tj, ti+1) - Tem_active(tk,tj,ti);
                // }
                //
                // if (dTx_old*dTx_new < 0.0){
                //   // if overshot happens, then force it to go to thermal equilibrium between the adjacent grids
                //   Tem = Tem_active(tk, tj, ti-1);
                // }
                //
                // if (ti>0){
                //   dTx_new = Tem - Tem_active(tk,tj,ti-1);
                // }
                

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

  // reset q_latent, q_diff after using them in RT scheme;
  for (int bn=0; bn<nblocal; ++bn) {
    MeshBlock *pmb = my_blocks(bn);
    LogicalLocation &loc = pmb->loc;
    if (loc.level == root_level) { // root level
      for (int k=pmb->ks; k<=pmb->ke; k++) {
        for (int j=pmb->js; j<=pmb->je; j++) {
          for (int i=pmb->is; i<=pmb->ie; i++) {
            // reset q_latent and copy its value to dfv_dt:
            if (Latent_heat_flag) {
              pmb->user_out_var(4,k,j,i) = pmb->pdustfluids->q_latent(k,j,i);
            } else {
              pmb->user_out_var(4,k,j,i) = 0.0;
            }
            // pmb->user_out_var(4,k,j,i) = pmb->pdustfluids->q_latent(k,j,i);
            //[25.07.01]zxl: why this latent q is not 0?

            pmb->user_out_var(10,k,j,i) = pmb->pdustfluids->q_diff(k,j,i);

            pmb->pdustfluids->q_latent(k,j,i) = 0.0;
            pmb->pdustfluids->q_diff(k,j,i) = 0.0;
          }
        }
      }
    }
  }
  ////////////////////////////////////////////////////////////////
  // get hydrostatic solution for the disk's density structure //
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
            Real fv = pmb->pdustfluids->df_prim(4*(NDUSTFLUIDS - NVapor),k,j,i) / pmb->phydro->w(IDN,k,j,i);
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
            Real nu_gas = nu_alpha*std::pow(rad/r0, nu_slope);
            
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
              rho_z(tk,tj,ti) = (pmb->phydro->w(IDN,k,j,pmb->is) -pmb->pdustfluids->df_prim(4*(NDUSTFLUIDS - NVapor),k,j,pmb->is)) * SQR(x1_active)/SQR(x1_ghost);
              // rho_z(tk,tj,ti) = std::exp(2.0*std::log(pmb->phydro->w(IDN, k, j, i+1)) - std::log(pmb->phydro->w(IDN, k, j, i+2)));
            }
            // outer bc
            // if(time > t_iterate and pmb->pbval->block_bcs[BoundaryFace::outer_x1] == BoundaryFlag::user and x1 > x1max){
            //   Real x1_active = pmb->pcoord->x1v(pmb->ie);
            //   Real x1_ghost = pmb->pcoord->x1v(i);
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

  // step 8: Normalize the surface density at inner boundary based on expected surface density
  // Get the surface density at inner boundary
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
            Real nu_gas = nu_alpha*std::pow(rad/r0, nu_slope);
            Real sigma0 = std::fabs(Mdot_gas)/(3.0*PI*nu_gas);
            rho_z(tk,tj,ti) *= sigma0/sigma_gas_est(i);
          }
        }
      }
    }
  }


  // Step 9: Get the total mass flux at outer boundary;
  // TBD: [25.07.10]zxl: I think we only need to do all of these when time is reached.
  Real Mdot_est = 0.0;
  Real Mdot_peb_est[NDUSTFLUIDS-NVapor] = {0.0};

  // this is the mass fluxes of ice
  Real Mdot_peb[NDUSTFLUIDS-NVapor] = {0.0};
  for (int pop_id=0; pop_id<NDUSTBIN; ++pop_id) {
    // [25.07.07]zxl: have tried to be very general, but a little bit tricky here. let me do it tomorrow.
    Mdot_peb[pop_id*NCOMPOS] = f_ICE_inter0*p2g_flux[pop_id]*Mdot_gas;
    Mdot_peb[pop_id*NCOMPOS + 1] = (1-f_ICE_inter0)*p2g_flux[pop_id]*Mdot_gas;

  }

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
          // Real gas_rho0 = DenProfileCyl_gas_fv_T(rad_active, phi_active, z_active,0.0,Tem_active);
          Real gas_rho0 = pmb->pmy_mesh->ruser_mesh_data[7](tk, tj, ti);
          Real vr0 = Vr_ProfileCyl_gas_fv_T(rad_ghost, phi_ghost, z_ghost, 0.0, Tem_mid_ghost);
          // assign the gas specific flux and calculate Mdot

          pmb->phydro->flux0_R(j) = gas_rho0*(vr0*std::sin(x2));
          Mdot_est += pmb->phydro->flux0_R(j)*ds_ghost;

          ///////////////////////////////////////////////////////////
          // pebble flux:
          Real v_drift_peb = -0.004; // random value, doesn't matter since no backreaction
          Real dust_vel1 = v_drift_peb*std::sin(x2);

          for (int pop_id=0; pop_id<NDUSTBIN; ++pop_id) {
            //[25.07.03]zxl: the 2.0 factor is not general, so I changed it
            //Real sigma_peb = 2.0*Mdot_peb[n] /(v_drift_peb * 2.0*PI*rad_active);

            // this is the sum of mass flux of one dust bin (dust population)
            Real Mdot_peb_pop = 0.0;
            for (int dust_id = pop_id*NCOMPOS; dust_id < (pop_id+1) * NCOMPOS; ++dust_id) {
              Mdot_peb_pop += Mdot_peb[dust_id];
            }
            // Real Mdot_peb_pop = std::accumulate(Mdot_peb[pop_id*NCOMPOS], Mdot_peb[(pop_id+1)*NCOMPOS], 0.0);
            
            Real sigma_peb = Mdot_peb_pop /(v_drift_peb * 2.0*PI*rad_active);

            Real cs0 = std::sqrt(Tem_mid/(mu_xy*KELVIN));
            Real h_peb = Hratio[pop_id*NCOMPOS]* cs0/std::sqrt(gm0/(rad_active*SQR(rad_active)));
            Real rho_peb_mid = sigma_peb / (sqrt(2.0*PI)*h_peb);
            Real dust_rho = rho_peb_mid* std::exp(-0.5*SQR(z_active/h_peb));
            dust_rho = (dust_rho > dffloor) ? dust_rho : dffloor;

            //[25.07.07]zxl: assign the dust fluxes to the dustfluids
            for (int dust_id = pop_id*NCOMPOS; dust_id < (pop_id+1)*NCOMPOS; ++dust_id) {
              pmb->pdustfluids->flux0_R_dust(dust_id, j) = dust_rho*dust_vel1;
              Mdot_peb_est[dust_id] += pmb->pdustfluids->flux0_R_dust(dust_id, j)*ds_ghost;
              // std::cout << "Mdot_peb_" << dust_id <<" = " << Mdot_peb[dust_id] << std::endl;
              // std::cout << "flux_" << dust_id << " = " << pmb->pdustfluids->flux0_R_dust(dust_id, j) << std::endl;
            }
            // pmb->pdustfluids->flux0_R_dust(j) = dust_rho*dust_vel1;
            // Mdot_peb_est += pmb->pdustfluids->flux0_R_dust(j)*ds_ghost;
          }

        }
      }
    }
  }

  Real flx_ratio = Mdot_gas / Mdot_est;

  Real flx_peb_ratio [NDUSTFLUIDS - NVapor] ; 
  for (int dust_id = 0; dust_id <NDUSTFLUIDS-NVapor; ++dust_id){ 
    flx_peb_ratio[dust_id] = Mdot_peb[dust_id] / Mdot_peb_est[dust_id];
  }

  for (int bn=0; bn<nblocal; ++bn) {
    MeshBlock *pmb = my_blocks(bn);
    if (pmb->pbval->block_bcs[BoundaryFace::outer_x1] == BoundaryFlag::user){

      for (int k=pmb->ks; k<=pmb->ke; k++) {
        for (int j=pmb->js; j<=pmb->je; j++) {

          for (int dust_id =0; dust_id < NDUSTFLUIDS-NVapor; ++dust_id) {
            pmb->phydro->flux0_R(dust_id, j) *= flx_ratio;
            pmb->pdustfluids->flux0_R_dust(dust_id, j) *= flx_peb_ratio[dust_id];

            if(time < dust_start_injection){
              pmb->pdustfluids->flux0_R_dust(dust_id, j) = 0.0;
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

  // wave damping at disk upper boundary
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

        // [25.07.04]zxl: make this copy processes all to a function, hopefully this will be correct.
        // copy velocities from gas to vapor
        copy_velocities(0, NDUSTFLUIDS-NVapor, k, j, i, this, "gas"); 
        // copy velocities from silicate to ice 
        copy_velocities(1, 0, k, j, i, this, "dust");
        copy_velocities(3, 2, k, j, i, this, "dust");

        // this is to calculate the temperature for the ghost cells defining boundary conditions
        if(i < is or i > ie or j < js or j > je){
          Real E_kg = 0.5*(SQR(gas_vel1) + SQR(gas_vel2) + SQR(gas_vel3))*gas_den;
          Real rhoe_g = phydro->u(IEN,k,j,i) - E_kg;
          Real fv = pdustfluids->df_prim(4*(NDUSTFLUIDS-NVapor), k, j, i)/gas_den;
          Real T_from_erg = Get_T_rhoe_g(rhoe_g,gas_den,fv);
          
          phydro->Tem(k, j, i) = T_from_erg;          
        }

        // get stopping time and diffusivity:
        Real rad, phi, z;
        GetCylCoord(pcoord, rad, phi, z, i, j, k);
        AthenaArray<Real> rho_dustfluid_array;
        Real omega_dyn = std::sqrt(gm0/std::pow(rad,3.0));

        rho_dustfluid_array.NewAthenaArray(NDUSTFLUIDS);
        for (int n=0; n<NDUSTFLUIDS; ++n) {
          rho_dustfluid_array(n) = pdustfluids->df_prim(4*n, k, j, i);
        }
        // rho_dustfluid_array(0) = pdustfluids->df_prim(0, k, j, i);
        // rho_dustfluid_array(1) = pdustfluids->df_prim(4*1, k, j, i);
        // rho_dustfluid_array(2) = pdustfluids->df_prim(4*2, k, j, i);

        Real Tem = phydro->Tem(k, j, i);

        // [25.07.10]zxl: only calculate the properties of pebbles when start to inject them.
        if(time>dust_start_injection){
          for (int pop_id = 0; pop_id < NDUSTBIN; ++pop_id) {

            Real &m_p = pdustfluids->m_p_array(pop_id,k,j,i);
            Real &s_p = pdustfluids->s_p_array(pop_id,k,j,i);   

            AthenaArray<Real> density_slice;
            // here  we only get the density of non-tracers, no vapors
            density_slice.InitWithShallowSlice(rho_dustfluid_array, 1, pop_id*NCOMPOS, NCOMPOS); 
            // std::cout << pop_id <<std::endl;
            // std::cout << rho_dustfluid_array (0) <<std::endl;
            // std::cout << density_slice(0) <<std::endl;
            // if (pop_id ==1) {
            //   std::cout << std::endl;
            // }

            Real t_stop = Get_stopping_time(
            density_slice, rho_dustfluid_array(NDUSTFLUIDS - NVapor),Tem, gas_den, m_p0[pop_id], rad, m_p, s_p
            );

            // std::cout << "t_stop = " <<t_stop << std::endl;
            // std::cout << "m_p=" << m_p <<std::endl;
            // if (std::isnan(m_p)){
            //   std::cout <<"hello" <<std::endl;
            // }
            // std::this_thread::sleep_for(std::chrono::seconds(1));

            //check the st_time calculation: 
            // if (rad*UNIT_LENGTH/14959787070000.0 > 3.9){
            // std::cout << "silicate density" << density_slice(1) << std::endl;
            // std::cout << "temperature" << Tem << std::endl;
            // std::cout << "m_p0_" << pop_id <<'=' << m_p0[pop_id] <<std::endl;
            // std::cout << "location = " << rad << std::endl;
            // std::cout << "t_stop = " << t_stop <<std::endl;
            // std::cout << std::endl;}

            // ===== upper limit, St can be extremely high for upper layer =====
            t_stop = (t_stop > 0.5) ? 0.5 : t_stop;
            //[25.07.19]zxl: fatal error here, b/c of the } is not in the correct location, the stokes numbers are not assigned to the dustfluids....
            if(std::fabs(z) > 3.0 *std::pow((rad/r0), 1.5 + qvalue/2)){
              t_stop = 1.e-4;
            }

            // Assign the stopping time and diffusivity to the dustfluids 
            Real &gas_nu = phydro->hdif.nu(HydroDiffusion::DiffProcess::alpha, k, j, i);
            gas_nu = nu_alpha* std::pow(rad/r0, nu_slope); // fix the gas viscosity.

            for (int dust_id=pop_id*NCOMPOS; dust_id<(pop_id+1)*NCOMPOS; ++dust_id) {

              Real &diffusivity = pdustfluids->nu_dustfluids_array(dust_id, k, j, i);
              Real &st_time = pdustfluids->stopping_time_array(dust_id, k,j,i);

              if(!pdustfluids->istracer[dust_id]){
                st_time = t_stop;
                // std::cout << "time = " << time<< std::endl;
                // std::cout << "start to calculate the st" << std::endl;
                // std::cout << "st_time = " << st_time << std::endl;
                // apply st floor 
                st_time = (st_time > 1.e-8) ? st_time : 1.e-8;
              }

              //calculate diffusivity

              // artifical decay
              Real w_damp = 0.05*(x1max-x1min);
              Real f_decay_art = std::tanh(std::pow((rad-x1max)/w_damp ,2.0)); // outer bc decay
            
              Real taus_peb = t_stop*omega_dyn;

              if(!pdustfluids->istracer[dust_id]){
                diffusivity = gas_nu/(1.+SQR(taus_peb));
              }else{
                diffusivity = gas_nu*f_decay_art;
              }
            }

          // Real t_stop = Get_stopping_time(rho_dustfluid_array,Tem,gas_den,rad,m_p,s_p);

            
          }
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
        const Real &rho_v = pdustfluids->df_prim(4*(NDUSTFLUIDS - NVapor),k,j,i);
        Real E_kg = 0.5*(SQR(phydro->u(IM1,k,j,i)) + SQR(phydro->u(IM2,k,j,i)) + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
        Real rhoe_g = phydro->u(IEN,k,j,i) - E_kg; // the potential energy of gas
        Real fv;

        fv = rho_v/rho_g;
        Real T = Get_T_rhoe_g(rhoe_g,rho_g,fv);
        Real mu = Get_mu(fv);
        Real prs = rho_g*T/(mu*KELVIN);
        // output
        // note: the outpout 9 and 10 are defined in other places.
        user_out_var(0,k,j,i) = phydro->Tem(k,j,i);
        user_out_var(1,k,j,i) = pdustfluids->stopping_time_array(1,k,j,i);
        user_out_var(2,k,j,i) = pdustfluids->m_p_array(0,k,j,i);
        user_out_var(3,k,j,i) = pdustfluids->s_p_array(0,k,j,i);
        
        user_out_var(6,k,j,i) = pdustfluids->df_flux[X1DIR](0,k,j,i);
        user_out_var(7,k,j,i) = pdustfluids->df_flux[X2DIR](0,k,j,i);
        user_out_var(17,k,j,i)= pdustfluids->df_flux[X1DIR](8,k,j,i);
        user_out_var(18,k,j,i)= pdustfluids->df_flux[X2DIR](8,k,j,i);

        user_out_var(8,k,j,i) = pdustfluids->df_flux[X1DIR](4*(NDUSTFLUIDS - NVapor),k,j,i);
        user_out_var(9,k,j,i) = pdustfluids->df_flux[X2DIR](4*(NDUSTFLUIDS - NVapor),k,j,i);
        // 10 is q, output at other place
        user_out_var(11,k,j,i) = phydro->flux[X1DIR](IDN,k,j,i);
        user_out_var(12,k,j,i) = phydro->flux[X2DIR](IDN,k,j,i);
        user_out_var(13,k,j,i) = pdustfluids->nu_dustfluids_array(NDUSTFLUIDS - NVapor,k,j,i);
        user_out_var(14,k,j,i) = pdustfluids->stopping_time_array(3,k,j,i);
        user_out_var(15,k,j,i) = pdustfluids->m_p_array(1,k,j,i);
        user_out_var(16,k,j,i) = pdustfluids->s_p_array(1,k,j,i);
        user_out_var(19,k,j,i)= pdustfluids->df_flux[X1DIR](4,k,j,i);
        user_out_var(20,k,j,i)= pdustfluids->df_flux[X2DIR](4,k,j,i);
        user_out_var(21,k,j,i)= pdustfluids->df_flux[X1DIR](12,k,j,i);
        user_out_var(22,k,j,i)= pdustfluids->df_flux[X2DIR](12,k,j,i);
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
        // user_out_var(10,k,j,i) = pmy_mesh->ruser_mesh_data[1](tk, tj, ti);
        user_out_var(5,k,j,i) = pmy_mesh->ruser_mesh_data[2](tk, tj, ti);
      }
    }
  }

  return;
}

namespace { 
//----------------------------------------------------------------------------------------
//! transform to cylindrical coordinate

Real Get_T_rhoe_g(Real rhoe_g, Real rho_g, Real fv){
  // the equation 17 in Wang et al. 2023, with the phi_{z,i} = 0.
  Real fx = 1.0-fv; // He/H fraction
  Real e = rhoe_g/rho_g;
  Real T = e*KELVIN/(fx/mu_H2*0.71*2.5+fx/mu_He*0.29*1.5+fv/mu_z*3.0);
  return T;
}

Real Get_T_rhoe(Real rhoe, Real rho_g, AthenaArray<Real> rho_d_array, Real fv){
  // the euqation 18 in Wang et al. 2023.
  Real fx = 1.0-fv; // He/H fraction
  Real bottom = rho_g/KELVIN*(fx/mu_H2*0.71*2.5+fx/mu_He*0.29*1.5+fv/mu_z*3.0);
  for (int m = 0; m< NVOLATILE*NDUSTBIN; ++m){
    bottom += rho_d_array(m)*Cd_water;
  }
  Real T = rhoe/bottom;

  return T;
}

Real Get_rhomu_d(AthenaArray<Real> rho_d_array){
  // only apply for volatiles
  Real rhomu_d = 0.0;
  for (int m = 0; m< NVOLATILE*NDUSTBIN; ++m){
    rhomu_d += rho_d_array(m)*(-L_heat);
  }
  return rhomu_d;
}

Real Get_rhoe(Real rhoE_total, Real rho_g, Real E_kg, AthenaArray<Real> rho_d_array, AthenaArray<Real> E_kd_array){
  Real rhoe = 0.0;
  Real rhomu_d = Get_rhomu_d(rho_d_array); //chemical potential
  rhoe = rhoE_total- rho_g*E_kg- rhomu_d;

  for (int m = 0; m< NVOLATILE*NDUSTBIN; ++m){
    rhoe -= E_kd_array(m)*rho_d_array(m);
  }

  return rhoe;
}

Real Get_Z(Real rho_g, Real T){
  Real z,P_eq,rhoz,kB_mp;
  
  P_eq = P_eq0*exp(-T_a/T);
  kB_mp = 1.0/KELVIN;
  rhoz = P_eq * mu_z /(T*kB_mp);
  
  return rhoz;
}

Real Get_mu(Real fv){
  return 1./((1.-fv)/mu_xy + fv / mu_z);
}

void Get_E_kg(AthenaArray<Real> gas_vel_array, Real &E_kg){
  E_kg = 0.0;
  for (int n = 0; n<= NDIM-1; ++n){
    E_kg += 0.5*SQR(gas_vel_array(n));
  }
  return;
}

void Get_vel_new_fromMC(AthenaArray<Real> gas_vel_array,
    Real rho_g, Real rho_g1, AthenaArray<Real> rho_d_array, AthenaArray<Real> rho_d_array1, Real drho,
    AthenaArray<Real> rho_ratio_array, AthenaArray<Real> dust_vel_array, bool istracer[NDUSTFLUIDS],
    AthenaArray<Real> &gas_vel_array1, Real &E_kg, AthenaArray<Real> &E_kd_array){
  // momentum conservation, update E_kg, E_kd_array:
  for (int n = 0; n<= NDIM-1; ++n){
    gas_vel_array1(n) = (rho_g)*gas_vel_array(n);
  }
  
  Real denom_gas_vel_new = rho_g1;

  for (int m = 0; m< NVOLATILE*NDUSTBIN; ++m){
    if(istracer[m]){
      for (int n = 0; n<= NDIM-1; ++n){
        gas_vel_array1(n) += rho_d_array(m)*gas_vel_array(n);
      }
      denom_gas_vel_new += rho_d_array1(m);
    }else{
      for (int n = 0; n<= NDIM-1; ++n){
        gas_vel_array1(n) += drho*rho_ratio_array(m)*dust_vel_array(m,n);
      }
    }
  }

  for (int n = 0; n<= NDIM-1; ++n){
    gas_vel_array1(n) /= denom_gas_vel_new;
  }

  E_kg = 0.0;
  for (int n = 0; n<= NDIM-1; ++n){
    E_kg += 0.5*SQR(gas_vel_array1(n));
  }

  for (int m = 0; m< NVOLATILE*NDUSTBIN; ++m){
    if(istracer[m]){
      E_kd_array(m) = E_kg;
    }
  }

  return;
}


void phase_trans(Real rhoe, Real rho_g, AthenaArray<Real> rho_I, Real rho_v, Real &drho){
  Real T;
  Real fv, rhoz;

  fv = rho_v/rho_g;

  T = Get_T_rhoe(rhoe,rho_g,rho_I,fv);
  rhoz = Get_Z(rho_g,T);
  drho = rhoz-rho_v;

  return;
}


Real Get_stopping_time(AthenaArray<Real> rho_d, Real rho_v, Real T, Real rho_g, Real m_p0, Real rad, Real &m_p_out, Real &s_p_out){
  //[25.07.01]zxl: add the m_p0 in to input values since the m_p0 is not a constant now.
  Real rho_I = rho_d(0)*UNIT_DENSITY;
  Real rho_sil = rho_d(1)*UNIT_DENSITY;
  // [25.07.19]zxl: there is a mistake....
  // Real rho_v = rho_d(NDUSTFLUIDS - NVapor)*UNIT_DENSITY;
  Real rho_v_cgs = rho_v*UNIT_DENSITY;
  Real rho_g_cgs = rho_g*UNIT_DENSITY;

  Real fv = rho_v_cgs/rho_g_cgs;
  Real mu1 = Get_mu(fv);

  Real mu_cgs = mu1*CONST_amu;
  Real sigma_mol = 2.e-15; // collisional cross-section of H2, in cgs.
  Real l_mfp = mu_cgs/(std::sqrt(2)*rho_g_cgs*sigma_mol);

  Real rho_Np = rho_sil/(1.0-f_ICE_inter0)/m_p0;
  Real f_ice = rho_I/(rho_sil+rho_I);
  Real f_sil = rho_sil/(rho_sil+rho_I);
  Real rho_p_inter = rho_ice_inter*rho_sil_inter/(f_ice*rho_sil_inter + f_sil*rho_ice_inter);
  Real m_p = (rho_I + rho_sil)/(rho_Np);
  Real s_p = m_p/(FOUR_3RD*PI*rho_p_inter);
  s_p = std::pow(s_p,ONE_3RD);

  Real t_stop;
  Real cs = std::sqrt(T/(mu1*KELVIN));
  Real vth = std::sqrt(8.0/PI)*cs*UNIT_VELOCITY;
  if(s_p < (9.0/4.0*l_mfp)){
    // Epstein regime
    t_stop = rho_p_inter*s_p/(vth*rho_g_cgs);
  }else{
    // std::cout<< "Stokes regime" << std::endl;
    t_stop = 4.0*rho_p_inter*SQR(s_p)/(9.0*vth*rho_g_cgs*l_mfp);
  }

  if (t_stop < 1.e-8){

  }

  if (std::isinf(s_p)){
    std::cout << "Warning: s_p goes to infinity in stopping time calculation" << std::endl;
    std::cout << "s_p = " << s_p << std::endl;
    std::cout << "m_p =" << m_p << std::endl; 
    std::cout << "rho_Np = " << rho_Np << std::endl;
    std::cout << "rho_sil = " << rho_sil << std::endl;
    std::cout << "rho_I = " << rho_sil << std::endl;
    quick_exit(1);
  }

  t_stop /= UNIT_T;

  m_p_out = m_p;
  s_p_out = s_p;

  return t_stop;
}

Real Get_thermal_relaxation_time(Real omega_dyn, Real Tem, Real rhokappa, Real rhod, Real rhog, Real Hg, Real z, Real rad){
  // omega_dyn: the keplerian frequency
  // Tem: local temperature 
  // kappad: dust opacity 
  // rhod: dust density 
  // rhog: gas density 
  // Hg: scale height of gas 
  Real rhod_cgs = rhod*UNIT_DENSITY;
  Real rhog_cgs = rhog*UNIT_DENSITY;
  Real rhokappa_cgs = rhokappa/ (UNIT_LENGTH);
  Real Hg_cgs = Hg*UNIT_LENGTH;

  Real Cv = 8.2e7; // heat capacity in erg/g/K
  Real l_thin = 1/rhokappa_cgs; 
  Real inv_D_rad = (3*rhokappa_cgs*rhog_cgs*Cv)/(16*CONST_sigma*std::pow(Tem, 3.0)); 

  Real t_relax = (SQR(l_thin)*ONE_3RD + SQR(Hg_cgs)) *inv_D_rad ;

  Real t_cool = t_relax/UNIT_T;

  // Real Cv = 8.2e7; // heat capacity in erg/g/K
  // Cv *= KELVIN/SQR(UNIT_VELOCITY);
  // Real UNIT_SB = UNIT_DENSITY * UNIT_VELOCITY * SQR(UNIT_VELOCITY); // flx/(K^4)
  // Real sigma = CONST_sigma/UNIT_SB;
  //
  // Real l_thin = 1/kappad/rhog; 
  // Real inv_D_rad = (3*kappad*rhod*rhog*Cv)/(16*sigma*std::pow(Tem/KELVIN, 3.0)); 
  //
  // Real t_relax = (SQR(l_thin)*ONE_3RD + SQR(Hg)) *inv_D_rad ;
  // Real t_cool = t_relax;
  //

  return t_cool; 

}

void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s){
  
  RadiativeCondution(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);
  LocalIsothermalEOS(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);
  if(PHASE_CHANGE and time >dust_start_injection){
    phase_change(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);
  }

  if(TRIPOD and time > dust_start_injection){
    tripod_source(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);
  }

  return;
}

void tripod_source(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s){
     
}

Real powerlaw_tripod(Real sigma1, Real sigma0, Real amax, Real amin){
  Real aint = std::sqrt(amax*amin); 
  return std::log(sigma1/sigma0)/std::log(amax/aint) - 4;
}

void phase_change(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s){
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          // store some constants of pebbles.
          AthenaArray<Real> E_kd_array, rho_d_array0, rho_d_array, rho_d_array1, rho_ratio_array;
          E_kd_array.NewAthenaArray(NVOLATILE*NDUSTBIN);
          rho_d_array0.NewAthenaArray(NVOLATILE*NDUSTBIN);
          rho_d_array.NewAthenaArray(NVOLATILE*NDUSTBIN);
          rho_d_array1.NewAthenaArray(NVOLATILE*NDUSTBIN);
          rho_ratio_array.NewAthenaArray(NVOLATILE*NDUSTBIN);
          // store velocity
          AthenaArray<Real> gas_vel_array0, gas_vel_array, gas_vel_array1, dust_vel_array;
          gas_vel_array0.NewAthenaArray(NDIM);
          gas_vel_array.NewAthenaArray(NDIM);
          gas_vel_array1.NewAthenaArray(NDIM);
          dust_vel_array.NewAthenaArray(NVOLATILE*NDUSTBIN,NDIM);

          // pre-calculation
          Real &rho_g  = cons(IDN, k, j, i);
          Real &gas_mom1 = cons(IM1, k, j, i);
          Real &gas_mom2 = cons(IM2, k, j, i);
          Real &gas_mom3 = cons(IM3, k, j, i);
          Real &gas_erg  = cons(IEN, k, j, i);
          // gas kinetic energy==
          Real gas_vel1 = gas_mom1/rho_g;
          Real gas_vel2 = gas_mom2/rho_g;
          Real gas_vel3 = gas_mom3/rho_g;
          Real E_kg = 0.5*(SQR(gas_vel1) + SQR(gas_vel2) + SQR(gas_vel3));
          gas_vel_array(0) = gas_vel1;
          gas_vel_array(1) = gas_vel2;
          gas_vel_array(2) = gas_vel3;
          gas_vel_array1 = gas_vel_array;
          gas_vel_array0 = gas_vel_array;

          //[25.07.07]zxl: get the kinetic energy and velocities for volatiles of different population.
          // for (int vol_id = 0; vol_id < NVOLATILE; ++vol_id) {
          //    for (int pop_id = vol_id; pop_id < vol_id * NDUSTBIN; ++vol_id){
          //
          //    }
          // }
          for (int pop_id = 0; pop_id < NDUSTBIN; ++pop_id){ 
            //TBD: maybe we can think about a better way to get this.
            int dust_id = pop_id*NDUSTBIN;
            // int dust_id = vol_id*NCOMPOS ; // this really means the first dustfluid in every population is the ice. not general... [25.07.08]zxl: TBD
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            // [25.07.10]zxl: since we cannot store references into list, and here we have several populations, so we don't 
            //                use the reference here, but later we should update these parameters by it's absolute name.
            Real rho_I   = cons_df(rho_id, k, j, i);
            Real d1_mom1 = cons_df(v1_id,  k, j, i);
            Real d1_mom2 = cons_df(v2_id,  k, j, i);
            Real d1_mom3 = cons_df(v3_id,  k, j, i);
            // ICE kinetic energy
            Real d1_vel1 = d1_mom1/rho_I;
            Real d1_vel2 = d1_mom2/rho_I;
            Real d1_vel3 = d1_mom3/rho_I;

            // assign the calculated values to the arrays. 
            E_kd_array(pop_id) = 0.5*(SQR(d1_vel1) + SQR(d1_vel2) + SQR(d1_vel3));
            dust_vel_array(pop_id,0) = d1_vel1;
            dust_vel_array(pop_id,1) = d1_vel2;
            dust_vel_array(pop_id,2) = d1_vel3;

            rho_d_array(pop_id) = rho_I;

          }

          // tracer particle &  vapor fraction:
          //TBD: [25.07.09]zxl: this also needed to be generalised if we have other components.
          int dust_id = NDUSTFLUIDS - NVapor;
          int rho_id  = 4*dust_id;
          int v1_id   = rho_id + 1;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;

          Real &rho_v   = cons_df(rho_id, k, j, i);
          Real &v1_mom1 = cons_df(v1_id, k, j, i);
          Real &v1_mom2 = cons_df(v2_id, k, j, i);
          Real &v1_mom3 = cons_df(v3_id, k, j, i);

          Real fv = rho_v/rho_g;
          Real fv0 = fv;

          if(std::isnan(fv) or std::isnan(gas_mom1) or std::isnan(gas_erg)){
            Real rad, phi, z;
            GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
            std::cout << "Nan in phase change" << std::endl;
            std::cout << "fv = " << fv << std::endl;
            std::cout << "gas_mom1 = " << gas_mom1 << std::endl;
            std::cout << "gas_erg = " << gas_erg << std::endl;
            std::cout << "rad, phi, z = " << rad << ", " << phi << ", " << z << std::endl;
            quick_exit(1);
          }

          //////////////////////////////////////////////
          // assign initial values
          Real rho_sil[NDUSTBIN];
          for (int pop_id = 0; pop_id < NDUSTBIN; ++pop_id) {
          // zxl: here we require the silicate is the last component defined by user.
          // TBD: we need this to get the number density. so the silicate cannot be sublimated during simulations
            int sil_id = NCOMPOS*(pop_id+1) - 1;
            rho_sil [pop_id] = cons_df(4*sil_id);
          }
          // Real rho_sil = cons_df(4*1, k, j, i);
          // dust array initiation
          rho_d_array1 = rho_d_array;
          rho_d_array0 = rho_d_array;

          /////////////////////////////////////////////
          // store the total energy: the conserved quantity
          Real rhoe_g = gas_erg-E_kg*rho_g;
          Real Tem = Get_T_rhoe_g(rhoe_g,rho_g,fv);
          Real rhomu_d = Get_rhomu_d(rho_d_array); // chemical potential of dust. mu_g is set to be 0.
          Real rhoE_total = gas_erg + rhomu_d;
          for (int m = 0; m < NVOLATILE*NDUSTBIN; ++m){
            rhoE_total += E_kd_array(m)*rho_d_array(m) + Cd_water*(rho_d_array(m))*Tem;
          }

          // define variables used in intermediate steps
          Real drho, rho_g1, rho_v1;
          Real fx0, fx1, fx2, x0, x1, x2; // secant points
          Real rhoe, rhoe1; // total internal energy density
          rho_g1 = rho_g;

          // first calculation of phase change quantity and save their initial value
          rhoe = Get_rhoe(rhoE_total,rho_g,E_kg,rho_d_array,E_kd_array);
          phase_trans(rhoe,rho_g,rho_d_array,rho_v, drho);
          //[25.07.09]zxl: the drho and rho_v are given values in the phase trans function, but I think this may not very safe.
          fx0 = drho;
          x0 = rho_v;

          // Loop starts
          Real sign = (drho > 0. ? 1.0:-1.0);

          // the amount of material that pebble can supply to the vapor, i.e. sublimation
          Real rho_d_supply = 0.0;
          AthenaArray<Real> rho_d_supply_array;
          rho_d_supply_array.NewAthenaArray(NVOLATILE*NDUSTBIN);

          for (int m = 0; m< NVOLATILE*NDUSTBIN; ++m){
            rho_d_supply_array(m) = rho_d_array(m) - dfloor;
            // still need a floor, which should be much smaller than density floor
            rho_d_supply_array(m) = (rho_d_supply_array(m) < 1.e-21) ? 1.e-21 : rho_d_supply_array(m);
            // rho_d_supply += rho_d_supply_array(m);
          }

          // the amount of material that vopor can supply to the pebble, i.e. condensation
          Real rho_v_supply = rho_v - dffloor;
          rho_v_supply = (rho_v_supply < 1.e-21) ? 1.e-21 : rho_v_supply;
          Real rho_v_evap_limit = 0.0;
          //TBD: here is distribute the vapor to the different population according to the density, need to be changed to n*s^2
          Real denominator = 0.0;

          // TBD: [25.07.10]zxl: more general way...
          // for (int vol_id = 0; vol_id <NVOLATILE; ++vol_id){
          for (int m = 0; m < NDUSTBIN; ++m){
            // calculate analytical evaporation rate (schoonenberg+ 2017);
            ///////////////////////////////////////////////////////////////
            // Real &m_p = pmb->pdustfluids->m_p_array(m, k,j,i);
            Real &s_p = pmb->pdustfluids->s_p_array(m, k,j,i);
            Real P_z = rho_v/(KELVIN*mu_z)*Tem;
            Real P_eq = P_eq0*std::exp(-T_a/Tem);
            Real rho_Np = rho_sil[m]/(1.0-f_ICE_inter0)/(m_p0[m]); // unit not unified
            Real drhodt_ice = -std::sqrt(8.0*PI)* std::sqrt(Tem/KELVIN/mu_z) * \
              SQR(s_p)*rho_Np*(UNIT_DENSITY*UNIT_LENGTH) *rho_v *(1.0 - P_eq/P_z);

            // rho_ratio_array(m) = rho_d_supply_array(m)/rho_d_supply;  
            // [25.07.10] zxl: define the distribution law. We here will distribute it by the ns^2
            rho_ratio_array(m) = rho_Np*SQR(s_p);
            denominator += rho_ratio_array(m);
            
            // condensation/sublimation rate limit:
          //  calculate the tota amount of materials that vapor/pebbles can supply
            if(sign > 0.0){
              // sublimation
              rho_d_supply_array(m) = (rho_d_supply_array(m) > drhodt_ice*dt) ? drhodt_ice*dt : rho_d_supply_array(m);
              rho_d_supply += rho_d_supply_array(m);
            }else{
              // condensation
              // this is the total supplement
              rho_v_evap_limit += -drhodt_ice*dt; 
            }

          }

          rho_v_supply = (rho_v_supply > rho_v_evap_limit) ? rho_v_evap_limit : rho_v_supply;

          // normalization
          if (denominator!=0.0){
            // [25.07.14]zxl: sometimes the rho_Np will be 0, then the denominator will also be 0.
            for (int m = 0; m<NDUSTBIN; ++m){
              rho_ratio_array(m) /= denominator;
            }
          }

        //}
          // test1: condense all the vapor to the large population 
          rho_ratio_array(0) = 1.0; 
          rho_ratio_array(1) = 0.0;

          // test2: condense all the vapor to the small population 
          // rho_ratio_array(0) = 0.0; 
          // rho_ratio_array(1) = 1.0;
          ////////////////////////////////////////////////////////////////

          // supply limit:
          Real drho_supply = (sign > 0.0 ? rho_d_supply : -rho_v_supply);
          // update the density of gas and vapor. 
          rho_g1 = rho_g + drho_supply;
          rho_v1 = rho_v + drho_supply;
          // update the density of voletiles.
          for (int m = 0; m< NDUSTBIN; ++m){
            rho_d_array1(m) = rho_d_array(m) - drho_supply*rho_ratio_array(m);
          }
          // update rhoe
        //// get a drho and rho_vapor under the asumption that the evaporation can reach the supply limit
          rhoe1 = Get_rhoe(rhoE_total,rho_g1,E_kg,rho_d_array1,E_kd_array);
          phase_trans(rhoe1,rho_g1,rho_d_array1,rho_v1,drho); 
        // loop parameters
          fx1 = drho;
          x1 = rho_v1;

          // bisect prepare: store inital values
          Real rho_left = rho_v;
          Real rho_right = rho_v1;
          Real rho_g0 = rho_g;
          Real rho_v0 = rho_v;
          gas_vel_array0 = gas_vel_array;
          rho_d_array0 = rho_d_array;

          //***** 2nd step of secant: start from [rhov,rhov1]  *******//
          // (drho*sign) > 0.
          rho_g = rho_g1;
          rho_v = rho_v1;
          rho_d_array = rho_d_array1;
          gas_vel_array = gas_vel_array1;

          if( (drho*sign) < 0.){
            // secant
            Real drho_adp;
            Real f_err = 1.0;
            int nite = 0;
            bool bisect = false;

            while(f_err>min_tol){
              nite += 1;
              if(nite > 100){
                std::cout << "nite > 100 in secant, break" <<std::endl;
                bisect = true;
                break;
                // quick_exit(1);
              }
              drho_adp = -fx1/(fx1-fx0)*(x1-x0);
              // exchange the order of multiplication
              if(std::isnan(drho_adp) or std::isinf(drho_adp)){
                drho_adp = -(x1-x0)/(fx1-fx0)*fx1;
              }

              if(std::isnan(drho_adp) or std::isinf(drho_adp)){
                std::cout << "drho_adp = " << drho_adp<< std::endl;
                bisect = true;
                break;
              }

              x2 = x1 + drho_adp;

              // update the gas density 
              rho_g1 = rho_g + drho_adp;
              rho_v1 = rho_v + drho_adp;
              // condense to the pebbles according to the ratio difined upper. 
              for (int m = 0; m< NVOLATILE*NDUSTBIN; ++m){
                rho_d_array1(m) = rho_d_array(m) - drho_adp*rho_ratio_array(m);
              }

              // update rhoe
              rhoe1 = Get_rhoe(rhoE_total,rho_g1,E_kg,rho_d_array1,E_kd_array);
              phase_trans(rhoe1,rho_g1,rho_d_array1, rho_v1, drho);

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

            if(bisect == true){
              // bisection
              std::cout << "[Warning]: Start bisection in the phase_change module, please check." <<std::endl;
              Real rho_mid;
              rho_g = rho_g0;
              rho_v = rho_v0;
              rho_d_array = rho_d_array0;
              gas_vel_array = gas_vel_array0; 
              int nite = 0;

              while(f_err>min_tol){
                nite += 1;
                if(nite > 1000){
                  std::cout << "nite > 1000 in bisect, break" <<std::endl;
                  quick_exit(1);
                }
                rho_mid = (rho_left+rho_right)/2.0;
                drho_adp = rho_mid-rho_v;

                rho_g1 = rho_g + drho_adp;
                rho_v1 = rho_v + drho_adp;
                // only condense to become st = 0 dusts (already implemented in [phase_trans]).
                for (int m = 0; m< NVOLATILE*NDUSTBIN; ++m){
                  rho_d_array1(m) = rho_d_array(m) - drho_adp*rho_ratio_array(m);
                }

                // update rhoe
                rhoe1 = Get_rhoe(rhoE_total,rho_g1,E_kg,rho_d_array1,E_kd_array);
                phase_trans(rhoe1,rho_g1,rho_d_array1,rho_v1, drho);

                if( (drho*sign) > 0.){
                  rho_left = rho_v1;
                }else{
                  rho_right = rho_v1;
                }

                rho_g = rho_g1;
                rho_v = rho_v1;
                rho_d_array = rho_d_array1;
                gas_vel_array = gas_vel_array1;

                // bisection root error fraction
                // f_err = std::fabs(rho_right-rho_left)/rho_g;
                f_err = std::fabs(drho)/rho_v;
              }

              std::cout << "nite="<< nite <<std::endl;
            }
          }

          // calculate latent heat absorption/release rate
          pmb->pdustfluids->q_latent(k,j,i) += -(rho_v - rho_v0)/(pmb->pmy_mesh->dt) * L_heat;
          //// cons update
          // [25.07.10]zxl: we have NDUSTBIN ice density now 
          // update cons for VOLATILES
          for (int pop_id =0; pop_id < NDUSTBIN; ++pop_id){

            int dust_id = pop_id*NDUSTBIN; //in case the ice the the 1st composition in the array
            // int dust_id = vol_id*NCOMPOS ; // this really means the first dustfluid in every population is the ice. not general... [25.07.08]zxl: TBD
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            cons_df(rho_id, k, j, i) = rho_d_array(pop_id);
          // the velocity of dust will not be affected by the gas, which means here we don't consider the momentum conservation
            cons_df(v1_id,  k, j, i) = rho_d_array(pop_id)*dust_vel_array(pop_id, 0);
            cons_df(v2_id,  k, j, i) = rho_d_array(pop_id)*dust_vel_array(pop_id, 1);
            cons_df(v3_id,  k, j, i) = rho_d_array(pop_id)*dust_vel_array(pop_id, 2);
          }
          // rho_I = rho_d_array(0);

          // apply floor value
          // rho_g = (rho_g > dfloor ) ? rho_g : dfloor;
          // rho_I = (rho_I > dffloor ) ? rho_I : dffloor;
          // rho_v = (rho_v > dfloor ) ? rho_v : dfloor;
          // rho_v = (rho_v > initial_D2G[0]*rho_g/(1.0+initial_D2G[0])) ? initial_D2G[0]*rho_g/(1.0+initial_D2G[0]) : rho_v;
          // rho_d_array(0) = rho_I;

          //// update the cons for gas 
          // calculate pressure
          fv = rho_v/rho_g;
          
          Real mu1 = Get_mu(fv);
          Real prs = rho_g*Tem/(mu1*KELVIN);
          Real gamma = pmb->peos->calc_gamma(fv);

          // Cons update: gas energy, momentum. dust1, 2 momentum.
          // H-He gas
          gas_mom1 = rho_g*gas_vel_array(0);
          gas_mom2 = rho_g*gas_vel_array(1);
          gas_mom3 = rho_g*gas_vel_array(2);
          gas_erg = (prs/(gamma - 1.0) + rho_g*E_kg);

          // vapor (the density has been updated) 
          v1_mom1 = rho_v*gas_vel_array(0);
          v1_mom2 = rho_v*gas_vel_array(1);
          v1_mom3 = rho_v*gas_vel_array(2);

        
          if(std::isnan(fv) or std::isnan(gas_mom1) or std::isnan(gas_erg)){
            Real rad, phi, z;
            GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
            std::cout << "Nan in phase change" << std::endl;
            std::cout << "fv = " << fv << std::endl;
            std::cout << "rho_v" << rho_v << std::endl;
            std::cout << "rho_g" << rho_g << std::endl;
            std::cout << "gas_mom1 = " << gas_mom1 << std::endl;
            std::cout << "gas_erg = " << gas_erg << std::endl;
            std::cout << "rad, phi, z = " << rad << ", " << phi << ", " << z << std::endl;
            std::cout << "pressure = " << prs << std::endl;
            std::cout << "temperature =" << Tem << std::endl;
            std::cout << "rho_gas = " << rho_g << std::endl;
            std::cout << "gas kinetic energy= " << E_kg << std::endl;
            quick_exit(1);
          }
          // check the energy conservation
          // Real rhomu_d_new = Get_rhomu_d(rho_d_array); 
          // Real rhoE_total_new = gas_erg + rhomu_d_new;
          // Real E_kg_new = 0.5*(SQR(gas_vel_array(0)) + SQR(gas_vel_array(1)) + SQR(gas_vel_array(2)));
          //
          // AthenaArray<Real> E_kd_array_new; 
          // E_kd_array_new.NewAthenaArray(NDUSTBIN*NVOLATILE);
          // for (int pop_id = 0; pop_id <NDUSTBIN*NVOLATILE; ++pop_id){
          //   E_kd_array_new(pop_id) = 0.5*(SQR(dust_vel_array(pop_id, 0)) + SQR(dust_vel_array(pop_id,1)) + SQR(dust_vel_array(pop_id,2)));
          // }
          // //TBcheck: the tempearture here I think has been updated in the loop.
          // Real rhoe_g_new = gas_erg-E_kg_new*rho_g;
          // Real Tem_new = Get_T_rhoe_g(rhoe_g_new,rho_g,fv);
          //
          // for (int m = 0; m < NVOLATILE*NDUSTBIN; ++m){
          //   rhoE_total_new += E_kd_array_new(m)*rho_d_array(m) + Cd_water*(rho_d_array(m))*Tem_new;
          // }
          //
          // Real err = std::fabs(rhoE_total_new - rhoE_total)/rhoE_total;
          // // [25.07.14]zxl: need to be further checked. since there are always ~10^-11 errors.
          // if (err > 1e-9){
          // std::cout << "phase change: energy conservation violated" << std::endl;
          // std::cout << "total energy before loop" << rhoE_total << std::endl;
          // std::cout << "total energy after loop" << rhoE_total_new << std::endl;
          // std::cout << "err= " << err <<std::endl;
          // }


          // d1_mom1 = rho_I*dust_vel_array(0,0);
          // d1_mom2 = rho_I*dust_vel_array(0,1);
          // d1_mom3 = rho_I*dust_vel_array(0,2);
          // update temperature for radiative diffusion step
          // rhoe = Get_rhoe(rhoE_total,rho_g,E_kg,rho_d_array,E_kd_array);
          // Real T1 = Get_T_rhoe(rhoe,rho_g,rho_d_array,fv);
          // pmb->phydro->Tem_raddiff(k,j,i) = Tem;

          // allow temperature change:
          // pmb->phydro->Tem(k,j,i) = T1;
          // prs = rho_g*T1/(mu1*KELVIN);

        }
      }
    }
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

          Real &rho_v = cons_df(4*(NDUSTFLUIDS - NVapor), k, j, i);
          Real fv = rho_v/gas_dens;
          Real gas_vel1_0 = gas_mom1/gas_dens;
          Real gas_vel2_0 = gas_mom2/gas_dens;
          Real gas_vel3_0 = gas_mom3/gas_dens;
          Real Ek0 = 0.5*(SQR(gas_vel1_0) + SQR(gas_vel2_0) + SQR(gas_vel3_0));

          if(std::isnan(fv) or std::isnan(Ek0)){
            std::cout <<"fv= nan in LocalIsothermalEOS" << std::endl;
            std::cout << "rhov =" << rho_v << std::endl;
            std::cout << "gas_dens =" << gas_dens << std::endl;
            std::cout << "rad, phi, z = " << rad << ", " << phi << ", " << z << std::endl;
            std::cout << "i, j, k = " << i << ", " << j << ", " << k << std::endl;
            quick_exit(1);
          }

          // new velocity field
          Real gas_vel1_1 = gas_vel1_0;
          Real gas_vel2_1 = gas_vel2_0;
          Real gas_vel3_1 = gas_vel3_0;

          // temperature profile:
          Real Tem;
          Tem = pmb->phydro->Tem(k, j, i);

          // initialization procedure: use hydrostatic solution for gas density
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
            if (pmb->porb->orbital_advection_defined){
              gas_vel3_1 -= vel_K;
            }
          }
          ///////////////////////////////////////////////////////////////////
          
          // get pressure out of local temperature
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

          // set boundary to be not influenced by thermal diffusion
          Real rad = pmb->pcoord->x1v(i);
          Real w_damp = 0.05*(x1max-x1min);
          Real f_decay_art = std::tanh(std::pow((rad-x1max)/w_damp ,2.0)); // outer bc decay
          Real f_decay_art2 = std::tanh(std::pow((rad-1.1*x1min)/w_damp,2.0));
          if(rad <= 1.1*x1min){
            f_decay_art2 = 0.0;
          }
          Real f_decay_art3 = std::exp(-1.0/(tau_vi));
          pmb->pdustfluids->q_diff(k, j, i) += dt/pmb->pmy_mesh->dt*(x1area(i+1)*x1flux(k,j,i+1) - x1area(i)*x1flux(k,j,i))/(-vol(i)) *f_decay_art*f_decay_art2*f_decay_art3;
        //   if (std::isnan(pmb->pdustfluids->q_diff(k, j, i))){
        //   std::cout <<"x1flux = " << x1flux(k,j,i)<<std::endl; 
        //   std::cout <<"x1flux+1 = " << x1flux(k,j,i+1)<<std::endl; 
        //     std::cout<< "f_decay_art3=" << f_decay_art3<<std::endl;
        //     std::cout << "tauvi="<<tau_vi<<std::endl;
        // }
          // pmb->pdustfluids->q_diff(k, j, i) *= 0.0;
        }
        
        // do not include heat conduction in Athena++ steps
        for (int i=pmb->is-NGHOST; i<=pmb->ie+1+NGHOST; ++i) {
          x1flux(k,j,i) = 0.0;
        }

      }
    } 
}


void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time) {
////////////////////////////////////////////////////
// See MeshBlock::UserWorkInLoop()
////////////////////////////////////////////////////
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
//           Real Tem = pmb->phydro->Tem(k, j, i);
//           Real omega_dyn = std::sqrt(gm0/std::pow(rad_arr(i),3.0));
//           rho_dustfluid_array(0) = prim_df(0, k, j ,i);
//           rho_dustfluid_array(1) = prim_df(4*1, k, j ,i);
//           rho_dustfluid_array(2) = rho_v;

//           Real &m_p = pmb->pdustfluids->m_p_array(k,j,i);
//           Real &s_p = pmb->pdustfluids->s_p_array(k,j,i);
//           Real t_stop = Get_stopping_time(rho_dustfluid_array,Tem,rho_g,rad_arr(i),m_p,s_p);
//           // ===== upper limit, St can be extremely high for upper layer =====
//           t_stop = (t_stop > 0.5) ? 0.5 : t_stop;
//           if(std::fabs(z_arr(i)) > 2.0 * std::pow((rad_arr(i)/r0), 1.5 + qvalue/2)){
//             t_stop = 1.e-4;
//           }
//           // ================================================================
//           Real gas_nu = pmb->phydro->hdif.nu(HydroDiffusion::DiffProcess::alpha, k, j, i);
//           Real &diffusivity = pmb->pdustfluids->nu_dustfluids_array(dust_id, k, j, i);
//           Real &st_time = pmb->pdustfluids->stopping_time_array(dust_id, k,j,i);

//           if(!pmb->pdustfluids->istracer[dust_id]){  
//             if(time > dust_start_injection){
//               st_time = t_stop;
//               // st_time = 0.01;
//             }else{
//               st_time= 1.e-8;
//             }
//             // apply st floor 
//             st_time = (st_time > 1.e-8) ? st_time : 1.e-8;
//           }
//           //calculate diffusivity
//           Real taus_peb= t_stop*omega_dyn;
//           // artifical decay
//           Real w_damp = 0.05*(x1max-x1min);
//           Real f_decay_art = std::tanh(std::pow((rad_arr(i)-x1max)/w_damp ,2.0)); // outer bc decay
//           Real f_decay_art2 = std::tanh(std::pow((rad_arr(i)-1.2*x1min)/w_damp,2.0));
//           if(rad_arr(i) <= 1.2*x1min){
//             f_decay_art2 = 0.0;
//           }
//           if(!pmb->pdustfluids->istracer[dust_id]){
//             diffusivity = gas_nu/(1.+SQR(taus_peb));
//           }else{
//             diffusivity = gas_nu*f_decay_art*f_decay_art2;
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
//////////////////////////////////////////
// See MeshBlock::UserWorkInLoop()
/////////////////////////////////////////
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

void MyConductivity(HydroDiffusion *phdif, MeshBlock *pmb,
    const AthenaArray<Real> &w, const AthenaArray<Real> &bc,
    int is, int ie, int js, int je, int ks, int ke) {
  
  for (int k=ks; k<=ke; ++k) { // include ghost zone
    for (int j=js; j<=je; ++j) { // prim, cons
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        LogicalLocation &loc = pmb->loc;
        int ti = static_cast<int>(loc.lx1)*pmb->block_size.nx1+(i-is)+ NGHOST;
        int tj = static_cast<int>(loc.lx2)*pmb->block_size.nx2+(j-js)+ NGHOST;
        int tk = static_cast<int>(loc.lx3)*pmb->block_size.nx3+(k-ks);

        // reflecting boundary at disc midplane
        if(tj > pmb->pmy_mesh->mesh_size.nx2 + NGHOST -1){
          tj = 2*(pmb->pmy_mesh->mesh_size.nx2 + NGHOST) - tj -1;
        }

        Real tau_vi = pmb->pmy_mesh->ruser_mesh_data[0](tk,tj,ti);
        Real Tem = pmb->phydro->Tem(k, j, i);

        const Real &gas_rho = w(IDN, k, j, i);
        //TBD: [25.07.09]zxl: let's write a function to calculate this 
        Real fv = pmb->pdustfluids->df_prim(4*(NDUSTFLUIDS - NVapor), k, j, i) / gas_rho;
        Real mu = Get_mu(fv);
        Real kappa_R = Get_kappa(0.0, fv);
        Real UNIT_SB = UNIT_DENSITY * UNIT_VELOCITY * SQR(UNIT_VELOCITY); // flx/(K^4)

        Real &kappa_heat = phdif->kappa(HydroDiffusion::DiffProcess::aniso, k, j, i);
        Real f_decay = 0.1;
        kappa_heat = 1.0/(3.0*kappa_R*gas_rho)* 16.0* (CONST_sigma/UNIT_SB) *SQR(Tem)*Tem*f_decay;
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
  kappa_R *= UNIT_DENSITY*UNIT_LENGTH;
  return kappa_R;
}

Real Get_nu_gas(const Real Tem, const Real rad, Real fv) {

  // Real mu = Get_mu(fv);
  Real mu = mu_xy; // simple model, scale height indep. of mu
  Real cs2 = Tem/(mu*KELVIN);
  Real omega = std::sqrt(gm0/(rad*rad*rad));
  Real nu_gas = nu_alpha*cs2/omega;

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
  //not used
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
  //not used
  Real H_gas = std::pow(rad/r0, qvalue/2.0 + 1.5);
  Real vel_r = -nu_alpha *(1.0/r0)*std::pow(rad/r0, qvalue + 0.5)* (3.0*pvalue + 2.0*qvalue + 6.0 + (5.0*qvalue + 9.0)/2.0*SQR(z/H_gas));
  return vel_r;
}

Real Vr_ProfileCyl_gas_fv(const Real rad, const Real phi, const Real z, Real fv) {
  //not used
  Real H_gas = std::pow(rad/r0, qvalue/2.0 + 1.5);
  Real mu = Get_mu(fv);
  H_gas *= std::sqrt(mu_xy/mu); // scale height decreases due to vapor injection
  Real vel_r = -nu_alpha *(1.0/r0)*std::pow(rad/r0, qvalue + 0.5)* (3.0*pvalue + 2.0*qvalue + 6.0 + (5.0*qvalue + 9.0)/2.0*SQR(z/H_gas));
  return vel_r;
}

Real Vr_ProfileCyl_gas_fv_T(const Real rad, const Real phi, const Real z, Real fv, Real Tem) {
  Real mu = Get_mu(fv);
  Real omega = std::sqrt(gm0/(rad*rad*rad));
  Real H_gas = std::sqrt(Tem/(KELVIN*mu))/omega;
  Real vel_r = -nu_alpha *(1.0/r0)*std::pow(rad/r0, qvalue + 0.5)* (3.0*pvalue + 2.0*qvalue + 6.0 + (5.0*qvalue + 9.0)/2.0*SQR(z/H_gas));
  return vel_r;
}

Real Vr_ProfileCyl_gas_fv_T_pvalue(const Real rad, const Real phi, const Real z, Real fv, Real Tem, Real pnew, Real qnew) {
  Real mu = Get_mu(fv);
  Real omega = std::sqrt(gm0/(rad*rad*rad));
  Real H_gas = std::sqrt(Tem/(KELVIN*mu))/omega;
  Real vel_r = -nu_alpha *(1.0/r0)*std::pow(rad/r0, qnew + 0.5)* (3.0*pnew + 2.0*qnew + 6.0 + (5.0*qnew + 9.0)/2.0*SQR(z/H_gas));
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

template<typename T>
void print(const T& value, const char* name) {
    std::cout << name << " = " << value << std::endl;
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
//! User-defined boundary Conditions
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
        Real fv = prim_df(4*(NDUSTFLUIDS - NVapor), k, j, il)/gas_rho_active;
        
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
            // extrapolated pressure
            gas_pres_ghost = std::exp(2.0*std::log(prim(IPR, k, j, il-i+1)) - std::log(prim(IPR, k, j, il-i+2)));
            // gas_pres_ghost = gas_rho_ghost*Tem_ghost/(mu1*KELVIN);
          }
          gas_pres_ghost = (gas_pres_ghost > pfloor) ? gas_pres_ghost : pfloor;
        }

        // copy latent heat (not always necessary... depending on simulation domain):
        pmb->pdustfluids->q_latent(k,j,il-i) = pmb->pdustfluids->q_latent(k,j,il);

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;
            
            if(!pmb->pdustfluids->istracer[dust_id]){
              // const flux outflow for dust
              prim_df(rho_id,k,j,il-i) = prim_df(rho_id,k,j,il)*SQR(x1_active)/SQR(x1_ghost);
              prim_df(v1_id,k,j,il-i) = prim_df(v1_id,k,j,il);
              prim_df(v2_id,k,j,il-i) = prim_df(v2_id,k,j,il);
              prim_df(v3_id,k,j,il-i) = prim_df(v3_id,k,j,il);
            }else{
              Real &vapor_rho_ghost  = prim_df(rho_id, k, j, il-i);
              Real &vapor_vel1_ghost = prim_df(v1_id,  k, j, il-i);
              Real &vapor_vel2_ghost = prim_df(v2_id,  k, j, il-i);
              Real &vapor_vel3_ghost = prim_df(v3_id,  k, j, il-i);
              
              // keep the consistency between gas and vapor.
              vapor_vel1_ghost = gas_vel1_ghost;
              vapor_vel2_ghost = gas_vel2_ghost;
              vapor_vel3_ghost = gas_vel3_ghost;
              vapor_rho_ghost = fv*gas_rho_ghost;
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

        // calc rotational vel from midplane temperature
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

        // debug cell:
        if(std::isnan(gas_vel1_ghost)){
          std::cout <<"gas_vel1_ghost = nan" << std::endl;
          std::cout <<"Tem_mid = " <<Tem_mid <<std::endl;
          std::cout <<"rad_ghost = " << rad_ghost << std::endl;
          std::cout <<"z_ghost = "<< z_ghost <<std::endl;
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
          // gas_pres_ghost = gas_rho_ghost*Tem_mid/(mu_xy*KELVIN);
          // extrapolated pressure
          gas_pres_ghost = std::exp(2.0*std::log(prim(IPR, k, j, iu+i-1)) - std::log(prim(IPR, k, j, iu+i-2)));
          gas_pres_ghost = (gas_pres_ghost > pfloor) ? gas_pres_ghost : pfloor;

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

            if(!pmb->pdustfluids->istracer[dust_id]){
              dust_rho_ghost = prim_df(rho_id, k, j, iu);
              dust_vel1_ghost = prim_df(v1_id,  k, j, iu);
              dust_vel2_ghost = prim_df(v2_id,  k, j, iu);
              dust_vel3_ghost = vel_dust_phi;
            }else{
              dust_rho_ghost = initial_D2G[dust_id]*gas_rho_ghost;
              dust_rho_ghost  = (dust_rho_ghost > dffloor) ? dust_rho_ghost : dffloor;
              dust_vel1_ghost = gas_vel1_ghost;
              dust_vel2_ghost = gas_vel2_ghost;
              dust_vel3_ghost = gas_vel3_ghost;
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
  // for now not used
  // Real inv_lower_damp = 1.0/lower_altitude_damping;
  // Real inv_inner_damp = 1.0/inner_width_damping;
  // Real inv_outer_damp = 1.0/outer_width_damping;

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

  //define a upper y to identify the damping zone
  Real y_upper_damping = std::sin(theta_upper_damping) * 3*14959787070000.0/UNIT_LENGTH;

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
        Real ymax = x1max/std::tan(x2);

        //transfer the x2 to the y in other coordinate 
        Real y = x1/std::tan(x2);

        // if (y >= y_upper_damping and y<ymax){
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


void copy_velocities(int source_dust_id, int target_dust_id, int k, int j, int i, MeshBlock *pmb, std::string source_type = "dust") {

  if (source_type == "gas") {
    // copy the velocity from gas to vapor 
    const Real &gas_den = pmb-> phydro->w(IDN,k ,j ,i);
    const Real &gas_vel1 = pmb-> phydro->w(IVX, k, j, i);
    const Real &gas_vel2 = pmb-> phydro->w(IVY, k, j, i);
    const Real &gas_vel3 = pmb-> phydro->w(IVZ, k, j, i);

    int rho_id = 4*target_dust_id; 
    int v1_id = rho_id + 1;
    int v2_id = rho_id + 2;
    int v3_id = rho_id + 3;

    Real &dust_den = pmb-> pdustfluids->df_prim(rho_id,  k, j, i);
    Real &vel1 = pmb-> pdustfluids->df_prim(v1_id,  k, j, i);
    Real &vel2 = pmb-> pdustfluids->df_prim(v2_id,  k, j, i);
    Real &vel3 = pmb-> pdustfluids->df_prim(v3_id,  k, j, i);
    Real &mom1 = pmb-> pdustfluids->df_cons(v1_id,  k, j, i);
    Real &mom2 = pmb-> pdustfluids->df_cons(v2_id,  k, j, i);
    Real &mom3 = pmb-> pdustfluids->df_cons(v3_id,  k, j, i);

    vel1 = gas_vel1;
    vel2 = gas_vel2;
    vel3 = gas_vel3;
    mom1 = dust_den * gas_vel1;
    mom2 = dust_den * gas_vel2;
    mom3 = dust_den * gas_vel3;

  } else {
    // copy the velocity from source_dust_id to target_dust_id
    int source_rho_id = 4*source_dust_id; 
    int source_v1_id = source_rho_id + 1;
    int source_v2_id = source_rho_id + 2;
    int source_v3_id = source_rho_id + 3;

    int target_rho_id = 4*target_dust_id; 
    int target_v1_id = target_rho_id + 1;
    int target_v2_id = target_rho_id + 2;
    int target_v3_id = target_rho_id + 3;

    Real &vel1 = pmb-> pdustfluids->df_prim(target_v1_id,  k, j, i);
    Real &vel2 = pmb-> pdustfluids->df_prim(target_v2_id,  k, j, i);
    Real &vel3 = pmb-> pdustfluids->df_prim(target_v3_id,  k, j, i);
    Real &mom1 = pmb-> pdustfluids->df_cons(target_v1_id,  k, j, i);
    Real &mom2 = pmb-> pdustfluids->df_cons(target_v2_id,  k, j, i);
    Real &mom3 = pmb-> pdustfluids->df_cons(target_v3_id,  k, j, i);

    vel1 = pmb-> pdustfluids->df_prim(source_v1_id,  k, j, i);
    vel2 = pmb-> pdustfluids->df_prim(source_v2_id,  k, j, i);
    vel3 = pmb-> pdustfluids->df_prim(source_v3_id,  k, j, i);
    
    Real &dust_den = pmb-> pdustfluids->df_prim(target_rho_id, k,j,i);

    mom1 = dust_den * vel1;
    mom2 = dust_den * vel2;
    mom3 = dust_den * vel3;
  }
}
