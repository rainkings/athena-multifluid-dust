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
    Real gm0, r0, sigma0, Tem0, alpha_vis, gamma_gas, x1min, x1max, qvalue,dust_start_injection, dffloor, Tslope, p_over_rho_slope;
    Real pvalue, p0_over_r0, Omega0, dslope, M_dot_g, injection_Tsoft, rho0, st_floor;   
    Real Isothermal_Flag, PhaseChange_Flag, Relaxation_Flag;
    Real initial_D2G[NDUSTFLUIDS], H_ratio[NDUSTFLUIDS], Stokes_number[NDUSTFLUIDS], p2g_flux[NDUSTFLUIDS];
    Real Sigma_gas(const Real rad);
    Real nu_gas(const Real Tem, const Real rad, Real fv);
    Real get_mu(const Real fv); 
    Real Tem_gas(const Real rad);

    Real Get_eta_vk(AthenaArray<Real> rad, Real k, Real j, Real i, const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_df);
    void LocalIsothermalEOS(MeshBlock *pmb, const Real time, const Real dt,
        const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_df,
        const AthenaArray<Real> &bcc, AthenaArray<Real> &cons, AthenaArray<Real> &cons_df);
    void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
      const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time,
      const int il, const int iu, const int jl, const int ju, const int kl, const int ku);
    Real DenProfileCyl_dust(const Real rad, const Real phi, const Real z, const Real den_ratio, const Real H_ratio); 
    void MyDustDiffusivity(DustFluids *pdf, MeshBlock *pmb,
          const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
          const AthenaArray<Real> &stopping_time,
          AthenaArray<Real> &nu_dust, AthenaArray<Real> &cs_dust,
          int is, int ie, int js, int je, int ks, int ke);
    void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
        const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
        AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
    void sigma_interpolate_inner_nomatter(const Real r_active, const Real r_ghost, const Real sigma_active,
        Real &sigma_ghost, const Real vr_active, const Real vr_ghost);
    void drift_vel(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
        const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
        AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s); 
    // void MySource(MeshBlock *pmb, const Real time, const Real dt,
    //     const AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
    //     const AthenaArray<Real> &bcc,
    //     AthenaArray<Real> &cons, AthenaArray<Real> &cons_df); 
    Real VelProfileCyl_gas_fv_T(const Real rad, const Real phi, const Real z, const Real fv, const Real Tem); 
    Real VelProfileCyl_dust(const Real rad, const Real phi, const Real z); 
    Real DenProfileCyl_gas(const Real rad, const Real phi, const Real z);
} //namespace

void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh); 
void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);

void Mesh::InitUserMeshData(ParameterInput *pin) {
    gm0 = pin->GetOrAddReal("problem", "gm0", 0.0);
    r0  = pin->GetOrAddReal("problem", "r0", 3.0);
    sigma0 = pin->GetReal("problem", "sigma0");
    rho0 = pin->GetReal("problem", "rho0");
    Tem0 = pin->GetOrAddReal("problem", "Tem0", 150.0);
    alpha_vis = pin->GetOrAddReal("problem", "alpha_vis", 1.e-3);
    gamma_gas = pin->GetReal("hydro", "gamma");
    x1min = pin->GetReal("mesh", "x1min"); 
    x1max = pin->GetReal("mesh", "x1max");
    // qvalue = pin->GetReal("problem", "qvalue");
    dust_start_injection = pin->GetReal("problem", "dust_start_injection");
    dffloor = pin->GetOrAddReal("dust", "dffloor", 1.e-15);
    Tslope = pin->GetOrAddReal("problem", "Tslope", -0.5);
    p_over_rho_slope = pin->GetOrAddReal("problem", "p_over_rho_slope", -0.5);
    Omega0    = pin->GetOrAddReal("orbital_advection", "Omega0", 0.0);
    dslope = pin->GetOrAddReal("problem", "dslope", -1.0);
    M_dot_g = pin->GetOrAddReal("problem", "M_dot_g", 1.e-8); // M_sun/yr
    injection_Tsoft = pin->GetReal("problem", "injection_Tsoft");
    st_floor = pin->GetOrAddReal("dust", "st_floor", 1.e-8);

    Isothermal_Flag  = pin->GetBoolean("problem", "Isothermal_Flag");
    PhaseChange_Flag = pin->GetBoolean("problem", "PhaseChange_Flag");
    Relaxation_Flag  = pin->GetBoolean("problem", "Relaxation_Flag");


  if (NON_BAROTROPIC_EOS) {
        p0_over_r0 = pin->GetOrAddReal("problem", "p0_over_r0", 1.0);
        Tslope     = pin->GetReal("problem", "Tslope");
        p_over_rho_slope = Tslope;
        gamma_gas  = pin->GetReal("hydro", "gamma");
      } else {
        p0_over_r0 = SQR(pin->GetReal("hydro", "iso_sound_speed"));
      }
    for (int n=0; n<NDUSTFLUIDS; n++) {
        initial_D2G[n]   = pin->GetReal("dust", "initial_D2G_" + std::to_string(n+1));
        Stokes_number[n] = pin->GetReal("dust", "Stokes_number_" + std::to_string(n+1));
        H_ratio[n]       = pin->GetReal("dust", "Hratio_" + std::to_string(n+1));
        p2g_flux[n]      = pin->GetOrAddReal("dust", "p2g_flux_" + std::to_string(n+1), 0.2);
    }

  if (NDUSTFLUIDS > 0) {
    // Enroll user-defined dust stopping time
    EnrollUserDustStoppingTime(MyStoppingTime);
    // Enroll user-defined dust diffusivity
    EnrollDustDiffusivity(MyDustDiffusivity);
  }
  // EnrollUserExplicitSourceFunction(MySource);
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
        EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiskOuterX1);
      }
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
        EnrollUserBoundaryFunction(BoundaryFace::inner_x1, DiskInnerX1);
    }

      EnrollUserExplicitSourceFunction(MySource);
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin){
  AllocateUserOutputVariables(10+2+3 + 1);
  // Firstly, we output the variables related to non-tracer dustfluids 
  const std::vector<std::pair<int,const char*>> dustProp = {
    {0,  "st"},
    {1, "m_p"},
    {2, "s_p"},
    {3, "flx_sil_x1"},
    {4, "dif_sil"}
  };

  for (int np =0; np<N_P; np++) {
    for (auto &p : dustProp) {
      SetUserOutputVariableName(p.first + np*dustProp.size(), (std::string(p.second) + "_" + std::to_string(np+1)).c_str());
    }
  }

  // Then, we output the variables related to tracer dustfluids 
  int offset_vapor = dustProp.size()*N_P; 
  SetUserOutputVariableName(offset_vapor, "dif_vap");
  SetUserOutputVariableName(offset_vapor+1, "flx_vap_x1");

  // Finally, output the variables related to gas 
  int offset_gas = offset_vapor + 2;
  const std::vector<std::pair<int,const char*>> gasProp = {
    {0,  "Tem"},
    {1, "flx_x1"},
    {2, "dif"},
  };

  for (auto &p : gasProp) {
    SetUserOutputVariableName(p.first + offset_gas, p.second);
  }

    SetUserOutputVariableName(15, "mmax");


    return;
}

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

    for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
            for (int i=il; i<=iu; ++i) {
                rad = pcoord->x1v(i);

                Real omega_dyn = std::sqrt(gm0/std::pow(rad,3.0));
                Real vel_K = vK(porb, rad, pcoord->x2v(j), pcoord->x3v(k));

                Real den_gas = DenProfileCyl_gas(rad, phi, z);
                Real Tem = Tem_gas(rad);
                Real cs2 = Tem/(mu_xy*KELVIN); 
                Real h_gas = std::sqrt(cs2)/omega_dyn;
                Real sigma_gas = den_gas*std::sqrt(2.0*PI)*h_gas;

                Real &nu = phydro->hdif.nu(HydroDiffusion::DiffProcess::alpha, k, j, i);
                nu = nu_gas(Tem, rad, 0.0);
                Real v_acc_gas = -1.5*nu/rad; // negative accretion vel
                
                Real vel_gas_phi = VelProfileCyl_gas_fv_T(rad,phi,0.0,0.0,Tem);
                if (porb->orbital_advection_defined)
                  vel_gas_phi -= vel_K;
                
                phydro->u(IDN, k, j, i) = sigma_gas;
                phydro->u(IM1, k, j, i) = sigma_gas*v_acc_gas; //radial momentum
                phydro->u(IM2, k, j, i) = sigma_gas*vel_gas_phi; // azimuthal momentum
                phydro->u(IM3, k, j, i) = 0.0; // vertical momentum
                phydro->u(IEN, k, j, i) = sigma_gas*cs2*igm1;
                phydro->u(IEN, k, j, i) += 0.5*(SQR(phydro->u(IM1, k, j, i)) +SQR(phydro->u(IM2,k,j,i)))/phydro->u(IDN, k, j, i);
                phydro->Tem(k, j, i) = Tem;

                //then assige the preperties to dust 
                for (int n=0; n<N_P*N_Z + 1; ++n){
                    int dust_id = n;
                    int rho_id  = 4*dust_id;
                    int v1_id   = rho_id + 1;
                    int v2_id   = rho_id + 2;
                    int v3_id   = rho_id + 3;

                    // should not multiply the H_ratio
                    Real dust_dens = DenProfileCyl_dust(rad,phi,z, initial_D2G[n], H_ratio[n]);
                    Real dust_sigma = dust_dens*std::sqrt(2.0*PI)*h_gas; // surface density of dust
                    pdustfluids->df_u(rho_id, k, j, i) = dust_sigma; // dust density

                    Real vel_dust_phi = VelProfileCyl_dust(rad, phi, z);
                    if (porb->orbital_advection_defined)
                      vel_dust_phi -= vel_K;

                    if (dust_id == vapor_id){
                        pdustfluids->df_u(v1_id, k, j, i) = dust_sigma*v_acc_gas; // vapor radial momentum
                        pdustfluids->df_u(v2_id, k, j, i) = dust_sigma*vel_gas_phi; // vapor azimuthal momentum
                        pdustfluids->df_u(v3_id, k, j, i) = 0.0; // vapor vertical momentum
                    } else{  // else we have both dustfluids and number density fluids [26.03.26]
                        pdustfluids->df_u(v1_id, k, j, i) = 0.0; // dust radial momentum
                        pdustfluids->df_u(v2_id, k, j, i) = dust_sigma*vel_dust_phi; // dust azimuthal momentum
                        pdustfluids->df_u(v3_id, k, j, i) = 0.0; // dust vertical momentum

                        int dustn_id = N_P*N_Z + 1 + dust_id;
                        int n_id = 4*dustn_id;
                        int nv1_id = n_id + 1;
                        int nv2_id = n_id + 2;
                        int nv3_id = n_id + 3;

                        if (pphase_change != nullptr) {
                            Real rho_Np_column = dust_sigma/(1 - 0.0)/(pphase_change->m_p0_array(dust_id/N_Z)); // [code_number_density]
                            pdustfluids->df_u(n_id, k, j, i) = rho_Np_column; // number density of dust
                            // Real check = rho_Np_column/std::pow(pmy_mesh->punit->code_length_cgs, 2);
                            pdustfluids->df_u(nv1_id, k, j, i) = 0.0; // number density flux in radial direction
                            pdustfluids->df_u(nv2_id, k, j, i) = rho_Np_column*vel_dust_phi; // number density flux in azimuthal direction
                            pdustfluids->df_u(nv3_id, k, j, i) = 0.0; // number density flux in vertical direction
                        }
                    }
                // std::cout << "sigma_dust = " << pdustfluids->df_u(rho_id, k, j, i) << " at (k,j,i)=("<<k<<","<<j<<","<<i<<")!" << std::endl;
                // std::cout << "mom_dust = " << pdustfluids->df_u(v1_id, k, j, i) << " at (k,j,i)=("<<k<<","<<j<<","<<i<<")!" << std::endl;
                }
                //check the dust properties 
                // std::cout << "At (i,j,k)=(" << i << "," << j << "," << k << "):" << std::endl;
                // std::cout << "gas_sigma = " << sigma_gas << std::endl;
                // std::cout << "gas_vel1 = " << v_acc_gas<< std::endl;
                // std::cout << "gas_vel2 = " << vel_gas_phi<< std::endl;
                // std::cout << "gas_Tem = " << Tem << std::endl;
                // std::cout << "gas_energy = " << phydro->u(IEN, k, j, i) << std::endl;
                // std::cout << "dust_sigma_0 = " << pdustfluids->df_u(0,k,j,i) << std::endl;
                // std::cout << "st_0" << " = " << pdustfluids->stopping_time_array(0,k,j,i) << std::endl;
                // std::cout << "dif_0 = " << pdustfluids->nu_dustfluids_array(0,k,j,i) << std::endl;
                // std::cout << "dust_vel1_0 = " << pdustfluids->df_w(1,k,j,i) << std::endl;
                // std::cout << "dust_sigma_1 = " << pdustfluids->df_u(4,k,j,i) << std::endl;
                // std::cout << "dust_vel1_1 = " << pdustfluids->df_w(5,k,j,i) << std::endl;
                // std::cout << "st_1" << " = " << pdustfluids->stopping_time_array(1,k,j,i) << std::endl;
                // std::cout << "dif_1 = " << pdustfluids->nu_dustfluids_array(1,k,j,i) << std::endl;
                // std::cout << "vapor_sigma = " << pdustfluids->df_u(4*vapor_id,k,j,i) << std::endl;
                // std::cout << "vapor_vel1 = " << pdustfluids->df_w(4*vapor_id+1,k,j,i) << std::endl;
                // std::cout << "=========================================" << std::endl;

                // [26.03.26]Zhixuan: We don't need to calculate the number density, since now the number density is a fluid
                // if (pphase_change != nullptr && N_Z>0) {
                //     for (int p = 0; p < N_P; ++p) {
                //         int refrac_id = N_Z * (p+1) - 1;
                //         int rho_id = 4*refrac_id;
                //         Real sigma_sil_p = pdustfluids->df_u(rho_id, k, j, i);
                //         Real &rho_Np = pphase_change->rho_Np_array(p, k, j, i);
                //         //TBD: add f_ice_inter here
                //         rho_Np = sigma_sil_p /(1.0 - 0.0)/(pphase_change->m_p0_array(p)); // [code_number_density]
                //         // if (std::isinf(rho_Np) || std::isnan(rho_Np) || rho_Np == 0.0) {
                //         //   std::cout << "Warning: rho_Np is " << rho_Np << " at (k,j,i)=("<<k<<","<<j<<","<<i<<")!" << std::endl;
                //         //   std::cout << "sigma_sil_p = " << sigma_sil_p << ", m_p0 = " << pphase_change->m_p0_array(p, k, j, i) << std::endl;
                //         // }
                //     }
                // }

                // get pebble stopping times

                if (pphase_change != nullptr) {
                  // vapor diffusivity (with artificial decay for outer boundary)
                  Real &vapor_diffusivity = pdustfluids->nu_dustfluids_array(vapor_id, k, j, i);
                  vapor_diffusivity = nu;

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
                    // Real rho_Np_column = pphase_change->rho_Np_array(p, k, j, i);
                    // [26.03.26]Zhixuan: Now we should get the number density from dustfluids
                    int n_id = 4*(N_P*N_Z + 1 + p);
                    Real rho_Np_column = pdustfluids->df_u(n_id, k, j, i); // number density of the pebble
                    Real rho_Np = rho_Np_column/(std::sqrt(2.0*PI)*h_gas); // convert back to volume density for stopping time calculation

                    // (Yu, 2025-11-16) Pass vapor density separately since it's not in per-pebble array
                    Real rho_v = pdustfluids->df_u(4*vapor_id, k, j, i);
                    Real gas_dens = phydro->u(IDN, k, j, i)/std::sqrt(2.0*PI)/h_gas; // convert back to volume density for stopping time calculation
                        
                      // std::cout << "rho_dustfluid_array_pebble(" << z << ") = " << rho_dustfluid_array_pebble(z) << " at (k,j,i)=("<<k<<","<<j<<","<<i<<")!" << std::endl;
                    Real t_stop = pphase_change->Get_stopping_time(pmy_mesh->punit, rho_dustfluid_array_pebble, Tem, gas_dens, rho_v, rho_Np);
                    //[26.03.30]Zhixuan: we should specify the stopping time for number density 
                    Real &t_stop_n = pdustfluids->stopping_time_array(n_id/4, k, j, i); 
                    t_stop_n = t_stop; 

                    // ===== upper limit, St can be extremely high for upper layer =====
                    t_stop = (t_stop > 0.5) ? 0.5 : t_stop;

                    // apply stopping time to all compositions of this pebble
                    for (int z = 0; z < N_Z; ++z) {
                      int dust_id = N_Z * p + z;
                      Real &st_time = pdustfluids->stopping_time_array(dust_id, k, j, i);
                      
                        // if (dust_id == 0){
                        //     st_time = 1.e-1/omega_dyn;
                        // } else {
                        //     st_time = 0.3/omega_dyn;
                        // }
                      if(pmy_mesh->time > dust_start_injection){
                        st_time = t_stop;
                      }else{
                        st_time = 1.e-8; 
                      }
                      // apply st floor 
                      st_time = (st_time > st_floor) ? st_time : st_floor;
                      
                      // calculate diffusivity
                      Real taus_peb = t_stop*omega_dyn;
                      Real &diffusivity = pdustfluids->nu_dustfluids_array(dust_id, k, j, i);
                      diffusivity = nu/(1.+SQR(taus_peb));
                    }
                  }
                }  // end if (pphase_change != nullptr)
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

  // Real &dt   = pmy_mesh->dt;
  // Real &time = pmy_mesh->time;

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        // pre-calculation
        const Real &gas_sigma = phydro->w(IDN,k ,j ,i);
        const Real &gas_vel1 = phydro->w(IVX, k, j, i);
        const Real &gas_vel2 = phydro->w(IVY, k, j, i);
        const Real &gas_vel3 = phydro->w(IVZ, k, j, i);
        // reset gas properties to initial conditions to mimic a static background (TBD: add damping layer for better stability) 
        // gas_sigma = Sigma_gas(pcoord->x1v(i));
        // Real nu = nu_gas(Tem_gas(pcoord->x1v(i)), pcoord->x1v(i), 0.0);
        // Real v_acc_gas = -1.5*nu/pcoord->x1v(i); // negative accretion vel
        // gas_vel1 = v_acc_gas;
        // Real gp0 = phydro->w(IPR, k, j, 0);
        // Real gp1 = phydro->w(IPR, k, j, 1);
        // Real r0 = pcoord->x1v(0);
        // Real r1 = pcoord->x1v(1);
        // Real ppp = std::log(gp1)/std::log(gp0)/(std::log(r1)/std::log(r0));
        // std::cout << ppp << std::endl;

        // std::cout << "At (i,j,k)=(" << i << "," << j << "," << k << "):" << std::endl;
        // std::cout << "gas_sigma = " << gas_sigma << std::endl;
        // std::cout << "gas_vel1 = " << gas_vel1<< std::endl;
        if (std::isnan(gas_sigma) || std::isnan(gas_vel1)){
          std::cout << "Error: NaN detected in gas variables at (i,j,k)=(" << i << "," << j << "," << k << ")." << std::endl;
        }
        //also check dust properties 
        if (std::isnan(pdustfluids->df_u(5,k,j,i)) || std::isnan(pdustfluids->df_w(1,k,j,i))){
            std::cout << "At (i,j,k)=(" << i << "," << j << "," << k << "):" << std::endl;
            std::cout << "dust_sigma_0 = " << pdustfluids->df_u(0,k,j,i) << std::endl;
            std::cout << "st_0" << " = " << pdustfluids->stopping_time_array(0,k,j,i) << std::endl;
            std::cout << "dif_0 = " << pdustfluids->nu_dustfluids_array(0,k,j,i) << std::endl;
            std::cout << "dust_vel1_0 = " << pdustfluids->df_w(1,k,j,i) << std::endl;
            std::cout << "dust_sigma_1 = " << pdustfluids->df_u(4,k,j,i) << std::endl;
            std::cout << "dust_vel1_1 = " << pdustfluids->df_w(5,k,j,i) << std::endl;
            std::cout << "st_1" << " = " << pdustfluids->stopping_time_array(1,k,j,i) << std::endl; 
            std::cout << "dif_1 = " << pdustfluids->nu_dustfluids_array(1,k,j,i) << std::endl; 
            std::cout << "dust_vel1_1 = " << pdustfluids->df_w(5,k,j,i) << std::endl;
        }


        // copy gas velocity to tracer;
        int dust_id = vapor_id;
        int rho_id  = 4*dust_id;
        int v1_id   = rho_id + 1;
        int v2_id   = rho_id + 2;
        int v3_id   = rho_id + 3;
        
        const Real &vapor_sigma  = pdustfluids->df_u(rho_id, k, j, i);

        Real &vapor_vel1 = pdustfluids->df_w(v1_id,  k, j, i);
        Real &vapor_vel2 = pdustfluids->df_w(v2_id,  k, j, i);
        Real &vapor_vel3 = pdustfluids->df_w(v3_id,  k, j, i);

        Real &vapor_mom1 = pdustfluids->df_u(v1_id,  k, j, i);
        Real &vapor_mom2 = pdustfluids->df_u(v2_id,  k, j, i);
        Real &vapor_mom3 = pdustfluids->df_u(v3_id,  k, j, i);

        vapor_vel1 = gas_vel1;
        vapor_vel2 = gas_vel2;
        vapor_vel3 = gas_vel3;
        vapor_mom1 = vapor_sigma*gas_vel1;
        vapor_mom2 = vapor_sigma*gas_vel2;
        vapor_mom3 = vapor_sigma*gas_vel3;


        // std::cout << "At (i,j,k)=(" << i << "," << j << "," << k << "):" << std::endl;
        // std::cout << "gas_sigma = " << gas_sigma << std::endl;
        // std::cout << "gas_vel1 = " << gas_vel1<< std::endl;
        // std::cout << "gas_pres = " << phydro->w(IPR, k, j, i) << std::endl;
        //check the dust properties 
        // if (gas_vel1>0.0){
        // std::cout << "At (i,j,k)=(" << i << "," << j << "," << k << "):" << std::endl;
        // std::cout << "gas_sigma = " << gas_sigma << std::endl;
        // std::cout << "gas_vel1 = " << gas_vel1<< std::endl;
        // std::cout << "gas_vel2 = " << gas_vel2<< std::endl;
        // std::cout << "dust_sigma_0 = " << pdustfluids->df_u(0,k,j,i) << std::endl;
        // std::cout << "st_0" << " = " << pdustfluids->stopping_time_array(0,k,j,i) << std::endl;
        // std::cout << "dif_0 = " << pdustfluids->nu_dustfluids_array(0,k,j,i) << std::endl;
        // std::cout << "dust_vel1_0 = " << pdustfluids->df_w(1,k,j,i) << std::endl;
        // std::cout << "dust_sigma_1 = " << pdustfluids->df_u(4,k,j,i) << std::endl;
        // std::cout << "dust_vel1_1 = " << pdustfluids->df_w(5,k,j,i) << std::endl;
        // std::cout << "st_1" << " = " << pdustfluids->stopping_time_array(1,k,j,i) << std::endl;
        // std::cout << "dif_1 = " << pdustfluids->nu_dustfluids_array(1,k,j,i) << std::endl;
        // std::cout << "vapor_sigma = " << pdustfluids->df_u(4*vapor_id,k,j,i) << std::endl;
        // std::cout << "vapor_vel1 = " << pdustfluids->df_w(4*vapor_id+1,k,j,i) << std::endl;
        // std::cout << "=========================================" << std::endl;
        //         }
        // copy refractory velocity to ice for each pebble
        if (N_Z >=2) {
          for (int p = 0; p < N_P; ++p) {
            int refrac_id = N_Z*(p+1)-1; // last index in each pebble size bin 
            int ice_id = N_Z*p;          // first index in each pebble size bin 

            int refrac_rho_id  = 4*refrac_id;
            int refrac_v1_id   = refrac_rho_id + 1;
            int refrac_v2_id   = refrac_rho_id + 2;
            int refrac_v3_id   = refrac_rho_id + 3;
            
            const Real &d1_vel1 = pdustfluids->df_w(refrac_v1_id,  k, j, i);
            const Real &d1_vel2 = pdustfluids->df_w(refrac_v2_id,  k, j, i);
            const Real &d1_vel3 = pdustfluids->df_w(refrac_v3_id,  k, j, i);
            
            int ice_rho_id  = 4*ice_id;
            int ice_v1_id   = ice_rho_id + 1;
            int ice_v2_id   = ice_rho_id + 2;
            int ice_v3_id   = ice_rho_id + 3;
            
            Real &d0_vel1 = pdustfluids->df_w(ice_v1_id,  k, j, i);
            Real &d0_vel2 = pdustfluids->df_w(ice_v2_id,  k, j, i);
            Real &d0_vel3 = pdustfluids->df_w(ice_v3_id,  k, j, i);

            const Real &d0_den  = pdustfluids->df_u(ice_rho_id, k, j, i);
            Real &d0_mom1 = pdustfluids->df_u(ice_v1_id,  k, j, i);
            Real &d0_mom2 = pdustfluids->df_u(ice_v2_id,  k, j, i);
            Real &d0_mom3 = pdustfluids->df_u(ice_v3_id,  k, j, i);

            d0_vel1 = d1_vel1;
            d0_vel2 = d1_vel2;
            d0_vel3 = d1_vel3;
            d0_mom1 = d1_vel1*d0_den;
            d0_mom2 = d1_vel2*d0_den;
            d0_mom3 = d1_vel3*d0_den;
          }
        }

        // copy refractory velocity to number density for each population 
        // if (N_P >=2 ) {
        //   for (int p = 0; p < N_P; ++p) {
        //     int refrac_id = N_Z*(p+1)-1; // last index in each pebble size bin 
        //     int ice_id = N_Z*p;          // first index in each pebble size bin 
        //
        //     int refrac_rho_id  = 4*refrac_id;
        //     int refrac_v1_id   = refrac_rho_id + 1;
        //     int refrac_v2_id   = refrac_rho_id + 2;
        //     int refrac_v3_id   = refrac_rho_id + 3;
        //
        //     const Real &d1_vel1 = pdustfluids->df_w(refrac_v1_id,  k, j, i);
        //     const Real &d1_vel2 = pdustfluids->df_w(refrac_v2_id,  k, j, i);
        //     const Real &d1_vel3 = pdustfluids->df_w(refrac_v3_id,  k, j, i);
        //
        //     int dustn_id = N_P*N_Z + 1 + p;
        //     int n_id = 4*dustn_id;
        //     int nv1_id = n_id + 1;
        //     int nv2_id = n_id + 2;
        //     int nv3_id = n_id + 3;
        //
        //     Real &n_vel1 = pdustfluids->df_w(nv1_id,  k, j, i);
        //     Real &n_vel2 = pdustfluids->df_w(nv2_id,  k, j, i);
        //     Real &n_vel3 = pdustfluids->df_w(nv3_id,  k, j, i);
        //
        //     const Real &n_rho  = pdustfluids->df_u(n_id, k, j, i);
        //     Real &n_mom1 = pdustfluids->df_u(nv1_id,  k, j, i);
        //     Real &n_mom2 = pdustfluids->df_u(nv2_id,  k, j, i);
        //     Real &n_mom3 = pdustfluids->df_u(nv3_id,  k, j, i);
        //
        //     n_vel1 = d1_vel1;
        //     n_vel2 = d1_vel2;
        //     n_vel3 = d1_vel3; 
        //     n_mom1 = n_rho*d1_vel1;
        //     n_mom2 = n_rho*d1_vel2;
        //     n_mom3 = n_rho*d1_vel3;
        //
        //   }
        //
        // }

        //apply floor value
        for (int n=0; n<NDUSTFLUIDS; n++) {
          int dust_id = n;
          int rho_id  = 4*dust_id;
          int v1_id   = rho_id + 1;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;

          Real &dust_rho = pdustfluids->df_w(rho_id, k, j, i);
          dust_rho = (dust_rho > dffloor) ? dust_rho : (dffloor);

          // if (dust_rho <= dffloor) {
          //     std::cout << "Applying floor for dustfluid " << n << " at (i,j,k)=(" << i << "," << j << "," << k << ")." << std::endl;
          //           }

          pdustfluids->df_u(rho_id, k, j, i) = dust_rho;
          pdustfluids->df_u(v1_id, k, j, i) = dust_rho*pdustfluids->df_w(v1_id, k, j, i);
          pdustfluids->df_u(v2_id, k, j, i) = dust_rho*pdustfluids->df_w(v2_id, k, j, i);
          pdustfluids->df_u(v3_id, k, j, i) = dust_rho*pdustfluids->df_w(v3_id, k, j, i);
        }

        }
      }
    }

  //[26.03.30]Zhixuan: update the pebble mass array here
    if (pphase_change != nullptr) {
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
          for (int i=il; i<=iu; ++i) {
            AthenaArray<Real> rho_comps, s_p_array;
            rho_comps.NewAthenaArray(N_Z); // +1 for vapor
            for (int p = 0; p < N_P; ++p) {
              //
              for (int z = 0; z < N_Z; ++z) {
                Real dust_rho_id = N_Z * p + z;
                rho_comps(z) = pdustfluids->df_u(4*dust_rho_id, k, j, i);
              }

              // update rho_Np for pebble p
              int dustn_id = 4*(N_P*N_Z + 1 + p);
              Real &rho_Np_p = pdustfluids->df_u(dustn_id, k, j, i); // number density of the pebble 
              Real rho_refrac = rho_comps(rho_comps.GetDim1() - 1); // refractory (always the last one)

              // Derive m_p and s_p from rho_Np and current densities (Yu, 2025-11-18)
              Real &m_p = pphase_change->m_p_array(p, k, j, i);
              if (N_Z >= 2){
                  m_p = pphase_change->Get_m_p_from_rho_Np(rho_comps, rho_Np_p); // [code_mass]
              } else{
                  m_p = rho_refrac/rho_Np_p; // if only one composition, directly derive mass from density and number density
              }

            }
           }
          }
        }
    }

      Real Mdot_gas = -1.e-8* Constants::solar_mass_cgs/ Constants::yr_cgs; // in cgs
      Mdot_gas /= (pmy_mesh->punit->code_mass_cgs /pmy_mesh->punit->code_time_cgs);

    if (pbval->block_bcs[BoundaryFace::outer_x1] == BoundaryFlag::user) {
        Real rad = pcoord->x1v(ie+1);
        Real Tem = Tem_gas(rad); 
        Real mu_gas = 2.34; // mean molecular weight in proton mass units
        Real cs = std::sqrt(Tem/mu_gas/KELVIN);
        Real omega_dyn = std::sqrt(gm0/std::pow(rad, 3));
        Real h_gas = cs/omega_dyn;
        phydro->inflx_x1(0,0,0) = Mdot_gas/(2*PI*pcoord->x1v(ie+1)); // convert to surface density flux for 2D

        for (int p =0; p<N_P; ++p){
            for (int zi = 0; zi<N_Z; ++zi){
                int dust_id = N_Z * p + zi;
                pdustfluids->inflx_dust_x1(dust_id,0,0,0) = phydro->inflx_x1(0,0,0)*p2g_flux[dust_id]*(1.0-std::exp(-0.5*SQR((pmy_mesh->time-dust_start_injection)/injection_Tsoft)));

                // int dustn_id = N_P*N_Z + 1 + p; 
                // pdustfluids->inflx_dust_x1(dustn_id,0,0,0) = pdustfluids->inflx_dust_x1(dust_id,0,0,0)/pphase_change->m_p_array(p,0,0,ie+1);
                // std::cout << "dustn_id = " << dustn_id << std::endl;
                // std::cout << "m_p = " << pphase_change->m_p_array(p,0,0,ie+1) << " at time=("<<pmy_mesh->time<<")!" << std::endl;
                // dust_sigma_ghost = sigma_peb*0.5*(1.0-std::exp(-0.5*SQR((time-dust_start_injection)/injection_Tsoft)));
            }
        }
            
      // phydro->inflx_x1(0,0,0) = 0.0; 
    };
    return;
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin){
  int dk     = NGHOST;
  int dj     = NGHOST;
  if (block_size.nx3 == 1) dk = 0;
  if (block_size.nx2 == 1) dj = 0;
  int kl = ks - dk;     int ku = ke + dk;
  int jl = js - dj;     int ju = je + dj;
  int il = is - NGHOST; int iu = ie + NGHOST;

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is-NGHOST; i<=ie+NGHOST; ++i) {
        
        user_out_var(12,k,j,i) = phydro->Tem(k,j,i);
        user_out_var(13,k,j,i) = phydro->flux[IDN](k,j,i);
        user_out_var(14,k,j,i) = phydro->hdif.nu(HydroDiffusion::DiffProcess::alpha, k, j, i);

        if (pphase_change != nullptr && N_P > 0) {
          for (int p=0; p<N_P; p++) {
            AthenaArray<Real> rho_comps; //get the uid of density and density for each composition
            rho_comps.NewAthenaArray(N_Z);
            for (int n=0; n<N_Z; n++) {
              Real comp_id = p*N_Z + n;
              rho_comps(n) = pdustfluids->df_w(4*comp_id, k, j, i); // [code_density] 
            }
            // Real rho_Np_column = pphase_change->rho_Np_array(p, k, j, i); // [code_number_density]
            //[26.03.27]Zhixuan: Now we should get the number density from dustfluids 
            int dustn_id = 4*(N_P*N_Z + 1 + p);
            Real rho_Np_column = pdustfluids->df_u(dustn_id, k, j, i); // number density of the pebble
            
            // Real rho_I = pdustfluids->df_w(4*p, k, j, i); // ice for pebble p [code_density]
            // Real rho_sil = pdustfluids->df_w(4*(p*N_Z+1), k, j, i); // refractory for pebble p [code_density]
            Units *punit = pmy_mesh->punit;
            Real m_p, s_p;
            if (N_Z > 1){
                m_p = pphase_change->Get_m_p_from_rho_Np(rho_comps, rho_Np_column); // [code_mass]
                s_p = pphase_change->Get_s_p_from_m_p(m_p, rho_comps); // [code_length]
            } else{
                // m_p = rho_comps(0)/rho_Np_column; 
                m_p = pphase_change->m_p_array(p, k, j, i); 
                s_p = std::pow(3.0*m_p/(4.0*PI*3.0/punit->code_density_cgs), 1.0/3.0);

            }

            // really no good idea to output this more elegently now.... [25.11.24]
            user_out_var(p*5,k,j,i) = pdustfluids->stopping_time_array(p*N_Z,k,j,i);
            user_out_var(p*5+1,k,j,i) = m_p * punit->code_mass_cgs; // Convert to CGS for output 
            user_out_var(p*5+2,k,j,i) = s_p * punit->code_length_cgs; // Convert to CGS for output
            user_out_var(p*5+3,k,j,i) = pdustfluids->df_flux[X1DIR](4*(p*N_Z),k,j,i);
            user_out_var(p*5+4,k,j,i) = pdustfluids->nu_dustfluids_array(p*N_Z,k,j,i); 
          }
        user_out_var(15,k,j,i) = pphase_change->mmax_array(k,j,i) * pmy_mesh->punit->code_mass_cgs; // Convert to CGS for output
        }
        
      }
    }
  }
  return;
}

namespace {

void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s){

  LocalIsothermalEOS(pmb, time, dt, prim, prim_df, bcc, cons, cons_df);
  // drift_vel(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);
  if(N_Z > 1 and time > dust_start_injection and PhaseChange_Flag){  // (Yu, 2025-11-16)
    pmb->pphase_change->PhaseChangeSource(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);
  }



  AthenaArray<Real> v_frag;
  v_frag.NewAthenaArray(N_P); 
  v_frag(0) = 1000.0/pmb->pmy_mesh->punit->code_velocity_cgs; // fragmentation velocity for pebble size bin 0 [code_velocity]
  v_frag(1) = 100.0/pmb->pmy_mesh->punit->code_velocity_cgs; // fragmentation velocity for pebble size bin 1 [code_velocity]

  // Real v1_sil_small = prim_df(4*0+1, 0, 0, 2);
  // Real v2_sil_small = prim_df(4*0+2, 0, 0, 2);
  // Real v3_sil_small = prim_df(4*0+3, 0, 0, 2);
  //
  // Real v1_sil_big = prim_df(4*1+1, 0, 0, 2);
  // Real v2_sil_big = prim_df(4*1+2, 0, 0, 2);
  // Real v3_sil_big = prim_df(4*1+3, 0, 0, 2);
  //
  // std::cout << "At (i,j,k)=(" << 2 << "," << 0 << "," << 0 << "):" << std::endl;
  // std::cout << "v1_sil_small = " << v1_sil_small << std::endl;
//
// std::cout << cons_df(4*0, 0, 0, 2) << std::endl;
// std::cout << cons_df(4*1, 0, 0, 2) << std::endl;
// std::cout << cons_df(4*3, 0, 0, 2) << std::endl;
// std::cout << cons_df(4*4, 0, 0, 2) << std::endl;

  if (N_P > 1 and time >dust_start_injection and Relaxation_Flag){
    pmb->pphase_change->RelaxationSource(pmb, time, dt, gm0, alpha_vis, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s, v_frag);
  }
// std::cout << cons_df(4*0, 0, 0, 2) << std::endl;
// std::cout << cons_df(4*1, 0, 0, 2) << std::endl;
// std::cout << cons_df(4*3, 0, 0, 2) << std::endl;
// std::cout << cons_df(4*4, 0, 0, 2) << std::endl;
// std::cout << "........" << std::endl;

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
        Real rad;
        rad = pcoord->x1v(i);
        Real omega_dyn = std::sqrt(gm0/std::pow(rad,3.0));

        // vapor diffusivity
        Real &vapor_diffusivity = nu_dust(vapor_id, k, j, i);
        Real &gas_nu = phydro->hdif.nu(HydroDiffusion::DiffProcess::alpha, k, j, i);
        // gas_nu = alpha_vis* std::pow(rad/r0, 1.0); // fix the gas viscosity.
        // artifical decay for outer boundary
        Real w_damp = 0.05*(x1max-x1min);
        Real f_decay_art = std::tanh(std::pow((rad-x1max)/w_damp ,2.0)); // outer bc decay
        vapor_diffusivity = gas_nu * f_decay_art;

        // Loop over pebbles to calculate diffusivity per pebble
        for (int p = 0; p < N_P; ++p) {
          // Get stopping time for this pebble (all compositions have same stopping time)
          int dust_id_first = N_Z * p; // first composition of pebble p
          int dustn_id = N_P*N_Z + 1 + p; // corresponding number density index for this pebble [26.03.31]Zhixuan
          Real t_stop = stopping_time(dust_id_first, k, j, i);
          Real t_stop_n = stopping_time(dustn_id, k, j, i);

          Real taus_n = t_stop_n*omega_dyn; 
          Real &diffusivity_n = nu_dust(dustn_id, k, j, i);
            diffusivity_n = gas_nu/(1.+SQR(taus_n));
          Real &soundspeed_n  = cs_dust(dustn_id, k, j, i);
            soundspeed_n        = std::sqrt(diffusivity_n/omega_dyn);

          // Calculate diffusivity for all compositions of this pebble
          for (int z = 0; z < N_Z; ++z) {
            int dust_id = N_Z * p + z;

            Real taus_peb = t_stop*omega_dyn;
            Real &diffusivity = nu_dust(dust_id, k, j, i);
            diffusivity = gas_nu/(1.+SQR(taus_peb));

            Real &soundspeed  = cs_dust(dust_id, k, j, i);
            soundspeed        = std::sqrt(diffusivity/omega_dyn);

            if (std::isnan(diffusivity) || std::isnan(soundspeed)){
              std::cout << "Error: NaN detected in diffusivity or soundspeed for pebble " << p << " composition " << z << " at (k,j,i)=("<<k<<","<<j<<","<<i<<")." << std::endl;
            }
            // std::cout << "Diffusivity for pebble " << p << " composition " << z << " at (k,j,i)=("<<k<<","<<j<<","<<i<<") is " << diffusivity << std::endl;
          }
        }
      }
    }
  }
  
  return;
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
  Real cs2_mid;

  for (int n=0; n<5; ++n) {
    prs_sigma(n) = prim(IPR, k, j, i+n-2);
    sigma_gas(n) = prim(IDN, k, j, i+n-2);
    sigma_vapor(n) = prim_df(4*vapor_id, k, j, i+n-2);
    fv(n) = sigma_vapor(n)/sigma_gas(n);

    Real mu_gas = get_mu(fv(n));
    Real Tem = Tem_gas(rad(n));
    Real cs2 = Tem/(mu_gas*KELVIN);
    if (n == 2){
      cs2_mid = cs2;
    }
    h_gas(n) = std::sqrt(cs2)/std::sqrt(gm0/(rad(n)*rad(n)*rad(n)));

    // h_gas(n) = Get_H_gas(rad(n),fv(n));

    // full model-- taking into account the change of mu
    // prs(n) = prs_sigma(n)/(std::sqrt(2.0*PI)*h_gas(n));

    // keep eta vk unchanged.-- simple model
    tem = Tem_gas(rad(n));
    rho_g_3D = Sigma_gas(rad(n))/(std::sqrt(2.0*PI)*h_gas(n)); 
    prs(n) = rho_g_3D*tem/(mu_xy*KELVIN);
  }

  Real dr = fabs(rad(1)-rad(0));
  // dfdx[i] = (-y[i+2]+8*y[i+1]-8*y[i-1]+y[i-2])/(12*dx);
  Real dprsdr = (-prs(4) + 8.0*prs(3)-8.0*prs(1)+prs(0))/(12.0*dr);
  // Real dprsdr = (prs(3)-prs(2))/(dr);
  Real vk = std::sqrt(gm0/rad(2));
  Real eta_vk = 0.5*cs2_mid/vk*(rad(2)/prs(2))*dprsdr;

  return eta_vk;
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
        Real rad;
        rad = pcoord->x1v(i);

        Real gas_sigma = prim(IDN, k, j, i);
        Real Tem = Tem_gas(rad);

        // vapor stopping time: set to a big value.
        Real omega_dyn = std::sqrt(gm0/std::pow(rad,3.0));
        Real &vapor_stopping_time = stopping_time(vapor_id, k, j, i);
        vapor_stopping_time = Stokes_number[vapor_id]/omega_dyn;

        Real mu = get_mu(prim_df(4*(vapor_id), k, j, i)/gas_sigma);
        Real cs2 = Tem/(mu*KELVIN); 
        Real h_gas = std::sqrt(cs2)/omega_dyn; 
        Real gas_dens = gas_sigma/(std::sqrt(2.0*PI)*h_gas);
        // Loop over pebbles to calculate stopping time per pebble
        for (int p = 0; p < N_P; ++p) {
          AthenaArray<Real> rho_dustfluid_array_pebble;
          rho_dustfluid_array_pebble.NewAthenaArray(N_Z);

          // collect dustfluid of the same pebble (primitive variables)
          // [25.11.26]lzx: let's avoid using same variable name 'z' here
          Real rho_Np;
          int n_id = 4*(N_P*N_Z + 1 + p);
          for (int zi = 0; zi < N_Z; ++zi) {
            int dust_id = N_Z * p + zi;
            int rho_id = 4*dust_id;

            Real St = stopping_time(dust_id, k, j, i)*omega_dyn;
            Real h_peb = h_gas/std::sqrt(1.0 + St/alpha_vis*(1.0 + 2.0*St)/(1.0 + St));
            rho_dustfluid_array_pebble(zi) = prim_df(rho_id, k, j, i)/std::sqrt(2.0*PI)/h_peb;
            if (rho_dustfluid_array_pebble(zi) <=0.0){
                std::cout << "Warning: rho_dustfluid_array_pebble < 0 for pebble " << p << " composition " << zi << " at (k,j,i)=("<<k<<","<<j<<","<<i<<")!" << std::endl;
                std::cout << "prim_df(rho_id) = " << prim_df(rho_id, k, j, i) << ", h_peb = " << h_peb << std::endl;
                std::cout << "time = " << pmesh->time << std::endl;
            }
              // Real rho_Np_column = pphase_change->rho_Np_array(p, k, j, i);
              Real rho_Np_column = prim_df(n_id, k, j, i);
              rho_Np = rho_Np_column/std::sqrt(2.0*PI)/h_peb;
          }

          // calculate stopping time for the pebble
          Real rho_v = prim_df(4*vapor_id, k, j, i);
          Real t_stop = pphase_change->Get_stopping_time(pmesh->punit, rho_dustfluid_array_pebble, Tem, gas_dens, rho_v, rho_Np);
        //[26.03.30]Zhixuan: we should specify the stopping time for number density 
        Real &t_stop_n = pdustfluids->stopping_time_array(n_id/4, k, j, i); 
        t_stop_n = t_stop; 

          // ===== upper limit, St can be extremely high for upper layer =====
          t_stop = (t_stop > 0.5) ? 0.5 : t_stop;

          // apply stopping time to all compositions of this pebble
          for (int z = 0; z < N_Z; ++z) {
            int dust_id = N_Z * p + z;
            Real &st_time = stopping_time(dust_id, k, j, i);

            // if (dust_id == 0){
            //     st_time = 1.e-1/omega_dyn;
            // } else {
            //     st_time = 0.3/omega_dyn;
            // }
            if(pmesh->time > dust_start_injection){
              st_time = t_stop;
            }else{
              st_time = 1.e-8;
            }
            // apply st floor 
            st_time = (st_time > st_floor) ? st_time : st_floor;
          }
        }
      }
    }
  }
  
  return;
}
Real Sigma_gas(const Real rad){
    return sigma0*std::sqrt(2*PI)*std::pow(rad/r0, dslope);
}
Real Tem_gas(const Real rad){
    return Tem0*std::pow(rad/r0, Tslope);
}
Real nu_gas(const Real Tem, const Real rad, Real fv){
    Real mu = get_mu(fv);
    Real cs = std::sqrt(Tem/(mu*KELVIN));
    Real omega = std::sqrt(gm0/(rad*rad*rad));
    Real h_gas = cs/omega;
    return alpha_vis*cs*h_gas;
}

Real get_mu(Real fv){
    return 1./((1.-fv)/mu_xy + fv / mu_z);
}
void sigma_interpolate_inner_nomatter(const Real r_active, const Real r_ghost, const Real sigma_active,
    Real &sigma_ghost, const Real vr_active, const Real vr_ghost) {
  //if (sigma_active < TINY_NUMBER)
    //vr_ghost = vr_active >= 0.0 ? ((sigma_active+TINY_NUMBER)*r_active*vr_active)/(sigma_ghost*r_ghost) : 0.0;
  //else
  //vr_ghost = vr_active >= 0.0 ? (sigma_active*r_active*vr_active)/(sigma_ghost*r_ghost) : 0.0;
  sigma_ghost = (sigma_active*r_active*vr_active)/(vr_ghost*r_ghost);
  return;
}


void LocalIsothermalEOS(MeshBlock *pmb, const Real time, const Real dt,
    const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_df,
    const AthenaArray<Real> &bcc, AthenaArray<Real> &cons, AthenaArray<Real> &cons_df) {

  // Local Isothermal equation of state
  Real rad;
  int is = pmb->is; int ie = pmb->ie;
  int js = pmb->js; int je = pmb->je;
  int ks = pmb->ks; int ke = pmb->ke;

  // Real igm1 = 1.0/(gamma_gas - 1.0);
  for (int k=ks; k<=ke; ++k) { // include ghost zone
    for (int j=js; j<=je; ++j) { // prim, cons
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        rad = pmb->pcoord->x1v(i);

        Real &gas_dens = cons(IDN, k, j, i);
        Real &gas_mom1 = cons(IM1, k, j, i);
        Real &gas_mom2 = cons(IM2, k, j, i);
        Real &gas_mom3 = cons(IM3, k, j, i);
        Real &gas_erg  = cons(IEN, k, j, i);

        const Real &rho_v = cons_df(4*vapor_id, k, j, i);

        // temperature profile: fixed
        Real Tem = Tem_gas(rad); 
        // Real gas_vel1 = gas_mom1/gas_dens;
        // Real gas_vel2 = gas_mom2/gas_dens;
        // Real gas_vel3 = gas_mom3/gas_dens;

        // gen density (do not allow H/He diffusion):        
        // gas_dens = rho_v + Sigma_gas(rad);
        // gas_mom1 = gas_dens*gas_vel1;
        // gas_mom2 = gas_dens*gas_vel2;
        // gas_mom3 = gas_dens*gas_vel3;

        // get pressure of local temperature
        Real fv = rho_v/gas_dens;
        if(std::isnan(fv)){
          std::cout <<"fv= nan in LocalIsothermalEOS" << std::endl;
          std::cout <<"At (k,j,i)=("<<k<<","<<j<<","<<i<<")"<< std::endl;
          std::cout << "rhov =" << rho_v << std::endl;
          std::cout << "gas_dens =" << gas_dens << std::endl;
          std::cout << "prim = "<< prim(IDN, k, j, i) << std::endl;
          // fv = 0.0;
        }
        Real mu = get_mu(fv);
        Real press = gas_dens * Tem /(KELVIN*mu);
        Real gamma = pmb->peos->calc_gamma(fv);

        gas_erg    = press/(gamma-1.0) + 0.5*(SQR(gas_mom1) + SQR(gas_mom2))/gas_dens;

      }
    }
  } 

  return;
}

Real PoverRho(const Real rad, const Real phi, const Real z) {
  Real poverr;
  poverr = p0_over_r0*std::pow(rad/r0, p_over_rho_slope);
  return poverr;
}
Real DenProfileCyl_dust(const Real rad, const Real phi, const Real z, const Real den_ratio, const Real H_ratio) {
  Real den;
  Real p_over_r = p0_over_r0;
  if (NON_BAROTROPIC_EOS) p_over_r = PoverRho(rad, phi, z);
  Real denmid = den_ratio*rho0*std::pow(rad/r0, -2.25);
  Real dentem = denmid*std::exp(gm0/(SQR(H_ratio)*p_over_r)*(1./std::sqrt(SQR(rad)+SQR(z))-1./rad));
  den         = dentem;
  return std::max(den,dffloor);
}
Real DenProfileCyl_gas(const Real rad, const Real phi, const Real z) {
  Real den;
  Real p_over_r = p0_over_r0;
  if (NON_BAROTROPIC_EOS) p_over_r = PoverRho(rad, phi, z);
  Real denmid = rho0*std::pow(rad/r0,-2.25);
  Real dentem = denmid*std::exp(gm0/p_over_r*(1./std::sqrt(SQR(rad)+SQR(z))-1./rad));
  den = dentem;
  return std::max(den,dffloor);
}

Real VelProfileCyl_gas_fv_T(const Real rad, const Real phi, const Real z, const Real fv, const Real Tem) {
    Real p_over_r = PoverRho(rad, phi, z);
    Real mu = get_mu(fv);
    Real cs2 = Tem/(mu*KELVIN); 
    Real vel = (dslope+p_over_rho_slope)*cs2/(gm0/rad) + (1.0+p_over_rho_slope)
             - p_over_rho_slope*rad/std::sqrt(rad*rad+z*z);
  // Real vel = (dslope + p_over_rho_slope - Tslope/2.0 - 1.5)*p_over_r/(gm0/rad) + (1.0+p_over_rho_slope)
  //            - p_over_rho_slope*rad/std::sqrt(rad*rad+z*z);
    vel = std::sqrt(gm0/rad)*std::sqrt(vel) - rad*Omega0;
    return vel;
}

Real VelProfileCyl_dust(const Real rad, const Real phi, const Real z) {
  Real dis = std::sqrt(SQR(rad) + SQR(z));
  Real vel = std::sqrt(gm0/dis) - rad*Omega0;
  return vel;
}
Real Get_H_gas(const Real rad, Real fv) {
  Real tem  = Tem_gas(rad);
  // Real mu = Get_mu(fv);
  Real mu = mu_xy; // simple model, scale height indep. of mu
  Real cs = std::sqrt(tem/(KELVIN*mu));
  Real rad3 = SQR(rad)*rad;
  Real omega = std::sqrt(gm0/rad3);

  return cs/omega;
  // return 1.0/std::sqrt(2.0*PI);
}

void drift_vel(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    //Real x3 = pmb->pcoord->x3v(k);
    for (int j=pmb->js; j<=pmb->je; ++j) {
      //Real x2 = pmb->pcoord->x2v(j);
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        //Real x1 = pmb->pcoord->x1v(i);
        AthenaArray<Real> r_five_point;
        r_five_point.NewAthenaArray(5);
        for (int n=0; n<5; ++n){
          r_five_point(n) = pmb->pcoord->x1v(i+n-2);
        }

        // Real dr = pmb->pcoord->dx1v(i);
        Real rad;
        rad = pmb->pcoord->x1v(i);
        const Real &gas_rho  = prim(IDN, k, j, i);
        const Real &gas_vel1 = prim(IM1, k, j, i);

        Real &gas_dens = cons(IDN, k, j, i);
        Real &gas_mom1 = cons(IM1, k, j, i);

        Real v_acc_gas = gas_mom1/gas_dens;

        if(std::isnan(gas_dens)){
          std::cout <<"gas_dens= nan" << std::endl;
          quick_exit(1);
        }
        //-----------------------------------------------------------
        // calcuate gas velocity analytically
        // simple model
        Real Tem = Tem_gas(rad);
        Real gas_nu = nu_gas(Tem, rad, 0.0);
        v_acc_gas = -1.5*gas_nu/rad; // negative accretion vel
        // self-consistent calculation
        // v_acc_gas = Get_v_gas(k, j, i, pmb, prim, prim_df);

        if(std::isnan(v_acc_gas)){
          std::cout <<"v_acc_gas= nan" << std::endl;
        }
        //
        // Real delta_mom1 = gas_dens*v_acc_gas-gas_mom1;
        // gas_mom1 += delta_mom1;
        //
        // if (NON_BAROTROPIC_EOS && (!Isothermal_Flag)) {
        //   Real &gas_erg  = cons(IEN, k, j, i);
        //   gas_erg       += (delta_mom1*gas_vel1);
        // }
        //-----------------------------------------------------------

        Real eta_vk = Get_eta_vk(r_five_point, k, j, i, prim, prim_df);
        // Real eta_vk = Get_eta_vk_old(rad,k,j,i,prim,fv,dr);

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;

            const Real &dust_rho  = prim_df(rho_id, k, j, i);
            const Real &dust_vel1 = prim_df(v1_id,  k, j, i);

            Real &dust_dens = cons_df(rho_id, k, j, i);
            Real &dust_mom1 = cons_df(v1_id,  k, j, i);

            if(dust_id == vapor_id){
              // copy gas velocity to tracer
              dust_mom1 = dust_dens*v_acc_gas;
            }else{
              // calculate stokes number
              Real omega_dyn   = std::sqrt(gm0/(rad*rad*rad));
              Real taus = pmb->pdustfluids->stopping_time_array(dust_id, k, j, i);
              taus *= omega_dyn;
              
              Real v_drift_peb = (v_acc_gas + 2.0*eta_vk*taus)/(1.+SQR(taus));
              if(std::isnan(v_drift_peb)){
                std::cout <<"v_drift_peb= nan" << std::endl;
                quick_exit(1);
              }
              dust_mom1 = dust_dens*v_drift_peb;
            }
          }
        }
      }
    }
  }
  return;
}
} // namespace
void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  Real rad_active(0.0), phi_active(0.0), z_active(0.0);
  Real rad_ghost(0.0),  phi_ghost(0.0),  z_ghost(0.0);
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        Real rad_active = pco->x1v(il); 
        Real rad_ghost = pco->x1v(il-i);
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

        Real Tem_ghost = Tem_gas(rad_ghost);
        Real gas_nu = nu_gas(Tem_ghost,rad_ghost,0.0);
        Real v_acc_gas = -1.5*gas_nu/rad_ghost; // negative accretion vel
        Real vel_gas_phi = VelProfileCyl_gas_fv_T(rad_ghost, phi_ghost, z_ghost, 0.0, Tem_gas(rad_ghost));

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
        
        Real fv = prim_df(4*vapor_id, k, j, il)/gas_sigma_active;
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
          Real T1 = Tem_gas(rad_ghost);
          Real mu1 = get_mu(fv);
          gas_pres_ghost = gas_sigma_ghost*T1/(mu1*KELVIN);
        }

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<N_P*N_Z + 1; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;
            
            if(dust_id != vapor_id){
              // free outflow for dust
              prim_df(rho_id,k,j,il-i) = prim_df(rho_id,k,j,il);
              prim_df(v1_id,k,j,il-i) = prim_df(v1_id,k,j,il);

              Real v_dust_phi = VelProfileCyl_dust(rad_ghost, phi_ghost, z_ghost);
                if (pmb->porb->orbital_advection_defined)
                  v_dust_phi -= vel_K;
              // prim_df(v2_id,k,j,il-i) = v_dust_phi; 
              prim_df(v2_id,k,j,il-i) = prim_df(v2_id,k,j,il);
              prim_df(v3_id,k,j,il-i) = prim_df(v3_id,k,j,il);
              //maybe we should also set the boundary condtion for number density 
              // check the number density: 
                // std::cout << "at (k,j,i)=("<<k<<","<<j<<","<<il-i<<"), dust_id = "<<dust_id<<" density = "<<prim_df(rho_id,k,j,il-i)<<std::endl;
                // std::cout << "number density_0 = " << pmb->pphase_change->rho_Np_array(dust_id/N_Z, k, j, 0)  << std::endl;          
                // std::cout << "number density_1 = " << pmb->pphase_change->rho_Np_array(dust_id/N_Z, k, j, 1)  << std::endl;          
                // std::cout << "number density_2 = " << pmb->pphase_change->rho_Np_array(dust_id/N_Z, k, j, 2)  << std::endl;          
              // [26.03.27]Zhixuan: now set the 
                //[26.03.27]Zhixuan: now set the number density to the number density function in the dustfluids
              //change number density accordingly. 
                // pmb->pphase_change->rho_Np_array(dust_id/N_Z, k,j,il-i) = pmb->pphase_change->rho_Np_array(dust_id/N_Z, k, j, il);
                int dustn_id = N_P*N_Z + 1 + dust_id; 
                int n_id = 4*dustn_id;
                int nv1_id = n_id + 1;
                int nv2_id = n_id + 2;
                int nv3_id = n_id + 3;

                prim_df(n_id, k, j, il-i) = prim_df(n_id, k, j, il);
                prim_df(nv1_id, k, j, il-i) = prim_df(nv1_id, k, j, il);
                prim_df(nv2_id, k, j, il-i) = prim_df(nv2_id, k, j, il);
                prim_df(nv3_id, k, j, il-i) = prim_df(nv3_id, k, j, il);

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
// void DiskInnerX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df, FaceField &b,
//                  Real time, Real dt,
//                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
//   OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
//   Real rad_active(0.0), phi_active(0.0), z_active(0.0);
//   Real rad_ghost(0.0),  phi_ghost(0.0),  z_ghost(0.0);
//   for (int k=kl; k<=ku; ++k) {
//     for (int j=jl; j<=ju; ++j) {
//       for (int i=1; i<=ngh; ++i) {
//         rad_active = pco->x1v(il); 
//         rad_ghost = pco->x1v(il-i);
//         Real vel_K     = vK(pmb->porb, pco->x1v(il-i), pco->x2v(j), pco->x3v(k));
//
//         // std::cout<< prim(IDN, k, j, il-i) << std::endl;
//         // std::cout<< prim(IM1, k, j, il-i) << std::endl;
//         // std::cout<< prim(IEN, k, j, il-i) << std::endl;
//         //
//         // std::cout<< prim(IDN, k, j, il) << std::endl;
//         // std::cout<< prim(IM1, k, j, il) << std::endl;
//
//         Real &gas_sigma_ghost  = prim(IDN, k, j, il-i);
//         Real &gas_vel1_ghost = prim(IVX, k, j, il-i);
//         Real &gas_vel2_ghost = prim(IVY, k, j, il-i);
//
//         Real &gas_pres_ghost = prim(IPR, k, j, il-i);
//
//         Real &gas_sigma_active  = prim(IDN, k, j, il); 
//         Real &gas_vel1_active = prim(IM1, k, j, il);
//
//         Real Tem = Tem_gas(rad_ghost);
//         Real cs = std::sqrt(Tem/(mu_xy*KELVIN));
//         Real h_gas = cs/std::sqrt(gm0/(rad_ghost*rad_ghost*rad_ghost));
//
//         Real gas_nu = nu_gas(Tem, rad_ghost,0.0);
//         Real v_acc_gas = -1.5*gas_nu/rad_ghost; // negative accretion vel
//         Real v_phi_gas = VelProfileCyl_gas_fv_T(rad_ghost, 0.0, 0.0, 0.0, Tem);
//         if (pmb->porb->orbital_advection_defined)
//           v_phi_gas -= vel_K;
//         // Real v_acc_gas = gas_vel1_active; // negative accretion vel
//         gas_vel1_ghost = v_acc_gas;
//         gas_vel2_ghost = v_phi_gas; 
//
//         // Real Mdot_xy = M_dot_g*CONST_Msun/CONST_yr;
//         // Mdot_xy /= (SQR(UNIT_LENGTH)*UNIT_LENGTH*UNIT_DENSITY/UNIT_T);
//         // Real xy_sigma_ghost = Mdot_xy/(2.0*PI*rad_ghost*fabs(v_acc_gas));
//
//         Real fv = prim_df(4*(NDUSTFLUIDS-1), k, j, il)/gas_sigma_active;
//         gas_sigma_ghost = DenProfileCyl_gas(rad_ghost, 0.0, 0.0)*std::sqrt(2.0*PI)*h_gas/(1.0-fv); 
//         //check 
//         // std::cout << "gas_sigma_ghost = " << gas_sigma_ghost << std::endl;
//         // std::cout << "v_phi_gas = " << v_phi_gas << std::endl;
//         // std::cout << "gas_vel1_ghost = " << gas_vel1_ghost << std::endl;
//
//         // sigma_interpolate_inner_nomatter(rad_active, rad_ghost, gas_sigma_active, gas_sigma_ghost,
//         //     gas_vel1_active, gas_vel1_ghost);
//
//         // Real &vapor_sigma_ghost  = prim_df(4*(NDUSTFLUIDS-1), k, j, il-i);
//         // Real &vapor_vel1_ghost = prim_df(4*(NDUSTFLUIDS-1) + 1, k, j, il-i);
//         // Real &vapor_sigma_active  = prim_df(4*(NDUSTFLUIDS-1), k, j, il);
//         // Real &vapor_vel1_active = prim_df(4*(NDUSTFLUIDS-1) + 1, k, j, il);
//
//         // vapor_vel1_ghost = gas_vel1_ghost;
//         // sigma_interpolate_inner_nomatter(rad_active, rad_ghost, vapor_sigma_active, vapor_sigma_ghost,
//         //     vapor_vel1_active, vapor_vel1_ghost);
//
//         // gas_sigma_ghost = xy_sigma_ghost + vapor_sigma_ghost;
//
//         if (NON_BAROTROPIC_EOS){
//           Real T1 = Tem_gas(rad_ghost);
//           Real mu1 = get_mu(fv);
//           gas_pres_ghost = gas_sigma_ghost*T1/(mu1*KELVIN);
//           // gas_pres_ghost = gas_sigma_ghost*T1/(mu_xy*KELVIN);
//         }
//
//         if (NDUSTFLUIDS > 0) {
//           for (int n=0; n<NDUSTFLUIDS; ++n) {
//             int dust_id = n;
//             int rho_id  = 4*dust_id;
//             int v1_id   = rho_id + 1;
//             int v2_id   = rho_id + 2;
//             int v3_id   = rho_id + 3;
//
//             if(dust_id != vapor_id){
//               // free outflow for dust
//                 prim_df(rho_id,k,j,il-i) = prim_df(rho_id,k,j,il+i-1);
//                 prim_df(v1_id,k,j,il-i) = prim_df(v1_id,k,j,il+i-1);
//                 prim_df(v2_id,k,j,il-i) = prim_df(v2_id,k,j,il+i-1);
//                 prim_df(v3_id,k,j,il-i) = prim_df(v3_id,k,j,il+i-1);
//             }else{
//               Real &vapor_sigma_ghost  = prim_df(rho_id, k, j, il-i);
//               Real &vapor_vel1_ghost = prim_df(v1_id,  k, j, il-i);
//               Real &vapor_vel2_ghost = prim_df(v2_id,  k, j, il-i);
//               Real &vapor_vel3_ghost = prim_df(v3_id,  k, j, il-i);
//
//               // keep the consistency between gas and vapor.
//               vapor_vel1_ghost = gas_vel1_ghost;
//               vapor_vel2_ghost = v_phi_gas;
//               vapor_vel3_ghost = 0.0;
//               vapor_sigma_ghost = fv*gas_sigma_ghost;
//             //check 
//             }
//           }
//         }
//       }
//     }
//   }
// }

// void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
//                   FaceField &b, Real time, Real dt,
//                   int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
//   Real rad(0.0),  phi(0.0),  z(0.0);
//   Real rad_active(0.0), phi_active(0.0), z_active(0.0);
//   for (int k=kl; k<=ku; ++k) {
//     for (int j=jl; j<=ju; ++j) {
//       for (int i=1; i<=ngh; ++i) {
//         rad_active = pco->x1v(iu);
//         Real &gas_sigma_active  = prim(IDN, k, j, iu);
//         Real &gas_vel1_active = prim(IM1, k, j, iu);
//         Real &gas_pres_active = prim(IPR, k, j, iu);
//
//         rad = pco->x1v(iu+i);
//         Real &gas_sigma  = prim(IDN, k, j, iu+i);
//         Real &gas_vel1 = prim(IVX, k, j, iu+i);
//         Real &gas_vel2 = prim(IVY, k, j, iu+i);
//         Real &gas_pres = prim(IEN, k, j, iu+i);
//
//         Real Tem = Tem_gas(rad);
//         Real cs2 = Tem/(mu_xy*KELVIN);
//         Real omega_dyn = std::sqrt(gm0/std::pow(rad,3.0));
//         Real h_gas = std::sqrt(cs2)/omega_dyn;
//
//
//         gas_sigma = Sigma_gas(rad);
//         Real fv = prim_df(4*(NDUSTFLUIDS-1), k, j, iu+i)/gas_sigma;
//
//         Real vk = std::sqrt(gm0/rad);
//         Real eta_vk = 0.5*cs2/vk*(Tslope + dslope);
//
//         Real v_acc_gas=0.0;
//         Real gas_nu = nu_gas(Tem, rad,0.0);
//         v_acc_gas = -1.5*gas_nu/rad; // negative accretion vel
//         Real v_phi_gas = VelProfileCyl_gas_fv_T(rad, 0.0, 0.0, fv, Tem);
//         if (pmb->porb->orbital_advection_defined)
//           v_phi_gas -= std::sqrt(gm0/rad);
//         // v_acc_gas = gas_vel1_active; // negative accretion vel
//         // v_acc_gas = Get_v_gas(k,j,i,pmb,prim,prim_df);
//
//         // Real Mdot_xy = -1.e-8*Constants::solar_mass_cgs/Constants::yr_cgs;
//         // Mdot_xy /= pmb->pmy_mesh->punit->code_mass_cgs/ pmb->pmy_mesh->punit->code_time_cgs; // in code unit      
//
//         // gas_sigma = gas_sigma_active;
//         gas_vel1 = v_acc_gas; 
//         gas_vel2 = v_phi_gas; 
//         v_acc_gas = gas_vel1;
//
//         if (NON_BAROTROPIC_EOS){
//           gas_pres = gas_sigma*cs2; 
//         }
//
//         if (NDUSTFLUIDS > 0) {
//           for (int p=0; p<N_P; ++p) {
//             for (int z = 0; z < N_Z; ++z) {
//               int dust_id = p*N_Z + z;
//               int rho_id  = 4*dust_id;
//               int v1_id   = rho_id + 1;
//               int v2_id   = rho_id + 2;
//
//               Real &dust_sigma  = prim_df(rho_id, k, j, iu+i);
//               Real &dust_vel1 = prim_df(v1_id,  k, j, iu+i);
//               Real &dust_vel2 = prim_df(v2_id,  k, j, iu+i);
//
//               Real t_stop = pmb->pdustfluids->stopping_time_array(dust_id, k,j,iu-i+1);
//               Real taus = t_stop*omega_dyn;
//               Real v_drift_peb = (v_acc_gas + 2.0*eta_vk*taus)/(1.+SQR(taus));
//
//               dust_vel1 = v_drift_peb;
//               dust_vel2 = 0.0;    
//
//               dust_sigma = prim_df(rho_id, k, j, iu+i-1);
//               // dust_sigma = prim_df(rho_id, k, j, iu); 
//               // dust_sigma = 0.0;
//             }
//           }
//           Real &vapor_sigma = prim_df(vapor_id*4, k, j, iu+i);
//           Real &vapor_vel1 = prim_df(vapor_id*4+1, k, j, iu+i);
//           Real &vapor_vel2 = prim_df(vapor_id*4+2, k, j, iu+i);
//           vapor_vel1 = gas_vel1;
//           vapor_vel2 = gas_vel2; 
//           vapor_sigma = gas_sigma*initial_D2G[vapor_id];
//
//
//         }
//       }
//     }
//   }
// }
void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad_ghost(0.0),  phi_ghost(0.0),  z_ghost(0.0);
  Real rad_active(0.0), phi_active(0.0), z_active(0.0);
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        rad_active = pco->x1v(iu);
        Real &gas_rho_active  = prim(IDN, k, j, iu);
        Real &gas_vel1_active = prim(IM1, k, j, iu);
        Real &gas_pres_active = prim(IPR, k, j, iu);
       
        rad_ghost = pco->x1v(iu+i);
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
        
        Real Tem = Tem_gas(rad_ghost);
        Real cs = std::sqrt(Tem/(mu_xy*KELVIN)); 
        Real vk_0 = std::sqrt(gm0/rad_ghost);

        //[26.03.23]TBD: not sure why here we shoud use slope of sigma rather than the slope of rho
        Real eta_vk = 0.5*SQR(cs)/vk_0*(Tslope+dslope);

        // viscous solution
        Real v_acc_gas;
        Real gas_nu = nu_gas(Tem, rad_ghost,0.0);
        v_acc_gas = -1.5*gas_nu/rad_ghost; // negative accretion vel

        Real vel_gas_phi = VelProfileCyl_gas_fv_T(rad_ghost, phi_ghost, z_ghost, 0.0, Tem);
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
          for (int n=0; n<N_P*N_Z + 1; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real &dust_sigma_ghost  = prim_df(rho_id, k, j, iu+i);
            Real &dust_vel1_ghost = prim_df(v1_id,  k, j, iu+i);
            Real &dust_vel2_ghost = prim_df(v2_id,  k, j, iu+i);
            Real &dust_vel3_ghost = prim_df(v3_id,  k, j, iu+i);
            // Real &rho_Np_ghost = pmb->pphase_change->rho_Np_array(dust_id/N_Z, k, j, iu+i);

            if(dust_id != vapor_id){
                Real t_stop = pmb->pdustfluids->stopping_time_array(dust_id, k,j,iu-i+1);
                Real taus = t_stop*omega_dyn;
                Real v_drift_peb = (v_acc_gas + 2.0*eta_vk*taus)/(1.+SQR(taus));
                // std::cout << "taus = " << taus << std::endl;
                // std::cout << "v_drift_peb = " << v_drift_peb << std::endl;
                // std::cout << "eta_vK = " << eta_vk << std::endl;
                // std::cout << "v_acc_gas = " << v_acc_gas << std::endl;
              
                // dust_vel1_ghost = v_drift_peb;
                // // Real H_peb = h_gas*std::sqrt(nu_alpha/(nu_alpha+taus));
                // Real Mdot_peb = p2g_flux[dust_id]*M_dot_g*Constants::solar_mass_cgs/Constants::yr_cgs;
                // Mdot_peb /= (SQR(UNIT_LENGTH)*std::sqrt(2.0*PI)*H_peb*UNIT_LENGTH*UNIT_DENSITY/UNIT_T); // in code unit
                // Mdot_peb /= (pmb->pmy_mesh->punit->code_mass_cgs/ pmb->pmy_mesh->punit->code_time_cgs); // in code unit 
                // Real sigma_peb = Mdot_peb/(2.0*PI*rad_ghost*fabs(v_drift_peb));
                //
                // dust_sigma_ghost = sigma_peb*0.5*(1.0-std::exp(-0.5*SQR((time-dust_start_injection)/injection_Tsoft)));
                Real vel_dust_phi = VelProfileCyl_dust(rad_ghost, phi_ghost, z_ghost);
                if (pmb->porb->orbital_advection_defined)
                    vel_dust_phi -= vel_K;

                dust_sigma_ghost = prim_df(rho_id, k, j, iu)*SQR(rad_active/rad_ghost);
                dust_vel1_ghost = v_drift_peb;
                dust_vel2_ghost = vel_dust_phi;
                // dust_vel1_ghost = prim_df(v1_id,  k, j, iu);
                dust_vel3_ghost = 0.0;

                //[26.03.27]Zhixuan: now set the number density to the number density function in the dustfluids
                // rho_Np_ghost = pmb->pphase_change->rho_Np_array(dust_id/N_Z, k, j, iu);
                int dustn_id = N_P*N_Z + 1 + dust_id/N_Z; 
                int n_id = 4*dustn_id;
                int nv1_id = n_id + 1;
                int nv2_id = n_id + 2;
                int nv3_id = n_id + 3;

                prim_df(n_id, k, j, iu+i) = prim_df(n_id, k, j, iu)*SQR(rad_active/rad_ghost);
                // prim_df(n_id, k, j, iu+i) = dust_sigma_ghost/pmb->pphase_change->m_p_array(dust_id/N_Z, k, j, iu+i);
                prim_df(nv1_id, k, j, iu+i) = v_drift_peb;
                prim_df(nv2_id, k, j, iu+i) = vel_dust_phi;
                prim_df(nv3_id, k, j, iu+i) = 0.0;

            }else{
                dust_sigma_ghost = DenProfileCyl_gas(rad_ghost, phi_ghost, z_ghost)*initial_D2G[dust_id]*std::sqrt(2.*PI)*h_gas;
                dust_vel1_ghost = gas_vel1_ghost;
                dust_vel2_ghost = vel_gas_phi;
                dust_vel3_ghost = 0.0;
            }

          }
        }
      }
    }
  }
}
