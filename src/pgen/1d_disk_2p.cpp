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
    Real pvalue, p0_over_r0, Omega0, dslope;   
    Real initial_D2G[NDUSTFLUIDS], H_ratio[NDUSTFLUIDS], Stokes_number[NDUSTFLUIDS], p2g_flux[NDUSTFLUIDS];
    Real Sigma_gas(const Real rad);
    Real nu_gas(const Real Tem, const Real rad, Real fv);
    Real get_mu(const Real fv); 
    Real Tem_gas(const Real rad);

    void LocalIsothermalEOS(MeshBlock *pmb, const Real time, const Real dt,
        const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_df,
        const AthenaArray<Real> &bcc, AthenaArray<Real> &cons, AthenaArray<Real> &cons_df);
    void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
      const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time,
      const int il, const int iu, const int jl, const int ju, const int kl, const int ku);
    void MyDustDiffusivity(DustFluids *pdf, MeshBlock *pmb,
          const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
          const AthenaArray<Real> &stopping_time,
          AthenaArray<Real> &nu_dust, AthenaArray<Real> &cs_dust,
          int is, int ie, int js, int je, int ks, int ke);
    void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
        const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
        AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
    void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                      FaceField &b, Real time, Real dt,
                      int il, int iu, int jl, int ju, int kl, int ku, int ngh); 
    void sigma_interpolate_inner_nomatter(const Real r_active, const Real r_ghost, const Real sigma_active,
        Real &sigma_ghost, const Real vr_active, const Real vr_ghost);
    void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df, FaceField &b,
                     Real time, Real dt,
                     int il, int iu, int jl, int ju, int kl, int ku, int ngh);
    // void MySource(MeshBlock *pmb, const Real time, const Real dt,
    //     const AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
    //     const AthenaArray<Real> &bcc,
    //     AthenaArray<Real> &cons, AthenaArray<Real> &cons_df); 
    Real VelProfileCyl_gas(const Real rad, const Real phi, const Real z); 
}


void Mesh::InitUserMeshData(ParameterInput *pin) {
    gm0 = pin->GetOrAddReal("problem", "gm0", 0.0);
    r0  = pin->GetOrAddReal("problem", "r0", 3.0);
    sigma0 = pin->GetReal("problem", "sigma0");
    Tem0 = pin->GetOrAddReal("problem", "Tem0", 150.0);
    alpha_vis = pin->GetOrAddReal("problem", "alpha_vis", 1.e-3);
    gamma_gas = pin->GetReal("hydro", "gamma");
    x1min = pin->GetReal("mesh", "x1min"); 
    x1max = pin->GetReal("mesh", "x1max");
    // qvalue = pin->GetReal("problem", "qvalue");
    dust_start_injection = pin->GetReal("problem", "dust_start_injection");
    dffloor = pin->GetOrAddReal("problem", "dffloor", 1.e-15);
    Tslope = pin->GetOrAddReal("problem", "Tslope", -0.5);
    p_over_rho_slope = pin->GetOrAddReal("problem", "p_over_rho_slope", -0.5);
    Omega0    = pin->GetOrAddReal("orbital_advection", "Omega0", 0.0);
    dslope = pin->GetOrAddReal("problem", "dslope", -1.0);

  if (NON_BAROTROPIC_EOS) {
        p0_over_r0 = pin->GetOrAddReal("problem", "p0_over_r0", 0.0025);
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
  AllocateUserOutputVariables(8+2+2);
  // Firstly, we output the variables related to non-tracer dustfluids 
  const std::vector<std::pair<int,const char*>> dustProp = {
    {0,  "st"},
    {1, "m_p"},
    {2, "s_p"},
    {3, "flx_sil_x1"},
  };

  for (int np =0; np<N_P; np++) {
    for (auto &p : dustProp) {
      SetUserOutputVariableName(p.first + np*4, (std::string(p.second) + "_" + std::to_string(np+1)).c_str());
    }
  }

  // Then, we output the variables related to tracer dustfluids 
  int offset_vapor = dustProp.size()*N_P; 
  SetUserOutputVariableName(offset_vapor, "dif");
  SetUserOutputVariableName(offset_vapor+1, "flx_vap_x1");

  // Finally, output the variables related to gas 
  int offset_gas = offset_vapor + 2;
  const std::vector<std::pair<int,const char*>> gasProp = {
    {0,  "Tem"},
    {1, "flx_x1"},
  };

  for (auto &p : gasProp) {
    SetUserOutputVariableName(p.first + offset_gas, p.second);
  }

  
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

                Real sigma_gas = Sigma_gas(rad);
                Real Tem = Tem_gas(rad);
                Real cs2 = Tem/(mu_xy*KELVIN); 
                Real &nu = phydro->hdif.nu(HydroDiffusion::DiffProcess::alpha, k, j, i);
                nu = nu_gas(Tem, rad, 0.0);
                Real v_acc_gas = -1.5*nu/rad; // negative accretion vel
                
                Real vel_gas_phi = VelProfileCyl_gas(rad,phi,0.0);
                
                phydro->u(IDN, k, j, i) = sigma_gas;
                phydro->u(IM1, k, j, i) = sigma_gas*v_acc_gas; //radial momentum
                phydro->u(IM2, k, j, i) = sigma_gas*vel_gas_phi; // azimuthal momentum
                phydro->u(IM3, k, j, i) = 0.0; // vertical momentum
                phydro->u(IEN, k, j, i) = sigma_gas*cs2*igm1;
                phydro->u(IEN, k, j, i) += 0.5*(SQR(phydro->u(IM1, k, j, i)) +SQR(phydro->u(IM2,k,j,i)))/phydro->u(IDN, k, j, i);
                phydro->Tem(k, j, i) = Tem;

                std::cout <<phydro->hdif.alpha_disk_model<<std::endl;
                Real omega_dyn = std::sqrt(gm0/std::pow(rad,3.0));
                Real h_gas = std::sqrt(cs2)/omega_dyn;
                
                //then assige the preperties to dust 
                for (int n=0; n<NDUSTFLUIDS; ++n){
                    int dust_id = n;
                    int rho_id  = 4*dust_id;
                    int v1_id   = rho_id + 1;
                    int v2_id   = rho_id + 2;
                    int v3_id   = rho_id + 3;

                    Real dust_sigma = initial_D2G[dust_id]*sigma_gas;
                    pdustfluids->df_u(rho_id, k, j, i) = dust_sigma; // dust density

                    Real vel_dust_phi = omega_dyn*rad; 

                    if (dust_id ==vapor_id){
                        pdustfluids->df_u(v1_id, k, j, i) = dust_sigma*v_acc_gas; // vapor radial momentum
                        pdustfluids->df_u(v2_id, k, j, i) = dust_sigma*vel_gas_phi; // vapor azimuthal momentum
                        pdustfluids->df_u(v3_id, k, j, i) = 0.0; // vapor vertical momentum
                    } else{
                        pdustfluids->df_u(v1_id, k, j, i) = 0.0; // dust radial momentum
                        pdustfluids->df_u(v2_id, k, j, i) = dust_sigma*vel_dust_phi; // dust azimuthal momentum
                        pdustfluids->df_u(v3_id, k, j, i) = 0.0; // dust vertical momentum
                    }
                // std::cout << "sigma_dust = " << pdustfluids->df_u(rho_id, k, j, i) << " at (k,j,i)=("<<k<<","<<j<<","<<i<<")!" << std::endl;
                // std::cout << "mom_dust = " << pdustfluids->df_u(v1_id, k, j, i) << " at (k,j,i)=("<<k<<","<<j<<","<<i<<")!" << std::endl;
                }
                //check the dust properties 
        std::cout << "At (i,j,k)=(" << i << "," << j << "," << k << "):" << std::endl;
        std::cout << "gas_sigma = " << sigma_gas << std::endl;
        std::cout << "gas_vel1 = " << v_acc_gas<< std::endl;
        std::cout << "gas_vel2 = " << vel_gas_phi<< std::endl;
        std::cout << "dust_sigma_0 = " << pdustfluids->df_u(0,k,j,i) << std::endl;
        std::cout << "st_0" << " = " << pdustfluids->stopping_time_array(0,k,j,i) << std::endl;
        std::cout << "dif_0 = " << pdustfluids->nu_dustfluids_array(0,k,j,i) << std::endl;
        std::cout << "dust_vel1_0 = " << pdustfluids->df_w(1,k,j,i) << std::endl;
        std::cout << "dust_sigma_1 = " << pdustfluids->df_u(4,k,j,i) << std::endl;
        std::cout << "dust_vel1_1 = " << pdustfluids->df_w(5,k,j,i) << std::endl;
        std::cout << "st_1" << " = " << pdustfluids->stopping_time_array(1,k,j,i) << std::endl;
        std::cout << "dif_1 = " << pdustfluids->nu_dustfluids_array(1,k,j,i) << std::endl;
        std::cout << "vapor_sigma = " << pdustfluids->df_u(4*vapor_id,k,j,i) << std::endl;
        std::cout << "vapor_vel1 = " << pdustfluids->df_w(4*vapor_id+1,k,j,i) << std::endl;
        std::cout << "=========================================" << std::endl;

                if (pphase_change != nullptr && N_Z>0) {
                    for (int p = 0; p < N_P; ++p) {
                        int refrac_id = N_Z * (p+1) - 1;
                        int rho_id = 4*refrac_id;
                        Real sigma_sil_p = pdustfluids->df_u(rho_id, k, j, i);
                        Real &rho_Np = pphase_change->rho_Np_array(p, k, j, i);
                        //TBD: add f_ice_inter here
                        rho_Np = sigma_sil_p /(1.0 - 0.0)/(pphase_change->m_p0_array(p)); // [code_number_density]
                        // if (std::isinf(rho_Np) || std::isnan(rho_Np) || rho_Np == 0.0) {
                        //   std::cout << "Warning: rho_Np is " << rho_Np << " at (k,j,i)=("<<k<<","<<j<<","<<i<<")!" << std::endl;
                        //   std::cout << "sigma_sil_p = " << sigma_sil_p << ", m_p0 = " << pphase_change->m_p0_array(p, k, j, i) << std::endl;
                        // }
                        
                    }
                }

                // get pebble stopping times

                if (pphase_change != nullptr) {
                  // vapor diffusivity (with artificial decay for outer boundary)
                  Real w_damp = 0.05*(x1max-x1min);
                  Real f_decay_art = std::tanh(std::pow((rad-x1max)/w_damp ,2.0)); // outer bc decay
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
                      diffusivity = nu/(1.+SQR(taus_peb));
                    }
                  }
                }  // end if (pphase_change != nullptr)

                
            }
        }
    }

}


void Mesh::UserWorkInLoop() {
    
  // Real Mdot_gas = -1.e-8* Constants::solar_mass_cgs/ Constants::yr_cgs; // in cgs
  // Mdot_gas /= (punit->code_mass_cgs /punit->code_time_cgs);
  //
  // // Get the total mass flux;
  // Real Mdot_est = 0.0;
  // AthenaArray<Real> Mdot_peb_est, Mdot_peb;
  // Mdot_peb_est.NewAthenaArray(NDUSTFLUIDS);
  // Mdot_peb.NewAthenaArray(NDUSTFLUIDS);
  // // Initialize to zero (only pebble compositions will be set, vapor is ignored)
  // for (int n=0; n<NDUSTFLUIDS; n++) {
  //   Mdot_peb_est(n) = 0.0;
  //   Mdot_peb(n) = p2g_flux[n] * Mdot_gas;
  // }
  // // Real Mdot_peb = f_ICE_inter0*p2g_flux*Mdot_gas;
  //
  //   MeshBlock *pmb = my_blocks(0);
  // if (pmb->pbval->block_bcs[BoundaryFace::outer_x1] == BoundaryFlag::user){
  //   int tk = 0;
  //   int ti = pmb->pmy_mesh->mesh_size.nx1 +NGHOST;
  //   int tj_mid = 0;
  //   Real Tem_mid = pmb->phydro->Tem(0,0,pmb->ie);
  //   Real Tem_mid_ghost = pmb->phydro->Tem(0,0,pmb->ie+1);
  //
  //   for (int k=pmb->ks; k<=pmb->ke; k++) {
  //     for (int j=pmb->js; j<=pmb->je; j++) {
  //       int tj = static_cast<int>(pmb->loc.lx2)*pmb->block_size.nx2+(j-pmb->js);
  //       Real x2 = pmb->pcoord->x2v(j);
  //       Real dx2 = pmb->pcoord->dx2v(j);
  //       Real x1_active = pmb->pcoord->x1v(pmb->ie);
  //       Real ds_ghost = 2.0*pmb->pcoord->GetFace1Area(k,j,pmb->ie+1);
  //
  //       Real rad_active(0.0);
  //       Real rad_ghost(0.0);
  //       rad_active = pmb->pcoord->x1v(pmb->ie); 
  //       rad_ghost = pmb->pcoord->x1v(pmb->ie+1);
  //
  //       Real Tem_active = pmb->phydro->Tem(k, j, pmb->ie);
  //       Real Tem_ghost = pmb->phydro->Tem(k, j, pmb->ie+1);
  //       Real gas_sigma0 = pmb->phydro->w(IDN, k, j, pmb->ie);
  //       Real gas_nu = nu_gas(Tem_active, rad_active, 0.0);
  //       Real vr0 = -1.5*gas_nu/rad_ghost; // negative accretion vel at ghost cell 
  //
  //       // gas flux
  //       pmb->phydro->inflx_x1(k,j,0) = gas_sigma0*vr0;
  //       Mdot_est += pmb->phydro->inflx_x1(k,j,0)*ds_ghost;
  //
  //       // pebble flux:
  //       Real v_drift_peb = -0.004; // random drift velocity
  //       Real dust_vel1 = v_drift_peb;
  //
  //       for (int p=0; p<N_P; p++) {
  //         for (int n=0; n<N_Z; n++) {
  //           int dust_id = N_Z * p + n;
  //
  //           Real sigma_peb = 2.0*Mdot_peb(dust_id) /(v_drift_peb * 2.0*PI*rad_active);
  //           Real cs0 = std::sqrt(Tem_mid/(mu_xy*KELVIN));
  //           Real h_peb = H_ratio[n]* cs0/std::sqrt(gm0/(rad_active*SQR(rad_active)));
  //           Real rho_peb_mid = sigma_peb / (sqrt(2.0*PI)*h_peb);
  //
  //           pmb->pdustfluids->inflx_dust_x1(dust_id,k,j,0) = sigma_peb*dust_vel1;
  //           Mdot_peb_est(dust_id) += pmb->pdustfluids->inflx_dust_x1(dust_id,k,j,0)*ds_ghost;
  //         }
  //       }
  //     }
  //   }
  // }
  //
  // pmb->phydro->inflx_x1(0,0,0) *= Mdot_gas/Mdot_est; // scale gas flux to target Mdot
  // AthenaArray<Real> flx_peb_ratio;
  // flx_peb_ratio.NewAthenaArray(NDUSTFLUIDS);
  // // Calculate flux ratio for each pebble composition (vapor entries remain uninitialized but unused)
  // for (int p=0; p<N_P; p++) {
  //   for (int n=0; n<N_Z; n++) {
  //   int dust_id = N_Z * p + n;
  //     flx_peb_ratio(dust_id) = Mdot_peb(dust_id) / Mdot_peb_est(dust_id);
  //   }
  // }
  //
  // for (int p=0; p<N_P; p++) {
  //   for (int n=0; n<N_Z; n++) {
  //     int dust_id = N_Z * p + n;
  //     if(time < dust_start_injection){
  //       pmb->pdustfluids->inflx_dust_x1(dust_id,0,0,0) *= 0.0;
  //     } else {
  //       pmb->pdustfluids->inflx_dust_x1(dust_id,0,0,0) *= flx_peb_ratio(dust_id); // scale pebble flux to target Mdot
  //     }
  //   }
  // }
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
        if (gas_vel1>0.0){
        std::cout << "At (i,j,k)=(" << i << "," << j << "," << k << "):" << std::endl;
        std::cout << "gas_sigma = " << gas_sigma << std::endl;
        std::cout << "gas_vel1 = " << gas_vel1<< std::endl;
        std::cout << "gas_vel2 = " << gas_vel2<< std::endl;
        std::cout << "dust_sigma_0 = " << pdustfluids->df_u(0,k,j,i) << std::endl;
        std::cout << "st_0" << " = " << pdustfluids->stopping_time_array(0,k,j,i) << std::endl;
        std::cout << "dif_0 = " << pdustfluids->nu_dustfluids_array(0,k,j,i) << std::endl;
        std::cout << "dust_vel1_0 = " << pdustfluids->df_w(1,k,j,i) << std::endl;
        std::cout << "dust_sigma_1 = " << pdustfluids->df_u(4,k,j,i) << std::endl;
        std::cout << "dust_vel1_1 = " << pdustfluids->df_w(5,k,j,i) << std::endl;
        std::cout << "st_1" << " = " << pdustfluids->stopping_time_array(1,k,j,i) << std::endl;
        std::cout << "dif_1 = " << pdustfluids->nu_dustfluids_array(1,k,j,i) << std::endl;
        std::cout << "vapor_sigma = " << pdustfluids->df_u(4*vapor_id,k,j,i) << std::endl;
        std::cout << "vapor_vel1 = " << pdustfluids->df_w(4*vapor_id+1,k,j,i) << std::endl;
        std::cout << "=========================================" << std::endl;
                }
        // copy refractory velocity to ice for each pebble
        if (N_Z > 1) {
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

        //apply floor value
        for (int n=0; n<NDUSTFLUIDS; n++) {
          int dust_id = n;
          int rho_id  = 4*dust_id;
          int v1_id   = rho_id + 1;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;

          Real &dust_rho = pdustfluids->df_w(rho_id, k, j, i);
          dust_rho = (dust_rho > dffloor) ? dust_rho : (dffloor);

          pdustfluids->df_u(rho_id, k, j, i) = dust_rho;
          pdustfluids->df_u(v1_id, k, j, i) = dust_rho*pdustfluids->df_w(v1_id, k, j, i);
          pdustfluids->df_u(v2_id, k, j, i) = dust_rho*pdustfluids->df_w(v2_id, k, j, i);
          pdustfluids->df_u(v3_id, k, j, i) = dust_rho*pdustfluids->df_w(v3_id, k, j, i);
        }

        }
       }
    }
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
        
        user_out_var(10,k,j,i) = phydro->Tem(k,j,i);
        user_out_var(11,k,j,i) = phydro->flux[IDN](k,j,i);

        if (pphase_change != nullptr && N_P > 0) {
          for (int p=0; p<N_P; p++) {
            AthenaArray<Real> rho_comps; //get the uid of density and density for each composition
            rho_comps.NewAthenaArray(N_Z);
            for (int n=0; n<N_Z; n++) {
              Real comp_id = p*N_Z + n;
              rho_comps(n) = pdustfluids->df_w(4*comp_id, k, j, i); // [code_density] 
            }
            Real rho_Np = pphase_change->rho_Np_array(p, k, j, i); // [code_number_density]
            
            // Real rho_I = pdustfluids->df_w(4*p, k, j, i); // ice for pebble p [code_density]
            // Real rho_sil = pdustfluids->df_w(4*(p*N_Z+1), k, j, i); // refractory for pebble p [code_density]
            Real m_p = pphase_change->Get_m_p_from_rho_Np(rho_comps, rho_Np); // [code_mass]
            Real s_p = pphase_change->Get_s_p_from_m_p(m_p, rho_comps); // [code_length]
            Units *punit = pmy_mesh->punit;

            // really no good idea to output this more elegently now.... [25.11.24]
            user_out_var(p*4,k,j,i) = pdustfluids->stopping_time_array(p*N_Z,k,j,i);
            user_out_var(p*4+1,k,j,i) = m_p * punit->code_mass_cgs; // Convert to CGS for output 
            user_out_var(p*4+2,k,j,i) = s_p * punit->code_length_cgs; // Convert to CGS for output
            user_out_var(p*4+3,k,j,i) = pdustfluids->df_flux[X1DIR](4*(p*N_Z),k,j,i);
          }
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

  // LocalIsothermalEOS(pmb, time, dt, prim, prim_df, bcc, cons, cons_df);
  // if(N_Z > 1 and time > t_iterate){  // (Yu, 2025-11-16)
  //   pmb->pphase_change->PhaseChangeSource(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);
  // }
  AthenaArray<Real> v_frag;
  v_frag.NewAthenaArray(N_P); 
  v_frag(0) = 1000.0/pmb->pmy_mesh->punit->code_velocity_cgs; // fragmentation velocity for pebble size bin 0 [code_velocity]
  v_frag(1) = 100.0/pmb->pmy_mesh->punit->code_velocity_cgs; // fragmentation velocity for pebble size bin 1 [code_velocity]
  
  // if (N_P > 1 and time >dust_start_injection){
  //   pmb->pphase_change->TriPodSource(pmb, time, dt, gm0, alpha_vis, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s, v_frag);
  // }

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
        gas_nu = alpha_vis* std::pow(rad/r0, 1.0); // fix the gas viscosity.
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
            // std::cout << "Diffusivity for pebble " << p << " composition " << z << " at (k,j,i)=("<<k<<","<<j<<","<<i<<") is " << diffusivity << std::endl;
          }
        }
      }
    }
  }
  
  return;
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
          // [25.11.26]lzx: let's avoid using same variable name 'z' here
          for (int zi = 0; zi < N_Z; ++zi) {
            int dust_id = N_Z * p + zi;
            int rho_id = 4*dust_id;
            Real Tem = phydro->Tem(k, j, i);
            Real mu = get_mu(prim_df(4*(vapor_id), k, j, i)/gas_sigma);
            Real cs2 = Tem/(mu*KELVIN); 
            Real h_gas = std::sqrt(cs2)/omega_dyn; 
            Real St = stopping_time(dust_id, k, j, i)*omega_dyn;
            Real h_peb = h_gas/std::sqrt(1.0 + St/alpha_vis*(1.0 + 2.0*St)/(1.0 + St));
            rho_dustfluid_array_pebble(zi) = prim_df(rho_id, k, j, i)/std::sqrt(2.0*PI)/h_peb;
            // if (rho_dustfluid_array_pebble(zi) <=0.0){
            //     std::cout << "Warning: rho_dustfluid_array_pebble < 0 for pebble " << p << " composition " << zi << " at (k,j,i)=("<<k<<","<<j<<","<<i<<")!" << std::endl;
            //     std::cout << "prim_df(rho_id) = " << prim_df(rho_id, k, j, i) << ", h_peb = " << h_peb << std::endl;
            //     std::cout << "time = " << pmesh->time << std::endl;
            // }
          }
          
          // calculate stopping time for the pebble
          Real rho_Np = pphase_change->rho_Np_array(p, k, j, i);
          Real rho_v = prim_df(4*vapor_id, k, j, i);
          Real t_stop = pphase_change->Get_stopping_time(pmesh->punit, rho_dustfluid_array_pebble, Tem, gas_sigma, rho_v, rho_Np);
          
          // ===== upper limit, St can be extremely high for upper layer =====
          t_stop = (t_stop > 0.5) ? 0.5 : t_stop;
          
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
Real Sigma_gas(const Real rad){
    return sigma0*std::pow(rad/r0, dslope);
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
void DiskInnerX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad_active(0.0), phi_active(0.0), z_active(0.0);
  Real rad_ghost(0.0),  phi_ghost(0.0),  z_ghost(0.0);
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        rad_active = pco->x1v(il); 
        rad_ghost = pco->x1v(il-i);

        // std::cout<< prim(IDN, k, j, il-i) << std::endl;
        // std::cout<< prim(IM1, k, j, il-i) << std::endl;
        // std::cout<< prim(IEN, k, j, il-i) << std::endl;
        //
        // std::cout<< prim(IDN, k, j, il) << std::endl;
        // std::cout<< prim(IM1, k, j, il) << std::endl;

        Real &gas_sigma_ghost  = prim(IDN, k, j, il-i);
        Real &gas_vel1_ghost = prim(IVX, k, j, il-i);
        Real &gas_vel2_ghost = prim(IVY, k, j, il-i);

        Real &gas_pres_ghost = prim(IPR, k, j, il-i);
        
        Real &gas_sigma_active  = prim(IDN, k, j, il); 
        Real &gas_vel1_active = prim(IM1, k, j, il);

        Real Tem = Tem_gas(rad_ghost);

        Real gas_nu = nu_gas(Tem, rad_ghost,0.0);
        Real v_acc_gas = -1.5*gas_nu/rad_ghost; // negative accretion vel
        // Real v_acc_gas = gas_vel1_active; // negative accretion vel
        
        // Real Mdot_xy = M_dot_g*CONST_Msun/CONST_yr;
        // Mdot_xy /= (SQR(UNIT_LENGTH)*UNIT_LENGTH*UNIT_DENSITY/UNIT_T);
        // Real xy_sigma_ghost = Mdot_xy/(2.0*PI*rad_ghost*fabs(v_acc_gas));
        
        gas_vel1_ghost = v_acc_gas;
        gas_vel2_ghost = VelProfileCyl_gas(rad_ghost, 0.0, 0.0);
        gas_sigma_ghost = Sigma_gas(rad_ghost);
        // sigma_interpolate_inner_nomatter(rad_active, rad_ghost, gas_sigma_active, gas_sigma_ghost,
        //     gas_vel1_active, gas_vel1_ghost);

        // Real &vapor_sigma_ghost  = prim_df(4*(NDUSTFLUIDS-1), k, j, il-i);
        // Real &vapor_vel1_ghost = prim_df(4*(NDUSTFLUIDS-1) + 1, k, j, il-i);
        // Real &vapor_sigma_active  = prim_df(4*(NDUSTFLUIDS-1), k, j, il);
        // Real &vapor_vel1_active = prim_df(4*(NDUSTFLUIDS-1) + 1, k, j, il);

        // vapor_vel1_ghost = gas_vel1_ghost;
        // sigma_interpolate_inner_nomatter(rad_active, rad_ghost, vapor_sigma_active, vapor_sigma_ghost,
        //     vapor_vel1_active, vapor_vel1_ghost);

        // gas_sigma_ghost = xy_sigma_ghost + vapor_sigma_ghost;

        if (NON_BAROTROPIC_EOS){
          Real fv = prim_df(4*(NDUSTFLUIDS-1), k, j, il-i)/gas_sigma_ghost;
          Real T1 = Tem_gas(rad_ghost);
          Real mu1 = get_mu(fv);
          gas_pres_ghost = gas_sigma_ghost*T1/(mu1*KELVIN);
          // gas_pres_ghost = gas_sigma_ghost*T1/(mu_xy*KELVIN);
        }

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            
            if(dust_id != vapor_id){
              // free outflow for dust
                prim_df(rho_id,k,j,il-i) = prim_df(rho_id,k,j,il+i-1);
                prim_df(v1_id,k,j,il-i) = prim_df(v1_id,k,j,il+i-1);
                prim_df(v2_id,k,j,il-i) = prim_df(v2_id,k,j,il+i-1);
            }else{
              prim_df(rho_id,k,j,il-i) = prim_df(rho_id,k,j,il);
              Real &vapor_sigma_ghost  = prim_df(rho_id, k, j, il-i);
              Real &vapor_vel1_ghost = prim_df(v1_id, k, j, il-i);
                Real &vapor_vel2_ghost = prim_df(v2_id, k, j, il-i);

              Real &vapor_sigma_active  = prim_df(rho_id, k, j, il);
              Real &vapor_vel1_active = prim_df(v1_id, k, j, il);

              vapor_vel1_ghost = gas_vel1_ghost;
              vapor_vel2_ghost = gas_vel2_ghost;
              sigma_interpolate_inner_nomatter(rad_active, rad_ghost, vapor_sigma_active, vapor_sigma_ghost,
                  vapor_vel1_active, vapor_vel1_ghost);
            }
          }
        }
      }
    }
  }
}

void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0),  phi(0.0),  z(0.0);
  Real rad_active(0.0), phi_active(0.0), z_active(0.0);
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        rad_active = pco->x1v(iu);
        Real &gas_sigma_active  = prim(IDN, k, j, iu);
        Real &gas_vel1_active = prim(IM1, k, j, iu);
        Real &gas_pres_active = prim(IPR, k, j, iu);

        rad = pco->x1v(iu+i);
        Real &gas_sigma  = prim(IDN, k, j, iu+i);
        Real &gas_vel1 = prim(IVX, k, j, iu+i);
        Real &gas_vel2 = prim(IVY, k, j, iu+i);
        Real &gas_pres = prim(IEN, k, j, iu+i);

        Real Tem = Tem_gas(rad);
        Real cs2 = Tem/(mu_xy*KELVIN);
        Real omega_dyn = std::sqrt(gm0/std::pow(rad,3.0));
        Real h_gas = std::sqrt(cs2)/omega_dyn;


        gas_sigma = Sigma_gas(rad);

        Real vk = std::sqrt(gm0/rad);
        Real eta_vk = 0.5*cs2/vk*(Tslope + dslope);

        Real v_acc_gas=0.0;
        Real gas_nu = nu_gas(Tem, rad,0.0);
        v_acc_gas = -1.5*gas_nu/rad; // negative accretion vel
        // v_acc_gas = gas_vel1_active; // negative accretion vel
        // v_acc_gas = Get_v_gas(k,j,i,pmb,prim,prim_df);

        // Real Mdot_xy = -1.e-8*Constants::solar_mass_cgs/Constants::yr_cgs;
        // Mdot_xy /= pmb->pmy_mesh->punit->code_mass_cgs/ pmb->pmy_mesh->punit->code_time_cgs; // in code unit      

        // gas_sigma = gas_sigma_active;
        gas_vel1 = v_acc_gas; 
        gas_vel2 = VelProfileCyl_gas(rad, 0.0, 0.0);
        v_acc_gas = gas_vel1;

        if (NON_BAROTROPIC_EOS){
          gas_pres = gas_sigma*cs2; 
        }

        if (NDUSTFLUIDS > 0) {
          for (int p=0; p<N_P; ++p) {
            for (int z = 0; z < N_Z; ++z) {
              int dust_id = p*N_Z + z;
              int rho_id  = 4*dust_id;
              int v1_id   = rho_id + 1;
              int v2_id   = rho_id + 2;

              Real &dust_sigma  = prim_df(rho_id, k, j, iu+i);
              Real &dust_vel1 = prim_df(v1_id,  k, j, iu+i);
              Real &dust_vel2 = prim_df(v2_id,  k, j, iu+i);

              Real t_stop = pmb->pdustfluids->stopping_time_array(dust_id, k,j,iu-i+1);
              Real taus = t_stop*omega_dyn;
              Real v_drift_peb = (v_acc_gas + 2.0*eta_vk*taus)/(1.+SQR(taus));

              dust_vel1 = v_drift_peb;
              dust_vel2 = std::sqrt(gm0/rad);    

              dust_sigma = prim_df(rho_id, k, j, iu+i-1);
              // dust_sigma = prim_df(rho_id, k, j, iu); 
              // dust_sigma = 0.0;
            }
          }
          Real &vapor_sigma = prim_df(vapor_id*4, k, j, iu+i);
          Real &vapor_vel1 = prim_df(vapor_id*4+1, k, j, iu+i);
          Real &vapor_vel2 = prim_df(vapor_id*4+2, k, j, iu+i);
          vapor_vel1 = gas_vel1;
          vapor_vel2 = gas_vel2; 
          vapor_sigma = gas_sigma*initial_D2G[vapor_id];


        }
      }
    }
  }
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
      for (int i=is-NGHOST; i<=ie+NGHOST; ++i) {
        rad = pmb->pcoord->x1v(i);

        Real &gas_dens = cons(IDN, k, j, i);
        Real &gas_mom1 = cons(IM1, k, j, i);
        Real &gas_mom2 = cons(IM2, k, j, i);
        Real &gas_mom3 = cons(IM3, k, j, i);
        Real &gas_erg  = cons(IEN, k, j, i);

        const Real &rho_v = cons_df(4*(NDUSTFLUIDS-1), k, j, i);

        // temperature profile: fixed
        Real Tem = pmb->phydro->Tem(k, j, i); 
        Real gas_vel1 = gas_mom1/gas_dens;
        Real gas_vel2 = gas_mom2/gas_dens;
        Real gas_vel3 = gas_mom3/gas_dens;

        // gen density (do not allow H/He diffusion):        
        gas_dens = rho_v + Sigma_gas(rad);
        gas_mom1 = gas_dens*gas_vel1;
        gas_mom2 = gas_dens*gas_vel2;
        gas_mom3 = gas_dens*gas_vel3;

        // get pressure of local temperature
        Real fv = rho_v/gas_dens;
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

Real VelProfileCyl_gas(const Real rad, const Real phi, const Real z) {
  Real p_over_r = PoverRho(rad, phi, z);
  // Real vel = (dslope+p_over_rho_slope)*p_over_r/(gm0/rad) + (1.0+p_over_rho_slope)
            //  - p_over_rho_slope*rad/std::sqrt(rad*rad+z*z);
  Real vel = (dslope + p_over_rho_slope - Tslope/2.0 - 1.5)*p_over_r/(gm0/rad) + (1.0+p_over_rho_slope)
             - p_over_rho_slope*rad/std::sqrt(rad*rad+z*z);
  vel = std::sqrt(gm0/rad)*std::sqrt(vel) - rad*Omega0;
  return vel;
}


}
