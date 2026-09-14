//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file dustfluids_noCs_solver.cpp
//! \brief HLLE Riemann solver for dust fludis (no dust sound speed)
//!
//! Computes 1D fluxes using the Harten-Lax-van Leer (HLL) Riemann solver.  This flux is
//! very diffusive, especially for contacts, and so it is not recommended for use in
//! applications.  However, as shown by Einfeldt et al.(1991), it is positively
//! conservative (cannot return negative densities or pressure), so it is a useful
//! option when other approximate solvers fail and/or when extra dissipation is needed.
//!
// C headers

// C++ headers
#include <algorithm>  // max(), min()
#include <cmath>      // sqrt()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../eos/eos.hpp"
#include "dustfluids.hpp"
#include "../phase_change/phase_change_constants.hpp"

//----------------------------------------------------------------------------------------
//! \fn void DustFluids::RiemannSolver_DustFluids
//! \brief The Riemann solver for Dust Fluids (no dust sound speed)

void DustFluids::RiemannSolverDustFluids_Penetration(const int k, const int j, const int il, const int iu,
                          const int index, AthenaArray<Real> &prim_df_l,
                          AthenaArray<Real> &prim_df_r, AthenaArray<Real> &dust_flux) {

  Real df_w_li[(NDUSTVARS)], df_w_ri[(NDUSTVARS)];

  // [26.09.06]Zhixuan: only solve for "real dust"
  for (int n=0; n<NDUSTFLUIDS_MOM; ++n) {
    int idust = n;
    int irho  = 4*idust;

    int v1_id = irho + 1;
    int v2_id = irho + 2;
    int v3_id = irho + 3;

    int ivx = irho + (IVX + (index-IVX)%3);
    int ivy = irho + (IVX + ((index-IVX)+1)%3);
    int ivz = irho + (IVX + ((index-IVX)+2)%3);

#pragma omp simd private(df_w_li, df_w_ri)
    for (int i=il; i<=iu; i++) {
      df_w_li[irho]  = prim_df_l(irho, i);
      df_w_li[v1_id] = prim_df_l(ivx,  i);
      df_w_li[v2_id] = prim_df_l(ivy,  i);
      df_w_li[v3_id] = prim_df_l(ivz,  i);

      df_w_ri[irho]  = prim_df_r(irho, i);
      df_w_ri[v1_id] = prim_df_r(ivx,  i);
      df_w_ri[v2_id] = prim_df_r(ivy,  i);
      df_w_ri[v3_id] = prim_df_r(ivz,  i);

      Real temp_li = (df_w_li[v1_id] > 0.0) ? 1.0 : 0.0;
      Real temp_ri = (df_w_ri[v1_id] < 0.0) ? 1.0 : 0.0;

      Real flx_rho_li = df_w_li[irho]*df_w_li[v1_id];
      Real flx_rho_ri = df_w_ri[irho]*df_w_ri[v1_id];

      dust_flux(irho, k, j, i) = temp_li*flx_rho_li + temp_ri*flx_rho_ri;
      dust_flux(ivx,  k, j, i) = temp_li*df_w_li[v1_id]*flx_rho_li + temp_ri*df_w_ri[v1_id]*flx_rho_ri;
      dust_flux(ivy,  k, j, i) = temp_li*df_w_li[v2_id]*flx_rho_li + temp_ri*df_w_ri[v2_id]*flx_rho_ri;
      dust_flux(ivz,  k, j, i) = temp_li*df_w_li[v3_id]*flx_rho_li + temp_ri*df_w_ri[v3_id]*flx_rho_ri;
    }
  }
  return;
}


void DustFluids::RiemannSolverDustFluids_noPenetration(const int k, const int j, const int il, const int iu,
                          const int index, AthenaArray<Real> &prim_df_l,
                          AthenaArray<Real> &prim_df_r, AthenaArray<Real> &dust_flux) {

  Real df_w_li[(NDUSTVARS)], df_w_ri[(NDUSTVARS)], df_w_roe[(NDUSTVARS)];

  // [26.09.06]Zhixuan: only solve for "real dust"
  for (int n=0; n<NDUSTFLUIDS_MOM; ++n) {
    int idust = n;
    int irho  = 4*idust;

    int v1_id = irho + 1;
    int v2_id = irho + 2;
    int v3_id = irho + 3;

    int ivx = (IVX + ((index-IVX))%3)   + irho;
    int ivy = (IVX + ((index-IVX)+1)%3) + irho;
    int ivz = (IVX + ((index-IVX)+2)%3) + irho;

#pragma omp simd private(df_w_li, df_w_ri, df_w_roe)
    for (int i=il; i<=iu; i++) {
      df_w_li[irho]  = prim_df_l(irho, i);
      df_w_li[v1_id] = prim_df_l(ivx,  i);
      df_w_li[v2_id] = prim_df_l(ivy,  i);
      df_w_li[v3_id] = prim_df_l(ivz,  i);

      df_w_ri[irho]  = prim_df_r(irho, i);
      df_w_ri[v1_id] = prim_df_r(ivx,  i);
      df_w_ri[v2_id] = prim_df_r(ivy,  i);
      df_w_ri[v3_id] = prim_df_r(ivz,  i);

      Real sqrtdl  = std::sqrt(df_w_li[irho]);
      Real sqrtdr  = std::sqrt(df_w_ri[irho]);
      Real isdlpdr = 1.0/(sqrtdl + sqrtdr);

      df_w_roe[v1_id] = (sqrtdl*df_w_li[v1_id] + sqrtdr*df_w_ri[v1_id])*isdlpdr;
      df_w_roe[v2_id] = (sqrtdl*df_w_li[v2_id] + sqrtdr*df_w_ri[v2_id])*isdlpdr;
      df_w_roe[v3_id] = (sqrtdl*df_w_li[v3_id] + sqrtdr*df_w_ri[v3_id])*isdlpdr;

      Real neg_li = df_w_li[v1_id] < 0.0 ? 1.0 : 0.0;
      Real pos_ri = df_w_ri[v1_id] > 0.0 ? 1.0 : 0.0;
      Real temp   = 1.0 - neg_li * pos_ri;

      Real fra_roe = (df_w_roe[v1_id] > 0.0) ? 1.0 : 0.0;
      if (df_w_roe[v1_id] == 0.0) fra_roe = 0.5;

      dust_flux(irho, k, j, i) = temp*(fra_roe*df_w_li[irho]*df_w_li[v1_id] +
                                 (1.0-fra_roe)*df_w_ri[irho]*df_w_ri[v1_id]);

      dust_flux(ivx, k, j, i) = temp*(fra_roe*df_w_li[v1_id]*dust_flux(irho, k, j, i) +
                                (1.0-fra_roe)*df_w_ri[v1_id]*dust_flux(irho, k, j, i));

      dust_flux(ivy, k, j, i) = temp*(fra_roe*df_w_li[v2_id]*dust_flux(irho, k, j, i) +
                                (1.0-fra_roe)*df_w_ri[v2_id]*dust_flux(irho, k, j, i));

      dust_flux(ivz, k, j, i) = temp*(fra_roe*df_w_li[v3_id]*dust_flux(irho, k, j, i) +
                                (1.0-fra_roe)*df_w_ri[v3_id]*dust_flux(irho, k, j, i));
    }
  }
  return;
}

// vapor flux solver. (Yu, 2025-11-18) [26.09.06]Zhixuan: also the number density 
void DustFluids::TracerUpwindFlux(const int k, const int j, const int il,
  const int iu, const int dir, // CoordinateDirection dir (1,2,3)
  const AthenaArray<Real> &prim_df,
  AthenaArray<Real> &r_l, AthenaArray<Real> &r_r, // 2D
  AthenaArray<Real> &gas_mass_flx,  // 3D
  AthenaArray<Real> &flx_out) { // 4D

  // Tracer fluids (vapor and number densities) are not Riemann-solved and their
  // primitive rows are not reconstructed. So we also need to copy the momentum fluxes  
  for (int n = vapor_id; n < NDUSTFLUIDS; ++n) {
    int rho_id = 4 * n;
    int v1_id  = rho_id + 1;
    int v2_id  = rho_id + 2;
    int v3_id  = rho_id + 3;
    for (int i = il; i <= iu; i++) {
      Real fluid_flx;
      if (n == vapor_id) {
        // vapor: carried by the gas
        fluid_flx = gas_mass_flx(k, j, i);
      } else {
        // number density of population p: carried by the silicate fluid of p
        int p = n - 1 - N_P*N_Z;
        int sil_id = (p+1)*N_Z - 1; // last (refractory) composition of population p
        fluid_flx = flx_out(sil_id*4, k, j, i);
      }
      // upwind cell along the face normal
      int kk = k, jj = j, ii = i;
      if (fluid_flx >= 0.0) {
        if (dir == 1)      ii = i - 1;
        else if (dir == 2) jj = j - 1;
        else               kk = k - 1;
      }
      Real dens_flx = fluid_flx * (fluid_flx >= 0.0 ? r_l(n, i) : r_r(n, i));
      flx_out(rho_id, k, j, i) = dens_flx;
      // maybe don't need this
      flx_out(v1_id,  k, j, i) = dens_flx * prim_df(v1_id, kk, jj, ii);
      flx_out(v2_id,  k, j, i) = dens_flx * prim_df(v2_id, kk, jj, ii);
      flx_out(v3_id,  k, j, i) = dens_flx * prim_df(v3_id, kk, jj, ii);
    }
  }
  return;
}
