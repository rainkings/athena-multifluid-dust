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
#include <string>

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
#include "relaxation.hpp"


Relaxation::Relaxation(MeshBlock *pmb, ParameterInput *pin): 
  pmy_block(pmb) {

  Tem0 = pin->GetOrAddReal("problem", "Tem0", 150.0);
  r0 = pin->GetOrAddReal("problem", "r0", 3.0);
  Tslope = pin->GetOrAddReal("problem", "Tslope", -0.5);
  T_relax_prefactor = pin->GetOrAddReal("problem", "T_relax_prefactor", 100.0);

  Units *punit = pmb->pmy_mesh->punit;
  Real rho_sil_inter_cgs = pin->GetOrAddReal("problem", "rho_sil_inter", 3.0); // [g/cm^3]
  Real rho_ice_inter_cgs = pin->GetOrAddReal("problem", "rho_ice_inter", 1.0); // [g/cm^3]
  rho_sil_inter_ = rho_sil_inter_cgs / punit->code_density_cgs; // Convert to code units
  rho_ice_inter_ = rho_ice_inter_cgs / punit->code_density_cgs; // Convert to code units
    //
  mmax_array.NewAthenaArray(pmb->ncells3, pmb->ncells2, pmb->ncells1);
  t_relax.NewAthenaArray(pmb->ncells3, pmb->ncells2, pmb->ncells1);
  m_p_array.NewAthenaArray(N_P, pmb->ncells3, pmb->ncells2, pmb->ncells1);
  m_p0_array.NewAthenaArray(N_P);

  mmin = pin->GetOrAddReal("problem", "mmin", 1.e-12); // minimum mass in the dust size distribution [code mass]
  mmin /= punit->code_mass_cgs; // Convert to code units
  

  // (Yu, 2025-11-18) Convert input parameters from CGS to code units
  for (int p = 0; p < N_P; ++p) {
    //[25.11.25]lzx: here we should multiply by N_Z to get the correct dust_id 
    Real m_p0_cgs = pin->GetReal("dust", "m_p0_" + std::to_string(p*N_Z+1)); // [g]
    m_p0_array(p) = m_p0_cgs / punit->code_mass_cgs; // Convert to code units

    // [26.06.02]Zhixuan: This should not be done since when restart, this should not be re-initialized
    // for (int k = 0; k < pmb->ncells3; ++k) {
    //   for (int j = 0; j < pmb->ncells2; ++j) {
    //     for (int i = 0; i < pmb->ncells1; ++i) {
    //       m_p_array(p, k, j, i) = m_p0_array(p); // Initialize m_p to m_p0
    //     }
    //   }
    // }

  }
}

void Relaxation::GetDivisionMasses(Real mmin, Real mmax, AthenaArray<Real> &m_div, std::string mode) {
  // Default: logarithmic spacing, i.e., equally spaced in log mass.
  // Override this method in a derived class to implement a different division law.
  if (mode == "log") {
    if (N_P <= 1) return;
    Real log_min = std::log(mmin);
    Real log_max = std::log(mmax);
    Real step = (log_max - log_min) / N_P;
    for (int p = 1; p < N_P; ++p) {
      m_div(p-1) = std::exp(log_min + p * step);
    }
  } else if (mode == "small") {
    Real mm = mmax;
    for (int p = 0; p< N_P; ++p) {
      m_div(p) = std::sqrt(mmin * mm);
      mm = m_div(p);
    }
    // inverse the m_div array to get the division masses from small to large 
    for (int p = 0; p < N_P/2; ++p) {
      Real temp = m_div(p);
      m_div(p) = m_div(N_P - 2 - p);
      m_div(N_P - 2 - p) = temp;
    }
  }
}


// a general version
// void Relaxation::RelaxationSource(MeshBlock *pmb, const Real time, const Real dt, const Real gm0, const Real alpha_vis,
//       const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_df,
//       const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
//       AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s, AthenaArray<Real> &v_frag){
//
//   for (int k=pmb->ks; k<=pmb->ke; ++k) {
//     for (int j=pmb->js; j<=pmb->je; ++j) {
// #pragma omp simd
//       for (int i=pmb->is; i<=pmb->ie; ++i) {
//         AthenaArray<Real> rhos, v1s, v2s, v3s, ns; 
//         int NFLUIDS = NDUSTFLUIDS-1-N_P; // exclude vapor and number density
//         rhos.NewAthenaArray(NFLUIDS); // exclude vapor and number density 
//         v1s.NewAthenaArray(NFLUIDS);
//         v2s.NewAthenaArray(NFLUIDS);
//         v3s.NewAthenaArray(NFLUIDS);
//         ns.NewAthenaArray(N_P);
//
//         Real rho_sil = 0.0;
//         Real rho_ice = 0.0;
//
//         for (int n=0; n<NFLUIDS; ++n){
//           rhos(n) = cons_df(4*n, k, j, i);
//
//           if (n%N_Z == 1 || N_Z==1){ // if it's a silicate population (or if there's only one population which is silicate)
//             rho_sil += rhos(n);
//           }else if (n%N_Z == 0){
//             rho_ice += rhos(n);
//           }
//           // if (rhos(n) <= dffloor_){
//           //   continue; // skip if the density is too low
//           // }
//           v1s(n) = cons_df(4*n+1, k, j, i)/rhos(n);
//           v2s(n) = cons_df(4*n+2, k, j, i)/rhos(n);
//           v3s(n) = cons_df(4*n+3, k, j, i)/rhos(n);
//
//           int n_id = N_Z*N_P + 1 + n/N_Z; 
//           ns(n/N_Z) = cons_df(4*n_id, k, j, i); // number density for this population
//         }
//
//         // get the maximum mass 
//         Real rad = pmb->pcoord->x1v(i); 
//         Real Tem = Tem_gas(rad); // [code_temperature]
//
//         Real sigma_g = cons(IDN, k, j, i);
//         Real sigma_v = cons_df(4*vapor_id, k, j, i);
//
//         Real OmegaK = std::sqrt(gm0/std::pow(rad, 3));
//
//         Real fv = sigma_v/sigma_g;
//         Real cs2 = Tem/KELVIN*Get_mu(fv);
//
//         Real H_gas = std::sqrt(cs2)/OmegaK;
//
//         Real rho_g = sigma_g/(std::sqrt(2*PI)*H_gas);
//
//         Real v_frag_tot = (v_frag(1)*rho_sil + v_frag(0)*rho_ice)/(rho_sil + rho_ice); // the fragmentation velocity for the current composition, using the composition as a proxy.
//         // Real v_frag_tot = v_frag(0); // now we just use the large velocity[26.04.02]
//
//         Real vth = std::sqrt(8.0/PI)*std::sqrt(cs2); // [code_velocity] 
//
//         // get internal density 
//         Real rho_inte_relax = (rho_sil_inter_*rho_sil + rho_ice_inter_*rho_ice)/(rho_sil + rho_ice); // the internal density of the relaxed state, using the current composition as a proxy.
//         //too small, for the moment artificially enlarge it
//         Real s_max =SQR(v_frag_tot)/cs2 * rho_g*vth/3/alpha_vis/OmegaK/rho_inte_relax;
//
//         Real m_max = FOUR_3RD*PI*SQR(s_max)*s_max*rho_inte_relax; // [code_mass]
//         //store the mmax 
//         mmax_array(k,j,i) = m_max;
//
//         AthenaArray<Real> rho_comps, s_p_array;
//         s_p_array.NewAthenaArray(N_P);
//         for (int p = 0; p < N_P; ++p) {
//           s_p_array(p) = std::pow(m_p_array(p,k,j,i)/(FOUR_3RD*PI*rho_inte_relax), ONE_3RD); // [code_length]
//         }
//
//         // calculate the power law before relaxation 
//         AthenaArray<Real> rho_pop_array;
//         rho_pop_array.NewAthenaArray(N_P);
//         for (int p = 0; p < N_P; ++p) {
//           for (int zi = 0; zi < N_Z; ++zi) {
//             int dust_id = N_Z*p + zi; 
//             rho_pop_array(p) += cons_df(4*dust_id, k, j, i); // total density for population p
//           }
//         }
//
//         Real q_before = std::log((rho_pop_array(1))/(rho_pop_array(0)))/std::log(m_p_array(1,k,j,i)/m_p_array(0,k,j,i)); // the effective powerlaw index of the current state
//
//         if (std::abs(q_before - 1.0/6.0)/1.0*6.0 > 1.e-6) {
//           Real M1_relax = 0.0;
//           for (int p = 0; p < N_P; ++p) {
//             M1_relax += rho_pop_array(p);
//           }
//
//           AthenaArray<Real> comp_ratio;
//           comp_ratio.NewAthenaArray(N_Z);
//           if (N_Z == 1){
//             comp_ratio(0) = 1.0;
//           } else {
//             comp_ratio(0) = rho_ice/(rho_ice + rho_sil);
//             comp_ratio(1) = 1.0 - comp_ratio(0); 
//           }
//
//           //get the distribution of the relaxation state: f*(m) = c* m^(-11/6) (assume the relaxed state is the MRN distribution)
//           Real c_relax = M1_relax/6/(std::pow(m_max, 1.0/6.0) - std::pow(mmin,1.0/6.0));
//           Real M2_relax = c_relax*6/7*(std::pow(m_max, 7.0/6.0) - std::pow(mmin,7.0/6.0)); // the 2nd moment of the relaxed state 
//
//           // the division mass 
//           Real m_01 = std::pow(mmin*m_max, 0.5); 
//
//           AthenaArray<Real> M1_relax_array, M2_relax_array;
//           M1_relax_array.NewAthenaArray(N_P);
//           M2_relax_array.NewAthenaArray(N_P);
//           M1_relax_array(0) = c_relax*6*(std::pow(m_01, 1.0/6.0) - std::pow(mmin, 1.0/6.0)); 
//           M1_relax_array(1) = M1_relax - M1_relax_array(0);  
//           M2_relax_array(0) = c_relax*6/7*(std::pow(m_01, 7.0/6.0) - std::pow(mmin, 7.0/6.0));
//           M2_relax_array(1) = M2_relax - M2_relax_array(0);
//
//           //from the moments we get the relaxed state per population.
//           AthenaArray<Real> rho_pop_relax, n_relax_array, rho_relax_array;
//           rho_pop_relax.NewAthenaArray(N_P);
//           n_relax_array.NewAthenaArray(N_P);
//           rho_relax_array.NewAthenaArray(NFLUIDS);
//
//           for (int p = 0; p < N_P; ++p) {
//             rho_pop_relax(p) = M1_relax_array(p);
//             n_relax_array(p) = SQR(M1_relax_array(p))/M2_relax_array(p);
//
//             // divide the relaxed state into different compositions according to the current composition ratio of each population.
//             for (int zi=0; zi<N_Z; ++zi){
//               int dust_id = N_Z*p + zi;
//               rho_relax_array(dust_id) = rho_pop_relax(p)*comp_ratio(zi);
//             }
//           }
//
//           //get the relaxation timescale: the collisional timescale of large population.
//           Real t_stop = pmb->pdustfluids->stopping_time_array(N_P-1, k, j, i);
//           Real St1 = t_stop*OmegaK;
//           Real delV = std::sqrt(3*alpha_vis*St1*cs2); // relative velocity between the two populations, using the turbulent relative velocity for simplicity.
//
//           //// Here we use volumn number density rather than the column number density 
//           int n_large_id = NDUSTFLUIDS - 1;
//           Real rho_Np_vol = cons_df(4*n_large_id,k,j,i)/std::sqrt(2*PI)/H_gas; // the bulk number density of pebbles, using the number density of the big population as a proxy.
//           Real t_relax = 1.0/(4*rho_Np_vol *PI*SQR(s_p_array(1)) * delV )*100;
//
//           //relax the current state to the relaxed state:
//           Real relax_rate = dt/t_relax;
//
//           AthenaArray<Real> ns_new; 
//           ns_new.NewAthenaArray(N_P);
//           for (int p=0; p<N_P; ++p){
//             for (int zi=0; zi<N_Z; ++zi){
//               int dust_id = N_Z*p + zi;
//               Real rho_new = rhos(dust_id) + relax_rate*(rho_relax_array(dust_id) - rhos(dust_id));
//
//               //prevent overshoot 
//               Real sign_before = rho_relax_array(dust_id) - rhos(dust_id); 
//               Real sign_after = rho_relax_array(dust_id) - rho_new;
//               // if (sign_before*sign_after < 0.0){
//               //   rho_new = rho_relax_array(dust_id);
//               // }
//
//               cons_df(4*dust_id, k, j, i) = rho_new; // avoid negative density after relaxation 
//               cons_df(4*dust_id+1, k, j, i) = cons_df(4*dust_id, k, j, i) * v1s(dust_id);
//               cons_df(4*dust_id+2, k, j, i) = cons_df(4*dust_id, k, j, i) * v2s(dust_id);
//               cons_df(4*dust_id+3, k, j, i) = cons_df(4*dust_id, k, j, i) * v3s(dust_id);
//             }
//
//             int n_id = N_Z*N_P + 1 + p;
//             ns_new(p)= ns(p) + relax_rate*(n_relax_array(p) - ns(p));
//
//             // prevent overshoot for number density as well 
//             Real sign_before = n_relax_array(p) - ns(p); 
//             Real sign_after = n_relax_array(p) - ns_new(p);
//             // if (sign_before*sign_after < 0.0){
//             //   ns_new(p) = n_relax_array(p);
//             // }
//
//             cons_df(4*n_id, k, j, i) = ns_new(p); // number density
//           }
//
//           //calculate the power law after relaxation for checking
//           AthenaArray<Real> rho_pop_new, m_p_new;
//           rho_pop_new.NewAthenaArray(N_P);
//           m_p_new.NewAthenaArray(N_P);
//           for (int p = 0; p < N_P; ++p) {
//             for (int zi = 0; zi < N_Z; ++zi) {
//               int dust_id = N_Z*p + zi; 
//               rho_pop_new(p) += cons_df(4*dust_id, k, j, i); // total density for population p
//             }
//
//             m_p_new(p) = rho_pop_new(p)/ns_new(p); // new characteristic mass for population p
//             // m_p_array(p,k,j,i) = m_p_new(p); // update the characteristic mass for population 0
//           }
//
//           Real q_after = std::log((rho_pop_new(1))/(rho_pop_new(0)))/std::log(m_p_new(1)/m_p_new(0)); // the effective powerlaw index of the relaxed state
//           //
//           Real m_0_relax = rho_pop_relax(0)/n_relax_array(0);
//           Real m_1_relax = rho_pop_relax(1)/n_relax_array(1); 
//           Real q_relax = std::log((rho_pop_relax(1))/(rho_pop_relax(0)))/std::log(m_1_relax/m_0_relax);
//
//           if ((q_after-1.0/6.0)/(q_before-1.0/6.0)<0.0){
//             std::cout << "overshoot" << std::endl;
//           }
//
//           Real mass_before = rho_pop_array(0) + rho_pop_array(1);
//           Real mass_after = rho_pop_new(0) + rho_pop_new(1);
//
//           if (std::abs(mass_after - mass_before)/mass_before > 1.e-10){
//             Real eee;
//             for (int d = 0; d< NFLUIDS; ++d){
//               eee += rho_relax_array(d) - rhos(d);
//             }
//             std::cout << "### WARNING: Mass not conserved during relaxation at cell (" << k << "," << j << "," << i << ")" << std::endl;
//             std::cout << "Mass before relaxation: " << mass_before << std::endl;
//             std::cout << "Mass after relaxation: " << mass_after << std::endl;
//           }
//
//         }
//       }
//     }
//   }
// }

Real Relaxation::Tem_gas(const Real rad){
    return Tem0*std::pow(rad/r0, Tslope);
}
Real Relaxation::Get_mu(Real fv) {
  return 1.0/((1.0 - fv)/mu_xy + fv/mu_z);
}

void Relaxation::RelaxationSource(MeshBlock *pmb, const Real time, const Real dt,
      const Real gm0, const Real alpha_vis,
      const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
      AthenaArray<Real> &cons, AthenaArray<Real> &cons_df,
      AthenaArray<Real> &cons_s, AthenaArray<Real> &v_frag) {

  //           std::cout << time <<std::endl;
  // std::cout<< dt <<std::endl;
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
#pragma omp simd
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        //1. Gather current state
        int NFLUIDS = NDUSTFLUIDS - 1 - N_P;   // number of dust fluids (excluding vapor & number density)
        AthenaArray<Real> rhos, v1s, v2s, v3s, ns;
        rhos.NewAthenaArray(NFLUIDS);
        v1s.NewAthenaArray(NFLUIDS);
        v2s.NewAthenaArray(NFLUIDS);
        v3s.NewAthenaArray(NFLUIDS);
        ns.NewAthenaArray(N_P);               // number density for each bin

        Real rho_sil = 0.0, rho_ice = 0.0;

        for (int n = 0; n < NFLUIDS; ++n) {
          rhos(n) = cons_df(4*n, k, j, i);
          // composition: assume N_Z = 2 for silicate/ice, else treat as single population
          if (N_Z == 1) {
            rho_sil += rhos(n);
          } else {
            if (n % N_Z == 1 || N_Z == 1) rho_sil += rhos(n);
            else if (n % N_Z == 0)         rho_ice += rhos(n);
          }
          v1s(n) = cons_df(4*n+1, k, j, i) / rhos(n);
          v2s(n) = cons_df(4*n+2, k, j, i) / rhos(n);
          v3s(n) = cons_df(4*n+3, k, j, i) / rhos(n);

          int bin_id = n / N_Z;
          int n_id = N_Z * N_P + 1 + bin_id;
          ns(bin_id) = cons_df(4 * n_id, k, j, i);
        }

        //2. Disk properties
        Real rad = pmb->pcoord->x1v(i);
        Real Tem = Tem_gas(rad);
        Real rho_g = cons(IDN, k, j, i);
        Real rho_v = cons_df(4 * vapor_id, k, j, i);
        Real OmegaK = std::sqrt(gm0 / std::pow(rad, 3));
        Real fv = rho_v / rho_g;
        Real cs2 = Tem / KELVIN * Get_mu(fv);
        Real vth = std::sqrt(8.0 / PI * cs2);

        // total dust mass (excluding vapor and number densities)
        Real total_dust_mass = 0.0;
        for (int n = 0; n < NFLUIDS; ++n) total_dust_mass += rhos(n);

        // composition ratio (global, as in original)
        AthenaArray<Real> comp_ratio;
        comp_ratio.NewAthenaArray(N_Z);
        if (N_Z == 1) {
          comp_ratio(0) = 1.0;
        } else {
          comp_ratio(0) = rho_ice / (rho_ice + rho_sil + 1e-30);
          comp_ratio(1) = 1.0 - comp_ratio(0);
        }

        Real rho_total = rho_sil + rho_ice; // avoid division by zero
        Real f_ice = rho_ice / rho_total;
        Real f_sil = rho_sil / rho_total;
        
        // Calculate effective internal density (harmonic mean of ice and silicate densities)
        Real rho_inte_relax = rho_ice_inter_ * rho_sil_inter_ / 
                           (f_ice * rho_sil_inter_ + f_sil * rho_ice_inter_);

        // maximum grain mass (fragmentation limit)
        Real v_frag_tot = (v_frag(1) * rho_sil + v_frag(0) * rho_ice) / (rho_sil + rho_ice);
        Real St = SQR(v_frag_tot)/cs2/3.0/alpha_vis;
        Real s_max = St* rho_g * vth / (OmegaK * rho_inte_relax);
        Real m_max = FOUR_3RD * PI * SQR(s_max)*s_max * rho_inte_relax;
        mmax_array(k, j, i) = m_max;

        AthenaArray<Real> rho_comps, s_p_array;
        s_p_array.NewAthenaArray(N_P);
        for (int p = 0; p < N_P; ++p) {
          Real &m_p = m_p_array(p, k, j, i);

          if (m_p== 0.0){
            //[26.06.04]Zhixuan: if restart, since the m_p_array is not registered to a restart property, so it will be 0, and we can get it by the density and number density 
            Real rho_pop = 0.0;
            for (int zi = 0; zi < N_Z; ++zi) {
              int dust_id = N_Z*p + zi;
              rho_pop += cons_df(4*dust_id, k, j, i); // total density for population p
            }
            int n_id = 4*(N_Z*N_P + 1 + p);
            Real n_pop = cons_df(n_id, k, j, i);
            m_p = rho_pop / (n_pop);
          }

          //get internal density for this pop 
          AthenaArray<Real> rho_comp_pop;
          rho_comp_pop.NewAthenaArray(N_Z);
          for (int zi = 0; zi < N_Z; ++zi) {
            int dust_id = N_Z*p + zi;
            rho_comp_pop(zi) = cons_df(4*dust_id, k, j, i);
          }
          Real rho_inte_pop = 0.0;
          if (N_Z == 1) {
            rho_inte_pop = rho_sil_inter_;
          } else {
            Real f_ice_pop = rho_comp_pop(0) / (rho_comp_pop(0) + rho_comp_pop(1) + 1e-30);
            Real f_sil_pop = rho_comp_pop(1) / (rho_comp_pop(0) + rho_comp_pop(1) + 1e-30);
            rho_inte_pop = rho_ice_inter_ * rho_sil_inter_ / 
                           (f_ice_pop * rho_sil_inter_ + f_sil_pop * rho_ice_inter_);
          }
          s_p_array(p) = std::pow(m_p/(FOUR_3RD*PI*rho_inte_pop), ONE_3RD); // [code_length]
          
          if (std::isnan(s_p_array(p))) {
            std::cout << "Error: s_p_array(" << p << ") is NaN at cell (" << k << "," << j << "," << i << ")" << std::endl;
          }
        }
        //3. Compute division masses between bins
        AthenaArray<Real> m_div;
        m_div.NewAthenaArray(N_P - 1);
        GetDivisionMasses(mmin, m_max, m_div, "small");   // fill m_div

        //4. Relaxed distribution (MRN, slope -11/6)
        Real c_relax = total_dust_mass / (6.0 * (std::pow(m_max, 1.0/6.0) - std::pow(mmin, 1.0/6.0)));
        AthenaArray<Real> M1_relax, M2_relax, n_relax, rho_pop_relax;
        M1_relax.NewAthenaArray(N_P);
        M2_relax.NewAthenaArray(N_P);
        n_relax.NewAthenaArray(N_P);
        rho_pop_relax.NewAthenaArray(N_P);

        for (int p = 0; p < N_P; ++p) {
          Real m_low, m_high;
          if (p == 0) {
            m_low = mmin;
            m_high = m_div(0);
          } else if (p == N_P - 1) {
            m_low = m_div(N_P - 2);
            m_high = m_max;
          } else {
            m_low = m_div(p - 1);
            m_high = m_div(p);
          }
          // First moment: ∫ m f*(m) dm = c_relax * 6 * (m_high^{1/6} - m_low^{1/6})
          M1_relax(p) = c_relax * 6.0 * (std::pow(m_high, 1.0/6.0) - std::pow(m_low, 1.0/6.0));
          // Second moment: ∫ m^2 f*(m) dm = c_relax * (6/7) * (m_high^{7/6} - m_low^{7/6})
          M2_relax(p) = c_relax * (6.0/7.0) * (std::pow(m_high, 7.0/6.0) - std::pow(m_low, 7.0/6.0));
          // number density ('monodisperse' approximation)
          // [26.05.17]Zhixuan: The M2 can be very small b/c of the mass integration, so here we shouldn't 
          //                  add the 1.e-30 factor, which was a bug here...
          n_relax(p) = SQR(M1_relax(p)) / (M2_relax(p));
          rho_pop_relax(p) = M1_relax(p);
        }

        //5. Relaxation timescale (based on largest bin)
        int largest_bin = N_P - 1;
        Real t_stop = pmb->pdustfluids->stopping_time_array(largest_bin, k, j, i);
        Real St1 = t_stop * OmegaK;
        Real delV = std::sqrt(3.0 * alpha_vis * St1 * cs2);
        // number density of largest bin (volumetric)
        Real rho_Np_vol = cons_df(4 * (N_P*N_Z + 1 + largest_bin), k, j, i);
        Real trelax = 1.0 / (4.0 * rho_Np_vol * PI * SQR(s_p_array(largest_bin)) * delV) * T_relax_prefactor;
        t_relax(k,j,i) = trelax; // store for diagnostics
        Real relax_rate = dt / trelax;
        // [26.06.24]Zhixuan: Not an elegent way ...
        if (relax_rate > 1.0) relax_rate = 1.0;   // limit to full relaxation

        //6. Perform relaxation for each dust fluid and number density
        AthenaArray<Real> rho_new, ns_new;
        rho_new.NewAthenaArray(NFLUIDS);
        ns_new.NewAthenaArray(N_P);

        // Store old values for mass check
        Real mass_before = 0.0;
        for (int n = 0; n < NFLUIDS; ++n) mass_before += cons_df(4*n, k, j, i);

        for (int p = 0; p < N_P; ++p) {
          Real rho_pop_new = 0.0;
          for (int zi = 0; zi < N_Z; ++zi) {
            int dust_id = N_Z * p + zi;
            Real rho_target = rho_pop_relax(p) * comp_ratio(zi);
            Real rho_current = cons_df(4*dust_id, k, j, i);
            Real rho_new_val = rho_current + relax_rate * (rho_target - rho_current);
            cons_df(4*dust_id, k, j, i) = rho_new_val;
            cons_df(4*dust_id+1, k, j, i) = rho_new_val * v1s(dust_id);
            cons_df(4*dust_id+2, k, j, i) = rho_new_val * v2s(dust_id);
            cons_df(4*dust_id+3, k, j, i) = rho_new_val * v3s(dust_id);

            //check for NaN values in the updated density
            if (std::isnan(rho_new_val) ||rho_new_val<=0.0 || std::isnan(v1s(dust_id)) || std::isnan(v2s(dust_id)) || std::isnan(v3s(dust_id))) {
              int dk     = NGHOST;
              int dj     = NGHOST;
              if (pmb->block_size.nx3 == 1) dk = 0;
              if (pmb->block_size.nx2 == 1) dj = 0;
              int kl = pmb->ks - dk;     int ku = pmb->ke + dk;
              int jl = pmb->js - dj;     int ju = pmb->je + dj;
              int il = pmb->is - NGHOST; int iu = pmb->ie + NGHOST;
              int ti = static_cast<int>(pmb->loc.lx1)*pmb->block_size.nx1+(i-il);
              int tj = static_cast<int>(pmb->loc.lx2)*pmb->block_size.nx2+(j-jl);
              int tk = static_cast<int>(pmb->loc.lx3)*pmb->block_size.nx3+(k-kl);
              std::cout << "### WARNING: NaN velocity for dust_id " << dust_id 
                        << " at cell (" << tk << "," << tj << "," << ti << ")" << std::endl;
              std::cout << "rho_target: " << rho_target << " rho_current: " << rho_current 
                        << " rho_new_val: " << rho_new_val << std::endl;
              std::cout << "v1: " << v1s(dust_id) << " v2: " << v2s(dust_id) << " v3: " << v3s(dust_id) << std::endl;
            }

            rho_pop_new += rho_new_val;

            // std::cout << "Relaxation: bin " << p << " zi " << zi << " rho_target " << rho_target 
            //           << " rho_current " << rho_current << " rho_new " << rho_new_val << std::endl;
          }

          int n_id = N_Z * N_P + 1 + p;
          int dust_id = N_Z*p;
          Real n_target = n_relax(p);
          Real n_current = cons_df(4*n_id, k, j, i);
          Real n_new_val = n_current + relax_rate * (n_target - n_current);
          cons_df(4*n_id, k, j, i) = n_new_val;
          //[26.06.09]Zhixuan: this will somehow lead to the decrease of time step, no idea why
          cons_df(4*n_id+1, k, j, i) = n_new_val * v1s(dust_id);
          cons_df(4*n_id+2, k, j, i) = n_new_val * v2s(dust_id);
          cons_df(4*n_id+3, k, j, i) = n_new_val * v3s(dust_id);
          ns_new(p) = n_new_val;
          // std::cout << "Relaxation: bin " << p << " n_target " << n_target 
          //           << " n_current " << n_current << " n_new " << n_new_val << std::endl;
          if (std::isnan(n_new_val) || n_new_val <= 0.0) {
            int dk     = NGHOST;
            int dj     = NGHOST;
            if (pmb->block_size.nx3 == 1) dk = 0;
            if (pmb->block_size.nx2 == 1) dj = 0;
            int kl = pmb->ks - dk;     int ku = pmb->ke + dk;
            int jl = pmb->js - dj;     int ju = pmb->je + dj;
            int il = pmb->is - NGHOST; int iu = pmb->ie + NGHOST;
            int ti = static_cast<int>(pmb->loc.lx1)*pmb->block_size.nx1+(i-il);
            int tj = static_cast<int>(pmb->loc.lx2)*pmb->block_size.nx2+(j-jl);
            int tk = static_cast<int>(pmb->loc.lx3)*pmb->block_size.nx3+(k-kl);
            std::cout << "### WARNING: NaN number density for bin " << p 
                      << " at cell (" << tk << "," << tj << "," << ti << ")" << std::endl;
            std::cout << "n_target: " << n_target << " n_current: " << n_current 
                      << " n_new_val: " << n_new_val << std::endl;
          }

          m_p_array(p,k,j,i) = rho_pop_new / ns_new(p); // update characteristic mass for this bin 
        }

        // //7. Update characteristic masses for each bin
        // for (int p = 0; p < N_P; ++p) {
        //   Real rho_pop_new = 0.0;
        //   for (int zi = 0; zi < N_Z; ++zi) {
        //     int dust_id = N_Z * p + zi;
        //     rho_pop_new += cons_df(4*dust_id, k, j, i);
        //   }
        //   if (ns_new(p) > 0.0) {
        //     m_p_array(p, k, j, i) = rho_pop_new / ns_new(p);
        //   } else {
        //     m_p_array(p, k, j, i) = m_p0_array(p);   // fallback
        //   }
        // }
        

        //mass conservation check
        Real mass_after = 0.0;
        for (int n = 0; n < NFLUIDS; ++n) mass_after += cons_df(4*n, k, j, i);
        if (std::abs(mass_after - mass_before) / (mass_before + 1e-30) > 1e-10) {
          std::cout << "### WARNING: Mass not conserved during relaxation at cell ("
                    << k << "," << j << "," << i << ")" << std::endl;
          std::cout << "Mass before: " << mass_before << " after: " << mass_after << std::endl;
        }
      }
    }
  }
}
