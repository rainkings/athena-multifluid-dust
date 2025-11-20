#ifndef PHASE_CHANGE_PHASE_CHANGE_CONSTANTS_HPP_
#define PHASE_CHANGE_PHASE_CHANGE_CONSTANTS_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file phase_change_constants.hpp
//! \brief Physical constants for phase change module
//!
//! User-specified constants for gas/dust/ice species properties used in phase change
//! calculations. Modify these values as needed for your problem setup.

// Athena++ headers
#include "../athena.hpp"

#ifndef PHASE_CHANGE_CONSTANTS_DEFINED
#define PHASE_CHANGE_CONSTANTS_DEFINED

// Physical constants for gas/dust/ice species
// These are user-specified and should be modified for your specific problem

//! Latent heat of vaporization/sublimation [erg/g]
constexpr Real L_heat_cgs = 2.75e10;

//! Heat capacity of water ice [erg/(g*K)]
constexpr Real Cd_water_cgs = 1.0;

//! Equilibrium vapor pressure prefactor [erg/cm^3]
constexpr Real P_eq0_cgs = 1.14e13;

//! Temperature scale for phase change [K]
constexpr Real T_a = 6062.0;

//! Mean molecular weight of vapor species (e.g., H2O) [amu]
constexpr Real mu_z = 18.0;

//! Mean molecular weight of gas mixture (H2+He) [amu]
constexpr Real mu_xy = 2.34;
constexpr Real mu_H2 = 2.0;
constexpr Real mu_He = 4.0;

//! KELVIN 1/(kb/m_p)
constexpr Real KELVIN = 6.4102564103e+01;

//! Collisional cross-section of H2 [cm^2]
constexpr Real sigma_mol_cgs = 2.e-15;

// number of vapor species
#define        NVapor    1  // (Yu) only one vapor is allowed now.
#define        vapor_id  (NDUSTFLUIDS-1)  // last dust fluid as vapor. must have this parenthesis when defining!
#define        FLX_COR   1


#endif // PHASE_CHANGE_CONSTANTS_DEFINED

#endif // PHASE_CHANGE_PHASE_CHANGE_CONSTANTS_HPP_

