//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file ideal.cpp
//! \brief implements ideal EOS in general EOS framework, mostly for debuging
//======================================================================================

// C headers

// C++ headers
#include <sstream>    // stringstream

// Athena++ headers
#include "../eos.hpp"
#include "../../phase_change/phase_change_constants.hpp"

Real EquationOfState::Get_mu(Real fv){
  return 1./((1.-fv)/mu_xy + fv / mu_z);
}

Real EquationOfState::calc_gamma(Real fv){
  if(std::isnan(fv)){
    std::stringstream msg;
    msg << "### FATAL ERROR in EquationOfState::calc_gamma" << std::endl
        << "fv= nan" << std::endl;
    ATHENA_ERROR(msg);
    return -1.0;
  }
  Real mu = Get_mu(fv);
  Real fx = 1.0-fv;
  Real gamma_minus1_inv = mu*(fx/mu_H2*0.71*2.5+fx/mu_He*0.29*1.5+fv/mu_z*3.0);

  return 1.0/gamma_minus1_inv + 1.0;
}
//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::PresFromRhoEg_fv(Real rho, Real egas, Real fv)
//! \brief Return gas pressure
Real EquationOfState::PresFromRhoEg_fv(Real rho, Real egas, Real fv) {
  if(std::isnan(fv)){
    std::stringstream msg;
    msg << "### FATAL ERROR in EquationOfState::PresFromRhoEg_fv" << std::endl
        << "fv= nan" << std::endl;
    ATHENA_ERROR(msg);
    return -1.0;
  }
  Real gm1 = calc_gamma(fv) - 1.0;
  return gm1 * egas;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::EgasFromRhoP_fv(Real rho, Real pres, Real fv)
//! \brief Return internal energy density
Real EquationOfState::EgasFromRhoP_fv(Real rho, Real pres, Real fv) {
  if(std::isnan(fv)){
    std::stringstream msg;
    msg << "### FATAL ERROR in EquationOfState::EgasFromRhoP_fv" << std::endl
        << "fv= nan" << std::endl;
    ATHENA_ERROR(msg);
    return -1.0;
  }
  Real gm1 = calc_gamma(fv) - 1.0;
  return pres / gm1;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::AsqFromRhoP_fv(Real rho, Real pres, Real fv)
//! \brief Return adiabatic sound speed squared
Real EquationOfState::AsqFromRhoP_fv(Real rho, Real pres, Real fv) {
  if(std::isnan(fv)){
    std::stringstream msg;
    msg << "### FATAL ERROR in EquationOfState::AsqFromRhoP_fv" << std::endl
        << "fv= nan" << std::endl;
    ATHENA_ERROR(msg);
    return -1.0;
  }
  Real gamma_calc = calc_gamma(fv);
  return gamma_calc * pres / rho;
}

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::InitEosConstants(ParameterInput* pin)
//! \brief Initialize constants for EOS
void EquationOfState::InitEosConstants(ParameterInput *pin) {
  return;
}


//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::PresFromRhoEg(Real rho, Real egas)
//! \brief Return gas pressure
Real EquationOfState::PresFromRhoEg(Real rho, Real egas) {
  return (gamma_ - 1.) * egas;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::EgasFromRhoP(Real rho, Real pres)
//! \brief Return internal energy density
Real EquationOfState::EgasFromRhoP(Real rho, Real pres) {
  return pres / (gamma_ - 1.);
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::AsqFromRhoP(Real rho, Real pres)
//! \brief Return adiabatic sound speed squared
Real EquationOfState::AsqFromRhoP(Real rho, Real pres) {
  return gamma_ * pres / rho;
}