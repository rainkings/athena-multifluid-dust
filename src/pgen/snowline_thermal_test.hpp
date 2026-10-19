//========================================================================================
// Fixed-density irradiation, cooling, and viscous-heating tests for the snowline pgen.
// Uses the pgen's plane-parallel theta columns (ds = r dtheta), not spherical
// flux divergence. Radial radiation, phase change, and hydrodynamics are disabled.
//========================================================================================
#ifndef PGEN_SNOWLINE_THERMAL_TEST_HPP_
#define PGEN_SNOWLINE_THERMAL_TEST_HPP_

#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>

#include "snowline_twostream.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

namespace SnowlineThermalTest {
using SnowlineTwoStream::Column;
using SnowlineTwoStream::Fourth;
using SnowlineTwoStream::Scalar;
using SnowlineTwoStream::Vector;

// Variables exposed only in the opt-in thermal test; normal pgen output is unchanged.
enum Output { kEnergy, kAnalytic, kDiscrete, kRelativeError, kTau, kIrradiation,
              kPsi, kInitialDensity, kCooling, kEqStatus, kIrrEq, kPulseEnd,
              kPulseEq, kPulseStatus, kPulseT4, kViscousNative, kViscousPrescribed,
              kPrescribedEquilibrium, kCount };

// Athena adapter for frozen-density experiments. It builds the same Column as
// production but owns pulse controls, steady references, and diagnostic output.
// The viscous audit measures native hydro operators without evolving the flow.
struct Comparison {
  bool enabled = false, initialized = false;
  bool viscous_test = false;
  bool start_equilibrium = false;
  Real initial_factor = 1, temperature_ref = 150;
  Real cooling_ratio = 0, cooling_start = 0, cooling_stop = 1;
  Real cooling_radius = 2, cooling_radial_width = 0.25;
  std::vector<Column> columns;
  std::vector<Real> initial_density;
  std::vector<Real> viscous_native, viscous_prescribed, prescribed_equilibrium;
  std::vector<Real> viscous_normalized;
  Scalar max_energy_defect = 0;

  void Configure(ParameterInput *pin, Mesh *mesh) {
    enabled = pin->GetOrAddBoolean("problem", "thermal_test", false);
    if (!enabled) return;
    viscous_test = pin->GetOrAddBoolean("problem", "thermal_test_viscous", false);
    initial_factor = pin->GetOrAddReal("problem", "thermal_test_initial_factor", 0.5);
    temperature_ref = pin->GetOrAddReal("problem", "thermal_test_Tref", 150);
    start_equilibrium = pin->GetOrAddBoolean("problem", "thermal_test_start_equilibrium", false);
    cooling_ratio = pin->GetOrAddReal("problem", "thermal_test_cooling_ratio", 0);
    cooling_start = pin->GetOrAddReal("problem", "thermal_test_cooling_start", 0);
    cooling_stop = pin->GetOrAddReal("problem", "thermal_test_cooling_stop", 1);
    cooling_radius = pin->GetOrAddReal("problem", "thermal_test_cooling_radius_au", 2);
    cooling_radial_width = pin->GetOrAddReal("problem", "thermal_test_cooling_width_au", 0.25);
    if (!std::isfinite(cooling_ratio) || cooling_ratio < 0
        || !std::isfinite(cooling_start) || cooling_start < 0
        || !std::isfinite(cooling_stop) || cooling_stop <= cooling_start
        || !std::isfinite(cooling_radius) || cooling_radius <= 0
        || !std::isfinite(cooling_radial_width) || cooling_radial_width <= 0) {
      std::stringstream msg;
      msg << "### FATAL ERROR: invalid prescribed thermal cooling parameters\n";
      ATHENA_ERROR(msg);
    }
    if (mesh->fluid_setup != FluidFormulation::fixed || mesh->nlim <= 0
        || mesh->tlim <= mesh->time || mesh->adaptive || mesh->multilevel
        || mesh->mesh_size.nx3 != 1 || mesh->mesh_size.nx2 <= 1
        || std::strcmp(COORDINATE_SYSTEM, "spherical_polar") != 0
        || !NON_BAROTROPIC_EOS || initial_factor <= 0 || temperature_ref <= 0
        || !std::isfinite(initial_factor) || !std::isfinite(temperature_ref)
        || pin->GetBoolean("problem", "ACTIVE")
        || pin->GetBoolean("problem", "PhaseChange_Flag")
        || pin->GetBoolean("problem", "Relaxation_Flag")
        || !pin->GetBoolean("problem", "Allow_T_change_Flag")
        || (!viscous_test && pin->GetOrAddInteger("orbital_advection", "OAorder", 0) != 0)
        || (viscous_test && (cooling_ratio != 0
            || STS_ENABLED
            || pin->GetOrAddReal("problem", "alpha_vis", 0) <= 0
            || pin->GetOrAddReal("problem", "nu_iso", 0) != 0
            || pin->GetOrAddReal("problem", "nu_aniso", 0) != 0))
        || pin->GetOrAddReal("problem", "kappa_aniso", 0) != 0
        || pin->GetOrAddReal("problem", "dust_start_injection", 0) <= mesh->tlim
        || std::abs(mesh->mesh_size.x2max-PI/2) > 1e-8) {
      std::stringstream msg;
      msg << "### FATAL ERROR: thermal_test requires a fixed spherical 2D half disk, "
          << "positive nlim/tlim and initial factor, no refinement, "
          << "OAorder=0 unless thermal_test_viscous=true, "
          << "ACTIVE/phase change/relaxation off, kappa_aniso=0, "
          << "Allow_T_change_Flag=true, and dust injection after tlim. "
          << "The viscous audit also requires positive alpha_vis, nu_iso=nu_aniso=0, "
          << "no prescribed cooling, and no STS.\n";
      ATHENA_ERROR(msg);
    }
  }

  int GlobalIndex(MeshBlock *block, int j, int i) const {
    return (block->loc.lx2*block->block_size.nx2+j-block->js)
           *block->pmy_mesh->mesh_size.nx1
           +block->loc.lx1*block->block_size.nx1+i-block->is;
  }

  // Measure du/dt = dE/dt - v.dm/dt with the production viscous flux,
  // divergence, coordinate, and orbital source operators on a prescribed
  // cylindrical Keplerian flow. Only this measured heat is retained: density
  // and shear remain fixed, separating the thermal test from disk evolution.
  // Analytic ghost profiles avoid boundary artifacts in this operator audit.
  void PrepareViscosity(Mesh *mesh, Real gm,
                        Real (*density)(Real, Real, Real, Real, Real),
                        Real (*profile)(Real, Real, Real)) {
    const int count = mesh->mesh_size.nx1*mesh->mesh_size.nx2;
    viscous_native.assign(count, 0);
    viscous_prescribed.assign(count, 0);
    prescribed_equilibrium.assign(count, 0);
    viscous_normalized.assign(count, 0);
    for (int b = 0; b < mesh->nblocal; ++b) {
      MeshBlock *p = mesh->my_blocks(b);
      Hydro *h = p->phydro;
      AthenaArray<Real> physical(NHYDRO, p->ncells3, p->ncells2, p->ncells1);
      AthenaArray<Real> residual(NHYDRO, p->ncells3, p->ncells2, p->ncells1);
      AthenaArray<Real> rate(NHYDRO, p->ncells3, p->ncells2, p->ncells1);
      const int k = p->ks;
      for (int j = p->js-NGHOST; j <= p->je+NGHOST; ++j) {
        for (int i = p->is-NGHOST; i <= p->ie+NGHOST; ++i) {
          const Real radius = p->pcoord->x1v(i)*std::sin(p->pcoord->x2v(j));
          const Real z = p->pcoord->x1v(i)*std::cos(p->pcoord->x2v(j));
          const Real base = profile(radius, 0, z), rho = density(radius, 0, z, 0, base);
          const Real velocity = std::sqrt(gm/radius)-p->porb->Omega0*radius;
          physical(IDN,k,j,i) = residual(IDN,k,j,i) = rho;
          physical(IPR,k,j,i) = residual(IPR,k,j,i) = rho*base/(KELVIN*mu_xy);
          physical(IM3,k,j,i) = velocity;
          residual(IM3,k,j,i) = velocity;
          if (p->porb->orbital_advection_defined)
            residual(IM3,k,j,i) -= p->porb->OrbitalVelocity(p->porb,
                p->pcoord->x1v(i), p->pcoord->x2v(j), p->pcoord->x3v(k));
        }
      }
      h->hdif.CalcDiffusionFlux(residual, physical, p->pfield->bcc);
      // Apply only the viscous fluxes, not advection or pressure fluxes. The
      // resulting unit-time increment is a derivative, not a finite fluid step.
      for (int dim = 0; dim < 3; ++dim)
        for (int n = 0; n < h->flux[dim].GetSize(); ++n)
          h->flux[dim](n) = h->hdif.visflx[dim](n);
      h->AddFluxDivergence(1, rate);
      p->pcoord->AddCoordTermsDivergence(1, h->flux, residual, p->pfield->bcc, rate);
      if (p->porb->orbital_advection_defined)
        h->hsrc.OrbitalAdvectionSourceTerms(1, h->flux, residual, rate);
      for (int j = p->js; j <= p->je; ++j) {
        for (int i = p->is; i <= p->ie; ++i) {
          const int g = GlobalIndex(p, j, i);
          const Real radius = p->pcoord->x1v(i)*std::sin(p->pcoord->x2v(j));
          const Real rho = physical(IDN,k,j,i);
          const Real fv = p->pdustfluids->df_u(4*vapor_id,k,j,i)/rho;
          viscous_native[g] = rate(IEN,k,j,i)-residual(IM3,k,j,i)*rate(IM3,k,j,i);
          // Keep the original H/He-only factor in the independent reference.
          // This audit uses negligible vapor; finite-fv viscosity is a separate
          // physical choice since native stresses use the total gas density.
          viscous_prescribed[g] = 2.25*rho*(1-fv)*gm/std::pow(radius,3)
              *h->hdif.nu(HydroDiffusion::DiffProcess::alpha,k,j,i);
        }
      }
    }
#ifdef MPI_PARALLEL
    MPI_Allreduce(MPI_IN_PLACE, viscous_native.data(), count, MPI_ATHENA_REAL,
                  MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, viscous_prescribed.data(), count, MPI_ATHENA_REAL,
                  MPI_SUM, MPI_COMM_WORLD);
#endif
  }

  void Initialize(Mesh *mesh, Real opacity, Real f_vi, Real luminosity,
                  Real r0, Real qvalue, Real gm,
                  Real (*density)(Real, Real, Real, Real, Real),
                  Real (*profile)(Real, Real, Real)) {
    if (viscous_test) PrepareViscosity(mesh, gm, density, profile);
    const int nr = mesh->mesh_size.nx1, nt = mesh->mesh_size.nx2, count = nr*nt;
    // Gather rho, vapor fraction, cylindrical R, path length, T, and ownership.
    // Active cells are indexed independently of MeshBlock iteration order.
    // A collective is needed here because a theta column may span MPI ranks.
    std::vector<Real> data(6*count, 0);
    for (int b = 0; b < mesh->nblocal; ++b) {
      MeshBlock *block = mesh->my_blocks(b);
      for (int j = block->js; j <= block->je; ++j) {
        for (int i = block->is; i <= block->ie; ++i) {
          const int g = GlobalIndex(block, j, i), k = block->ks;
          const Real rho = block->phydro->u(IDN,k,j,i);
          const Real fv = block->pdustfluids->df_u(4*vapor_id,k,j,i)/rho;
          const Real gamma = block->peos->calc_gamma(fv), mu = 1/((1-fv)/mu_xy+fv/mu_z);
          data[g] = rho;
          data[count+g] = fv;
          data[2*count+g] = block->pcoord->x1v(i)*std::sin(block->pcoord->x2v(j));
          data[3*count+g] = block->pcoord->x1v(i)*block->pcoord->dx2f(j);
          data[4*count+g] = block->phydro->u(IEN,k,j,i)*KELVIN*mu*(gamma-1)/rho;
          data[5*count+g] = 1;
        }
      }
    }
#ifdef MPI_PARALLEL
    MPI_Allreduce(MPI_IN_PLACE, data.data(), data.size(), MPI_ATHENA_REAL,
                  MPI_SUM, MPI_COMM_WORLD);
#endif
    const Scalar flux_unit = mesh->punit->code_energydensity_cgs
                             *mesh->punit->code_velocity_cgs;
    const Scalar sigma = Constants::sigma_cgs/flux_unit;
    const Scalar reference_flux = sigma*std::pow(Scalar(temperature_ref), 4);
    initial_density.assign(data.begin(), data.begin()+count);
    columns.reserve(nr);
    for (int i = 0; i < nr; ++i) {
      columns.emplace_back(nt);
      Column &column = columns.back();
      Scalar total_tau = 0;
      for (int j = 0; j < nt; ++j) {
        const int g = j*nr+i;
        if (data[5*count+g] != 1 || !(data[g] > 0)) {
          std::stringstream msg;
          msg << "### FATAL ERROR: incomplete snowline thermal comparison grid\n";
          ATHENA_ERROR(msg);
        }
        const Scalar fv = data[count+g], rho = data[g], chi = rho*opacity*(1-fv);
        const Scalar mu = 1/((1-fv)/mu_xy+fv/mu_z);
        const Scalar gamma = mesh->my_blocks(0)->peos->calc_gamma(fv);
        const Scalar cv = rho/(KELVIN*mu*(gamma-1));
        if (!(chi > 0) || !std::isfinite(chi)) {
          std::stringstream msg;
          msg << "### FATAL ERROR: thermal_test requires positive finite IR opacity\n";
          ATHENA_ERROR(msg);
        }
        column.width[j] = chi*data[3*count+g];
        column.capacity[j] = cv/(chi*sigma*std::pow(Scalar(temperature_ref), 3));
        column.theta[j] = data[4*count+g]/temperature_ref;
        total_tau += column.width[j];
      }
      Scalar tau = 0;
      for (int j = 0; j < nt; ++j) {
        const Scalar rad = data[2*count+j*nr+i];
        const Scalar height = std::pow(rad/r0, qvalue/2+1.5);
        const Scalar incidence = (qvalue/2+0.5)*(4*height/rad);
        const Scalar luminosity_unit = flux_unit*mesh->punit->code_length_cgs
                                                *mesh->punit->code_length_cgs;
        const Scalar e0 = luminosity*Constants::solar_lum_cgs
                          /(8*PI*rad*rad*luminosity_unit);
        const Scalar center = tau+column.width[j]/2;
        if (!(incidence > 0) || !(f_vi > 0) || !(e0 > 0)) {
          std::stringstream msg;
          msg << "### FATAL ERROR: thermal_test requires positive "
              << "irradiation/incidence\n";
          ATHENA_ERROR(msg);
        }
        column.source[j] = e0*f_vi/reference_flux
            *(std::exp(-f_vi*center/incidence)
              +std::exp(-f_vi*(2*total_tau-center)/incidence));
        tau += column.width[j];
      }
      if (viscous_test) {
        // Independently construct the former prescribed-q_vis equilibrium.
        Column reference(nt);
        reference.width = column.width;
        reference.theta = column.theta;
        for (int j = 0; j < nt; ++j) {
          const int g = j*nr+i;
          const Scalar chi = data[g]*opacity*(1-data[count+g]);
          reference.source[j] = column.source[j]+viscous_prescribed[g]/(chi*reference_flux);
          viscous_normalized[g] = viscous_native[g]/(chi*reference_flux);
          column.source[j] += viscous_normalized[g];
        }
        reference.Initialize();
        for (int j = 0; j < nt; ++j)
          prescribed_equilibrium[j*nr+i] = temperature_ref*reference.analytic[j];
      }
      column.Initialize();
      column.radius_au = data[2*count+(nt-1)*nr+i]
                         *mesh->punit->code_length_cgs/Constants::au_cgs;
      // Smooth midplane sink in optical depth, normalized so its integrated
      // strength is A*F_irr times a radial Gaussian. This prescribes energy
      // extraction only; it does not convert vapor or ice mass.
      Scalar center_tau = 0, normalization = 0;
      for (int j = 0; j < nt; ++j) {
        const Scalar fraction = (center_tau+column.width[j]/2)/total_tau;
        column.cooling[j] = std::exp(-Scalar(0.5)*std::pow((fraction-0.9L)/0.1L, 2));
        normalization += column.width[j]*column.cooling[j];
        center_tau += column.width[j];
      }
      const Scalar amplitude = cooling_ratio*column.absorbed*std::exp(-Scalar(0.5)
          *std::pow((column.radius_au-cooling_radius)/cooling_radial_width, 2));
      for (int j = 0; j < nt; ++j) column.cooling[j] *= amplitude/normalization;
      column.SetSource(true);
      column.pulse_eq = column.analytic;
      column.pulse_t4 = column.eq_t4;
      column.pulse_status = column.eq_status;
      column.SetSource(false);
      if (start_equilibrium) {
        column.theta = column.irradiation_eq;
        Vector emission(nt);
        for (int j = 0; j < nt; ++j)
          emission[j] = column.width[j]*4*Fourth(column.theta[j]);
        column.psi = column.Solve(column.width, emission);
      }
      column.pulse_end = column.theta;
      column.initial_energy = column.Energy();
      column.SetSource(cooling_ratio > 0 && mesh->time >= cooling_start
                       && mesh->time < cooling_stop);
    }
    initialized = true;
    if (start_equilibrium) Scatter(mesh);
    if (cooling_ratio > 0 && Globals::my_rank == 0) {
      std::ofstream history("thermal_cooling.csv");
      history << "time,column,radius_au,min_T,midplane_T,eq_status,energy,initial_energy,"
                 "irradiation_integral,cooling_integral,escape_integral,energy_defect\n";
    }
    History(mesh->time);
  }

  void History(Scalar time) const {
    if (cooling_ratio <= 0 || Globals::my_rank != 0) return;
    std::ofstream history("thermal_cooling.csv", std::ios::app);
    history << std::setprecision(18);
    for (std::size_t i = 0; i < columns.size(); ++i) {
      const Column &c = columns[i];
      history << time << ',' << i << ',' << c.radius_au << ','
              << temperature_ref*(*std::min_element(c.theta.begin(), c.theta.end()))
              << ',' << temperature_ref*c.theta.back() << ',' << c.eq_status << ','
              << c.Energy() << ',' << c.initial_energy << ',' << c.injected << ','
              << c.extracted << ',' << c.escaped << ',' << max_energy_defect << '\n';
    }
  }

  void Advance(Mesh *mesh) {
    // Split exactly at source on/off times, even when a global step straddles
    // the pulse. All columns see the same source clock. No source is clipped.
    // Match Athena's Real-valued clock addition. Extending only this addition
    // to long double creates spurious substeps of ~1e-18 at pulse boundaries.
    Scalar now = mesh->time, end = mesh->time+mesh->dt;
    if (cooling_ratio > 0) {
      const Scalar clock_roundoff = 8*std::numeric_limits<Real>::epsilon()
                                    *std::max(std::abs(now), std::abs(end));
      for (Real event : {cooling_start, cooling_stop}) {
        if (std::abs(now-event) <= clock_roundoff) now = event;
        if (std::abs(end-event) <= clock_roundoff) end = event;
      }
    }
    while (now < end) {
      const bool active = cooling_ratio > 0 && now >= cooling_start && now < cooling_stop;
      Scalar stop = end;
      if (cooling_ratio > 0 && now < cooling_start)
        stop = std::min(stop, Scalar(cooling_start));
      if (active) stop = std::min(stop, Scalar(cooling_stop));
      for (auto &column : columns) {
        column.SetSource(active);
        Scalar elapsed = 0;
        int substeps = 0;
        while (elapsed < stop-now) {
          Scalar dt = stop-now-elapsed;
          // Ensure a positive material RHS even before incoming diffuse
          // radiation is known. This is a stored-energy bound, not a floor.
          for (std::size_t j = 0; j < column.width.size(); ++j)
            if (column.source[j] < 0)
              dt = std::min(dt, -Scalar(0.5)*column.capacity[j]
                                *column.theta[j]/column.source[j]);
          Scalar defect = 0;
          int retries = 0;
          while (!column.Step(dt, defect)) {
            dt /= 2;
            if (++retries > 50) break;
          }
          if (retries > 50 || ++substeps > 100000 || !(elapsed+dt > elapsed)) {
            std::stringstream msg;
            msg << "### FATAL ERROR: prescribed cooling exhausted thermal energy "
                << "or thermal solve failed; shorten/reduce the pulse.\n";
            ATHENA_ERROR(msg);
          }
          max_energy_defect = std::max(max_energy_defect, defect);
          elapsed += dt;
        }
        if (active && stop == cooling_stop) column.pulse_end = column.theta;
      }
      now = stop;
      History(now);
    }
    Scatter(mesh);
    if (Globals::my_rank == 0 && (mesh->ncycle%mesh->ncycle_out == 0
                                  || mesh->ncycle+1 == mesh->nlim)) {
      Scalar analytic_error = 0, discrete_error = 0, flux_error = 0;
      int invalid = 0, negative_flux = 0;
      for (const auto &column : columns) {
        invalid += column.eq_status != 0;
        negative_flux += (column.eq_status & 1) != 0;
        if (column.eq_status) continue;
        for (std::size_t j = 0; j < column.width.size(); ++j) {
          analytic_error = std::max(analytic_error,
                                    std::abs(column.theta[j]/column.analytic[j]-1));
          discrete_error = std::max(discrete_error,
                                    std::abs(column.theta[j]/column.discrete[j]-1));
        }
        flux_error = std::max(flux_error, std::abs(column.surface*column.psi[0]
                                                  /column.absorbed-1));
      }
      std::cout << "[thermal_test] cycle=" << mesh->ncycle+1
                << " analytic_error=" << Real(analytic_error)
                << " discrete_error=" << Real(discrete_error)
                << " flux_error=" << Real(flux_error)
                << " invalid_columns=" << invalid
                << " negative_flux_columns=" << negative_flux
                << " energy_defect=" << Real(max_energy_defect) << std::endl;
    }
  }

  // In the frozen tests the integrated velocity is zero (residual zero with
  // OA). Thus the stored conserved energy here contains only gas internal heat.
  void Scatter(Mesh *mesh) {
    for (int b = 0; b < mesh->nblocal; ++b) {
      MeshBlock *block = mesh->my_blocks(b);
      for (int j = block->js; j <= block->je; ++j) {
        for (int i = block->is; i <= block->ie; ++i) {
          const int g = GlobalIndex(block, j, i), nr = mesh->mesh_size.nx1;
          const int ti = g%nr, tj = g/nr, k = block->ks;
          const Real rho = block->phydro->u(IDN,k,j,i);
          const Real fv = block->pdustfluids->df_u(4*vapor_id,k,j,i)/rho;
          const Real mu = 1/((1-fv)/mu_xy+fv/mu_z);
          const Real temperature = temperature_ref*columns[ti].theta[tj];
          block->phydro->Tem(k,j,i) = temperature;
          block->phydro->u(IEN,k,j,i) = rho*temperature
              /(KELVIN*mu*(block->peos->calc_gamma(fv)-1));
          block->phydro->w(IPR,k,j,i) = rho*temperature/(KELVIN*mu);
          mesh->ruser_mesh_data[4](0,tj+NGHOST,ti+NGHOST) = temperature;
        }
      }
    }
  }

  void OutputBlock(MeshBlock *block) const {
    const int nr = block->pmy_mesh->mesh_size.nx1;
    for (int j = block->js; j <= block->je; ++j) {
      for (int i = block->is; i <= block->ie; ++i) {
        const int g = GlobalIndex(block, j, i), ti = g%nr, tj = g/nr, k = block->ks;
        const Column &column = columns[ti];
        Scalar tau = column.width[tj]/2;
        for (int l = 0; l < tj; ++l) tau += column.width[l];
        block->user_out_var(kEnergy,k,j,i) = temperature_ref*column.theta[tj];
        block->user_out_var(kAnalytic,k,j,i)
            = column.eq_status ? -1 : temperature_ref*column.analytic[tj];
        block->user_out_var(kDiscrete,k,j,i)
            = column.eq_status ? -1 : temperature_ref*column.discrete[tj];
        block->user_out_var(kRelativeError,k,j,i)
            = column.eq_status ? -1 : column.theta[tj]/column.analytic[tj]-1;
        block->user_out_var(kTau,k,j,i) = tau;
        block->user_out_var(kIrradiation,k,j,i) = column.irradiation[tj]
            -(viscous_test ? viscous_normalized[g] : 0);
        block->user_out_var(kPsi,k,j,i) = column.psi[tj];
        block->user_out_var(kInitialDensity,k,j,i) = initial_density[g];
        block->user_out_var(kCooling,k,j,i) = -column.cooling[tj];
        block->user_out_var(kEqStatus,k,j,i) = column.eq_status;
        block->user_out_var(kIrrEq,k,j,i) = temperature_ref*column.irradiation_eq[tj];
        block->user_out_var(kPulseEnd,k,j,i) = temperature_ref*column.pulse_end[tj];
        block->user_out_var(kPulseEq,k,j,i)
            = column.pulse_status ? -1 : temperature_ref*column.pulse_eq[tj];
        block->user_out_var(kPulseStatus,k,j,i) = column.pulse_status;
        block->user_out_var(kPulseT4,k,j,i) = column.pulse_t4[tj];
        block->user_out_var(kViscousNative,k,j,i) = viscous_test ? viscous_native[g] : 0;
        block->user_out_var(kViscousPrescribed,k,j,i) = viscous_test ? viscous_prescribed[g] : 0;
        block->user_out_var(kPrescribedEquilibrium,k,j,i)
            = viscous_test ? prescribed_equilibrium[g] : 0;
      }
    }
  }
};
}  // namespace SnowlineThermalTest
#endif  // PGEN_SNOWLINE_THERMAL_TEST_HPP_
