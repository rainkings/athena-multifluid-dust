// Coupled finite-rate phase change and two-stream / beta thermal relaxation.
// All temperatures are in kelvin; other quantities use Athena code units.
// This file deliberately has no MeshBlock dependency: trials never write to
// the live simulation, and each OpenMP thread owns its entire column workspace.
#ifndef PHASE_CHANGE_COUPLED_COLUMN_HPP_
#define PHASE_CHANGE_COUPLED_COLUMN_HPP_

#include <string>
#include <vector>

#include "../athena.hpp"

namespace coupled_column {

struct Population {
  Real ice = 0.0;
  Real refractory = 0.0;
  Real number = 0.0;
};

struct Cell {
  Real temperature = 0.0;
  Real gas = 0.0;                 // Includes vapor, as in the existing pgen.
  Real vapor = 0.0;
  Real path_length = 0.0;         // r * dtheta, surface toward midplane.
  Real cylindrical_radius = 0.0;
  Real viscosity = 0.0;
  Real q_diff = 0.0;              // Frozen throughout this column solve.
  std::vector<Population> populations;
};

struct Settings {
  Real dt = 0.0;
  Real beta = 50.0;
  Real gm = 0.0;
  Real r0 = 1.0;
  Real temperature_slope = -0.5;
  Real stellar_luminosity = 0.0;  // Converted to code energy / code time.
  Real sigma = 0.0;               // Stefan-Boltzmann constant in code units.
  Real kappa0 = 0.0;
  Real visible_opacity_ratio = 1.0;
  Real latent_heat = 0.0;
  Real equilibrium_pressure = 0.0;
  Real ice_density = 0.0;
  Real refractory_density = 0.0;
  Real source_floor = 0.0;
  Real tolerance = 1.e-7;
  int max_iterations = 40;
  bool phase_change = true;
  bool viscous_heating = false;
};

struct Solution {
  std::vector<Cell> cells;
  std::vector<Real> equilibrium_temperature;
  std::vector<Real> tau, rhokappa, q_latent, q_total, integrated_heating;
  int iterations = 0;
  Real residual = 0.0;
};

// The input is immutable. Failure returns an explanation and no accepted state.
// Cells must be ordered from the upper surface to the reflecting midplane.
bool Solve(const std::vector<Cell> &saved, const Settings &settings,
           Solution &solution, std::string &error);

}  // namespace coupled_column
#endif  // PHASE_CHANGE_COUPLED_COLUMN_HPP_
