#include "coupled_column.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <utility>

#include "phase_change_constants.hpp"

namespace coupled_column {
namespace {

Real FourthPower(Real x) { return x*x*x*x; }

Real IceFloor(const Population &p, const Settings &s) {
  // A bound on transfer, not a floor applied repeatedly to the live state.
  // A starting reservoir below source_floor is left intact, never increased.
  return std::min(p.ice, s.source_floor);
}

Real RadiusSquared(Real ice, const Population &p, const Settings &s) {
  if (p.number == 0.0) return 0.0;
  const Real volume = ice/s.ice_density + p.refractory/s.refractory_density;
  const Real radius = std::cbrt(3.0*volume/(4.0*PI*p.number));
  return radius*radius;
}

// At fixed trial T and vapor density, solve the backward-Euler equation for
// ONE population's final ice. Its area is evaluated at the FINAL ice mass.
// All populations will subsequently be tied to the same vapor reservoir.
Real SolveIce(const Population &p, Real vapor, Real vapor_eq, Real available_vapor,
              Real speed, const Settings &s, Real &vapor_derivative) {
  vapor_derivative = 0.0;
  if (p.number == 0.0 || vapor == vapor_eq) return p.ice;
  const Real coefficient = s.dt*speed*p.number*(vapor_eq-vapor);
  Real left = coefficient > 0.0 ? IceFloor(p, s) : p.ice;
  Real right = coefficient > 0.0 ? p.ice : p.ice + available_vapor;
  auto residual = [&](Real ice) {
    return ice-p.ice + coefficient*RadiusSquared(ice, p, s);
  };
  if (residual(left) >= 0.0) return left;  // Ice supply exhausted.
  if (residual(right) <= 0.0) return right;

  // Safeguarded Newton: the bracket also handles the rapidly varying grain
  // area near depletion. No trial can produce a negative ice density.
  Real ice = 0.5*(left+right);
  for (int iter = 0; iter < 80; ++iter) {
    const Real area = RadiusSquared(ice, p, s);
    const Real value = ice-p.ice + coefficient*area;
    const Real scale = std::max(p.ice+available_vapor, Real(1.e-100));
    if (std::abs(value) <= 2.e-14*scale) break;
    if (value > 0.0) right = ice;
    else left = ice;
    if (right-left <= 2.e-13*scale) {
      ice = 0.5*(left+right);
      break;
    }
    const Real volume = ice/s.ice_density + p.refractory/s.refractory_density;
    const Real derivative = 1.0 + coefficient*area*2.0/(3.0*s.ice_density*volume);
    const Real next = ice-value/derivative;
    ice = std::isfinite(next) && next > left && next < right
        ? next : 0.5*(left+right);
  }
  const Real volume = ice/s.ice_density + p.refractory/s.refractory_density;
  const Real area = RadiusSquared(ice, p, s);
  vapor_derivative = s.dt*speed*p.number*area
      /(1.0+coefficient*area*2.0/(3.0*s.ice_density*volume));
  return ice;
}

Cell SolvePhase(const Cell &saved, Real temperature, const Settings &s) {
  Cell trial = saved;
  trial.temperature = temperature;
  if (!s.phase_change) return trial;

  const Real vapor_eq = s.equilibrium_pressure*std::exp(-T_a/temperature)
      *KELVIN*mu_z/temperature;
  const Real speed = std::sqrt(8.0*PI*temperature/(KELVIN*mu_z));
  const Real minimum_vapor = std::min(saved.vapor, s.source_floor);
  const Real available_vapor = saved.vapor-minimum_vapor;
  Real maximum_vapor = saved.vapor;
  for (const Population &p : saved.populations)
    maximum_vapor += p.ice-IceFloor(p, s);

  auto evaluate = [&](Real vapor) {
    Real returned_vapor = saved.vapor;
    Real derivative = 1.0;
    for (std::size_t p = 0; p < saved.populations.size(); ++p) {
      Real ice_derivative;
      const Real ice = SolveIce(saved.populations[p], vapor, vapor_eq,
                                available_vapor, speed, s, ice_derivative);
      trial.populations[p].ice = ice;
      returned_vapor += saved.populations[p].ice-ice;
      derivative += ice_derivative;
    }
    return std::make_pair(vapor-returned_vapor, derivative);
  };

  // At fixed temperature each population moves toward the same saturation
  // density. The scalar mass-balance root enforces their shared vapor supply.
  Real left = std::max(minimum_vapor, std::min(saved.vapor, vapor_eq));
  Real right = std::min(maximum_vapor, std::max(saved.vapor, vapor_eq));
  const Real mass_scale = std::max(maximum_vapor, Real(1.e-100));
  Real vapor = saved.vapor;
  for (int iter = 0; iter < 80; ++iter) {
    const auto value = evaluate(vapor);
    if (std::abs(value.first) <= 2.e-13*mass_scale) break;
    if (value.first > 0.0) right = vapor;
    else left = vapor;
    if (right-left <= 2.e-13*mass_scale) {
      vapor = 0.5*(left+right);
      break;
    }
    const Real next = vapor-value.first/value.second;
    vapor = std::isfinite(next) && next > left && next < right
        ? next : 0.5*(left+right);
  }
  evaluate(vapor);

  // Obtain vapor from mass conservation, rather than independently rounding
  // both sides of the transfer. Hydrogen/helium density remains unchanged.
  trial.vapor = saved.vapor;
  for (std::size_t p = 0; p < saved.populations.size(); ++p)
    trial.vapor += saved.populations[p].ice-trial.populations[p].ice;
  trial.gas = saved.gas + (trial.vapor-saved.vapor);
  return trial;
}

// This is the existing two-stream equilibrium formula, written WITHOUT
// dividing by the surface flux. That division is unnecessary and is singular
// when signed latent/diffusion terms make the net flux vanish in a trial.
// q_latent is negative for sublimation; it is never clipped or made positive.
void Radiation(const std::vector<Cell> &saved, const Settings &s,
               Solution &trial, std::vector<Real> &equilibrium_fourth_power) {
  const std::size_t n = saved.size();
  trial.tau.resize(n);
  trial.rhokappa.resize(n);
  trial.q_latent.resize(n);
  trial.q_total.resize(n);
  trial.integrated_heating.resize(n);
  equilibrium_fourth_power.resize(n);
  Real tau_face = 0.0;
  for (std::size_t j = 0; j < n; ++j) {
    const Cell &cell = trial.cells[j];
    // Exactly the current Get_kappa(d2g, fv) = kappa0*(1-fv).
    trial.rhokappa[j] = s.kappa0*(cell.gas-cell.vapor);
    const Real dtau = trial.rhokappa[j]*cell.path_length;
    trial.tau[j] = tau_face+0.5*dtau;
    tau_face += dtau;
  }
  const Real full_tau = 2.0*tau_face;
  Real heating_face = 0.0;
  for (std::size_t j = 0; j < n; ++j) {
    const Cell &cell = trial.cells[j];
    const Real radius = cell.cylindrical_radius;
    const Real height = std::pow(radius/s.r0, s.temperature_slope/2.0+1.5);
    const Real incidence = (s.temperature_slope/2.0+0.5)*4.0*height/radius;
    const Real incident_flux = s.stellar_luminosity/(8.0*PI*radius*radius);
    const Real q_irr = incident_flux*s.visible_opacity_ratio*trial.rhokappa[j]
        *(std::exp(-s.visible_opacity_ratio*trial.tau[j]/incidence)
          +std::exp(-s.visible_opacity_ratio*(full_tau-trial.tau[j])/incidence));
    const Real omega_squared = s.gm/(radius*radius*radius);
    const Real q_vis = s.viscous_heating
        ? 2.25*(cell.gas-cell.vapor)*cell.viscosity*omega_squared : 0.0;
    trial.q_latent[j] = -s.latent_heat*(cell.vapor-saved[j].vapor)/s.dt;
    trial.q_total[j] = q_irr+q_vis+saved[j].q_diff+trial.q_latent[j];
    const Real dheat = trial.q_total[j]*cell.path_length;
    trial.integrated_heating[j] = heating_face+0.5*dheat;
    heating_face += dheat;
  }
  const Real surface_flux = heating_face;
  Real moment_face = 0.0;
  for (std::size_t j = 0; j < n; ++j) {
    const Real dmoment = trial.rhokappa[j]
        *(surface_flux-trial.integrated_heating[j])*trial.cells[j].path_length;
    const Real moment_center = moment_face+0.5*dmoment;
    equilibrium_fourth_power[j] = (0.75*moment_center
        +std::sqrt(3.0)/4.0*surface_flux
        +trial.q_total[j]/(4.0*trial.rhokappa[j]))/s.sigma;
    moment_face += dmoment;
  }
}

struct Evaluation {
  Solution trial;
  std::vector<Real> residual;
  std::vector<Real> radiation_fourth;
  Real norm = 0.0;
  Real temperature_defect = 0.0;
};

// Unknowns are log(T_eq / T_saved). Beta relaxation always starts from
// T_saved. Working with the fourth-power radiation residual lets Newton pass
// through trial states whose signed heating has no positive equilibrium;
// an ACCEPTED solution must still have a positive radiative equilibrium.
Evaluation Evaluate(const std::vector<Cell> &saved, const Settings &s,
                    const std::vector<Real> &log_equilibrium) {
  Evaluation result;
  result.trial.cells.resize(saved.size());
  result.trial.equilibrium_temperature.resize(saved.size());
  result.residual.resize(saved.size());
  for (std::size_t j = 0; j < saved.size(); ++j) {
    const Cell &cell = saved[j];
    const Real omega = std::sqrt(s.gm/std::pow(cell.cylindrical_radius, 3));
    const Real fraction = std::min(s.dt*omega/s.beta, Real(1.0));
    const Real equilibrium = cell.temperature*std::exp(log_equilibrium[j]);
    const Real temperature = cell.temperature+fraction*(equilibrium-cell.temperature);
    result.trial.equilibrium_temperature[j] = equilibrium;
    result.trial.cells[j] = SolvePhase(cell, temperature, s);
  }
  Radiation(saved, s, result.trial, result.radiation_fourth);
  for (std::size_t j = 0; j < saved.size(); ++j) {
    const Real radiation_fourth = result.radiation_fourth[j];
    const Real predicted = FourthPower(result.trial.equilibrium_temperature[j]);
    const Real scale = std::max(FourthPower(saved[j].temperature),
                               std::abs(radiation_fourth));
    result.residual[j] = (predicted-radiation_fourth)/scale;
    if (!std::isfinite(result.residual[j])) result.norm = INFINITY;
    else result.norm = std::max(result.norm, std::abs(result.residual[j]));
    if (radiation_fourth < 0.0 || !std::isfinite(radiation_fourth)) {
      result.temperature_defect = INFINITY;
    } else {
      const Real target = std::sqrt(std::sqrt(radiation_fourth));
      const Real omega = std::sqrt(s.gm/std::pow(saved[j].cylindrical_radius, 3));
      const Real fraction = std::min(s.dt*omega/s.beta, Real(1.0));
      const Real beta_temperature = saved[j].temperature
          +fraction*(target-saved[j].temperature);
      result.temperature_defect = std::max(result.temperature_defect,
          std::abs(beta_temperature-result.trial.cells[j].temperature)/saved[j].temperature);
    }
  }
  return result;
}

// Small dense solve for one column. Partial pivoting is sufficient here; no
// global matrix, external linear algebra library, or shared scratch is needed.
bool LinearSolve(std::vector<std::vector<Real>> matrix, std::vector<Real> rhs,
                 std::vector<Real> &answer) {
  const int n = rhs.size();
  for (int k = 0; k < n; ++k) {
    int pivot = k;
    for (int i = k+1; i < n; ++i)
      if (std::abs(matrix[i][k]) > std::abs(matrix[pivot][k])) pivot = i;
    if (!std::isfinite(matrix[pivot][k]) || std::abs(matrix[pivot][k]) < 1.e-14)
      return false;
    std::swap(matrix[pivot], matrix[k]);
    std::swap(rhs[pivot], rhs[k]);
    for (int i = k+1; i < n; ++i) {
      const Real factor = matrix[i][k]/matrix[k][k];
      for (int j = k+1; j < n; ++j) matrix[i][j] -= factor*matrix[k][j];
      rhs[i] -= factor*rhs[k];
    }
  }
  answer.assign(n, 0.0);
  for (int i = n-1; i >= 0; --i) {
    Real value = rhs[i];
    for (int j = i+1; j < n; ++j) value -= matrix[i][j]*answer[j];
    answer[i] = value/matrix[i][i];
  }
  return true;
}

bool Validate(const std::vector<Cell> &saved, const Settings &s, std::string &error) {
  if (saved.empty() || !(s.dt > 0.0 && s.beta > 0.0 && s.gm > 0.0
      && s.r0 > 0.0 && s.sigma > 0.0 && s.kappa0 > 0.0
      && s.ice_density > 0.0 && s.refractory_density > 0.0
      && s.source_floor >= 0.0 && s.stellar_luminosity >= 0.0
      && s.visible_opacity_ratio >= 0.0 && s.temperature_slope > -1.0
      && s.tolerance > 0.0 && s.max_iterations > 0)) {
    error = "invalid column settings (positive opacity, beta, dt and grazing angle required)";
    return false;
  }
  for (const Cell &cell : saved) {
    if (!(cell.temperature > 0.0 && cell.gas > cell.vapor && cell.vapor >= 0.0
        && cell.path_length > 0.0 && cell.cylindrical_radius > 0.0
        && cell.viscosity >= 0.0 && std::isfinite(cell.q_diff))) {
      error = "invalid starting temperature, density, geometry or frozen q_diff";
      return false;
    }
    for (const Population &p : cell.populations) {
      if (!(p.ice >= 0.0 && p.refractory > 0.0 && p.number >= 0.0)) {
        error = "invalid starting ice, refractory or particle number density";
        return false;
      }
    }
  }
  return true;
}

std::string FailureDescription(const Evaluation &current, const char *reason) {
  std::ostringstream message;
  message << reason << "; residual=" << current.norm;
  for (std::size_t j = 0; j < current.radiation_fourth.size(); ++j) {
    if (current.radiation_fourth[j] < 0.0) {
      message << "; trial row " << j << " has negative radiation T_eq^4="
              << current.radiation_fourth[j]
              << " (q_total=" << current.trial.q_total[j]
              << ", q_latent=" << current.trial.q_latent[j] << ')';
      break;
    }
  }
  return message.str();
}

}  // namespace

bool Solve(const std::vector<Cell> &saved, const Settings &s,
           Solution &solution, std::string &error) {
  error.clear();
  if (!Validate(saved, s, error)) return false;
  const std::size_t n = saved.size();

  // The zero-heating limit has T_eq=0 exactly. Treat it directly rather than
  // representing log(0). With a finite beta step the accepted gas T stays
  // positive, unless the clipped cooling factor is already one.
  bool zero_heating = !s.phase_change && s.stellar_luminosity == 0.0;
  for (const Cell &cell : saved)
    zero_heating = zero_heating && cell.q_diff == 0.0
        && (!s.viscous_heating || cell.viscosity == 0.0);
  if (zero_heating) {
    Solution trial;
    trial.cells = saved;
    trial.equilibrium_temperature.assign(n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
      const Real omega = std::sqrt(s.gm/std::pow(saved[j].cylindrical_radius,3));
      trial.cells[j].temperature *= 1.0-std::min(s.dt*omega/s.beta, Real(1.0));
      if (!(trial.cells[j].temperature > 0.0)) {
        error = "zero heating and a fully relaxed beta step imply zero gas temperature";
        return false;
      }
    }
    std::vector<Real> fourth;
    Radiation(saved, s, trial, fourth);
    trial.iterations = 1;
    solution = std::move(trial);
    return true;
  }
  std::vector<Real> x(n, 0.0);  // First guess: T_eq = saved T.

  for (int iter = 0; iter < s.max_iterations; ++iter) {
    Evaluation current = Evaluate(saved, s, x);
    if (current.norm < s.tolerance && current.temperature_defect < s.tolerance) {
      current.trial.iterations = iter+1;
      current.trial.residual = current.norm;
      solution = std::move(current.trial);
      return true;
    }

    // First try the inexpensive phase -> two-stream -> beta fixed-point
    // iteration requested by the model. Use Newton only if feedback prevents
    // a clear improvement. Ordinary disk columns do not need a dense Jacobian.
    std::vector<Real> picard = x;
    bool positive_equilibrium = true;
    for (std::size_t j = 0; j < n; ++j) {
      const Real fourth = current.radiation_fourth[j];
      if (!(fourth > 0.0 && std::isfinite(fourth))) {
        positive_equilibrium = false;
        break;
      }
      picard[j] = 0.25*std::log(fourth)-std::log(saved[j].temperature);
    }
    if (positive_equilibrium) {
      Evaluation next = Evaluate(saved, s, picard);
      if (next.norm < 0.8*current.norm) {
        x = std::move(picard);
        continue;
      }
    }

    // A plain fixed-point iteration can oscillate at the snowline because
    // saturation pressure is exponential in T. A damped Newton iteration
    // solves the SAME phase -> radiation -> saved-T beta equations.
    std::vector<std::vector<Real>> jacobian(n, std::vector<Real>(n));
    const Real perturbation = 1.e-5;
    for (std::size_t column = 0; column < n; ++column) {
      std::vector<Real> perturbed = x;
      perturbed[column] += perturbation;
      Evaluation shifted = Evaluate(saved, s, perturbed);
      for (std::size_t row = 0; row < n; ++row)
        jacobian[row][column] = (shifted.residual[row]-current.residual[row])/perturbation;
    }
    std::vector<Real> rhs = current.residual, step;
    for (Real &value : rhs) value = -value;
    if (!LinearSolve(jacobian, rhs, step)) {
      error = FailureDescription(current, "singular column thermal Jacobian");
      return false;
    }
    Real largest_step = 0.0;
    for (Real value : step) largest_step = std::max(largest_step, std::abs(value));
    Real damping = largest_step > 1.0 ? 1.0/largest_step : 1.0;
    bool accepted = false;
    for (int search = 0; search < 24; ++search) {
      std::vector<Real> candidate = x;
      for (std::size_t j = 0; j < n; ++j) candidate[j] += damping*step[j];
      Evaluation next = Evaluate(saved, s, candidate);
      if (next.norm < current.norm*(1.0-1.e-4*damping)) {
        x = std::move(candidate);
        accepted = true;
        break;
      }
      damping *= 0.5;
    }
    if (!accepted) {
      error = FailureDescription(current, "column line search failed");
      return false;
    }
  }
  error = FailureDescription(Evaluate(saved, s, x), "column iteration limit reached");
  return false;
}

}  // namespace coupled_column
