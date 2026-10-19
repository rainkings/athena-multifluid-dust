//========================================================================================
// Shared quasi-static two-stream solver for the snowline disk and fixed-disk tests.
// This header has no Mesh/MeshBlock dependency; callers assemble each theta column.
//========================================================================================
#ifndef PGEN_SNOWLINE_TWOSTREAM_HPP_
#define PGEN_SNOWLINE_TWOSTREAM_HPP_

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace SnowlineTwoStream {
// Very small surface optical depths make the column matrix ill-conditioned.
// Retain extended precision throughout the solve, including the small powers.
using Scalar = long double;
using Vector = std::vector<Scalar>;

// These powers dominate the material Newton loop if dispatched to powl.
// Preserve long-double arithmetic without a general transcendental evaluation.
inline Scalar Cube(Scalar x) { return x*x*x; }
inline Scalar Fourth(Scalar x) { const Scalar square = x*x; return square*square; }
inline Scalar FourthRoot(Scalar x) { return std::sqrt(std::sqrt(x)); }

// A half-disk column, ordered from surface (j=0) to midplane (j=n-1).
// Geometry is plane-parallel along ds=r*dtheta, as in the original pgen.
//
// With F_ref=sigma*T_ref^4 and chi=rho*kappa_IR, the stored variables are:
//   width    = chi*ds                       (cell optical depth)
//   theta    = T/T_ref                      (material temperature)
//   psi      = Psi/F_ref, Psi=4*pi*J=c*E_rad (thermal radiation field)
//   source   = q_external/(chi*F_ref)        (positive for heating)
//   capacity = C_V*T_ref/(chi*F_ref)         (time in caller's code units)
// C_V is heat capacity per volume; the caller may include ice at the common T.
//
// The coupled equations are:
//   capacity*d(theta)/dt = source + psi - 4*theta^4
//   -(1/3)*d^2(psi)/d(tau)^2 + psi = 4*theta^4.
// Radiation has no time derivative: its field adjusts to the evolving material
// temperature. This is not the assumption that the material is in equilibrium.
// Outward flux is F/F_ref=(1/3)*d(psi)/d(tau), with tau increasing inward.
struct Column {
  Vector width, source, capacity, theta, analytic, discrete, psi, lower, upper;
  // Optional equilibrium/pulse diagnostics used by the fixed-disk tests. The
  // production evolution does not use these reference temperatures as targets.
  // "irradiation" stores the baseline source (also viscous heat in its audit).
  Vector irradiation, cooling, irradiation_eq, pulse_end, pulse_eq, pulse_t4, eq_t4;
  Scalar surface, absorbed, net_heating = 0, initial_energy = 0;
  Scalar injected = 0, extracted = 0, escaped = 0, radius_au = 0;
  int eq_status = 0, pulse_status = 0;

  explicit Column(int n) : width(n), source(n), capacity(n), theta(n), analytic(n),
      discrete(n), psi(n), lower(n, 0), upper(n, 0), irradiation(n), cooling(n, 0),
      irradiation_eq(n), pulse_end(n), pulse_eq(n), pulse_t4(n), eq_t4(n),
      surface(0), absorbed(0) {}

  // Thermal energy per column area divided by F_ref (therefore a code time).
  Scalar Energy() const {
    Scalar value = 0;
    for (std::size_t j = 0; j < width.size(); ++j)
      value += capacity[j]*width[j]*theta[j];
    return value;
  }

  // Test helper: subtract a prescribed cooling pulse from the baseline source.
  void SetSource(bool active) {
    net_heating = 0;
    for (std::size_t j = 0; j < width.size(); ++j) {
      source[j] = irradiation[j]-(active ? cooling[j] : 0);
      net_heating += width[j]*source[j];
    }
    Equilibrium();
  }

  // Independent steady references: analytic follows the original pgen's
  // quadrature; discrete solves the equilibrium of this finite-volume stencil.
  // Their difference measures spatial error, not failure to converge in time.
  // Status bit 1: negative boundary flux; bit 2: negative radiation field;
  // bit 4: negative material T^4. Invalid temperatures use the diagnostic
  // sentinel -1, never a clipped physical target or an input to the evolution.
  void Equilibrium(bool compute_temperatures = true) {
    eq_status = net_heating < 0 ? 1 : 0;
    Scalar integral = 0, weighted = 0, discrete_psi = 0;
    for (std::size_t j = 0; j < width.size(); ++j) {
      const Scalar cell_heating = source[j]*width[j];
      const Scalar flux_center = net_heating-integral-cell_heating/2;
      const Scalar field = std::sqrt(Scalar(3))*net_heating
                           +3*(weighted+width[j]*flux_center/2);
      discrete_psi = j == 0 ? net_heating/surface : discrete_psi
          +Scalar(1.5)*(width[j-1]+width[j])*(net_heating-integral);
      const Scalar t4 = (field+source[j])/4;
      const Scalar fv_t4 = (discrete_psi+source[j])/4;
      if (field < 0 || discrete_psi < 0) eq_status |= 2;
      if (t4 < 0 || fv_t4 < 0) eq_status |= 4;
      if (compute_temperatures) {
        analytic[j] = t4 >= 0 ? FourthRoot(t4) : -1;
        discrete[j] = fv_t4 >= 0 ? FourthRoot(fv_t4) : -1;
      }
      eq_t4[j] = t4;
      weighted += width[j]*flux_center;
      integral += cell_heating;
    }
    if (eq_status && compute_temperatures) {
      std::fill(analytic.begin(), analytic.end(), -1);
      std::fill(discrete.begin(), discrete.end(), -1);
    }
  }

  // Solve the tridiagonal transport operator plus a supplied diagonal term.
  // "mass" is the algebraic reaction coefficient, not material cell mass.
  Vector Solve(const Vector &mass, const Vector &rhs) const {
    const int n = width.size();
    Vector d(n), b(rhs), answer(n);
    // Carry the effective left conductance without subtracting nearly equal
    // large pivots. Surface cells can have d(tau)~1e-11; ordinary Thomas
    // elimination loses the outgoing boundary term on this disk grid.
    Scalar effective_left = surface;
    d[0] = -upper[0]+mass[0]+effective_left;
    for (int j = 1; j < n; ++j) {
      const Scalar conductance = -lower[j];
      effective_left = conductance*(mass[j-1]+effective_left)/d[j-1];
      d[j] = -upper[j]+mass[j]+effective_left;
      b[j] += conductance*b[j-1]/d[j-1];
    }
    answer[n-1] = b[n-1]/d[n-1];
    for (int j = n-2; j >= 0; --j) answer[j] = (b[j]-upper[j]*answer[j+1])/d[j];
    return answer;
  }

  // Face fluxes point outward; face 0 escapes and face n has midplane symmetry.
  Vector Flux(const Vector &field) const {
    const int n = width.size();
    Vector flux(n+1, 0);
    flux[0] = surface*field[0];
    for (int j = 1; j < n; ++j) flux[j] = -lower[j]*(field[j]-field[j-1]);
    return flux;
  }

  void Initialize(bool equilibrium_diagnostics = true) {
    const int n = width.size();
    // No incoming diffuse IR: psi_surface=sqrt(3)*F_surface/F_ref.
    // Include transport across the first half-cell in the surface conductance.
    // The zero-initialized lower/upper endpoints impose zero midplane flux.
    surface = 1/(std::sqrt(Scalar(3)) + Scalar(1.5)*width[0]);
    for (int j = 1; j < n; ++j) {
      const Scalar conductance = 2/(3*(width[j-1]+width[j]));
      lower[j] = -conductance;
      upper[j-1] = -conductance;
    }
    absorbed = 0;
    for (int j = 0; j < n; ++j) {
      absorbed += source[j]*width[j];
    }
    irradiation = source;
    net_heating = absorbed;
    if (equilibrium_diagnostics) {
      SetSource(false);
      irradiation_eq = discrete;
      pulse_end = theta;
    }
    Vector emission(n);
    for (int j = 0; j < n; ++j) emission[j] = width[j]*4*Fourth(theta[j]);
    psi = Solve(width, emission);  // Consistent radiation field at the supplied T.
  }

  Vector MaterialTemperature(const Vector &old, const Vector &alpha,
                             const Vector &field) const {
    const int n = width.size();
    Vector result(n);
    for (int j = 0; j < n; ++j) {
      // The implicit material equation is a monotone scalar quartic. Starting
      // from this upper bound gives positive, decreasing Newton iterates.
      // theta + 4*alpha*theta^4 = old + alpha*(source+psi), alpha=dt/capacity.
      // Callers restrict negative external sources so the RHS stays positive.
      const Scalar rhs = old[j]+alpha[j]*(source[j]+field[j]);
      Scalar t = std::min(rhs, FourthRoot(rhs/(4*alpha[j])));
      for (int iteration = 0; iteration < 30; ++iteration) {
        const Scalar cube = Cube(t);
        const Scalar correction = (t+4*alpha[j]*cube*t-rhs)/(1+16*alpha[j]*cube);
        t -= correction;
        if (std::abs(correction) < 2e-18L*t) break;
      }
      result[j] = t;
    }
    return result;
  }

  // Backward Euler for C_V dT/dt = q_external + chi(Psi-4 sigma T^4).
  // Eliminate the local material equation nonlinearly, then solve the remaining
  // radiation equation with a tridiagonal Newton step. This avoids competing
  // radiation/material residual scales in optically thin surface cells.
  // Geometry, opacity, capacity, and external source remain fixed in this step.
  // On failure, restore theta/psi so the caller can safely retry with smaller dt.
  bool Step(Scalar dt, Scalar &defect) {
    const int n = width.size();
    const Vector old(theta), old_psi(psi);
    Vector alpha(n), mass(n), rhs(n);
    for (int j = 0; j < n; ++j) alpha[j] = dt/capacity[j];
    bool converged = false;
    Scalar final_norm = 0;
    for (int iteration = 0; iteration < 80; ++iteration) {
      theta = MaterialTemperature(old, alpha, psi);
      for (int j = 0; j < n; ++j) {
        const Scalar local = 1+16*alpha[j]*Cube(theta[j]);
        mass[j] = width[j]/local;
        rhs[j] = width[j]*(psi[j]/local+(old[j]-theta[j])/alpha[j]+source[j]);
      }
      const Vector new_psi = Solve(mass, rhs);
      Scalar damping = 1;
      final_norm = 0;
      for (int j = 0; j < n; ++j) {
        if (!std::isfinite(new_psi[j])) {
          final_norm = std::numeric_limits<Scalar>::infinity();
          break;
        }
        if (new_psi[j] <= 0) {
          // Shorten the Newton update to keep radiation positive; do not floor it.
          damping = std::min(damping, Scalar(0.8)*psi[j]/(psi[j]-new_psi[j]));
        }
        final_norm = std::max(final_norm, std::abs(new_psi[j]-psi[j])/(1+psi[j]));
      }
      if (!std::isfinite(final_norm)) break;
      for (int j = 0; j < n; ++j) psi[j] += damping*(new_psi[j]-psi[j]);
      if (final_norm < 2e-13L) {
        theta = MaterialTemperature(old, alpha, psi);
        converged = true;
        break;
      }
    }
    if (!converged) {
      theta = old;
      psi = old_psi;
      return false;
    }
    // Audit material heat change = supplied heat - escaping surface radiation.
    // This is a planar column balance; radial sources are external inputs and
    // radiation energy storage is absent under the quasi-static approximation.
    Scalar delta = 0, old_energy = 0;
    for (int j = 0; j < n; ++j) {
      delta += capacity[j]*width[j]*(theta[j]-old[j]);
      old_energy += capacity[j]*width[j]*old[j];
    }
    const Scalar supplied = dt*(net_heating-surface*psi[0]);
    defect = std::abs(delta-supplied)/std::max(old_energy, dt*absorbed);
    injected += dt*absorbed;
    extracted += dt*(absorbed-net_heating);
    escaped += dt*surface*psi[0];
    return true;
  }
};

}  // namespace SnowlineTwoStream
#endif  // PGEN_SNOWLINE_TWOSTREAM_HPP_
