#!/usr/bin/env python3
"""Standalone irradiation-only benchmark for a two-stream thermal energy update.

This is a numerical prototype, not an Athena++ run. Density, grey IR opacity,
and volumetric heat capacity are constant. There is no hydrodynamics, radial
transport, phase change, or imposed equilibrium-temperature relaxation. It tests
the radiation/heating substep needed before adding an operator-split phase step.

Optical depth tau increases from the surface (0) to the midplane (tau_mid).
The infrared flux F is positive outwards. In dimensionless variables,

    d theta / d s = Q + psi - 4 theta**4,
    d F / d tau = psi - 4 theta**4,
    d psi / d tau = 3 F,
    psi(0) = sqrt(3) F(0),       F(tau_mid) = 0.

Here theta=T/T_ref, psi=4*pi*J/(sigma_SB*T_ref**4), and
s=t/t_ref, with t_ref=C_V/(rho*kappa_IR*sigma_SB*T_ref**3).
The incident stellar heating mirrors Eq. (22) of Wang et al. (2025):

    Q = A f [exp(-a tau) + exp(-a (2 tau_mid - tau))],  a=f/mu,

where f=kappa_visible/kappa_IR and A=E0/(sigma_SB*T_ref**4).
Both surfaces are irradiated symmetrically; only half the slab is simulated.
The independent continuous equilibrium solution is

    F(tau) = A mu [exp(-a tau) - exp(-a (2 tau_mid - tau))],
    psi(tau) = sqrt(3) F(0)
               + 3 A mu/a [1-exp(-a tau)-exp(-a(2 tau_mid-tau))
                            +exp(-2 a tau_mid)],
    theta_eq**4 = (psi + Q)/4.

References:
https://arxiv.org/html/2502.08936v1#S2.SS5
https://arxiv.org/html/1901.06921#A1

Backward Euler couples material temperature to a quasi-static radiation field.
Each Newton iteration eliminates the local temperature increments and solves a
tridiagonal radiation system. Irradiation is included in this same thermal step;
splitting heating from cooling would introduce a separate timestep bias.

Run with Python, NumPy, SciPy, and Matplotlib, for example:
    python twostream_column.py --output-dir /tmp/twostream-column

The default suite checks hot/cold starts, spatial convergence, timestep
independence of the final state, optically thin/thick columns, and conservation.
Results are written as JSON, CSV, and a PNG/PDF figure. Failed checks exit nonzero.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_banded


class Column:
    """Finite-volume column in optical depth, with a fixed irradiation source."""

    def __init__(self, cells, tau_mid, mu=0.1, visible_ratio=1.0):
        if cells < 4 or tau_mid <= 0.0 or not 0.0 < mu <= 1.0:
            raise ValueError("Need cells >= 4, tau_mid > 0, and 0 < mu <= 1")
        if visible_ratio <= 0.0:
            raise ValueError("The visible/IR opacity ratio must be positive")
        self.cells = cells
        self.tau_mid = tau_mid
        self.mu = mu
        self.visible_ratio = visible_ratio
        self.attenuation = visible_ratio / mu
        # This normalization gives theta_eq -> 1 deep inside an opaque slab.
        self.amplitude = 4.0 / (np.sqrt(3.0) * mu + 3.0 * mu**2 / visible_ratio)
        # Resolve the irradiated skin without oversampling the isothermal interior.
        skin = 1.0 / self.attenuation
        x = np.linspace(0.0, 1.0, cells + 1)
        self.faces = skin * np.expm1(x * np.log1p(tau_mid / skin))
        self.width = np.diff(self.faces)
        self.tau = 0.5 * (self.faces[:-1] + self.faces[1:])
        a = self.attenuation
        average = -np.expm1(-a * self.width) / (a * self.width)
        self.heating = self.amplitude * visible_ratio * average * (
            np.exp(-a * self.faces[:-1])
            + np.exp(-a * (2.0 * tau_mid - self.faces[1:])))
        self.total_heating = float(self.heating @ self.width)
        self.conductance = 1.0 / (3.0 * np.diff(self.tau))
        self.surface_conductance = 1.0 / (np.sqrt(3.0) + 3.0 * self.tau[0])

        # L psi = (F_left - F_right)/cell_width; radiation solves (I+L)psi=4T^4.
        self.lower = np.zeros(cells)
        self.upper = np.zeros(cells)
        self.lower[1:] = -self.conductance / self.width[1:]
        self.upper[:-1] = -self.conductance / self.width[:-1]
        self.diffusion_diagonal = -self.lower - self.upper
        self.diffusion_diagonal[0] += self.surface_conductance / self.width[0]

    def solve(self, diagonal, rhs):
        bands = np.zeros((3, self.cells))
        bands[0, 1:] = self.upper[:-1]
        bands[1] = diagonal
        bands[2, :-1] = self.lower[1:]
        return solve_banded((1, 1), bands, rhs, check_finite=False)

    def diffusion(self, psi):
        flux = self.flux(psi)
        return (flux[:-1] - flux[1:]) / self.width

    def radiation(self, theta):
        return self.solve(1.0 + self.diffusion_diagonal, 4.0 * theta**4)

    def flux(self, psi):
        flux = np.zeros(self.cells + 1)
        flux[0] = self.surface_conductance * psi[0]
        flux[1:-1] = self.conductance * np.diff(psi)
        return flux

    def analytic(self, tau):
        """Continuous closed form; not used by the time integrator."""
        a = self.attenuation
        first = np.exp(-a * tau)
        second = np.exp(-a * (2.0 * self.tau_mid - tau))
        flux = self.amplitude * self.mu * (first - second)
        surface_flux = self.amplitude * self.mu * (-np.expm1(
            -2.0 * a * self.tau_mid))
        # Product form avoids cancellation for optically thin slabs near tau=0.
        integrated_flux = self.amplitude * self.mu / a * (
            -np.expm1(-a * tau)) * (-np.expm1(-a * (2.0 * self.tau_mid - tau)))
        psi = np.sqrt(3.0) * surface_flux + 3.0 * integrated_flux
        heating = self.amplitude * self.visible_ratio * (first + second)
        theta = ((psi + heating) / 4.0)**0.25
        return theta, psi, flux

    def step(self, old_theta, dt, tolerance=2.0e-13):
        """Backward Euler with positivity-preserving damped Newton iteration."""
        theta = old_theta.copy()
        psi = self.radiation(theta)
        scale_t = max(1.0, float(np.max(old_theta)) + dt * (
            float(np.max(self.heating)) + 4.0 * float(np.max(old_theta))**4))
        scale_r = max(1.0, 4.0 * float(np.max(old_theta))**4,
                      float(np.max(self.heating)))

        def residual(trial_t, trial_psi):
            emission = 4.0 * trial_t**4
            rt = trial_t - old_theta - dt * (self.heating + trial_psi - emission)
            rr = self.diffusion(trial_psi) + trial_psi - emission
            # Measure the linear equation's backward error: fine optical-depth
            # cells have large diffusion coefficients and cancellation in A psi.
            radiation_scale = scale_r + 2.0 * self.diffusion_diagonal * abs(trial_psi)
            norm = max(float(np.max(np.abs(rt))) / scale_t,
                       float(np.max(np.abs(rr) / radiation_scale)))
            return rt, rr, norm

        for iteration in range(1, 61):
            rt, rr, norm = residual(theta, psi)
            if norm < tolerance:
                return theta, psi, iteration
            derivative = 16.0 * theta**3
            local_diagonal = 1.0 + dt * derivative
            # Schur complement: L + 1/(1+dt*16*T^3), still tridiagonal.
            delta_psi = self.solve(
                self.diffusion_diagonal + 1.0 / local_diagonal,
                -rr - derivative * rt / local_diagonal)
            delta_t = (-rt + dt * delta_psi) / local_diagonal
            damping = 1.0
            for _ in range(40):
                trial_t = theta + damping * delta_t
                trial_psi = psi + damping * delta_psi
                if np.all(trial_t > 0.0) and np.all(trial_psi > 0.0):
                    if residual(trial_t, trial_psi)[2] < norm:
                        theta, psi = trial_t, trial_psi
                        break
                damping *= 0.5
            else:
                raise RuntimeError("Newton line search failed")
        raise RuntimeError("Thermal/radiation step failed to converge")


def run_case(cells, tau_mid, start, dt, mu, visible_ratio, max_steps=3000):
    column = Column(cells, tau_mid, mu, visible_ratio)
    reference, _, _ = column.analytic(column.tau)
    initial = 0.5 if start == "cold" else 2.0
    theta = np.full(cells, initial)
    psi = column.radiation(theta)
    history = []
    cumulative_defect = 0.0
    max_step_defect = 0.0
    max_newton = 0
    time = 0.0
    heating_scale = float(np.max(column.heating))

    def record(step):
        rate = column.heating + psi - 4.0 * theta**4
        # Steady stopping criteria use the differential equations, not T_eq.
        rate_error = float(np.max(np.abs(rate))) / heating_scale
        flux_error = abs(column.flux(psi)[0] / column.total_heating - 1.0)
        history.append({
            "step": step, "time": time,
            "max_relative_temperature_error": float(np.max(np.abs(
                theta / reference - 1.0))),
            "thermal_residual": rate_error, "surface_flux_error": flux_error,
            "cumulative_energy_defect": cumulative_defect,
        })
        return max(rate_error, flux_error)

    record(0)
    for step in range(1, max_steps + 1):
        old_theta = theta
        theta, psi, iterations = column.step(old_theta, dt)
        max_newton = max(max_newton, iterations)
        time = step * dt
        delta_u = float((theta - old_theta) @ column.width)
        supplied = dt * (column.total_heating - column.flux(psi)[0])
        defect = delta_u - supplied
        cumulative_defect += defect
        max_step_defect = max(max_step_defect, abs(defect) / max(
            float(old_theta @ column.width), dt * column.total_heating, 1.0))
        error = record(step)
        if error < 2.0e-10:
            break
    else:
        raise RuntimeError("No thermal equilibrium after {} steps".format(max_steps))

    # Check the radiation solution itself, as well as the material equation.
    radiation_residual = float(np.max(np.abs(
        column.diffusion(psi) + psi - 4.0 * theta**4))) / max(
            1.0, float(np.max(4.0 * theta**4)))
    summary = {
        "cells": cells, "tau_mid": tau_mid, "start": start, "dt": dt,
        "steps": step, "time": time, "max_newton_iterations": max_newton,
        "max_relative_temperature_error": history[-1][
            "max_relative_temperature_error"],
        "rms_relative_temperature_error": float(np.sqrt(np.mean(
            (theta / reference - 1.0)**2))),
        "surface_flux_error": history[-1]["surface_flux_error"],
        "thermal_residual": history[-1]["thermal_residual"],
        "radiation_residual": radiation_residual,
        "max_step_energy_defect": max_step_defect,
        "relative_cumulative_energy_defect": abs(cumulative_defect) / max(
            1.0, time * column.total_heating),
        "min_theta": float(np.min(theta)), "max_theta": float(np.max(theta)),
    }
    print("tau={:5g} N={:4d} {:4s} dt={:g}: {:4d} steps, "
          "max dT/T={:.3e}, energy defect={:.2e}".format(
              tau_mid, cells, start, dt, step,
              summary["max_relative_temperature_error"], max_step_defect), flush=True)
    return {"column": column, "theta": theta, "psi": psi,
            "summary": summary, "history": history}


def write_results(output_dir, cases, resolutions, mu, visible_ratio, temperature_ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    reference_cases = [c for c in cases if c["summary"]["tau_mid"] == 10.0
                       and c["summary"]["dt"] == 2.0]
    finest = max(resolutions)
    cold = next(c for c in reference_cases if c["summary"]["cells"] == finest
                and c["summary"]["start"] == "cold")
    hot = next(c for c in reference_cases if c["summary"]["cells"] == finest
               and c["summary"]["start"] == "hot")
    timestep_cases = [c for c in cases if c["summary"]["cells"] == finest
                     and c["summary"]["tau_mid"] == 10.0
                     and c["summary"]["start"] == "cold"]
    errors = [next(c["summary"]["max_relative_temperature_error"]
                   for c in reference_cases if c["summary"]["cells"] == n
                   and c["summary"]["start"] == "cold") for n in resolutions]
    orders = [float(np.log(errors[i] / errors[i+1]) / np.log(
        resolutions[i+1] / resolutions[i])) for i in range(len(resolutions)-1)]
    start_difference = float(np.max(np.abs(cold["theta"] / hot["theta"] - 1.0)))
    timestep_difference = max(float(np.max(np.abs(
        c["theta"] / cold["theta"] - 1.0))) for c in timestep_cases)
    checks = {
        "finest_profile_error_below_0.1_percent": errors[-1] < 1.0e-3,
        "spatial_order_above_1.8": all(p > 1.8 for p in orders),
        "hot_cold_agreement_below_1e-7": start_difference < 1.0e-7,
        "timestep_agreement_below_1e-7": timestep_difference < 1.0e-7,
        "steady_equations_and_flux_below_1e-9": all(max(
            c["summary"]["thermal_residual"], c["summary"]["radiation_residual"],
            c["summary"]["surface_flux_error"]) < 1.0e-9 for c in cases),
        "energy_defect_below_1e-9": all(
            c["summary"]["max_step_energy_defect"] < 1.0e-9 for c in cases),
    }
    results = {
        "model": "static grey slab; irradiation only; no phase change or hydrodynamics",
        "mu": mu, "kappa_visible_over_kappa_ir": visible_ratio,
        "temperature_ref_K": temperature_ref,
        "time_unit": "C_V / (rho*kappa_IR*sigma_SB*T_ref**3)",
        "spatial_orders": orders, "hot_cold_max_relative_difference": start_difference,
        "timestep_max_relative_difference": timestep_difference,
        "checks": checks, "passed": all(checks.values()),
        "cases": [c["summary"] for c in cases],
    }
    (output_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    for label, case in (("cold", cold), ("hot", hot)):
        with (output_dir / (label + "_history.csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(case["history"][0]))
            writer.writeheader()
            writer.writerows(case["history"])
    column = cold["column"]
    analytic_t, analytic_psi, analytic_f = column.analytic(column.tau)
    numerical_flux = 0.5 * (column.flux(cold["psi"])[:-1]
                            + column.flux(cold["psi"])[1:])
    with (output_dir / "equilibrium_profile.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["tau_IR", "T_analytic_K", "T_energy_cold_K", "T_energy_hot_K",
                         "psi_analytic", "psi_energy", "F_analytic", "F_energy"])
        writer.writerows(zip(column.tau, temperature_ref * analytic_t,
                             temperature_ref * cold["theta"],
                             temperature_ref * hot["theta"],
                             analytic_psi, cold["psi"], analytic_f, numerical_flux))

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(column.tau, temperature_ref * analytic_t, color="black", lw=2,
            label="Analytic two-stream equilibrium")
    sample = slice(None, None, max(1, finest // 28))
    ax.plot(column.tau[sample], temperature_ref * cold["theta"][sample], "o",
            ms=4, color="#167c9e", label="Energy update: cold start")
    ax.plot(column.tau[sample], temperature_ref * hot["theta"][sample], "+",
            ms=6, color="#c75b25", label="Energy update: hot start")
    ax.set(xscale="log", xlabel="Infrared optical depth from surface",
           ylabel="Temperature [K]", title="Both starts reach the analytic profile")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    for case, label, color in ((cold, "Cold start", "#167c9e"),
                               (hot, "Hot start", "#c75b25")):
        history = case["history"]
        ax.semilogy([h["time"] for h in history],
                    [h["max_relative_temperature_error"] for h in history],
                    label=label, color=color)
    ax.axhline(errors[-1], ls=":", color="gray", label="Spatial error floor")
    ax.set(xlabel="Time / thermal reference time", ylabel="Maximum |T / T analytic - 1|",
           title="Physical energy evolution; no prescribed T target")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    ax.loglog(resolutions, errors, "o-", color="#167c9e", label="Measured profile error")
    ax.loglog(resolutions, errors[0] * (np.array(resolutions) / resolutions[0])**-2.0,
              "--", color="gray", label="Second-order reference")
    ax.set(xlabel="Cells per half-column", ylabel="Maximum relative temperature error",
           title="Resolution convergence (last order: {:.2f})".format(orders[-1]))
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    history = cold["history"]
    ax.semilogy([h["time"] for h in history],
                [max(h["surface_flux_error"], 1.0e-16) for h in history],
                color="#167c9e", label="|Escaping IR / absorbed starlight - 1|")
    ax.semilogy([h["time"] for h in history],
                [max(h["thermal_residual"], 1.0e-16) for h in history],
                color="#c75b25", label="Material energy-equation residual")
    ax.set(xlabel="Time / thermal reference time", ylabel="Relative residual",
           title="Steady energy balance")
    ax.legend(fontsize=9)
    for ax in axes.flat:
        ax.grid(True, alpha=0.2)
    fig.suptitle("1D irradiated slab: energy evolution versus analytic two-stream RT\n"
                 r"$\tau_{\rm mid}=10$, $\mu_0=%.2g$, "
                 r"$\kappa_{\rm vis}/\kappa_{\rm IR}=%.2g$"
                 % (mu, visible_ratio), fontsize=14)
    fig.savefig(output_dir / "comparison.png", dpi=180)
    fig.savefig(output_dir / "comparison.pdf")
    plt.close(fig)
    print(json.dumps({"checks": checks, "spatial_orders": orders,
                      "hot_cold_difference": start_difference,
                      "timestep_difference": timestep_difference}, indent=2))
    return results["passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--mu", type=float, default=0.1)
    parser.add_argument("--visible-ratio", type=float, default=1.0)
    parser.add_argument("--temperature-ref", type=float, default=150.0)
    args = parser.parse_args()
    resolutions = sorted(set(args.resolutions))
    if len(resolutions) < 2 or resolutions[0] < 4 or args.temperature_ref <= 0.0:
        parser.error("Need two resolutions >= 4 and a positive temperature scale")
    cases = []
    for cells in resolutions:
        for start in ("cold", "hot"):
            cases.append(run_case(cells, 10.0, start, 2.0, args.mu, args.visible_ratio))
    for dt in (0.5, 8.0):
        cases.append(run_case(resolutions[-1], 10.0, "cold", dt,
                              args.mu, args.visible_ratio))
    for tau_mid in (0.1, 1.0):
        for start in ("cold", "hot"):
            cases.append(run_case(resolutions[-1], tau_mid, start, 2.0,
                                  args.mu, args.visible_ratio))
    passed = write_results(args.output_dir, cases, resolutions, args.mu,
                           args.visible_ratio, args.temperature_ref)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
