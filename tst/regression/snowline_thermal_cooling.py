#!/usr/bin/env python3
"""Native Athena++ prescribed-cooling tests on the fixed 2D snowline disk.

Run after building the same binary as snowline_thermal_disk.py:
  python tst/regression/snowline_thermal_cooling.py --athena /path/to/athena \
      --output-dir /tmp/snowline-cooling

Each run starts in the irradiation-only finite-volume equilibrium. A smooth
sink centered at 2 au and tau/tau_mid=0.9 removes A*F_irr per column times a
radial Gaussian. Weak cooling remains on until steady state; stronger sinks
last 0.05 code time units, followed by recovery. There is no phase conversion,
hydro, radial transport, opacity change, or relaxation toward an analytic T.

eq_status is a bit mask: 1=negative outgoing equilibrium flux, 2=negative
equilibrium radiation field, 4=negative equilibrium T^4. An invalid equilibrium
temperature is output as -1 K, never used by the time evolution. Energy history
records the numerical backward-Euler boundary flux integral, enabling an audit
of the discrete equation independently of the solver's per-step residual.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
from snowline_thermal_disk import ROOT, read


def run(athena, output_dir, name, amplitude, stop, end=200, steps=400,
        block_theta=24):
    directory = output_dir / name
    directory.mkdir(parents=True, exist_ok=True)
    args = [str(athena), "-i", str(ROOT/"inputs/snowline/athinput.thermal_compare"),
            "mesh/nx1=40", "meshblock/nx1=10", "mesh/nx2=96",
            "meshblock/nx2={}".format(block_theta),
            "time/tlim={}".format(end), "time/nlim={}".format(steps),
            "time/ncycle_out=100", "problem/thermal_test_start_equilibrium=true",
            "problem/thermal_test_cooling_ratio={}".format(amplitude),
            "problem/thermal_test_cooling_stop={}".format(stop)]
    # Athena's fixed-fluid clock may end a few ulps below tlim on the cycle
    # limit. Schedule the final dump just before that boundary.
    args += ["output{}/dt={}".format(i, end*(1-1e-8)) for i in (1, 2, 3)]
    start = time.perf_counter()
    with (directory/"run.log").open("w") as log:
        completed = subprocess.run(args, cwd=directory, stdout=log,
                                   stderr=subprocess.STDOUT, timeout=180)
    log = (directory/"run.log").read_text()
    if completed.returncode or "FATAL ERROR" in log:
        raise RuntimeError(name+":\n"+log[-3000:])
    data, initial = read(directory, 2), read(directory, 2, final=False)
    prim, prim0, cons = read(directory, 1), read(directory, 1, False), read(directory, 3)
    history = np.genfromtxt(directory/"thermal_cooling.csv", delimiter=",",
                            names=True, dtype=np.longdouble)
    residual = (history["energy"]-history["initial_energy"]
                -history["irradiation_integral"]+history["cooling_integral"]
                +history["escape_integral"])/history["initial_energy"]
    status = data["pulse_eq_status"][0, 0].astype(int)
    pulse_ratio = data["T_pulse_end"]/initial["T_energy"]
    rho = cons["dens"]
    fv = 1e-20/rho
    cv = ((1-fv)*(0.71/2*2.5+0.29/4*1.5)+fv/18*3)/64.102564103
    from_energy = cons["Etot"]/(rho*cv)
    metrics = dict(name=name, amplitude=amplitude, pulse_stop=stop,
                   final_time=end, dt=end/steps,
                   wall_seconds=time.perf_counter()-start,
                   invalid_equilibrium_columns=int(np.count_nonzero(status)),
                   negative_flux_columns=int(np.count_nonzero(status & 1)),
                   local_invalid_positive_flux_columns=int(np.count_nonzero(
                       (status != 0) & ((status & 1) == 0))),
                   minimum_T=float(np.min(history["min_T"])),
                   max_pulse_cooling_fraction=float(1-np.min(pulse_ratio)),
                   max_cumulative_energy_error=float(np.max(np.abs(residual))),
                   max_step_energy_error=float(np.max(history["energy_defect"])),
                   temperature_from_energy_error=float(np.max(np.abs(
                       from_energy/data["T_energy"]-1))),
                   density_drift=float(np.max(np.abs(prim["rho"]/prim0["rho"]-1))),
                   max_velocity=max(float(np.max(np.abs(prim["vel"+str(i)])))
                                    for i in (1, 2, 3)))
    if end > stop:
        metrics["recovery_error"] = float(np.max(np.abs(
            data["T_energy"]/data["T_irr_eq"]-1)))
    if not np.any(status):
        metrics["steady_analytic_error"] = float(np.max(np.abs(
            data["T_energy"]/data["T_2sRT"]-1)))
        metrics["steady_discrete_error"] = float(np.max(np.abs(
            data["T_energy"]/data["T_FV_eq"]-1)))
    print("{}: invalid={} (negative flux={}), min T={:.2f} K, energy error={:.2e}"
          .format(name, metrics["invalid_equilibrium_columns"],
                  metrics["negative_flux_columns"], metrics["minimum_T"],
                  metrics["max_cumulative_energy_error"]), flush=True)
    return dict(data=data, initial=initial, history=history, metrics=metrics)


def plot(output_dir, weak, local, strong, refinement, differences, order):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = ["#157ca0", "#e39825", "#ae345c"]
    for case, label, color in zip((weak, local, strong),
                                  ("Weak continuous sink", "Local equilibrium invalid",
                                   "Negative column heating"), colors):
        h = case["history"]
        radius = h["radius_au"][h["time"] == 0]
        column = int(np.argmin(np.abs(radius-2)))
        h = h[h["column"] == column]
        t, temp = h["time"].astype(float), h["midplane_T"].astype(float)
        axes[0, 0].plot(t, temp/temp[0], label=label, color=color)
        if case is strong:
            before = t <= 2
            axes[0, 1].plot(t[before], temp[before]/temp[0], "o-", color=color)
            axes[0, 1].axvspan(0, 0.05, alpha=0.18, color=color,
                                 label="Prescribed cooling on")
            energy = h["energy"]-h["initial_energy"]
            supplied = (h["irradiation_integral"]-h["cooling_integral"]
                        -h["escape_integral"])
            axes[1, 0].plot(t, (energy/h["initial_energy"]).astype(float),
                              label="Change in stored heat", color=color)
            axes[1, 0].plot(t, (supplied/h["initial_energy"]).astype(float),
                              "--", label="Integrated heating minus losses", color="black")
    axes[0, 0].set(xlabel="Time [code units]", ylabel="Midplane T / initial T",
                    title="Evolution near 2 au")
    axes[0, 0].set_xscale("symlog", linthresh=0.1)
    axes[0, 0].legend(fontsize=9)
    axes[0, 1].set(xlabel="Time [code units]", ylabel="Midplane T / initial T",
                    title="Cooling during a pulse with no equilibrium")
    axes[0, 1].legend()
    axes[1, 0].set(xlabel="Time [code units]", ylabel="Energy change / initial heat",
                    title="Column energy budget (same 2 au column)")
    axes[1, 0].set_xscale("symlog", linthresh=0.1)
    axes[1, 0].legend()
    dt = [c["metrics"]["dt"] for c in refinement[:-1]]
    axes[1, 1].loglog(dt, differences, "o-", label="Successive timestep difference")
    axes[1, 1].loglog(dt, differences[0]*np.array(dt)/dt[0], "--",
                        color="gray", label="First order")
    axes[1, 1].set(xlabel="Timestep [code units]", ylabel="Max |delta T| / initial T",
                    title="Pulse convergence (last order {:.2f})".format(order))
    axes[1, 1].legend()
    fig.suptitle("Athena++ snowline disk: prescribed cooling, fixed density\n"
                 "Implicit material energy + quasi-static theta two-stream radiation")
    fig.savefig(output_dir/"cooling_evolution.png", dpi=180)
    fig.savefig(output_dir/"cooling_evolution.pdf")
    plt.close(fig)

    data = strong["data"]
    r, th = np.meshgrid(data["x1f"]*1.8983553591e12/1.495978707e13, data["x2f"])
    radius, height = r*np.sin(th), r*np.cos(th)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    maps = [(data["pulse_eq_status"][0], "Equilibrium status during pulse", "Bit mask",
             "viridis"),
            (100*(data["T_pulse_end"][0]/data["T_irr_eq"][0]-1),
             "Temperature change at pulse end", "Temperature change [%]", "RdBu_r"),
            (data["T_energy"][0], "Temperature after recovery", "Temperature [K]", "inferno")]
    for ax, (field, title, label, cmap) in zip(axes, maps):
        img = ax.pcolormesh(radius, height, field, shading="flat", cmap=cmap,
                            rasterized=True)
        ax.set(title=title, xlabel="Cylindrical radius [au]", ylabel="Height [au]")
        fig.colorbar(img, ax=ax, label=label)
    fig.savefig(output_dir/"cooling_disk_maps.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--athena", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    athena, output_dir = args.athena.resolve(), args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    weak = run(athena, output_dir, "weak_steady", 0.001, 1000)
    local = run(athena, output_dir, "local_invalid_pulse", 0.1, 0.05)
    strong = run(athena, output_dir, "negative_column_pulse", 1.2, 0.05)
    refinement = [run(athena, output_dir, "pulse_dt_{}".format(n), 1.2, 0.05,
                      end=0.05, steps=n) for n in (1, 2, 4, 8)]
    decomposition = run(athena, output_dir, "pulse_full_column", 1.2, 0.05,
                        end=0.05, steps=8, block_theta=96)
    differences = [float(np.max(np.abs((a["data"]["T_energy"]
        -b["data"]["T_energy"])/a["initial"]["T_energy"])))
        for a, b in zip(refinement[:-1], refinement[1:])]
    orders = [float(np.log2(a/b)) for a, b in zip(differences[:-1], differences[1:])]
    decomposition_error = float(np.max(np.abs(decomposition["data"]["T_energy"]
                                              /refinement[-1]["data"]["T_energy"]-1)))
    cases = [weak, local, strong]+refinement+[decomposition]
    checks = dict(
        weak_equilibrium_valid=weak["metrics"]["invalid_equilibrium_columns"] == 0,
        weak_matches_analytic=weak["metrics"]["steady_analytic_error"] < 1e-4,
        weak_matches_discrete=weak["metrics"]["steady_discrete_error"] < 1e-8,
        local_invalid_with_positive_net_heating=(
            local["metrics"]["local_invalid_positive_flux_columns"] > 0
            and local["metrics"]["negative_flux_columns"] == 0),
        strong_negative_column_heating=strong["metrics"]["negative_flux_columns"] > 0,
        finite_positive_temperatures=all(np.isfinite(c["metrics"]["minimum_T"])
                                         and c["metrics"]["minimum_T"] > 0 for c in cases),
        measurable_pulse_cooling=strong["metrics"]["max_pulse_cooling_fraction"] > 0.02,
        energy_conserved=all(c["metrics"]["max_cumulative_energy_error"] < 1e-10
                             for c in cases),
        gas_energy_updated=all(c["metrics"]["temperature_from_energy_error"] < 1e-10
                               for c in cases),
        hydro_frozen=all(c["metrics"]["density_drift"] == 0
                         and c["metrics"]["max_velocity"] == 0 for c in cases),
        recovery=all(c["metrics"]["recovery_error"] < 1e-8 for c in (local, strong)),
        first_order_timestep_convergence=all(0.7 < p < 1.3 for p in orders),
        decomposition_independent=decomposition_error < 1e-12)
    summary = dict(cases=[c["metrics"] for c in cases],
                   successive_dt_differences=differences, timestep_orders=orders,
                   decomposition_error=decomposition_error, checks=checks)
    (output_dir/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    plot(output_dir, weak, local, strong, refinement, differences, orders[-1])
    print(json.dumps(checks, indent=2), flush=True)
    if not all(checks.values()):
        raise SystemExit("Failed cooling checks; inspect summary.json")


if __name__ == "__main__":
    main()
