#!/usr/bin/env python3
"""Run the native Athena++ irradiation-only snowline temperature comparison.

Use the existing spherical/general-eos_phase_change snowline build, with HDF5
double precision, 7 dust fluids, and N_P=N_Z=2. Build in separate directories to
preserve the current binary and objects, e.g. from the repository root:

  make -j4 OBJ_DIR=/tmp/snowline-thermal/obj/ EXE_DIR=/tmp/snowline-thermal/bin/ \
      CXXFLAGS='-O3 -std=c++11 -fopenmp'
  python tst/regression/snowline_thermal_disk.py \
      --athena /tmp/snowline-thermal/bin/athena --output-dir /tmp/disk-comparison

This tests an actual 2D Athena++ disk mesh, with independent plane-parallel theta
columns matching the original pgen approximation. Density is fixed. Hydro,
orbital advection, radial diffusion, and phase change are off. The energy
temperature is an implicit evolution of gas internal energy, never a relaxation
to the analytic target. The suite includes the original pgen's analytic path,
hot/cold starts, theta resolution, timestep, and MeshBlock decomposition checks.
"""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vis/python"))
import athena_read  # noqa: E402


def read(directory, output, final=True):
    files = sorted(directory.glob("thermal_disk.out{}.*.athdf".format(output)))
    if len(files) < 2:
        raise RuntimeError("Expected initial and final outputs in " + str(directory))
    def theta_faces(xmin, xmax, ratio, count):
        return xmin+(xmax-xmin)*np.sqrt(np.linspace(0, 1, count))
    return athena_read.athdf(str(files[-1] if final else files[0]), dtype=np.float64,
                            face_func_2=theta_faces)


def run(athena, input_file, output_dir, name, nt, factor=0.5, dt=500,
        block_theta=24, legacy=False):
    directory = output_dir / name
    directory.mkdir(parents=True, exist_ok=True)
    steps = 128 if dt == 500 else 256
    arguments = [str(athena), "-i", str(input_file),
                 "mesh/nx2={}".format(nt), "meshblock/nx2={}".format(block_theta),
                 "problem/thermal_test_initial_factor={}".format(factor),
                 "time/nlim={}".format(steps), "time/tlim={}".format(steps*dt)]
    if legacy:
        # The existing pgen's beta relaxation reaches its valid target in one
        # sufficiently large fixed-fluid step; density is unchanged by hydro.
        arguments += ["problem/thermal_test=false", "time/nlim=1", "time/tlim=64000"]
    start = time.perf_counter()
    with (directory / "run.log").open("w") as log:
        completed = subprocess.run(arguments, cwd=directory, stdout=log,
                                   stderr=subprocess.STDOUT, timeout=180)
    elapsed = time.perf_counter() - start
    log_text = (directory / "run.log").read_text()
    if completed.returncode or "FATAL ERROR" in log_text:
        raise RuntimeError("Athena++ failed in {}:\n{}".format(name, log_text[-3000:]))
    data = read(directory, 2)
    prim, initial = read(directory, 1), read(directory, 1, final=False)
    if legacy:
        print("{}: original analytic pgen completed ({:.2f} s)".format(name, elapsed),
              flush=True)
        return {"data": data, "prim": prim, "directory": directory}
    matches = re.findall(r"\[thermal_test\].*energy_defect=[^\n]+", log_text)
    if not matches:
        raise RuntimeError("Missing thermal comparison diagnostics in " + name)
    diagnostics = {k: float(v) for k, v in re.findall(
        r"(\w+)=([+\-\d.eE]+)", matches[-1])}
    theta = data["T_energy"][0]
    reference = data["T_2sRT"][0]
    conservative = read(directory, 3)
    # Read actual gas internal energy, not just the diagnostic temperature array.
    fields = list(conservative)
    density_key = "dens" if "dens" in conservative else "rho"
    energy_key = "Etot" if "Etot" in conservative else "ener"
    if density_key not in conservative or energy_key not in conservative:
        raise RuntimeError("Unrecognized conserved variables: " + str(fields))
    fv = 1e-20 / conservative[density_key][0]
    cv_specific = ((1-fv)*(0.71/2*2.5 + 0.29/4*1.5) + fv/18*3) / 64.102564103
    from_energy = conservative[energy_key][0] / (conservative[density_key][0]*cv_specific)
    drift = float(np.max(np.abs(prim["rho"] / initial["rho"] - 1)))
    metrics = {
        "name": name, "nx1": theta.shape[1], "nx2": nt, "initial_factor": factor,
        "dt_code": dt, "block_theta": block_theta, "wall_seconds": elapsed,
        "max_relative_analytic_error": float(np.max(np.abs(theta/reference-1))),
        "rms_relative_analytic_error": float(np.sqrt(np.mean((theta/reference-1)**2))),
        "max_relative_discrete_error": float(np.max(np.abs(theta/data["T_FV_eq"][0]-1))),
        "density_drift": drift,
        "initial_density_diagnostic_error": float(np.max(np.abs(
            prim["rho"] / data["rho_initial"] - 1))),
        "max_velocity": max(float(np.max(np.abs(prim["vel"+str(v)])))
                            for v in (1, 2, 3)),
        "temperature_from_energy_error": float(np.max(np.abs(from_energy/theta-1))),
        "final_diagnostics": diagnostics,
    }
    print("{}: max dT/T={:.3e}, FV equilibrium error={:.2e}, "
          "density drift={:.1e} ({:.2f} s)".format(
              name, metrics["max_relative_analytic_error"],
              metrics["max_relative_discrete_error"], drift, elapsed), flush=True)
    return {"data": data, "prim": prim, "metrics": metrics, "directory": directory}


def plot(output_dir, cold, hot, cases, orders):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    data = cold["data"]
    length_au = 1.8983553591e12 / 1.495978707e13
    r, th = np.meshgrid(data["x1f"]*length_au, data["x2f"])
    radial, height = r*np.sin(th), r*np.cos(th)
    analytic, energy = data["T_2sRT"][0], data["T_energy"][0]
    error = 100*(energy/analytic-1)
    norm = Normalize(vmin=min(analytic.min(), energy.min()),
                     vmax=max(analytic.max(), energy.max()))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, temperature, title in ((axes[0, 0], analytic, "Two-stream equilibrium"),
                                   (axes[0, 1], energy, "Athena++ energy evolution")):
        image = ax.pcolormesh(radial, height, temperature, norm=norm, cmap="inferno",
                              shading="flat", rasterized=True)
        ax.set(title=title, xlabel="Cylindrical radius [au]", ylabel="Height [au]")
    fig.colorbar(image, ax=list(axes[0]), label="Temperature [K]", shrink=0.9)
    limit = max(float(np.max(np.abs(error))), 1e-6)
    image = axes[1, 0].pcolormesh(radial, height, error, cmap="RdBu_r", vmin=-limit,
                                 vmax=limit, shading="flat", rasterized=True)
    axes[1, 0].set(title="Energy / analytic - 1", xlabel="Cylindrical radius [au]",
                   ylabel="Height [au]")
    fig.colorbar(image, ax=axes[1, 0], label="Temperature difference [%]")
    ax = axes[1, 1]
    ns = np.array([c["metrics"]["nx2"] for c in cases])
    errors = np.array([c["metrics"]["max_relative_analytic_error"] for c in cases])
    ax.loglog(ns, errors, "o-", label="Maximum relative difference", color="#167c9e")
    ax.loglog(ns, errors[0]*(ns/ns[0])**-2, "--", color="gray", label="Second order")
    ax.set(xlabel="Theta cells", ylabel="Maximum |T energy / T analytic - 1|",
           title="Resolution convergence (last order {:.2f})".format(orders[-1]))
    ax.grid(alpha=0.2)
    ax.legend()
    fig.suptitle("Snowline disk in Athena++: fixed density, irradiation only\n"
                 "Hydrodynamics, radial diffusion, and phase change disabled",
                 fontsize=14)
    fig.savefig(output_dir / "temperature_comparison.png", dpi=180)
    fig.savefig(output_dir / "temperature_comparison.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    radius = data["x1v"]*length_au
    axes[0].plot(radius, analytic[-1], "k-", label="Analytic")
    axes[0].plot(radius, energy[-1], "--", label="Energy: cold start")
    axes[0].plot(radius, hot["data"]["T_energy"][0, -1], ":", label="Energy: hot")
    axes[0].set(xlabel="Midplane radius [au]", ylabel="Temperature [K]",
                 title="Midplane temperature")
    axes[0].legend()
    for i in (0, len(radius)//2, -1):
        label = "{:.2f} au".format(radius[i])
        axes[1].plot(data["x2v"], analytic[:, i], "-", label=label)
        color = axes[1].lines[-1].get_color()
        axes[1].plot(data["x2v"], energy[:, i], "--", color=color)
    axes[1].set(xlabel="Polar angle [rad]", ylabel="Temperature [K]",
                 title="Theta profiles: analytic solid, energy dashed")
    axes[1].legend()
    fig.savefig(output_dir / "temperature_profiles.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--athena", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[48, 96, 192])
    args = parser.parse_args()
    athena, output_dir = args.athena.resolve(), args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_file = ROOT / "inputs/snowline/athinput.thermal_compare"
    resolutions = sorted(set(args.resolutions))
    if len(resolutions) < 2 or any(n % 24 for n in resolutions):
        parser.error("Use at least two resolutions, each a multiple of 24")
    cases = [run(athena, input_file, output_dir, "cold_n{}".format(nt), nt)
             for nt in resolutions]
    cold, finest = cases[-1], resolutions[-1]
    hot = run(athena, input_file, output_dir, "hot_n{}".format(finest), finest, factor=2)
    smaller_dt = run(athena, input_file, output_dir, "half_dt", finest, dt=250)
    decomposition = run(athena, input_file, output_dir, "full_theta_blocks", finest,
                        block_theta=finest)
    legacy = run(athena, input_file, output_dir, "original_analytic", finest, legacy=True)
    errors = [c["metrics"]["max_relative_analytic_error"] for c in cases]
    orders = [float(np.log(errors[i]/errors[i+1]) / np.log(
        resolutions[i+1]/resolutions[i])) for i in range(len(resolutions)-1)]
    reference = cold["data"]["T_energy"]
    differences = {name: float(np.max(np.abs(case["data"]["T_energy"]/reference-1)))
                   for name, case in (("hot_start", hot), ("half_dt", smaller_dt),
                                      ("decomposition", decomposition))}
    differences["original_pgen_reference"] = float(np.max(np.abs(
        legacy["data"]["Tem"] / cold["data"]["T_2sRT"] - 1)))
    all_cases = cases + [hot, smaller_dt, decomposition]
    checks = {
        "analytic_max_error_below_0.5_percent": errors[-1] < 0.005,
        "spatial_orders_above_1.7": all(p > 1.7 for p in orders),
        "same_discrete_equilibrium_below_1e-7": all(
            c["metrics"]["max_relative_discrete_error"] < 1e-7 for c in all_cases),
        "density_and_velocity_frozen": all(c["metrics"]["density_drift"] == 0
            and c["metrics"]["max_velocity"] == 0
            and c["metrics"]["initial_density_diagnostic_error"] == 0 for c in all_cases),
        "gas_energy_matches_temperature": all(
            c["metrics"]["temperature_from_energy_error"] < 1e-10 for c in all_cases),
        "hot_start_timestep_decomposition_and_legacy_agree": all(
            d < 1e-7 for d in differences.values()),
        "column_energy_conserved": all(
            c["metrics"]["final_diagnostics"]["energy_defect"] < 1e-9 for c in all_cases),
    }
    summary = {"passed": all(checks.values()), "checks": checks, "spatial_orders": orders,
               "differences": differences, "cases": [c["metrics"] for c in all_cases]}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(output_dir, cold, hot, cases, orders)
    print(json.dumps({"checks": checks, "orders": orders, "differences": differences},
                     indent=2), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
