#!/usr/bin/env python3
"""Compare legacy and energy evolution from the supplied snowline 01500 restart.

  python tst/regression/snowline_restart_energy.py --athena /path/to/athena \
      --output-dir /tmp/snowline-restart

Build in a fresh OBJ_DIR/EXE_DIR after the PhaseChange header changes; the
repository Makefile does not track header dependencies.

The bundled restart is at cycle 20219108. The energy input advances to cycle
20220200 with the original hydro, OA, growth, phase-change and radial-conduction
settings. Legacy thermal relaxation is expected to fail on its first update.
Use --analyze-only to inspect runs already saved in the output directory.

The phase checks audit each local gas+ice sensible/kinetic/latent energy update
and gas+ice mass update. The radiation check audits each planar column step,
including radial heating as an external source. These are discrete component
checks, not a global conservation or resolution study of the full disk model.
"""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

sys.dont_write_bytecode = True
from snowline_thermal_disk import ROOT, athena_read


def read(path):
    return athena_read.athdf(str(path), dtype=np.float64,
        face_func_2=lambda a, b, ratio, n: a+(b-a)*np.sqrt(np.linspace(0, 1, n)))


def execute(athena, output_dir):
    commands = {}
    for mode in ("legacy", "energy"):
        directory = output_dir/mode
        directory.mkdir(parents=True, exist_ok=True)
        args = [str(athena), "-r", str(ROOT/"tst/data_bug/iceline.01500.rst"),
                "-i", str(ROOT/"inputs/snowline/athinput.snowline_energy_restart")]
        if mode == "legacy":
            args += ["problem/thermal_energy=false"]
        with (directory/"run.log").open("w") as log:
            p = subprocess.run(args, cwd=directory, stdout=log,
                               stderr=subprocess.STDOUT, timeout=900)
        commands[mode] = dict(command=args, returncode=p.returncode)
        if mode == "energy" and p.returncode != 0:
            raise RuntimeError("Energy restart failed; inspect "+str(directory/"run.log"))
    (output_dir/"commands.json").write_text(json.dumps(commands, indent=2)+"\n")


def analyze(output_dir):
    legacy = (output_dir/"legacy/run.log").read_text()
    energy = (output_dir/"energy/run.log").read_text()
    rows = [{k: float(v) for k, v in re.findall(r"(\w+)=([+\-\d.eE]+)", line)}
            for line in energy.splitlines() if line.startswith("[thermal_energy]")]
    if not rows:
        raise RuntimeError("Missing energy diagnostics")
    initial = read(ROOT/"tst/data_bug/iceline.out2.01500.athdf")
    files = sorted((output_dir/"energy").glob("iceline.out2.*.athdf"))
    snapshots = [read(p) for p in files]
    final = snapshots[-1]
    prim = read(sorted((output_dir/"energy").glob("iceline.out1.*.athdf"))[-1])
    cons = read(sorted((output_dir/"energy").glob("iceline.out5.*.athdf"))[-1])
    rho = cons["dens"]
    fv = cons["dust_5_dens"]/rho
    cv = ((1-fv)*(0.71/2*2.5+0.29/4*1.5)+fv/18*3)/64.102564103
    kinetic = sum(cons["mom"+str(i)]**2 for i in (1, 2, 3))/(2*rho)
    temperature = (cons["Etot"]-kinetic)/(rho*cv)
    failure = re.search(r"cycle, time = (\d+) ([\d.eE+-]+)", legacy)
    if not failure:
        raise RuntimeError("Legacy failure did not reproduce")
    checks = dict(
        legacy_invalid_equilibrium_reproduced="Tem = -nan" in legacy or "Tem = nan" in legacy,
        energy_run_terminated_normally=("Terminating on cycle limit" in energy
            and not any(s in energy for s in ("FATAL ERROR", "invalid pointer", "double free"))),
        completed_1092_steps=int(final["NumCycles"])-int(initial["NumCycles"]) == 1092,
        passed_failure_by_over_one_time_unit=float(final["Time"])-float(failure[2]) > 1,
        all_physical_fields_finite=all(np.all(np.isfinite(prim[k])) for k in prim
                                       if isinstance(prim[k], np.ndarray)
                                       and np.issubdtype(prim[k].dtype, np.number)),
        positive_density_pressure_temperature=(np.min(rho) > 0
            and np.min(prim["press"]) > 0 and np.min(temperature) > 0),
        latent_exchange_active=any(np.max(np.abs(s["q_latent"])) > 0 for s in snapshots),
        gas_energy_matches_temperature=float(np.max(np.abs(temperature/final["Tem"]-1))) < 1e-9,
        radiation_energy_conserved=max(r["radiation_energy_defect"] for r in rows) < 1e-10,
        phase_energy_conserved=max(r["phase_energy_defect"] for r in rows) < 1e-10,
        phase_mass_conserved=max(r["phase_mass_defect"] for r in rows) < 1e-10)
    checks = {key: bool(value) for key, value in checks.items()}
    metrics = dict(initial_cycle=int(initial["NumCycles"]), initial_time=float(initial["Time"]),
        legacy_failure_cycle=int(failure[1]), legacy_failure_time=float(failure[2]),
        final_cycle=int(final["NumCycles"]), final_time=float(final["Time"]),
        final_min_T=float(temperature.min()), final_max_T=float(temperature.max()),
        max_relative_temperature_change=float(np.max(np.abs(final["Tem"]/initial["Tem"]-1))),
        temperature_energy_error=float(np.max(np.abs(temperature/final["Tem"]-1))),
        max_radiation_energy_defect=max(r["radiation_energy_defect"] for r in rows),
        max_phase_energy_defect=max(r["phase_energy_defect"] for r in rows),
        max_phase_mass_defect=max(r["phase_mass_defect"] for r in rows), checks=checks)
    (output_dir/"summary.json").write_text(json.dumps(metrics, indent=2)+"\n")
    plot(output_dir, initial, final, snapshots, metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    if not all(checks.values()):
        raise SystemExit("Restart checks failed; inspect summary.json")


def plot(output_dir, initial, final, snapshots, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    r, th = np.meshgrid(final["x1f"]*1.8983553591e12/1.495978707e13, final["x2f"])
    radius, height = r*np.sin(th), r*np.cos(th)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    norm = Normalize(min(initial["Tem"].min(), final["Tem"].min()),
                     max(initial["Tem"].max(), final["Tem"].max()))
    for ax, data, title in ((axes[0, 0], initial, "Restart 01500"),
                             (axes[0, 1], final, "After 1,092 energy steps")):
        im = ax.pcolormesh(radius, height, data["Tem"][0], cmap="inferno", norm=norm,
                           shading="flat", rasterized=True)
        ax.set(title=title, xlabel="Cylindrical radius [au]", ylabel="Height [au]")
    fig.colorbar(im, ax=list(axes[0]), label="Temperature [K]")
    change = 100*(final["Tem"][0]/initial["Tem"][0]-1)
    bound = np.max(np.abs(change))
    im = axes[1, 0].pcolormesh(radius, height, change, cmap="RdBu_r", vmin=-bound,
                               vmax=bound, shading="flat", rasterized=True)
    axes[1, 0].set(title="Temperature response", xlabel="Cylindrical radius [au]",
                    ylabel="Height [au]")
    fig.colorbar(im, ax=axes[1, 0], label="Temperature change [%]")
    series = [initial]+snapshots
    times = [float(d["Time"])-metrics["initial_time"] for d in series]
    # Legacy ti=39,tj=43 includes two ghosts at the inner and upper boundaries.
    axes[1, 1].plot(times, [d["Tem"][0, 41, 37] for d in series], "o-", markersize=3,
                      label="Cell where legacy T_eq failed")
    axes[1, 1].axvline(metrics["legacy_failure_time"]-metrics["initial_time"],
                         color="#ae345c", linestyle="--", label="Legacy failure")
    axes[1, 1].set(xlabel="Time since restart [code units]", ylabel="Temperature [K]",
                    title="Energy evolution through the old failure")
    axes[1, 1].legend(fontsize=9)
    fig.suptitle("Snowline restart 01500: implicit radiation + material energy\n"
                 "Hydro, orbital advection, radial transport and phase change enabled")
    fig.savefig(output_dir/"restart_temperature.png", dpi=180)
    fig.savefig(output_dir/"restart_temperature.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--athena", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        execute(args.athena.resolve(), output_dir)
    analyze(output_dir)


if __name__ == "__main__":
    main()
