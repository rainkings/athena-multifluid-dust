#!/usr/bin/env python3
"""Compare native viscous heating plus irradiation with the snowline T_eq.

Run with the spherical/general-eos_phase_change build, seven dust fluids,
N_P=N_Z=2, HDF5 double precision, and OpenMP. Example:

  python tst/regression/snowline_thermal_viscosity.py --athena /path/to/athena \
      --before /path/to/athena_before_fix --output-dir /tmp/viscous-comparison

The disk density and cylindrical Keplerian shear are prescribed. Measure the
instantaneous internal-energy rate using Athena's native viscous fluxes,
geometric divergence and orbital sources: du/dt=dE/dt-v.dm/dt. Evolve this heat
and irradiation with the same implicit Column solver used by thermal_energy.
Compare with an independently assembled analytic two-stream equilibrium using
the old q_vis=(9/4)*rho*(1-fv)*nu*Omega_K^2. This is an equilibrium/operator audit,
not a freely evolving accretion disk. Phase change and radial diffusion are off.
In this general-EOS build alpha_vis sets nu=alpha_vis*(R/23.64118024), not
nu=alpha*c_s^2/Omega. The viscosity is independent of the evolving temperature.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import numpy as np
from snowline_thermal_disk import ROOT, read


def run(binary, output, name, nr=160, nt=48, oa=2, factor=0.5, dt=500,
        block_r=20, block_t=24, analyze_only=False):
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    steps = int(64000/dt)
    command = [str(binary), '-i', str(ROOT/'inputs/snowline/athinput.thermal_compare'),
               'problem/thermal_test_viscous=true', 'problem/alpha_vis=0.003',
               'problem/L_star=7', f'orbital_advection/OAorder={oa}',
               f'mesh/nx1={nr}', f'mesh/nx2={nt}', f'meshblock/nx1={block_r}',
               f'meshblock/nx2={block_t}', f'problem/thermal_test_initial_factor={factor}',
               f'time/nlim={steps}', 'time/tlim=64000']
    if not analyze_only:
        (directory/'command.json').write_text(json.dumps(command, indent=2)+'\n')
        with (directory/'run.log').open('w') as log:
            result = subprocess.run(command, cwd=directory, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=180)
        if result.returncode:
            raise RuntimeError(name+': process exited '+str(result.returncode))
    text = (directory/'run.log').read_text()
    if 'FATAL ERROR' in text:
        raise RuntimeError(name+': '+text[-2500:])
    data, prim = read(directory, 2), read(directory, 1)
    initial, cons = read(directory, 1, final=False), read(directory, 3)
    energy, reference = data['T_energy'][0], data['T_prescribed_eq'][0]
    native, prescribed = data['q_vis_native'][0], data['q_vis_prescribed'][0]
    # ds=r*dtheta is the plane-parallel column measure used by the pgen.
    ds = np.diff(data['x2f'])[:, None]*data['x1v'][None, :]
    ratio = np.sum(native*ds, axis=0)/np.sum(prescribed*ds, axis=0)
    rho = cons['dens'][0]
    fv = cons['dust_5_dens'][0]/rho
    cv = ((1-fv)*(0.71/2*2.5+0.29/4*1.5)+fv/18*3)/64.102564103
    # Athena writes orbital-advection outputs in the physical frame: subtract
    # their kinetic energy, even though the integrated residual velocity is zero.
    kinetic = sum(cons['mom'+str(v)][0]**2 for v in (1, 2, 3))/(2*rho)
    from_energy = (cons['Etot'][0]-kinetic)/(rho*cv)
    diagnostics = re.findall(r'\[thermal_test\][^\n]+', text)[-1]
    diagnostics = {key: float(value) for key, value in re.findall(
        r'(\w+)=([+\-\d.eE]+)', diagnostics)}
    metrics = dict(name=name, nr=nr, nt=nt, oa=oa, dt=dt, initial_factor=factor,
                   max_relative_T_error=float(np.max(np.abs(energy/reference-1))),
                   rms_relative_T_error=float(np.sqrt(np.mean((energy/reference-1)**2))),
                   max_midplane_relative_T_error=float(np.max(np.abs(energy[-1]/reference[-1]-1))),
                   midplane_inner_K=float(energy[-1, 0]),
                   midplane_outer_K=float(energy[-1, -1]),
                   reference_midplane_inner_K=float(reference[-1, 0]),
                   reference_midplane_outer_K=float(reference[-1, -1]),
                   native_heating_column_ratio_min=float(ratio.min()),
                   native_heating_column_ratio_max=float(ratio.max()),
                   max_midplane_heating_error=float(np.max(np.abs(native[-1]/prescribed[-1]-1))),
                   max_discrete_equilibrium_error=float(np.max(np.abs(energy/data['T_FV_eq'][0]-1))),
                   energy_temperature_error=float(np.max(np.abs(from_energy/energy-1))),
                   density_drift=float(np.max(np.abs(prim['rho']/initial['rho']-1))),
                   max_velocity_drift=max(float(np.max(np.abs(
                       prim['vel'+str(v)]-initial['vel'+str(v)]))) for v in (1, 2, 3)),
                   diagnostics=diagnostics)
    print(name, 'max |dT/T|={:.5g}, q column ratio=[{:.6g},{:.6g}]'.format(
        metrics['max_relative_T_error'], ratio.min(), ratio.max()), flush=True)
    return dict(data=data, metrics=metrics)


def plot(output, before, inertial, orbital, fine, orders):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/snowline-thermal-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    data = orbital['data']
    au = 1.8983553591e12/1.495978707e13
    r, theta = np.meshgrid(data['x1f']*au, data['x2f'])
    radius, height = r*np.sin(theta), r*np.cos(theta)
    reference = data['T_prescribed_eq'][0]
    energy = data['T_energy'][0]
    norm = Normalize(vmin=min(reference.min(), energy.min()),
                     vmax=max(reference.max(), energy.max()))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, values, title in ((axes[0, 0], reference, 'Analytic: irradiation + prescribed viscous heat'),
                              (axes[0, 1], energy, 'Energy solver: native viscous heat, OA corrected')):
        image = ax.pcolormesh(radius, height, values, norm=norm, cmap='inferno',
                             rasterized=True, shading='flat')
        ax.set(title=title, xlabel='Cylindrical radius [au]', ylabel='Height [au]')
    fig.colorbar(image, ax=list(axes[0]), label='Temperature [K]', shrink=0.9)
    error = 100*(energy/reference-1)
    limit = max(float(np.max(np.abs(error))), 1e-5)
    image = axes[1, 0].pcolormesh(radius, height, error, cmap='RdBu_r',
                                 vmin=-limit, vmax=limit, rasterized=True, shading='flat')
    axes[1, 0].set(title='Corrected energy / analytic - 1', xlabel='Cylindrical radius [au]',
                   ylabel='Height [au]')
    fig.colorbar(image, ax=axes[1, 0], label='Temperature difference [%]')
    ax = axes[1, 1]
    midradius = data['x1v']*au
    ax.plot(midradius, reference[-1], 'k', label='Analytic')
    ax.plot(midradius, energy[-1], '--', label='Energy: OA corrected')
    ax.plot(midradius, inertial['data']['T_energy'][0, -1], ':', label='Energy: OA off')
    if before is not None:
        ax.plot(midradius, before['data']['T_energy'][0, -1], color='#c03e36',
                label='Energy: OA before correction')
    ax.set(xlabel='Midplane radius [au]', ylabel='Temperature [K]', title='Midplane comparison')
    ax.legend(fontsize=8)
    fig.suptitle('Snowline disk: alpha_vis = 0.003, L_star = 7 solar luminosities\n'
                 'Fixed density and Keplerian shear; phase change and radial diffusion off')
    fig.savefig(output/'viscous_temperature_comparison.png', dpi=180)
    fig.savefig(output/'viscous_temperature_comparison.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for case, label in ((inertial, 'OA off'), (orbital, 'OA corrected')):
        d = case['data']
        axes[0].plot(d['x1v']*au, d['q_vis_native'][0, -1]/d['q_vis_prescribed'][0, -1], label=label)
    if before is not None:
        axes[0].plot(midradius, before['data']['q_vis_native'][0, -1]/
                     before['data']['q_vis_prescribed'][0, -1], label='OA before correction')
    axes[0].axhline(1, color='gray', linestyle='--')
    axes[0].set(xlabel='Midplane radius [au]', ylabel='Measured / prescribed viscous heating')
    axes[0].legend()
    for mode in ('inertial', 'orbital'):
        cases = fine[mode]
        n = np.array([c['metrics']['nr'] for c in cases])
        e = np.array([c['metrics']['max_relative_T_error'] for c in cases])
        axes[1].loglog(n, e, 'o-', label=mode+' (order {:.2f})'.format(orders[mode][-1]))
    axes[1].set(xlabel='Radial cells (theta refined proportionally)', ylabel='Maximum |dT/T|')
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.savefig(output/'viscous_heating_convergence.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--athena', required=True, type=Path)
    parser.add_argument('--before', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--analyze-only', action='store_true')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    binary = args.athena.resolve()
    def case(binary, name, **kwargs):
        return run(binary, output, name, analyze_only=args.analyze_only, **kwargs)
    before = case(args.before.resolve(), 'before_oa2') if args.before else None
    fine = {}
    for mode, oa in (('inertial', 0), ('orbital', 2)):
        fine[mode] = [case(binary, f'{mode}_{nr}x{nt}', nr=nr, nt=nt, oa=oa)
                      for nr, nt in ((160, 48), (320, 96), (640, 192))]
    hot = case(binary, 'hot', nr=320, nt=96, factor=2)
    half = case(binary, 'half_dt', nr=320, nt=96, dt=250)
    blocks = case(binary, 'different_blocks', block_r=40, block_t=48)
    orders = {mode: [float(np.log(cases[i]['metrics']['max_relative_T_error']/
                                  cases[i+1]['metrics']['max_relative_T_error'])/np.log(2))
                     for i in range(2)] for mode, cases in fine.items()}
    all_cases = sum(fine.values(), [])+[hot, half, blocks]
    baseline = fine['orbital'][1]['data']['T_energy']
    differences = dict(hot=float(np.max(np.abs(hot['data']['T_energy']/baseline-1))),
                       half_dt=float(np.max(np.abs(half['data']['T_energy']/baseline-1))),
                       blocks=float(np.max(np.abs(blocks['data']['T_energy']/
                                                  fine['orbital'][0]['data']['T_energy']-1))))
    checks = dict(
        finest_temperature_error_below_0_02_percent=all(
            cases[-1]['metrics']['max_relative_T_error'] < 2e-4 for cases in fine.values()),
        spatial_orders_above_1_7=all(p > 1.7 for values in orders.values() for p in values),
        native_heating_agrees_at_midplane=all(
            cases[-1]['metrics']['max_midplane_heating_error'] < 2e-4 for cases in fine.values()),
        reaches_native_discrete_equilibrium=all(
            c['metrics']['max_discrete_equilibrium_error'] < 1e-9 for c in all_cases),
        actual_gas_energy_matches=all(c['metrics']['energy_temperature_error'] < 1e-10 for c in all_cases),
        density_and_flow_frozen=all(c['metrics']['density_drift'] == 0 and
                                  c['metrics']['max_velocity_drift'] == 0 for c in all_cases),
        hot_start_timestep_and_blocks_agree=all(v < 1e-9 for v in differences.values()),
        thermal_column_energy_conserved=all(
            c['metrics']['diagnostics']['energy_defect'] < 1e-9 for c in all_cases))
    if before:
        checks['reproduces_missing_orbital_heating'] = (
            before['metrics']['native_heating_column_ratio_max'] < 1e-10 and
            before['metrics']['max_relative_T_error'] > 0.1)
    summary = dict(passed=all(checks.values()), checks=checks, alpha_vis=0.003,
                   L_star=7, orders=orders, differences=differences,
                   setup=dict(density='fixed pgen initial disk profile',
                              shear='fixed cylindrical Keplerian',
                              heating='native hydro viscous operator plus irradiation',
                              reference='analytic two-stream with prescribed 9/4 rho (1-fv) nu Omega_K^2',
                              phase_change=False, radial_diffusion=False,
                              viscosity='nu=alpha_vis*(R/23.64118024) in code units'),
                   before=before['metrics'] if before else None,
                   cases=[c['metrics'] for c in all_cases])
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    plot(output, before, fine['inertial'][0], fine['orbital'][0], fine, orders)
    print(json.dumps(dict(checks=checks, orders=orders), indent=2), flush=True)
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
