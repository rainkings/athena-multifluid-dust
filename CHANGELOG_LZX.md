# CHANGELOG_LZX

Date: 2026-09-06

## Purpose

Remove the passive tracer fluids — the **vapor** (gas-phase species) and the **number-density** (grain-count) fluids — from the dust-fluids physics solves. Tracers are advected only, as concentrations riding a carrier flux (gas for vapor, the population's silicate fluid for the number density). They must not be Riemann-solved, reconstructed, dragged, or constrain the timestep as if they were real dust.

In the current 1p-2z configuration: `NDUSTFLUIDS = 4` = `N_P*N_Z = 2` real (momentum-carrying) composition fluids + 1 vapor + 1 number density. All changes are no-ops when a layout has no tracers (`NDUSTFLUIDS == NDUSTFLUIDS_MOM`).

## New layout macro

- `src/defs.hpp.in` (+ regenerated `src/defs.hpp`): `#define NDUSTFLUIDS_MOM (N_P*N_Z)` — number of real momentum dust fluids; tracer fluids are indices `NDUSTFLUIDS_MOM..NDUSTFLUIDS-1`. Defined in the configure template so it survives `configure.py`.

## Changes by file

1. **Riemann solve / fluxes** — `src/dustfluids/dustfluids_Riemann_solver.cpp`, `src/dustfluids/dustfluids_hlle_solver.cpp`, `src/dustfluids/calculate_dustfluids_fluxes.cpp`
   - Tracer rows are excluded from all Riemann/HLLE solves (loops tightened from `NDUSTFLUIDS` to `N_P*N_Z` fluids).
   - `TracerUpwindFlux` advects tracer density rows as `carrier mass flux × upwind concentration` and fills the tracer momentum rows as `dens_flux × upwind cell-centred tracer velocity` (bookkeeping only); vapor rides the gas flux, number densities ride their population's silicate flux.

2. **Reconstruction** — `src/reconstruct/ppm_dustfluids.cpp`, `plm_dustfluids.cpp`, `dc_dustfluids.cpp`
   - Reconstruction row range capped at the real dust rows (`4*NDUSTFLUIDS_MOM`); tracer rows use cell-centred values (upwinding handled by the concentration reconstruction).

3. **Drag** — `src/dustfluids/dustfluids_drags/vl2implicit_integrator.cpp`, `dust_gas_drag.hpp`, `inversion.cpp`, `multiplication.cpp`, `addition.cpp`
   - Per-fluid drag loops capped at `NDUSTFLUIDS_MOM` in `VL2ImplicitFeedback`/`VL2ImplicitNoFeedback`; tracer rows never read their stopping time and never exchange momentum.
   - The VL2 implicit solve is reduced from the full `NDUSTGAS` system to the active block of size `NDUSTFLUIDS_MOM + 1` (gas + real dust); matrix helpers (`Add`, `MultiplyVector`, `LUdecompose`, `SolveLinearEquation`, `SolveMultipleLinearEquation`, `Inverse`) take an `nsize` argument (default `NDUSTGAS`, so the other integrators are unchanged); `nsize` is propagated through the internal helper calls.
   - Whole-buffer `ZeroClear()` of scratch arrays replaced by bounded sub-block clears (`ClearDustScratch1/2/3`) over the active rows/columns and active cells.
   - `LUdecompose` initialises the pivot row each elimination step (`mmax_arr(i) = l`) — deterministic behaviour for degenerate columns.
   - Other drag integrators (`rk2implicit`, `backwardEuler`, `trbdf2`, `trapezoid`, `explicit`) are **not** converted; banner comments added in each file (and at the `DragIntegrate` dispatch) noting they still treat tracer rows as real dust.

4. **Diffusion (parabolic dt only)** — `src/dustfluids/dustfluids_diffusion/dustfluids_diffusion.cpp`
   - `NewDiffusionDt()` skips the number-density rows: they share their population's diffusivity, so their parabolic constraint is identical to the real fluids already looped over. Concentration diffusion of the number density itself is **kept** (it must stay turbulently mixed with its parent population — physics).

5. **Profiling** — `src/globals.hpp`, `src/task_list/task_list.cpp`, `src/main.cpp`
   - `TimeProfile` dust bucket split into `prop / flux / int / diff / drag / src` columns printed as a second `[profile]` line (total `dust=` unchanged).


## Notes

- Physical behaviour of the real dust fluids is unchanged; vapor and number-density *density/concentration* fields are unaffected (their advection is carrier-exact by construction); their velocity/momentum rows are passive bookkeeping and are not used in analysis.

---

## Addendum 2026-09-18 — floor-gated dust speed ceiling in the CFL constraint

**Problem.** A single cell whose dust density has been driven onto the dust floor (`dffloor`) can hold a `|v| = |mom|/dffloor` spike that is numerically meaningless (the momentum row is not rescaled when the EOS floors the conserved density, `src/eos/eos_dustfluids.cpp:106`). That spike contains no mass, yet it limited `dt` globally: `NewAdvectionDt` reported `d(2) x2 (3, 26, 0)` with `dt ~ 8e-3` while the cell held ~1e-15 of dust.

**Change (timestep determination only).** The primitive and conserved dust states are **not** modified anywhere:

- `src/dustfluids/dustfluids.hpp` — two new members, `vel_ceiling_factor_` and `dffloor_`.
- `src/dustfluids/dustfluids.cpp` — constructor reads `dust/st_vel_ceiling` (default `3.0`) and `dust/dffloor` (default `sqrt(1024*float_min)`, same expression the EOS uses).
- `src/dustfluids/new_blockdt_dustfluids.cpp` — in `NewAdvectionDt`, the denominators of `dt = dx/|v|` use `min(|v|, vel_ceiling_factor_*cs_gas)`, with `cs_gas` recovered from the gas primitive state as `sqrt(p/rho)` (no gamma/mu; valid for `general/phase_change`). The ceiling is engaged **only** in cells with `rho_d <= dffloor`; above the floor the raw velocity is used unchanged. A diagnostic printout reports how many `(cell, direction)` pairs were capped, with the capped minimum `dt` and its location.

**Properties.** `min(|v|, v_ceil) <= |v|`, so `dt` can only become *larger*: no new CFL/stability constraint is introduced and the pre-existing `dt <= 1e-3` bail-out cannot be triggered by the ceiling. The cap is one-way safe, and it is the only behaviour that changes — the min-dt bookkeeping, `dt_hyperbolic *= cfl_number`, `Mesh::NewTimeStep()`, and all `df_u`/`df_w` values are untouched.

**Input switch.** `<dust> st_vel_ceiling = 3.0` (units of the local gas sound speed; set `<= 0` to disable the ceiling and its diagnostic).

