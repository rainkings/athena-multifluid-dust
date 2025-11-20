# Phase Change Module User Guide

## Overview

The Phase Change module enables simulations of phase transitions (sublimation/condensation) between ice and vapor for multi-composition pebbles in protoplanetary disks. The module handles:
- Multiple pebble sizes (N_P) with multiple compositions per pebble (N_Z)
- Latent heat absorption and release during phase transitions
- Vapor-dependent equation of state and Riemann solver
- Dynamic pebble number density tracking

## Features

### 1. Multi-Composition Pebble Support
- **N_P**: Configurable number of pebble sizes (compile-time parameter)
- **N_Z**: Configurable number of compositions per pebble (compile-time parameter, typically 2: ice + refractory)
- **NVapor**: Number of vapor tracers (currently fixed at 1)

### 2. Phase Transition Physics
- **Sublimation/Condensation**: Phase transitions between ice and vapor based on equilibrium conditions
- **Latent Heat**: Energy exchange during phase changes tracked via `q_latent`

### 3. Dynamic Pebble Properties
- **Number Density Tracking**: Stores `rho_Np_array` (pebble number density) as primary quantity
- **Derived Properties**: Pebble mass (`m_p`) and size (`s_p`) calculated on-the-fly from number density
- **Stopping Time**: Calculates stopping time for each pebble size based on local conditions and drag regimes (customly specified)

### 4. Vapor as a Tracer
- **Passive Tracer**: Vapor is modeled as a passive tracer using a dustfluid
- **Concentration Tracking**: Vapor concentration (density normalized by gas density) reconstructed at cell faces
- **Tracer Solver**: Vapor flux calculated using `TracerUpwindFlux()` function instead of standard dust fluid Riemann solvers.

> **Note:** We did **not** use Athena++'s passive scalar module because it does **not** support user-defined boundary conditions.

### 5. Vapor-Dependent EOS
- **Sound Speed**: Depends on vapor fraction when phase change is enabled
- **Pressure/Energy**: Internal energy and pressure calculations account for vapor concentration
- **Mean Molecular Weight**: Varies with vapor fraction

### 6. Restart Capability
- **State Preservation**: `rho_Np_array` saved to and loaded from restart files
- **I/O**: Integrated into Athena++ restart system

## How to Use

### 1. Configuration

#### Compile-Time Parameters

Configure the number of pebble sizes and compositions when building Athena++:

```bash
python configure.py --np <N_P> --nz <N_Z> [other options...]
```

- `--np`: Number of pebble sizes (N_P)
- `--nz`: Number of compositions per pebble (N_Z), typically 2 (ice + refractory)

**Example:**
```bash
python configure.py --np 3 --nz 2 --prob disk_snowline_2D_RT_erg_2
```

This creates:
- 3 pebble sizes (p = 0, 1, 2)
- 2 compositions per pebble (ice + refractory)
- Total: 3 × 2 = 6 dust fluids + 1 vapor = 7 total dust fluids

#### Dust Fluid Layout

For pebble size `p` (0 to N_P-1):
- **Ice composition**: `dust_id = N_Z * p`
- **Refractory composition**: `dust_id = N_Z * p + 1`
- **Vapor tracer**: `dust_id = NDUSTFLUIDS - 1` (always the last dust fluid)

**Verification:** `NDUSTFLUIDS = N_P * N_Z + NVapor`

### 2. ParameterInput Configuration

Add the following parameters to your input file:

```ini
<problem>
  # Enable phase change module (N_Z > 0 activates module)
  # Module is automatically created if N_Z > 0
  
  # Initial ice fraction in pebbles
  f_ICE_inter0 = 0.5
  
  # Initial pebble masses [g] (one per pebble size)
  m_p0_1 = 1.0e-3
  m_p0_2 = 1.0e-2
  m_p0_3 = 1.0e-1
  
  # Material densities [g/cm^3]
  rho_sil_inter = 3.0    # Refractory material density
  rho_ice_inter = 1.0    # Ice density
  
  # Numerical tolerance for iterative solver
  min_tol = 1.0e-7

<hydro>
  # Minimum gas density (used by phase change module)
  dfloor = 1.0e-10

<dust>
  # Minimum dust density (used by phase change module)
  dffloor = 1.0e-10
```

**Required Parameters:**
- `m_p0_<N>`: Initial pebble mass for pebble size N (must provide N_P values: m_p0_1, m_p0_2, ..., m_p0_N_P)

**Optional Parameters:**
- `f_ICE_inter0`: Initial ice mass fraction (default: 0.5)
- `rho_sil_inter`: Refractory material density in CGS [g/cm^3] (default: 3.0)
- `rho_ice_inter`: Ice density in CGS [g/cm^3] (default: 1.0)
- `min_tol`: Numerical tolerance of phase change solver (default: 1.0e-7)

### 3. Problem Generator Setup

#### Include Phase Change Header

```cpp
#include "../phase_change/phase_change.hpp"
```

#### Initialize Pebble Number Density

In `MeshBlock::ProblemGenerator()`, after setting initial dust densities:

```cpp
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // ... initialize dust densities ...
  
  // Initialize rho_Np_array from initial refractory densities
  if (pphase_change != nullptr && N_Z > 0) {
    for (int p = 0; p < N_P; ++p) {
      int refrac_id = N_Z * p + 1; // refractory composition
      int rho_id = 4*refrac_id;
      
      for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
          for (int i = is; i <= ie; ++i) {
            Real rho_sil_p = pdustfluids->df_u(rho_id, k, j, i);
            Real &rho_Np = pphase_change->rho_Np_array(p, k, j, i);
            Real f_ICE_inter0 = pin->GetOrAddReal("problem", "f_ICE_inter0", 0.5);
            rho_Np = rho_sil_p / (1.0 - f_ICE_inter0) / pphase_change->m_p0_array(p);
          }
        }
      }
    }
  }
}
```

#### Implement Source term
```cpp
void MySource(MeshBlock *pmb, const Real time, const Real dt,
    const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_df,
    const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {
  
  // Other source terms (e.g., radiative conduction, local isothermal EOS)
  
  // Phase change source term
  if (N_Z > 0 && pmb->pphase_change != nullptr) {
    pmb->pphase_change->PhaseChangeSource(pmb, time, dt, prim, prim_df, prim_s, 
                                          bcc, cons, cons_df, cons_s);
  }
}
```

#### Implement Stopping Time Function

The `MyStoppingTime` function calculates the stopping time for each dust fluid. The PhaseChange module provides `Get_stopping_time()` as an example implementation that handles:
- Epstein drag regime (for small particles)
- Stokes drag regime (for larger particles)
- Dependence on pebble properties (mass, size) derived from `rho_Np_array`
- Vapor density effects on gas properties

**Note**: Users can define their own stopping time calculation logic. The PhaseChange module's `Get_stopping_time()` is provided as a convenient example, but you are free to implement any stopping time formula that fits your physical model.

```cpp
void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time,
    const int il, const int iu, const int jl, const int ju, const int kl, const int ku) {
  
  if (N_Z == 0 || pmb->pphase_change == nullptr) {
    // Fallback for no phase change - use constant stopping time or custom logic
    return;
  }
  
  // ... calculate gas properties (density, temperature, etc.) ...
  
  // Vapor stopping time (vapor is essentially a passive tracer)
  int vapor_id = NDUSTFLUIDS - 1;
  stopping_time(vapor_id, k, j, i) = 1.e-8;  // Very small stopping time
  
  // Per-pebble stopping time
  for (int p = 0; p < N_P; ++p) {
    // Collect densities for all compositions of this pebble
    AthenaArray<Real> rho_dustfluid_array_pebble;
    rho_dustfluid_array_pebble.NewAthenaArray(N_Z);
    
    for (int z = 0; z < N_Z; ++z) {
      int dust_id = N_Z * p + z;
      int rho_id = 4*dust_id;
      rho_dustfluid_array_pebble(z) = prim_df(rho_id, k, j, i);
    }
    
    // Get pebble number density
    Real rho_Np = pmb->pphase_change->rho_Np_array(p, k, j, i);
    Real rho_v = prim_df(4*vapor_id, k, j, i);
    
    // Example: Use PhaseChange module's Get_stopping_time()
    // This handles Epstein/Stokes drag and uses rho_Np to derive pebble properties
    Real t_stop = pmb->pphase_change->Get_stopping_time(
        pmb->pmy_mesh->punit, rho_dustfluid_array_pebble, T, rho_g, rho_v, rho_Np);
    
    // Alternative: Users can implement their own stopping time calculation here
    // For example:
    // Real s_p = ...;  // calculate pebble size
    // Real m_p = ...;  // calculate pebble mass
    // Real t_stop = ...;  // your custom stopping time formula
    
    // Apply stopping time to all compositions of this pebble
    for (int z = 0; z < N_Z; ++z) {
      int dust_id = N_Z * p + z;
      stopping_time(dust_id, k, j, i) = t_stop;
    }
  }
}
```

### 4. Accessing Phase Change Data

#### Public Member Variables

```cpp
// Pebble number density [1/cm^3] in code units
pphase_change->rho_Np_array(p, k, j, i)

// Latent heat rate [code unit]
pphase_change->q_latent(k, j, i)

// Heat conduction rate [code unit]
pphase_change->q_diff(k, j, i)

// Initial pebble masses [code unit]
pphase_change->m_p0_array(p)

// Physical constants (converted to code units)
pphase_change->L_heat      // Latent heat of vaporization
pphase_change->Cd_water    // Heat capacity of water ice
pphase_change->P_eq0       // Equilibrium pressure prefactor
```

## Implementation Details

The Phase Change module is integrated throughout Athena++ at multiple levels. This section provides an overview of where and how it is implemented.

### 1. Core Phase Change Module (`src/phase_change/`)

#### `phase_change.hpp`
- **Class Definition**: `PhaseChange` class with public and private members
- **Public Members**: `rho_Np_array`, `q_latent`, `q_diff`, `m_p0_array`, physical constants
- **Public Functions**:
  - `PhaseChange()`: Constructor
  - `PhaseChangeSource()`: Main source term function
  - `Get_m_p_from_rho_Np()`: Derive pebble mass
  - `Get_s_p_from_m_p()`: Derive pebble size
  - `Get_stopping_time()`: Calculate stopping time
  - `Get_T_rhoe_g()`: Get temperature from gas energy

#### `phase_change.cpp`
- **Constructor** (`PhaseChange::PhaseChange()`):
  - Allocates `rho_Np_array` (4D: N_P × ncells3 × ncells2 × ncells1)
  - Registers arrays for restart I/O
  - Reads and converts input parameters from CGS to code units
  - Validates dust fluid layout consistency
  
- **Main Source Term** (`PhaseChange::PhaseChangeSource()`):
  - Iterates over all grid cells
  - For each pebble size p:
    - Collects ice and refractory densities
    - Updates `rho_Np_array`
    - Calculates pebble mass and size
    - Computes sublimation/condensation rates
    - Updates dust and gas conservation equations
    - Updates latent heat source term
  
- **Helper Functions**:
  - `Get_m_p_from_rho_Np()`: Calculates pebble mass from number density
  - `Get_s_p_from_m_p()`: Calculates pebble size from mass
  - `Get_stopping_time()`: Handles Epstein and Stokes drag regimes
  - `Get_T_rhoe_g()`: Temperature from gas internal energy
  - `phase_trans()`: Phase transition rate calculation
  - `GetSublimationRate()`: Sublimation rate calculation

#### `phase_change_constants.hpp`
- **Physical Constants**: `L_heat_cgs`, `Cd_water_cgs`, `P_eq0_cgs`, molecular weights
- **Compile-Time Constants**: `NVapor`, `vapor_id`, `FLX_COR`
- **Temperature Constant**: `KELVIN`

> **⚠️ CRITICAL BUG WARNING**: The `vapor_id` macro **MUST** be defined with parentheses:
> ```cpp
> #define vapor_id (NDUSTFLUIDS-1)  // ✅ CORRECT - parentheses required!
> ```
> 
> **DO NOT** define it without parentheses:
> ```cpp
> #define vapor_id NDUSTFLUIDS-1    // ❌ WRONG - will cause subtle bugs!
> ```
> 
> Without parentheses, the macro expansion can cause incorrect operator precedence when used in expressions like `4*vapor_id`, leading to hard-to-debug errors. This bug can be very difficult to track down!

### 2. Problem Generator Integration (`src/pgen/`)

#### Problem Generator File (e.g., `disk_snowline_2D_RT_erg_2.cpp`)
- **Initialization** (`MeshBlock::ProblemGenerator()`):
  - Sets initial dust densities for all pebble sizes
  - Initializes `rho_Np_array` from refractory densities
  - Formula: `rho_Np = rho_sil / (1.0 - f_ICE_inter0) / m_p0`
  
- **Source Function** (`MySource()`):
  - Calls `pphase_change->PhaseChangeSource()` after other source terms
  
- **Stopping Time** (`MyStoppingTime()`):
  - Loops over all pebble sizes
  - Collects per-pebble dust densities
  - Calls `pphase_change->Get_stopping_time()` for each pebble
  - Applies stopping time to all compositions of each pebble
  
- **Dust Diffusivity** (`MyDustDiffusivity()`):
  - Calculates vapor diffusivity
  - Calculates pebble diffusivity based on stopping time
  
- **Boundary Flux** (`Mesh::UserWorkInLoop()`):
  - Calculates mass flux for each dust fluid
  - Applies flux ratio correction per dust_id
  - Handles dust injection timing

### 3. Mesh and MeshBlock Infrastructure (`src/mesh/`)

#### `mesh.hpp`
- **Forward Declaration**: `class PhaseChange;`
- **Member Variable**: `PhaseChange *pphase_change;` in `MeshBlock` class

#### `meshblock.cpp`
- **Constructor (Normal)** (`MeshBlock::MeshBlock()`):
  - Creates `PhaseChange` object if `N_Z > 0`
  - Called after `DustFluids` is created
  
- **Constructor (Restart)** (`MeshBlock::MeshBlock()`):
  - Same initialization as normal constructor
  - PhaseChange object created before loading restart data
  
- **Destructor** (`MeshBlock::~MeshBlock()`):
  - Deletes `PhaseChange` object if `N_Z > 0`
  
- **Restart I/O**:
  - `LoadRestartData()`: Loads `rho_Np_array` from restart file
  - `GetBlockSizeInBytes()`: Includes `rho_Np_array` size in restart file size

### 4. DustFluids Module Integration (`src/dustfluids/`)

#### `dustfluids.hpp`
- **Vapor Concentration Arrays**: 
  - `AthenaArray<Real> r_`: Vapor concentration array (density normalized by gas density)
  - `AthenaArray<Real> rl_, rr_, rlb_`: Left and right reconstructed states for vapor concentration
  
- **Tracer Flux Function**:
  - `TracerUpwindFlux()`: Calculates vapor flux using upwind method based on gas mass flux

#### `dustfluids.cpp`
- **Constructor** (`DustFluids::DustFluids()`):
  - Allocates vapor concentration arrays (`r_`, `rl_`, `rr_`, `rlb_`)
  - Arrays have dimensions: NDUSTFLUIDS × ncells1 (for `rl_`, `rr_`, `rlb_`) and NDUSTFLUIDS × ncells3 × ncells2 × ncells1 (for `r_`)

#### `calculate_dustfluids_fluxes.cpp`
- **Flux Calculation** (`CalculateDustFluidsFluxes()`):
  - For each coordinate direction (x1, x2, x3):
    - Reconstructs vapor concentration using PiecewiseLinear or PiecewiseParabolic reconstruction
    - Calls regular dust fluid Riemann solver for pebble dust fluids
    - Calls `TracerUpwindFlux()` to overwrite vapor flux with tracer-based calculation
  - **X1 Direction** (lines 95, 98, 129):
    - Reconstructs `rl_`, `rr_` from `r_`
    - Calls `TracerUpwindFlux()` after Riemann solver
  - **X2 Direction** (lines 210, 213, 229, 232, 252):
    - Reconstructs vapor concentration at faces
    - Calls `TracerUpwindFlux()` for vapor flux
  - **X3 Direction** (lines 331, 334, 349, 352, 371):
    - Same pattern as X2 direction

#### `dustfluids_Riemann_solver.cpp`
- **Tracer Upwind Flux** (`DustFluids::TracerUpwindFlux()`):
  - Iterates over all dust fluids
  - For vapor dustfluid (`n == vapor_id`):
    - Uses gas mass flux (`gas_mass_flx`)
    - Selects left or right reconstructed vapor concentration based on flux direction
    - Calculates: `flx_out = gas_mass_flx * r_l` (if flux ≥ 0) or `gas_mass_flx * r_r` (if flux < 0)
  - **Purpose**: Ensures vapor advects exactly with gas, appropriate for passive tracer

**Key Implementation Details:**
- Vapor concentration `r_(vapor_id, k, j, i) = vapor_density(k, j, i) / gas_density(k, j, i)`
- Reconstruction uses same order as dust fluids (piecewise linear or parabolic)
- Upwind flux ensures numerical stability and correct tracer advection

### 6. Hydro Module Integration (`src/hydro/`)

#### `calculate_fluxes.cpp`
- **Flux Correction** (`CalculateFluxes()`):
  - Applies flux correction only at outer simulation boundary
  - Condition: `pmb->pbval->block_bcs[BoundaryFace::outer_x1] == BoundaryFlag::user`
  
#### `rsolvers/hydro/hllc.cpp`
- **HLLC Riemann Solver** (`RiemannSolver()`):
  - Uses vapor-dependent sound speed when `N_Z > 0`
  - Calls `peos->SoundSpeed_fv()` with vapor fraction from reconstructed states
  
#### `new_blockdt.cpp`
- **Time Step Calculation** (`NewBlockTimeStep()`):
  - Uses vapor-dependent sound speed for CFL condition
  - Calls `peos->SoundSpeed_fv()` when `N_Z > 0`

### 7. Equation of State Integration (`src/eos/`)

#### `general/phase_change.cpp`
- **EOS Functions with Vapor**:
  - `Get_mu()`: Mean molecular weight from vapor fraction
  - `calc_gamma()`: Adiabatic index from vapor fraction
  - `PresFromRhoEg_fv()`: Pressure from energy with vapor
  - `EgasFromRhoP_fv()`: Energy from pressure with vapor
  - `AsqFromRhoP_fv()`: Sound speed squared with vapor
  
- **Constants Access**: Includes `phase_change_constants.hpp` for molecular weights

#### `general/general_hydro.cpp`
- **Primitive to Conservative** (`PrimToCons()`):
  - Uses vapor-dependent pressure calculation when `N_Z > 0`
  - Accesses vapor density via `vapor_id`
  - Calls `PresFromRhoEg_fv()` instead of standard EOS
  
- **Conservative to Primitive** (`ConsToPrim()`):
  - Uses vapor-dependent energy calculation when `N_Z > 0`
  - Calls `EgasFromRhoP_fv()` instead of standard EOS

### 6. Output Integration (`src/outputs/`)

#### `restart.cpp`
- **Write Restart Data** (`WriteRestartFile()`):
  - Writes `rho_Np_array` to restart file
  - Positioned after dustfluids data, before scalars
  - Uses `std::memcpy()` for binary I/O

### 8. Compile-Time Configuration

#### `configure.py`
- **Build Arguments**:
  - `--np`: Sets `N_P` (number of pebble sizes)
  - `--nz`: Sets `N_Z` (number of compositions per pebble)
  
- **Template Replacement**: Replaces `@NUMBER_PEBBLES@` and `@NUMBER_COMPOSITIONS@` in `defs.hpp.in`

#### `defs.hpp.in`
- **Macro Definitions**:
  - `#define N_P @NUMBER_PEBBLES@`
  - `#define N_Z @NUMBER_COMPOSITIONS@`

### 8. Unit Conversion Strategy

**Principle**: All internal calculations use code units. Conversions only at input/output boundaries.

**Input Conversion** (ParameterInput → PhaseChange constructor):
- `m_p0_array`: CGS [g] → code units [code_mass]
- `rho_sil_inter_`: CGS [g/cm³] → code units [code_density]
- `rho_ice_inter_`: CGS [g/cm³] → code units [code_density]
- `L_heat`: CGS [erg/g] → code units [code_velocity²]
- `P_eq0`: CGS [erg/cm³] → code units [code_pressure]

**Internal Calculations**:
- All arrays (`rho_Np_array`, `q_latent`, `q_diff`) in code units
- Helper functions operate in code units
- `Get_stopping_time()` returns code_time

**Output Conversion**:
- Convert from code units to CGS only when writing to output files
- Example: `s_p * punit->code_length_cgs` for pebble size output

## Common Issues and Solutions

### Issue 1: Module Not Created
**Symptom**: `pphase_change` is `nullptr`  
**Solution**: Ensure `N_Z > 0` is set in `configure.py` (via `--nz` flag)

### Issue 2: Inconsistent Dust Fluid Count
**Symptom**: Error message about `NDUSTFLUIDS != N_P * N_Z + NVapor`  
**Solution**: Check that `NDUSTFLUIDS` matches `N_P * N_Z + 1` in your configuration

### Issue 3: Missing Initial Pebble Mass
**Symptom**: ParameterInput error about missing `m_p0_<N>`  
**Solution**: Provide initial pebble mass for each pebble size: `m_p0_1`, `m_p0_2`, ..., `m_p0_<N_P>`

### Issue 4: Incorrect Dust Fluid Indexing
**Symptom**: Wrong pebble properties accessed  
**Solution**: Use `dust_id = N_Z * p + z` where `p` is pebble index (0..N_P-1) and `z` is composition index (0..N_Z-1)

### Issue 5: Restart File Incompatibility
**Symptom**: Restart file cannot be loaded  
**Solution**: Restart files store `rho_Np_array`. Ensure same `N_P` and `N_Z` values when restarting

### Issue 6: Incorrect Macro Definition (CRITICAL BUG)
**Symptom**: Subtle bugs when accessing vapor density, incorrect array indexing, wrong vapor fraction calculations, or mysterious errors in expressions involving `vapor_id`  
**Root Cause**: Missing parentheses in `vapor_id` macro definition causes incorrect operator precedence  
**Solution**: 
```cpp
// ✅ CORRECT - MUST use parentheses
#define vapor_id (NDUSTFLUIDS-1)

// ❌ WRONG - Missing parentheses causes operator precedence bugs
#define vapor_id NDUSTFLUIDS-1
```

**Why This Matters**: 
- When you write `4*vapor_id`, without parentheses it expands to `4*NDUSTFLUIDS-1` which evaluates as `(4*NDUSTFLUIDS)-1` instead of `4*(NDUSTFLUIDS-1)`
- This causes incorrect array indexing and can lead to accessing wrong memory locations or calculating wrong vapor fractions
- The bug is very difficult to track down because the code may compile and run, but produce incorrect results
- **Always use parentheses when defining macros that involve arithmetic operations!**

