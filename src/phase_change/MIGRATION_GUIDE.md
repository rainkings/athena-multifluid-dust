# Phase Change Module Migration Guide

## Overview

This document describes the modifications made to Athena++ to implement the Phase Change module for multi-composition pebbles. The module handles phase transitions (sublimation/condensation) between ice and vapor for pebbles with multiple sizes and compositions.

## Key Concepts

### Dust Fluid Layout

The module supports:
- **N_P**: Number of pebble sizes (compile-time parameter, set via `configure.py --np`)
- **N_Z**: Number of compositions per pebble (compile-time parameter, set via `configure.py --nz`)
- **NVapor**: Number of vapor tracers (currently fixed at 1)

**Dust fluid indexing:**
- For pebble size `p` (0 to N_P-1):
  - `dust_id = N_Z * p`: ice composition (z=0)
  - `dust_id = N_Z * p + 1`: refractory composition (z=1)
  - Additional compositions can be added by extending N_Z
- `vapor_id = NDUSTFLUIDS - 1`: vapor tracer

**Consistency check:** `NDUSTFLUIDS = N_P * N_Z + NVapor`

### Data Structure Evolution

**Current design (Yu, 2025-11-18):**
- `rho_Np_array`: pebble number density array [1/cm^3] in code units
- Helper functions: `Get_m_p_from_rho_Np()` and `Get_s_p_from_m_p()` to derive mass and size on-the-fly

**Benefits:**
- More fundamental quantity (number density)
- Reduces storage (one array instead of two)
- Mass and size can be derived when needed

## Files Modified

### 1. Core Phase Change Module

#### `src/phase_change/phase_change.hpp`
**Purpose:** Class definition for PhaseChange module

**Key Changes:**
- Added public member `rho_Np_array` (replaced `m_p_array` and `s_p_array`)
- Added public member `q_diff` for heat conduction rate
- Added public helper functions:
  - `Get_m_p_from_rho_Np()`: Derive pebble mass from number density
  - `Get_s_p_from_m_p()`: Derive pebble size from mass
  - `Get_stopping_time()`: Calculate stopping time for a pebble
  - `Get_T_rhoe_g()`: Get temperature from gas internal energy (moved from private to public)
- Member variables use code units internally (converted from CGS in constructor)

#### `src/phase_change/phase_change.cpp`
**Purpose:** Implementation of PhaseChange class

**Key Changes:**
- Constructor:
  - Allocates `rho_Np_array` (4D: N_P × ncells3 × ncells2 × ncells1)
  - Registers `rho_Np_array` for restart I/O
  - Converts input parameters (`m_p0_array`, `rho_sil_inter_`, `rho_ice_inter_`) from CGS to code units
- `PhaseChangeSource()`:
  - Updates `rho_Np_array` during phase transitions
  - Uses `Get_m_p_from_rho_Np()` and `Get_s_p_from_m_p()` to derive pebble properties
  - All calculations in code units
- `Get_stopping_time()`:
  - Moved from problem generator to PhaseChange module
  - Uses member variables `rho_sil_inter_` and `rho_ice_inter_`
  - All calculations in code units
  - Handles Epstein and Stokes drag regimes
- **Bug Fixes:**
  - Replaced hardcoded `2*p` with `N_Z * p` for general N_Z support (lines 139, 221, 406, 425)

#### `src/phase_change/phase_change_constants.hpp`
**Purpose:** Physical constants for phase change calculations

**Key Constants:**
- `L_heat_cgs`: Latent heat of vaporization [erg/g]
- `Cd_water_cgs`: Heat capacity of water ice [erg/(g*K)]
- `P_eq0_cgs`: Equilibrium vapor pressure prefactor [erg/cm^3]
- `mu_xy`, `mu_z`, `mu_H2`, `mu_He`: Mean molecular weights
- `KELVIN`: Temperature scale constant
- `sigma_mol_cgs`: Collisional cross-section of H2 [cm^2]
- `NVapor`: Number of vapor species (currently 1)
- `vapor_id`: Index of vapor dust fluid (`NDUSTFLUIDS - 1`)
- `FLX_COR`: Flag for flux correction at boundaries

### 2. Problem Generator

#### `src/pgen/disk_snowline_2D_RT_erg_2.cpp`
**Purpose:** Main problem generator for disk snowline simulation

**Key Changes:**

1. **Initialization (ProblemGenerator):**
   - Initializes `rho_Np_array` from refractory densities (lines 447-456)
   - Uses formula: `rho_Np = rho_sil / (1.0 - f_ICE_inter0) / m_p0_array(p)`
   - All calculations in code units

2. **Stopping Time Calculation:**
   - **Removed:** Inline stopping time calculation from `UserWorkInLoop`
   - **Moved to:** `MyStoppingTime()` function (lines 1495-1566)
   - Uses `pphase_change->Get_stopping_time()` for per-pebble calculation
   - Includes `dust_start_injection` check

3. **Dust Diffusivity Calculation:**
   - **Removed:** Inline diffusivity calculation from `UserWorkInLoop`
   - **Moved to:** `MyDustDiffusivity()` function (lines 1568-1621)
   - Calculates diffusivity per pebble based on stopping time
   - Sets vapor diffusivity with artificial decay for outer boundary

4. **Boundary Flux Correction (UserWorkInLoop):**
   - Updated to use `inflx_dust_x1` array for per-dust-fluid fluxes (lines 1008-1104)
   - Loops over `N_P` pebbles and `N_Z` compositions
   - Applies flux ratio correction per dust_id
   - Includes division-by-zero protection

5. **Temperature Calculation:**
   - Uses `pphase_change->Get_T_rhoe_g()` instead of local function
   - Added null check with fallback

6. **User Output:**
   - Derives `s_p` from `rho_Np` using helper functions (lines 1269-1276)
   - Converts to CGS only for output

### 3. Mesh and MeshBlock Infrastructure

#### `src/mesh/mesh.hpp`
**Purpose:** MeshBlock class definition

**Key Changes:**
- **Forward declaration** (line 57):
  - Added `class PhaseChange;` forward declaration
- **Member variable** (line 139):
  - Added `PhaseChange *pphase_change;` as a public member of MeshBlock
  - Allows access to PhaseChange module from any MeshBlock member function

#### `src/mesh/meshblock.cpp`
**Purpose:** MeshBlock construction, destruction, and restart I/O

**Key Changes:**

1. **Include** (line 52):
   - Added `#include "../phase_change/phase_change.hpp"` to access PhaseChange class

2. **Constructor (normal)** (lines 224-229):
   - Initializes PhaseChange module if `N_Z > 0`:
     ```cpp
     if (N_Z > 0) {
       pphase_change = new PhaseChange(this, pin);
     } else {
       pphase_change = nullptr;
     }
     ```
   - Called after DustFluids is created (line 224)
   - PhaseChange constructor needs access to DustFluids for consistency checks

3. **Constructor (restart)** (lines 435-440):
   - Same initialization logic as normal constructor
   - PhaseChange object is created before loading restart data

4. **Destructor** (line 610):
   - Deletes PhaseChange object if `N_Z > 0`:
     ```cpp
     if (N_Z > 0) delete pphase_change;
     ```

5. **LoadRestartData()** (lines 509-513):
   - Loads `rho_Np_array` from restart file:
     ```cpp
     if (N_Z > 0) {
       std::memcpy(pphase_change->rho_Np_array.data(), &(mbdata[os]), 
                   pphase_change->rho_Np_array.GetSizeInBytes());
       os += pphase_change->rho_Np_array.GetSizeInBytes();
     }
     ```
   - **Replaced:** `m_p_array` and `s_p_array` loading
   - **Order:** Loaded after dustfluids data, before radiation/scalars

6. **GetBlockSizeInBytes()** (lines 720-722, 762-764):
   - Two overloaded functions for calculating restart file size
   - Both include `rho_Np_array` size:
     ```cpp
     if (N_Z > 0) {
       size += pphase_change->rho_Np_array.GetSizeInBytes();
     }
     ```
   - **Replaced:** `m_p_array` and `s_p_array` size calculations

### 4. Restart I/O

#### `src/outputs/restart.cpp`
**Purpose:** Writing restart files

**Key Changes:**
- **Include** (line 34):
  - Added `#include "../phase_change/phase_change.hpp"` to access PhaseChange class
- **WriteRestartFile()** (lines 214-219):
  - Writes `rho_Np_array` to restart file:
     ```cpp
     if (N_Z > 0) {
       std::memcpy(pdata, pmb->pphase_change->rho_Np_array.data(), 
                   pmb->pphase_change->rho_Np_array.GetSizeInBytes());
       pdata += pmb->pphase_change->rho_Np_array.GetSizeInBytes();
     }
     ```
  - **Replaced:** `m_p_array` and `s_p_array` writing
  - **Order:** Written after dustfluids data, before scalars

### 5. Hydro Module

#### `src/hydro/calculate_fluxes.cpp`
**Purpose:** Calculate hydrodynamic fluxes

**Key Changes:**
- **CalculateFluxes()** (lines 119-124):
  - Added boundary check: flux correction only applied at outer simulation boundary
  - Condition: `pmb->pbval->block_bcs[BoundaryFace::outer_x1] == BoundaryFlag::user`
  - Prevents flux correction at internal block boundaries
  - **Before:** Flux correction applied to all blocks when `FLX_COR` is enabled
  - **After:** Only applied to blocks at the outer simulation domain boundary
  - **Reason:** `inflx_x1` is only set for blocks with user boundary condition

#### `src/hydro/rsolvers/hydro/hllc.cpp`
**Purpose:** HLLC Riemann solver for hydrodynamics

**Key Changes:**
- **Include** (line 27):
  - Added `#include "../../../phase_change/phase_change_constants.hpp"` to access `vapor_id`
- **RiemannSolver()** (lines 69-76):
  - Added vapor concentration handling for sound speed calculation:
     ```cpp
     if (N_Z > 0) {
       int vapor_den_id = vapor_id*4;
       fv_l = rl_(vapor_den_id,i);
       fv_r = rr_(vapor_den_id,i);
       cl = pmy_block->peos->SoundSpeed_fv(wli,fv_l);
       cr = pmy_block->peos->SoundSpeed_fv(wri,fv_r);
     }
     ```
  - **Reason:** Sound speed depends on vapor fraction when phase change is enabled
  - Uses reconstructed vapor concentrations (`rl_`, `rr_`) at cell faces

#### `src/hydro/new_blockdt.cpp`
**Purpose:** Calculate new time step for each MeshBlock

**Key Changes:**
- **Include** (line 33):
  - Added `#include "../phase_change/phase_change_constants.hpp"` to access `vapor_id`
- **NewBlockTimeStep()** (lines 118-123):
  - Added vapor-dependent sound speed calculation:
     ```cpp
     if (N_Z > 0) {
       int vapor_den_id = vapor_id*4;
       Real fv = pmb->pdustfluids->df_w(vapor_den_id,k,j,i)/wi[IDN];
       cs = pmb->peos->SoundSpeed_fv(wi,fv);
     } else {
       cs = pmb->peos->SoundSpeed(wi);
     }
     ```
  - **Reason:** Time step calculation needs correct sound speed, which depends on vapor fraction
  - Uses `SoundSpeed_fv()` instead of `SoundSpeed()` when phase change is enabled

### 6. Equation of State Module

#### `src/eos/general/phase_change.cpp`
**Purpose:** EOS functions for phase change module

**Key Changes:**
- **Include** (line 17):
  - Added `#include "../../phase_change/phase_change_constants.hpp"` to access constants
- **Function implementations:**
  - `Get_mu()`: Uses `mu_xy` and `mu_z` from constants
  - `calc_gamma()`: Uses `mu_H2`, `mu_He`, `mu_z` from constants
  - `PresFromRhoEg_fv()`, `EgasFromRhoP_fv()`, `AsqFromRhoP_fv()`:
    - Fixed `ATHENA_ERROR` usage to accept `std::stringstream`
    - Added `return -1.0;` after `ATHENA_ERROR` calls to ensure return value
- **Purpose:** These functions handle EOS calculations when vapor is present

#### `src/eos/general/general_hydro.cpp`
**Purpose:** General hydro EOS functions

**Key Changes:**
- **Include** (line 30):
  - Added `#include "../../phase_change/phase_change_constants.hpp"` to access `vapor_id`
- **PrimToCons()** (lines 111-114):
  - Added vapor-dependent pressure calculation:
     ```cpp
     if (N_Z > 0) {
       int vapor_den_id = vapor_id*4;
       Real fv = pmy_block_->pdustfluids->df_w(vapor_den_id,k,j,i)/u_d;
       w_p = PresFromRhoEg_fv(u_d,u_e-ke,fv);
     }
     ```
  - **Reason:** Pressure depends on vapor fraction when phase change is enabled
- **ConsToPrim()** (lines 160-164):
  - Added vapor-dependent energy calculation:
     ```cpp
     if (N_Z > 0) {
       int vapor_den_id = vapor_id*4;
       Real fv = pmy_block_->pdustfluids->df_w(vapor_den_id,k,j,i)/w_d;
       u_e = EgasFromRhoP_fv(u_d, w_p, fv) + 0.5*w_d*(SQR(w_vx) + SQR(w_vy) + SQR(w_vz));
     }
     ```
  - **Reason:** Internal energy depends on vapor fraction when phase change is enabled

### 7. Compile-Time Configuration

#### `configure.py`
**Purpose:** Build configuration script

**Key Changes:**
- Added `--np` argument: sets `N_P` (number of pebble sizes)
- Added `--nz` argument: sets `N_Z` (number of compositions per pebble)
- Example: `python configure.py --np 2 --nz 2`

#### `src/defs.hpp.in`
**Purpose:** Template for compile-time macros

**Key Changes:**
- Added `#define N_P @NUMBER_PEBBLES@`
- Added `#define N_Z @NUMBER_COMPOSITIONS@`
- These are replaced by `configure.py` with actual values

## Unit Conversion Strategy

**Principle:** All internal calculations use code units. Conversions only at input/output boundaries.

**Input (ParameterInput → PhaseChange constructor):**
- `m_p0_array`: CGS [g] → code units [code_mass]
- `rho_sil_inter_`: CGS [g/cm^3] → code units [code_density]
- `rho_ice_inter_`: CGS [g/cm^3] → code units [code_density]
- `L_heat`: CGS [erg/g] → code units [code_velocity^2]
- `P_eq0`: CGS [erg/cm^3] → code units [code_pressure]

**Internal calculations:**
- All arrays (`rho_Np_array`, `q_latent`, `q_diff`) in code units
- Helper functions operate in code units
- `Get_stopping_time()` returns code_time

**Output (for user output):**
- Convert from code units to CGS only when writing to output files
- Example: `s_p * punit->code_length_cgs` (line 1276)

## Function Migration Summary

### Functions Moved to PhaseChange Module

1. **`Get_stopping_time()`**
   - **From:** Standalone function in problem generator
   - **To:** `PhaseChange::Get_stopping_time()` (public member)
   - **Benefits:** Can access `rho_sil_inter_` and `rho_ice_inter_` directly

2. **`Get_T_rhoe_g()`**
   - **From:** Private member function
   - **To:** Public member function
   - **Reason:** Needed in problem generator for temperature calculation

3. **`InitializeRhoNp()`** (planned but not implemented)
   - **Status:** Initialization currently done inline in ProblemGenerator
   - **Future:** Could be moved to PhaseChange module for better encapsulation

### Functions Removed

1. **Local `Get_T_rhoe_g()` in problem generator**
   - **Reason:** Now uses `pphase_change->Get_T_rhoe_g()`

2. **Standalone `Get_stopping_time()` in problem generator**
   - **Reason:** Moved to PhaseChange module

## Array Indexing Patterns

### Correct Pattern (use this):
```cpp
for (int p = 0; p < N_P; ++p) {
  for (int z = 0; z < N_Z; ++z) {
    int dust_id = N_Z * p + z;
    // Access dust fluid properties using dust_id
  }
}
```

## Common Bugs and Fixes

### Bug 1: Hardcoded `2*p` instead of `N_Z * p`
**Location:** `src/phase_change/phase_change.cpp` (lines 139, 221, 406, 425)
**Fix:** Replace `2*p` with `N_Z * p` for general N_Z support
**Status:** ✅ Fixed

### Bug 2: Division by zero in flux ratio calculation
**Location:** `src/pgen/disk_snowline_2D_RT_erg_2.cpp` (line 1081)
**Fix:** Add check: `if (Mdot_peb_est(dust_id) > 0.0)`
**Status:** ✅ Fixed

### Bug 3: Loop bounds using `NDUSTFLUIDS` instead of `N_P`
**Location:** `src/pgen/disk_snowline_2D_RT_erg_2.cpp` (lines 1058, 1084)
**Fix:** Replace `for (int p=0; p<NDUSTFLUIDS; p++)` with `for (int p=0; p<N_P; p++)`
**Status:** ✅ Fixed

### Bug 4: Flux correction applied to all blocks
**Location:** `src/hydro/calculate_fluxes.cpp` (line 120)
**Fix:** Add boundary check: only apply at outer simulation boundary
**Status:** ✅ Fixed

## Testing Checklist

- [ ] Verify `NDUSTFLUIDS = N_P * N_Z + NVapor` consistency
- [ ] Check restart file I/O: `rho_Np_array` is saved/loaded correctly
- [ ] Verify unit conversions: all internal calculations in code units
- [ ] Test with different `N_P` and `N_Z` values
- [ ] Verify stopping time calculation per pebble
- [ ] Check flux correction only at outer boundary
- [ ] Verify `vapor_id` is correctly excluded from pebble loops

## Notes

1. **Vapor handling:** Vapor is treated as a special case and excluded from pebble loops. Always use `vapor_id = NDUSTFLUIDS - 1` to access vapor.

2. **Initialization order:** `rho_Np_array` must be initialized after dust fluid densities are set in `ProblemGenerator`.

3. **Restart compatibility:** Old restart files with `m_p_array` and `s_p_array` are not compatible. New runs must start from scratch or convert restart files.

4. **Compile-time parameters:** `N_P` and `N_Z` must be set at compile time via `configure.py`. They cannot be changed at runtime.

5. **Constants:** Physical constants are defined in `phase_change_constants.hpp` as `constexpr` for compile-time evaluation.

## Future Improvements

1. Implement `InitializeRhoNp()` as a public member function for better encapsulation
2. Add runtime validation for `N_P` and `N_Z` consistency
3. Consider making `vapor_id` a compile-time constant instead of a macro
4. Add unit tests for helper functions
5. Document expected ranges for physical constants
