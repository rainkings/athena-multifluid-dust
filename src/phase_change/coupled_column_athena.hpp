// Athena storage adapter for the standalone column solver.
#ifndef PHASE_CHANGE_COUPLED_COLUMN_ATHENA_HPP_
#define PHASE_CHANGE_COUPLED_COLUMN_ATHENA_HPP_

#include <vector>

#include "coupled_column.hpp"

class Mesh;
class MeshBlock;
template <typename T> class AthenaArray;

namespace coupled_column {

// Output indices belong to the pgen, not to the mathematical solver.
struct OutputIndices {
  int temperature, latent, radiative, diffusive, cooling, integrated;
  int drho_vapor, drho_ice0, drho_ice1;
  std::vector<int> particle_mass, particle_size;
};

// Project only thermal energy onto stored T, retaining density and momentum.
// Used at hydro source evaluations so hydro IEN is not the thermal authority.
void RestoreStoredTemperature(MeshBlock *block, AthenaArray<Real> &gas,
                              const AthenaArray<Real> &dust);

// Restart migration: older energy-mode files can have a stale global T cache.
// Recover the initial temperature after restart conserved data have been read.
void ReadRestartTemperature(MeshBlock *block, bool replace_valid_temperature);

// One operator-split solve after all blocks finish the hydro timestep.
// All columns are solved before ANY column is committed.
void AdvanceColumns(Mesh *mesh, const Settings &settings, const OutputIndices &outputs);

}  // namespace coupled_column
#endif  // PHASE_CHANGE_COUPLED_COLUMN_ATHENA_HPP_
