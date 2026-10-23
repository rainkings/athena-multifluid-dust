#include "coupled_column_athena.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../dustfluids/dustfluids.hpp"
#include "../eos/eos.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/hydro_diffusion/hydro_diffusion.hpp"
#include "../mesh/mesh.hpp"
#include "../relaxation/relaxation.hpp"
#include "phase_change.hpp"
#include "phase_change_constants.hpp"

namespace coupled_column {
namespace {

struct CellAddress {
  MeshBlock *block = nullptr;
  int k = 0, j = 0, i = 0;
};

Real Pressure(Real gas, Real vapor, Real temperature) {
  const Real inverse_mu = (1.0-vapor/gas)/mu_xy + (vapor/gas)/mu_z;
  return gas*temperature*inverse_mu/KELVIN;
}

void SetGasEnergy(MeshBlock *block, AthenaArray<Real> &gas,
                  const AthenaArray<Real> &dust, int k, int j, int i, Real temperature) {
  const Real density = gas(IDN,k,j,i);
  const Real vapor = dust(4*vapor_id,k,j,i);
  const Real pressure = Pressure(density, vapor, temperature);
  const Real gamma = block->peos->calc_gamma(vapor/density);
  const Real kinetic = 0.5*(SQR(gas(IM1,k,j,i)) + SQR(gas(IM2,k,j,i))
                            + SQR(gas(IM3,k,j,i)))/density;
  gas(IEN,k,j,i) = pressure/(gamma-1.0)+kinetic;
}

Cell Gather(const CellAddress &address) {
  MeshBlock *block = address.block;
  const int k = address.k, j = address.j, i = address.i;
  Cell cell;
  cell.temperature = block->phydro->Tem(k,j,i);
  cell.gas = block->phydro->u(IDN,k,j,i);
  cell.vapor = block->pdustfluids->df_u(4*vapor_id,k,j,i);
  cell.path_length = block->pcoord->x1v(i)*block->pcoord->dx2f(j);
  cell.cylindrical_radius = block->pcoord->x1v(i)*std::sin(block->pcoord->x2v(j));
  cell.viscosity = block->phydro->hdif.nu(HydroDiffusion::DiffProcess::alpha,k,j,i);
  const int radial = block->loc.lx1*block->block_size.nx1+i-block->is;
  const int vertical = block->loc.lx2*block->block_size.nx2+j-block->js;
  cell.q_diff = block->pmy_mesh->ruser_mesh_data[8](0,vertical+NGHOST,radial+NGHOST);
  cell.populations.resize(N_P);
  for (int p = 0; p < N_P; ++p) {
    cell.populations[p].ice = block->pdustfluids->df_u(4*N_Z*p,k,j,i);
    cell.populations[p].refractory = block->pdustfluids->df_u(4*(N_Z*p+1),k,j,i);
    cell.populations[p].number = block->pdustfluids->df_u(4*(N_Z*N_P+1+p),k,j,i);
  }
  return cell;
}

void Commit(const CellAddress &address, const Cell &saved, const Cell &cell,
            const Solution &result, int row, int radial, int vertical,
            const Settings &settings, const OutputIndices &output) {
  MeshBlock *block = address.block;
  const int k = address.k, j = address.j, i = address.i;
  AthenaArray<Real> &gas = block->phydro->u;
  AthenaArray<Real> &dust = block->pdustfluids->df_u;
  AthenaArray<Real> &primitive = block->phydro->w;
  AthenaArray<Real> &dust_primitive = block->pdustfluids->df_w;

  // Keep the existing fixed constituent-velocity phase prescription. This is
  // thermal relaxation, not a new drag or momentum-exchange operator.
  for (int direction = 1; direction <= 3; ++direction)
    gas(direction,k,j,i) *= cell.gas/saved.gas;
  gas(IDN,k,j,i) = cell.gas;
  for (int p = 0; p < N_P; ++p) {
    const int ice_id = 4*N_Z*p;
    const int refractory_id = 4*(N_Z*p+1);
    dust(ice_id,k,j,i) = cell.populations[p].ice;
    dust_primitive(ice_id,k,j,i) = cell.populations[p].ice;
    for (int direction = 1; direction <= 3; ++direction) {
      const Real velocity = dust(refractory_id+direction,k,j,i)
          /cell.populations[p].refractory;
      dust(ice_id+direction,k,j,i) = cell.populations[p].ice*velocity;
      dust_primitive(ice_id+direction,k,j,i) = velocity;
    }
    const Real mass = (cell.populations[p].ice+cell.populations[p].refractory)
        /std::max(cell.populations[p].number, Real(1.e-100));
    const Real volume = cell.populations[p].ice/settings.ice_density
        +cell.populations[p].refractory/settings.refractory_density;
    const Real size = std::cbrt(3.0*volume
        /(4.0*PI*std::max(cell.populations[p].number, Real(1.e-100))));
    block->prelax->m_p_array(p,k,j,i) = mass;
    block->user_out_var(output.particle_mass[p],k,j,i) = mass;
    block->user_out_var(output.particle_size[p],k,j,i) = size;
  }
  dust(4*vapor_id,k,j,i) = cell.vapor;
  dust_primitive(4*vapor_id,k,j,i) = cell.vapor;
  primitive(IDN,k,j,i) = cell.gas;
  for (int direction = 1; direction <= 3; ++direction) {
    const Real velocity = gas(direction,k,j,i)/cell.gas;
    primitive(direction,k,j,i) = velocity;
    dust(4*vapor_id+direction,k,j,i) = cell.vapor*velocity;
    dust_primitive(4*vapor_id+direction,k,j,i) = velocity;
  }
  block->phydro->Tem(k,j,i) = cell.temperature;
  SetGasEnergy(block, gas, dust, k, j, i, cell.temperature);
  primitive(IPR,k,j,i) = Pressure(cell.gas, cell.vapor, cell.temperature);
  block->pphase_change->q_latent(k,j,i) = result.q_latent[row];
  block->pphase_change->P_eq_array(k,j,i)
      = settings.equilibrium_pressure*std::exp(-T_a/cell.temperature);

  // Preserve the pgen's cache numbering and allocation sizes. Only active
  // cells are written here; no block ghost cell owns another copy of a column.
  AthenaArray<Real> *cache = block->pmy_mesh->ruser_mesh_data;
  const int tj = vertical+NGHOST, ti = radial+NGHOST;
  cache[0](0,tj,ti) = result.tau[row];
  cache[1](0,tj,ti) = result.rhokappa[row];
  cache[2](0,tj,ti) = result.q_total[row];
  cache[3](0,tj,ti) = result.integrated_heating[row];
  cache[4](0,tj,ti) = cell.temperature;
  cache[8](0,tj,ti) = 0.0;  // Consume the frozen diffusion source exactly once.

  block->user_out_var(output.temperature,k,j,i) = cell.temperature;
  block->user_out_var(output.latent,k,j,i) = result.q_latent[row];
  block->user_out_var(output.radiative,k,j,i) = result.q_total[row];
  block->user_out_var(output.diffusive,k,j,i) = saved.q_diff;
  block->user_out_var(output.integrated,k,j,i) = result.integrated_heating[row];
  block->user_out_var(output.cooling,k,j,i) = settings.beta
      /std::sqrt(settings.gm/std::pow(cell.cylindrical_radius,3));
  block->user_out_var(output.drho_vapor,k,j,i) = (cell.vapor-saved.vapor)/settings.dt;
  block->user_out_var(output.drho_ice0,k,j,i)
      = (cell.populations[0].ice-saved.populations[0].ice)/settings.dt;
  block->user_out_var(output.drho_ice1,k,j,i) = N_P > 1
      ? (cell.populations[1].ice-saved.populations[1].ice)/settings.dt : 0.0;
}

}  // namespace

void RestoreStoredTemperature(MeshBlock *block, AthenaArray<Real> &gas,
                              const AthenaArray<Real> &dust) {
  for (int k = block->ks; k <= block->ke; ++k)
    for (int j = block->js; j <= block->je; ++j)
      for (int i = block->is; i <= block->ie; ++i)
        SetGasEnergy(block, gas, dust, k, j, i, block->phydro->Tem(k,j,i));
}

void ReadRestartTemperature(MeshBlock *block, bool replace_valid_temperature) {
  // The timestep callback also runs at the end of the first restarted hydro
  // step, before the mesh clock advances. Recover from energy ONLY once;
  // later hydro energy changes must not replace the authoritative stored T.
  static std::set<MeshBlock *> initialized_blocks;
  bool first_call = false;
#pragma omp critical(coupled_column_restart_initialization)
  {
    first_call = initialized_blocks.insert(block).second;
  }
  if (!first_call) return;
  const AthenaArray<Real> &gas = block->phydro->u;
  const AthenaArray<Real> &dust = block->pdustfluids->df_u;
  for (int j = block->js; j <= block->je; ++j) {
    for (int i = block->is; i <= block->ie; ++i) {
      const int k = block->ks;
      const Real stored = block->phydro->Tem(k,j,i);
      if (!replace_valid_temperature && stored > 0.0 && std::isfinite(stored)) continue;
      const Real density = gas(IDN,k,j,i);
      const Real kinetic = 0.5*(SQR(gas(IM1,k,j,i)) + SQR(gas(IM2,k,j,i))
                                +SQR(gas(IM3,k,j,i)))/density;
      block->phydro->Tem(k,j,i) = block->pphase_change->Get_T_rhoe_g(
          gas(IEN,k,j,i)-kinetic, density, dust(4*vapor_id,k,j,i)/density);
    }
  }
}

void AdvanceColumns(Mesh *mesh, const Settings &settings, const OutputIndices &output) {
  if (Globals::nranks != 1 || mesh->multilevel || mesh->mesh_size.nx3 != 1
      || mesh->mesh_size.nx2 <= 1 || N_Z != 2 || N_P < 1
      || std::strcmp(COORDINATE_SYSTEM,"spherical_polar") != 0
      || MAGNETIC_FIELDS_ENABLED || STS_ENABLED
      || std::abs(mesh->mesh_size.x2max-PI/2.0) > 1.e-6) {
    throw std::runtime_error("coupled columns require one MPI rank, a 2D root spherical "
        "mesh ending at the midplane, N_Z=2, N_P>=1, hydro and no STS");
  }
  const int nr = mesh->mesh_size.nx1, nz = mesh->mesh_size.nx2;
  std::vector<std::vector<CellAddress>> columns(nr, std::vector<CellAddress>(nz));

  // Logical coordinates, rather than block-list order, identify a column.
  // Therefore vertical decomposition and OpenMP scheduling cannot reorder it.
  for (int bn = 0; bn < mesh->nblocal; ++bn) {
    MeshBlock *block = mesh->my_blocks(bn);
    for (int j = block->js; j <= block->je; ++j) {
      for (int i = block->is; i <= block->ie; ++i) {
        const int radial = block->loc.lx1*block->block_size.nx1+i-block->is;
        const int vertical = block->loc.lx2*block->block_size.nx2+j-block->js;
        if (radial < 0 || radial >= nr || vertical < 0 || vertical >= nz
            || columns[radial][vertical].block != nullptr)
          throw std::runtime_error("duplicate or out-of-range coupled column cell");
        CellAddress &address = columns[radial][vertical];
        address.block = block;
        address.k = block->ks;
        address.j = j;
        address.i = i;
      }
    }
  }
  std::vector<std::vector<Cell>> saved(nr, std::vector<Cell>(nz));
  for (int radial = 0; radial < nr; ++radial)
    for (int vertical = 0; vertical < nz; ++vertical) {
      if (columns[radial][vertical].block == nullptr)
        throw std::runtime_error("incomplete coupled column");
      saved[radial][vertical] = Gather(columns[radial][vertical]);
    }

  std::vector<Solution> solutions(nr);
  std::vector<std::string> errors(nr);
#pragma omp parallel for schedule(dynamic)
  for (int radial = 0; radial < nr; ++radial) {
    // Exceptions must not escape an OpenMP worker. Report them after the join,
    // before changing any live column, so failure cannot partially commit.
    try {
      Solve(saved[radial], settings, solutions[radial], errors[radial]);
    } catch (const std::exception &exception) {
      errors[radial] = exception.what();
    }
  }
  for (int radial = 0; radial < nr; ++radial) {
    if (!errors[radial].empty()) {
      std::ostringstream message;
      message << "coupled column " << radial << " failed at t=" << mesh->time
              << ", dt=" << mesh->dt << ": " << errors[radial]
              << "; no columns committed";
      throw std::runtime_error(message.str());
    }
  }
#pragma omp parallel for schedule(static)
  for (int radial = 0; radial < nr; ++radial)
    for (int vertical = 0; vertical < nz; ++vertical)
      Commit(columns[radial][vertical], saved[radial][vertical],
             solutions[radial].cells[vertical], solutions[radial], vertical,
             radial, vertical, settings, output);

  // The task list computed dt BEFORE Mesh::UserWorkInLoop. Refresh the block
  // estimates after changing pressure; Mesh::NewTimeStep will reduce them in
  // its normal place. No integrator or EOS change is required.
#pragma omp parallel for schedule(static)
  for (int bn = 0; bn < mesh->nblocal; ++bn) {
    MeshBlock *block = mesh->my_blocks(bn);
    // Temperature is not an exchanged hydro primitive. Explicitly refresh
    // internal block ghosts from their active owner for the next diffusion
    // evaluation. Physical-boundary ghosts remain owned by the user BCs.
    for (int j = block->js-NGHOST; j <= block->je+NGHOST; ++j)
      for (int i = block->is-NGHOST; i <= block->ie+NGHOST; ++i) {
        const int radial = block->loc.lx1*block->block_size.nx1+i-block->is;
        const int vertical = block->loc.lx2*block->block_size.nx2+j-block->js;
        if (radial >= 0 && radial < nr && vertical >= 0 && vertical < nz)
          block->phydro->Tem(block->ks,j,i) = solutions[radial].cells[vertical].temperature;
      }
    block->phydro->NewBlockTimeStep();
  }
}

}  // namespace coupled_column
