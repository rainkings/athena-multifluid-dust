// Move native viscous dissipation out of hydro and, optionally, into column RT.
// This is scratch storage: it must not be added to ruser_mesh_data/restarts.
#ifndef PGEN_SNOWLINE_VISCOUS_HEATING_HPP_
#define PGEN_SNOWLINE_VISCOUS_HEATING_HPP_

#include <cmath>
#include <cstring>
#include <sstream>
#include <vector>

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../parameter_input.hpp"

namespace SnowlineThermalEnergy {

struct ViscousHeating {
  bool enabled = true;
  bool deferred = false;
  // Same logical indexing as the thermal column buffer, including radial ghosts.
  // Only active cells are written. Boundary-profile ghost sources remain zero.
  std::vector<Real> rate;

  void Configure(ParameterInput *pin, Mesh *mesh) {
    enabled = pin->GetOrAddBoolean("problem", "viscous_heating", true);
    const bool requested = pin->GetOrAddBoolean(
        "problem", "defer_viscous_heating", false);
    deferred = requested || !enabled;
    if (!deferred) return;  // Preserve the native hydro-heating path by default.

    if (STS_ENABLED || MAGNETIC_FIELDS_ENABLED || !NON_BAROTROPIC_EOS
        || Globals::nranks != 1 || mesh->mesh_size.nx3 != 1
        || std::strcmp(COORDINATE_SYSTEM, "spherical_polar") != 0) {
      std::stringstream msg;
      msg << "### FATAL ERROR: deferred/disabled viscous heating currently requires "
          << "nonmagnetic spherical 2D hydro, explicit viscosity, and one MPI rank.\n";
      ATHENA_ERROR(msg);
    }
    const int nr = mesh->mesh_size.nx1+2*NGHOST;
    rate.assign(nr*mesh->mesh_size.nx2, 0);
  }

  // Call at EVERY VL2 source stage, before recovering T or applying phase change.
  // prim must be the same stage primitive state used for the hydro operators;
  // its velocity is the residual velocity when orbital advection is enabled.
  void RemoveFromHydro(MeshBlock *p, Real dt, const AthenaArray<Real> &prim,
                       AthenaArray<Real> &cons) {
    if (!deferred) return;
    Hydro *h = p->phydro;
    const HydroDiffusion &hd = h->hdif;
    const bool viscous = hd.hydro_diffusion_defined
                        && (hd.alpha_disk_model || hd.nu_iso > 0 || hd.nu_aniso > 0);
    const int nr = p->pmy_mesh->mesh_size.nx1+2*NGHOST;
    if (!viscous) {
      for (int j = p->js; j <= p->je; ++j)
        for (int i = p->is; i <= p->ie; ++i) {
          const int ti = p->loc.lx1*p->block_size.nx1+i-p->is+NGHOST;
          const int tj = p->loc.lx2*p->block_size.nx2+j-p->js;
          rate[tj*nr+ti] = 0;
        }
      return;
    }

    // Arrays are zero-initialized. All scratch is private to this block call.
    AthenaArray<Real> rhs(NHYDRO, p->ncells3, p->ncells2, p->ncells1);
    AthenaArray<Real> zero_prim(NHYDRO, p->ncells3, p->ncells2, p->ncells1);
    AthenaArray<Real> a1(p->ncells1+1), a2(p->ncells1), a2p(p->ncells1);
    AthenaArray<Real> volume(p->ncells1);
    Coordinates *coord = p->pcoord;

    // Reproduce only the native viscous flux divergence. Do not overwrite
    // h->flux: other hydro tasks still use the complete flux array.
    for (int k = p->ks; k <= p->ke; ++k) {
      for (int j = p->js; j <= p->je; ++j) {
        coord->Face1Area(k, j, p->is, p->ie+1, a1);
        coord->CellVolume(k, j, p->is, p->ie, volume);
        if (p->block_size.nx2 > 1) {
          coord->Face2Area(k, j, p->is, p->ie, a2);
          coord->Face2Area(k, j+1, p->is, p->ie, a2p);
        }
        for (int n = IM1; n <= IEN; ++n) {
          for (int i = p->is; i <= p->ie; ++i) {
            Real divergence = a1(i+1)*hd.visflx[X1DIR](n,k,j,i+1)
                              -a1(i)*hd.visflx[X1DIR](n,k,j,i);
            if (p->block_size.nx2 > 1)
              divergence += a2p(i)*hd.visflx[X2DIR](n,k,j+1,i)
                            -a2(i)*hd.visflx[X2DIR](n,k,j,i);
            rhs(n,k,j,i) = -divergence/volume(i);
          }
        }
      }
    }

    // With zero rho/pressure/velocity, the existing spherical source operator
    // retains just its viscous geometric terms. zero_prim also serves as an
    // unused zero magnetic field; MHD is explicitly excluded above.
    coord->AddCoordTermsDivergence(1, hd.visflx, zero_prim, zero_prim, rhs);
    if (p->porb->orbital_advection_defined) {
      // Zero primitives and zero viscous mass flux eliminate the ordinary
      // orbital/gravity terms, leaving the existing background-shear work.
      h->hsrc.OrbitalAdvectionSourceTerms(1, hd.visflx, zero_prim, rhs);
    }

    for (int k = p->ks; k <= p->ke; ++k) {
      for (int j = p->js; j <= p->je; ++j) {
        for (int i = p->is; i <= p->ie; ++i) {
          // Internal-energy RHS, not simply the total-energy flux divergence.
          const Real heat = rhs(IEN,k,j,i)
              -prim(IVX,k,j,i)*rhs(IM1,k,j,i)
              -prim(IVY,k,j,i)*rhs(IM2,k,j,i)
              -prim(IVZ,k,j,i)*rhs(IM3,k,j,i);
          if (!std::isfinite(heat)) {
            std::stringstream msg;
            msg << "### FATAL ERROR: nonfinite native viscous heating in block "
                << p->gid << " at " << i << ',' << j << ',' << k << '\n';
            ATHENA_ERROR(msg);
          }
          const int ti = p->loc.lx1*p->block_size.nx1+i-p->is+NGHOST;
          const int tj = p->loc.lx2*p->block_size.nx2+j-p->js;
          rate[tj*nr+ti] = heat;  // Overwrite: the final VL2 evaluation wins.
          cons(IEN,k,j,i) -= dt*heat;
        }
      }
    }
  }
};

}  // namespace SnowlineThermalEnergy
#endif  // PGEN_SNOWLINE_VISCOUS_HEATING_HPP_
