// Opt-in snowline adapter for the shared implicit two-stream column solver.
// One full timestep is split into hydro/phase change, followed by this thermal
// update at fixed density and composition. Source accounting is deliberate:
//   q_irr:    absorbed stellar irradiation, rebuilt here from current opacity;
//   q_diff:   explicit radial conduction, recorded by the pgen's MySource;
//   q_vis:    already in hydro energy, including orbital background-shear work;
//   q_latent: already paid by PhaseChange, retained here only for diagnostics.
// Only q_irr+q_diff enter the additional material/radiation energy solve.
#ifndef PGEN_SNOWLINE_THERMAL_ENERGY_HPP_
#define PGEN_SNOWLINE_THERMAL_ENERGY_HPP_

#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#include "snowline_twostream.hpp"

namespace SnowlineThermalEnergy {
using SnowlineTwoStream::Column;
using SnowlineTwoStream::Scalar;
using SnowlineTwoStream::Vector;

// Normalization only: 150 K is neither an equilibrium target nor a T floor.
constexpr Scalar kReferenceTemperature = 150.L;

struct Evolution {
  // Field-major buffer: data[field*count + theta_index*nr + radial_index].
  // Scalar material quantities use Athena code units; temperatures are kelvin.
  enum GatheredField {
    kDensity, kVaporFraction, kCylindricalRadius, kPathLength, kGasInternalEnergy,
    kIceHeatCapacity, kRadialSource, kLatentDiagnostic, kOwnershipCount, kFieldCount
  };
  // Existing pgen mesh-array layout; keep it compatible with boundaries/output.
  enum MeshField {
    kOpticalDepth = 0, kEffectiveDepth = 1, kHeatingRate = 2,
    kIntegratedHeating = 3, kTemperature = 4, kRadialHeating = 8
  };

  bool enabled = false;
  Scalar max_energy_defect = 0;
  int steps = 0;
  std::vector<Real> data;
  std::vector<Column> columns;  // Reused allocations, one column per radial index.
  std::vector<Vector> radiation_sources;
  Vector irradiation_factors, ray_coefficients;

  void Configure(ParameterInput *pin, Mesh *mesh) {
    enabled = pin->GetOrAddBoolean("problem", "thermal_energy", false);
    if (!enabled) return;
    // "Static" refers to the mesh: hydro and phase change can still evolve.
    // These restrictions match the pgen's half-disk geometry and VL2 sources.
    if (pin->GetOrAddBoolean("problem", "thermal_test", false)
        || !pin->GetBoolean("problem", "Allow_T_change_Flag")
        || mesh->adaptive || mesh->multilevel || mesh->mesh_size.nx3 != 1
        || std::abs(mesh->mesh_size.x2max-PI/2) > 1e-8
        || std::strcmp(COORDINATE_SYSTEM, "spherical_polar") != 0
        || pin->GetString("time", "integrator") != "vl2"
        || pin->GetOrAddReal("problem", "kappa_iso", 0) != 0
        || mesh->mesh_bcs[BoundaryFace::inner_x1] != BoundaryFlag::user
        || mesh->mesh_bcs[BoundaryFace::outer_x1] != BoundaryFlag::user) {
      std::stringstream msg;
      msg << "### FATAL ERROR: thermal_energy requires a static spherical 2D half disk, "
          << "VL2, user radial boundaries, kappa_iso=0, Allow_T_change_Flag=true, "
          << "and thermal_test=false.\n";
      ATHENA_ERROR(msg);
    }
  }

  // Called once after the full hydro timestep. Assemble complete theta columns,
  // solve their material/radiation coupling, then scatter heat back to blocks.
  // Luminosity and illumination geometry must remain fixed for the cached rays.
  void Advance(Mesh *mesh, Real opacity, Real f_vi, Real luminosity,
               Real r0, Real qvalue) {
    const int nr = mesh->mesh_size.nx1+2*NGHOST, nt = mesh->mesh_size.nx2;
    const int count = nr*nt;
    const int nthreads = mesh->GetNumMeshThreads();
    // Include radial user ghosts for the pgen's boundary-profile construction.
    // Theta ghosts are excluded: each column runs from surface to midplane.
    data.assign(kFieldCount*count, 0);
    if (columns.empty()) {
      columns.reserve(nr);
      for (int i = 0; i < nr; ++i) columns.emplace_back(nt);
      radiation_sources.assign(nr, Vector(nt));
      irradiation_factors.resize(count);
      ray_coefficients.resize(count);
    }
    Real phase_defect = 0, mass_defect = 0;
#pragma omp parallel for num_threads(nthreads) schedule(static) reduction(max:phase_defect,mass_defect)
    for (int b = 0; b < mesh->nblocal; ++b) {
      MeshBlock *p = mesh->my_blocks(b);
      if (p->pphase_change) {
        phase_defect = std::max(phase_defect, p->pphase_change->max_energy_defect);
        mass_defect = std::max(mass_defect, p->pphase_change->max_mass_defect);
      }
      int il = p->is, iu = p->ie;
      if (p->pbval->block_bcs[BoundaryFace::inner_x1] == BoundaryFlag::user) il -= NGHOST;
      if (p->pbval->block_bcs[BoundaryFace::outer_x1] == BoundaryFlag::user) iu += NGHOST;
      for (int j = p->js; j <= p->je; ++j) {
        for (int i = il; i <= iu; ++i) {
          const int ti = p->loc.lx1*p->block_size.nx1+i-p->is+NGHOST;
          const int tj = p->loc.lx2*p->block_size.nx2+j-p->js, g = tj*nr+ti, k = p->ks;
          const Real rho = p->phydro->u(IDN,k,j,i);
          const Real kinetic = (SQR(p->phydro->u(IM1,k,j,i))
              +SQR(p->phydro->u(IM2,k,j,i))+SQR(p->phydro->u(IM3,k,j,i)))/(2*rho);
          data[kDensity*count+g] = rho;
          data[kVaporFraction*count+g] = p->pdustfluids->df_u(4*vapor_id,k,j,i)/rho;
          data[kCylindricalRadius*count+g] = p->pcoord->x1v(i)*std::sin(p->pcoord->x2v(j));
          data[kPathLength*count+g] = p->pcoord->x1v(i)*p->pcoord->dx2f(j);
          data[kGasInternalEnergy*count+g] = p->phydro->u(IEN,k,j,i)-kinetic;
          if (p->pphase_change && i >= p->is && i <= p->ie) {
            for (int pop = 0; pop < N_P; ++pop)
              data[kIceHeatCapacity*count+g] += p->pphase_change->Cd_water
                                *p->pdustfluids->df_u(4*N_Z*pop,k,j,i);
            data[kLatentDiagnostic*count+g] = p->pphase_change->q_latent(k,j,i);
          }
          data[kRadialSource*count+g] = mesh->ruser_mesh_data[kRadialHeating](0,tj+NGHOST,ti);
          data[kOwnershipCount*count+g] = 1;
        }
      }
    }
#ifdef MPI_PARALLEL
    // Reconstruct each column when MeshBlocks span ranks. Every rank currently
    // repeats the full solve; this is not distributed column ownership. The
    // surrounding pgen's boundary integrals still need MPI work/validation.
    MPI_Allreduce(MPI_IN_PLACE, data.data(), data.size(), MPI_ATHENA_REAL,
                  MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &phase_defect, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &mass_defect, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
    const Scalar flux_unit = mesh->punit->code_energydensity_cgs
                             *mesh->punit->code_velocity_cgs;
    const Scalar reference_flux = Constants::sigma_cgs/flux_unit
                                  *std::pow(kReferenceTemperature, 4);
    const Scalar luminosity_unit = flux_unit*mesh->punit->code_length_cgs
                                            *mesh->punit->code_length_cgs;
    int invalid = 0, negative_flux = 0;
    Scalar min_temperature = std::numeric_limits<Scalar>::infinity(), max_temperature = 0;
    Scalar step_energy_defect = 0;
    std::vector<std::string> errors(nr);
    // Each column owns its vectors and writes a distinct radial slice of the
    // global arrays. Only diagnostics are reduced; floating-point sums inside
    // a column retain their original order. MPI collectives stay on the caller.
#pragma omp parallel for num_threads(nthreads) schedule(static) reduction(+:invalid,negative_flux) reduction(min:min_temperature) reduction(max:max_temperature,step_energy_defect)
    for (int i = 0; i < nr; ++i) {
      Column &c = columns[i];
      Vector &radiation_source = radiation_sources[i];
      Scalar total_tau = 0;
      for (int j = 0; j < nt; ++j) {
        const int g = j*nr+i;
        const Scalar rho = data[kDensity*count+g], fv = data[kVaporFraction*count+g];
        const Scalar chi = rho*opacity*(1-fv);
        if (data[kOwnershipCount*count+g] != 1 || !(chi > 0) || !(data[kGasInternalEnergy*count+g] > 0)
            || !std::isfinite(chi) || !std::isfinite(fv) || fv < 0
            || !std::isfinite(data[kGasInternalEnergy*count+g])) {
          std::stringstream msg;
          msg << "### FATAL ERROR: invalid conserved state entering thermal_energy at "
              << i << ',' << j << " rho=" << Real(rho) << " fv=" << Real(fv)
              << " internal_energy=" << data[kGasInternalEnergy*count+g] << '\n';
          errors[i] = msg.str();
          break;
        }
        const Scalar mu = 1/((1-fv)/mu_xy+fv/mu_z);
        const Scalar gamma = mesh->my_blocks(0)->peos->calc_gamma(fv);
        // Infer T from gas heat alone. Ice is already at the same T by the
        // phase model, and contributes extra heat capacity during this step.
        const Scalar gas_cv = rho/(KELVIN*mu*(gamma-1));
        c.width[j] = chi*data[kPathLength*count+g];
        c.capacity[j] = (gas_cv+data[kIceHeatCapacity*count+g])*kReferenceTemperature
                        /(chi*reference_flux);
        c.theta[j] = data[kGasInternalEnergy*count+g]/(kReferenceTemperature*gas_cv);
        total_tau += c.width[j];
      }
      if (!errors[i].empty()) continue;
      Scalar tau = 0;
      for (int j = 0; j < nt; ++j) {
        const int g = j*nr+i;
        // Static mesh and illumination: cache geometry, but rebuild optical
        // depths and attenuation from the evolving density/composition.
        if (steps == 0) {
          const Scalar radius = data[kCylindricalRadius*count+g];
          const Scalar incidence = (qvalue/2+0.5)*4
                                   *std::pow(radius/r0, qvalue/2+1.5)/radius;
          const Scalar e0 = luminosity*Constants::solar_lum_cgs
                            /(8*PI*radius*radius*luminosity_unit);
          irradiation_factors[g] = e0*f_vi/reference_flux;
          ray_coefficients[g] = f_vi/incidence;
        }
        const Scalar center = tau+c.width[j]/2;
        const Scalar coefficient = ray_coefficients[g];
        c.source[j] = irradiation_factors[g]
            *(std::exp(-coefficient*center)
              +std::exp(-coefficient*(2*total_tau-center)));
        const Scalar chi = data[kDensity*count+g]*opacity*(1-data[kVaporFraction*count+g]);
        radiation_source[j] = c.source[j]+data[kRadialSource*count+g]/(chi*reference_flux);
        tau += c.width[j];
        mesh->ruser_mesh_data[kOpticalDepth](0,j+NGHOST,i) = center;
      }
      c.Initialize(false);
      // Hypothetical equilibrium of q_irr+q_diff+q_latent, for diagnostics only.
      // q_vis is excluded here because it is already in hydro's incoming heat;
      // this is not the complete old prescribed-viscosity equilibrium. Its sign
      // check can fail while a transient with stored material heat remains valid.
      c.net_heating = 0;
      for (int j = 0; j < nt; ++j) {
        const int g = j*nr+i;
        const Scalar chi = data[kDensity*count+g]*opacity*(1-data[kVaporFraction*count+g]);
        c.source[j] = radiation_source[j]+data[kLatentDiagnostic*count+g]/(chi*reference_flux);
        c.net_heating += c.width[j]*c.source[j];
      }
      c.Equilibrium(false);
      if (i >= NGHOST && i < nr-NGHOST) {
        invalid += c.eq_status != 0;
        negative_flux += (c.eq_status & 1) != 0;
      }
      // Preserve q_z/q_int/tau_eff outputs as equilibrium diagnostics. In this
      // mode q_int is not the escaping flux from the evolved radiation field.
      Scalar integral = 0, weighted = 0;
      for (int j = 0; j < nt; ++j) {
        const int g = j*nr+i;
        const Scalar heating = c.width[j]*c.source[j];
        const Scalar flux_center = c.net_heating-integral-heating/2;
        mesh->ruser_mesh_data[kHeatingRate](0,j+NGHOST,i)
            = reference_flux*c.source[j]*data[kDensity*count+g]*opacity
              *(1-data[kVaporFraction*count+g]);
        mesh->ruser_mesh_data[kIntegratedHeating](0,j+NGHOST,i) = reference_flux*(integral+heating/2);
        mesh->ruser_mesh_data[kEffectiveDepth](0,j+NGHOST,i) = c.net_heating != 0
            ? (weighted+c.width[j]*flux_center/2)/c.net_heating : 0;
        integral += heating;
        weighted += c.width[j]*flux_center;
      }
      // Restore the actual source before stepping: latent and viscous heat have
      // already changed the incoming conserved energy and must not be added twice.
      c.source = radiation_source;
      c.net_heating = 0;
      for (int j = 0; j < nt; ++j) c.net_heating += c.width[j]*c.source[j];
      Scalar elapsed = 0;
      int substeps = 0;
      while (elapsed < mesh->dt) {
        Scalar dt = mesh->dt-elapsed, defect = 0;
        // For a negative explicit source, spend at most half the available
        // sensible heat before accounting for radiation. Restrict time, not T
        // or the source itself; the implicit emission term is handled by Step.
        for (int j = 0; j < nt; ++j)
          if (c.source[j] < 0)
            dt = std::min(dt, -Scalar(0.5)*c.capacity[j]*c.theta[j]/c.source[j]);
        int retries = 0;
        while (!c.Step(dt, defect)) {
          dt /= 2;
          if (++retries > 50) break;
        }
        if (retries > 50 || ++substeps > 100000 || !(elapsed+dt > elapsed)) {
          errors[i] = "### FATAL ERROR: thermal_energy exhausted stored heat or solve failed\n";
          break;
        }
        step_energy_defect = std::max(step_energy_defect, defect);
        elapsed += dt;
      }
      for (int j = 0; j < nt; ++j) {
        mesh->ruser_mesh_data[kTemperature](0,j+NGHOST,i) = kReferenceTemperature*c.theta[j];
        if (i >= NGHOST && i < nr-NGHOST) {
          min_temperature = std::min(min_temperature, kReferenceTemperature*c.theta[j]);
          max_temperature = std::max(max_temperature, kReferenceTemperature*c.theta[j]);
        }
      }
    }
    for (const auto &error : errors) {
      if (!error.empty()) {
        std::stringstream msg;
        msg << error;
        ATHENA_ERROR(msg);
      }
    }
    max_energy_defect = std::max(max_energy_defect, step_energy_defect);
    // Preserve momenta and all material masses; update gas heat and primitive
    // pressure consistently. Ice sensible heat is represented at the common T.
#pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int b = 0; b < mesh->nblocal; ++b) {
      MeshBlock *p = mesh->my_blocks(b);
      int il = p->is, iu = p->ie;
      if (p->pbval->block_bcs[BoundaryFace::inner_x1] == BoundaryFlag::user) il -= NGHOST;
      if (p->pbval->block_bcs[BoundaryFace::outer_x1] == BoundaryFlag::user) iu += NGHOST;
      for (int j = p->js; j <= p->je; ++j) {
        for (int i = il; i <= iu; ++i) {
          const int ti = p->loc.lx1*p->block_size.nx1+i-p->is+NGHOST;
          const int tj = p->loc.lx2*p->block_size.nx2+j-p->js, k = p->ks;
          const Real rho = p->phydro->u(IDN,k,j,i);
          const Real fv = p->pdustfluids->df_u(4*vapor_id,k,j,i)/rho;
          const Real mu = 1/((1-fv)/mu_xy+fv/mu_z);
          const Real temperature = kReferenceTemperature*columns[ti].theta[tj];
          // Energy and momenta are in the same frame (residual if OA is on),
          // so subtracting/adding their kinetic energy leaves material heat.
          Real kinetic = 0;
          p->phydro->w(IDN,k,j,i) = rho;
          for (int dim = 1; dim <= 3; ++dim) {
            const Real momentum = p->phydro->u(dim,k,j,i);
            kinetic += momentum*momentum/(2*rho);
            p->phydro->w(dim,k,j,i) = momentum/rho;
          }
          p->phydro->Tem(k,j,i) = temperature;
          p->phydro->w(IPR,k,j,i) = rho*temperature/(KELVIN*mu);
          p->phydro->u(IEN,k,j,i) = kinetic
              +p->phydro->w(IPR,k,j,i)/(p->peos->calc_gamma(fv)-1);
        }
      }
    }
    ++steps;
    if (Globals::my_rank == 0 && (steps <= 3 || steps%100 == 0)) {
      std::cout << "[thermal_energy] cycle=" << mesh->ncycle+1
                << " time=" << mesh->time+mesh->dt
                << " Tmin=" << Real(min_temperature) << " Tmax=" << Real(max_temperature)
                << " invalid_eq_columns=" << invalid << " negative_flux_columns=" << negative_flux
                << " radiation_energy_defect=" << Real(max_energy_defect)
                << " phase_energy_defect=" << phase_defect
                << " phase_mass_defect=" << mass_defect << std::endl;
    }
  }
};
}  // namespace SnowlineThermalEnergy
#endif
