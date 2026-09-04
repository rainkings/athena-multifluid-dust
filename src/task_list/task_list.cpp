//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file task_list.cpp
//! \brief functions for TaskList base class


// C headers

// C++ headers
//#include <vector> // formerly needed for vector of MeshBlock ptrs in DoTaskListOneStage
#include <ctime>    // std::clock(), CLOCKS_PER_SEC

// Athena++ headers
#include "../athena.hpp"
#include "../globals.hpp"
#include "../mesh/mesh.hpp"
#include "task_list.hpp"

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

// Classify a task into a TimeProfile category based on its numeric id.
// Dust-task ids happen to be the contiguous range PROPERTIES_DFS..RECV_DFSSH.
namespace {
int ProfCategory(const TaskID &id) {
  using namespace HydroIntegratorTaskNames;
  // Gas hydro evolution
  if (id == CALC_HYDFLX || id == SEND_HYDFLX || id == RECV_HYDFLX ||
      id == INT_HYD || id == SEND_HYD || id == RECV_HYD || id == SETB_HYD ||
      id == SEND_HYDFLXSH || id == SEND_HYDSH || id == RECV_HYDFLXSH ||
      id == RECV_HYDSH || id == DIFFUSE_HYD || id == CALC_HYDORB ||
      id == SEND_HYDORB || id == RECV_HYDORB)
    return TimeProfile::HYDRO;
  // Dust fluids evolution
  if (id == PROPERTIES_DFS || id == DIFFUSE_DFS || id == CALC_DFSFLX ||
      id == SEND_DFSFLX || id == RECV_DFSFLX || id == INT_DFS ||
      id == SRCTERM_DFS || id == DRAG_DFS || id == SEND_DFS || id == RECV_DFS ||
      id == SETB_DFS || id == SEND_DFSFLXSH || id == SEND_DFSSH ||
      id == RECV_DFSFLXSH || id == RECV_DFSSH)
    return TimeProfile::DUSTFLUIDS;
  // User explicit source terms (MySource)
  if (id == SRC_TERM)
    return TimeProfile::SOURCE;
  // Conserved -> primitive conversion (gas + dust fluids)
  if (id == CONS2PRIM)
    return TimeProfile::CONS2PRIM;
  // Physical boundary conditions
  if (id == PHY_BVAL || id == CLEAR_ALLBND)
    return TimeProfile::BVAL;
  // Per-block user work (MeshBlock::UserWorkInLoop via the USERWORK task)
  if (id == USERWORK)
    return TimeProfile::USERWORK;
  return TimeProfile::OTHER;
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn TaskListStatus TaskList::DoAllAvailableTasks
//! \brief do all tasks that can be done (are not waiting for a dependency to be
//! cleared) in this TaskList, return status.

TaskListStatus TaskList::DoAllAvailableTasks(MeshBlock *pmb, int stage, TaskStates &ts) {
  int skip = 0;
  TaskStatus ret;
  if (ts.num_tasks_left == 0) return TaskListStatus::nothing_to_do;

  for (int i=ts.indx_first_task; i<ntasks; i++) {
    Task &taski = task_list_[i];
    if (ts.finished_tasks.IsUnfinished(taski.task_id)) { // task not done
      // check if dependency clear
      if (ts.finished_tasks.CheckDependencies(taski.dependency)) {
        if (taski.lb_time) pmb->StartTimeMeasurement();
        int prof_cat = -1;
        clock_t prof_c0 = 0;
        if (TimeProfile::enabled) {
          prof_cat = ProfCategory(taski.task_id);
          prof_c0 = std::clock();
        }
        ret = (this->*task_list_[i].TaskFunc)(pmb, stage);
        if (taski.lb_time) pmb->StopTimeMeasurement();
        if (TimeProfile::enabled && prof_cat >= 0) {
          double dt_cpu =
              static_cast<double>(std::clock() - prof_c0) / CLOCKS_PER_SEC;
#pragma omp atomic
          TimeProfile::accum[prof_cat] += dt_cpu;
        }
        if (ret != TaskStatus::fail) { // success
          ts.num_tasks_left--;
          ts.finished_tasks.SetFinished(taski.task_id);
          if (skip == 0) ts.indx_first_task++;
          if (ts.num_tasks_left == 0) return TaskListStatus::complete;
          if (ret == TaskStatus::next) continue;
          return TaskListStatus::running;
        }
      }
      skip++; // increment number of tasks processed

    } else if (skip == 0) { // this task is already done AND it is at the top of the list
      ts.indx_first_task++;
    }
  }
  // there are still tasks to do but nothing can be done now
  return TaskListStatus::stuck;
}

//----------------------------------------------------------------------------------------
//! \fn void TaskList::DoTaskListOneStage(Mesh *pmesh, int stage)
//! \brief completes all tasks in this list, will not return until all are tasks done

void TaskList::DoTaskListOneStage(Mesh *pmesh, int stage) {
  int nthreads = pmesh->GetNumMeshThreads();
  int nmb = pmesh->nblocal;

  // clear the task states, startup the integrator and initialize mpi calls
#pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
  for (int i=0; i<nmb; ++i) {
    pmesh->my_blocks(i)->tasks.Reset(ntasks);
    StartupTaskList(pmesh->my_blocks(i), stage);
  }

  int nmb_left = nmb;
  // cycle through all MeshBlocks and perform all tasks possible
  while (nmb_left > 0) {
    //! \note
    //! KNOWN ISSUE: Workaround for unknown OpenMP race condition. See #183 on GitHub.
#pragma omp parallel for reduction(- : nmb_left) num_threads(nthreads) schedule(dynamic,1)
    for (int i=0; i<nmb; ++i) {
      if (DoAllAvailableTasks(pmesh->my_blocks(i), stage, pmesh->my_blocks(i)->tasks)
          == TaskListStatus::complete) {
        nmb_left--;
      }
    }
  }
  return;
}
