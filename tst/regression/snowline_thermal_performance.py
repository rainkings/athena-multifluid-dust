#!/usr/bin/env python3
"""Benchmark and compare the snowline energy solver before/after optimization.

Preserve an optimized (-O3) baseline executable before rebuilding the new pgen:
  python tst/regression/snowline_thermal_performance.py \
      --baseline /path/to/previous/athena --optimized /path/to/new/athena \
      --output-dir /tmp/snowline-performance

Runs are sequential, from the same 01500 restart, with hydro, phase change,
growth, orbital advection and radial transport enabled. Both executables write
one final snapshot. Check the optimized result against the baseline and across
1, 2 and 4 threads. Speedups are measured, not pass/fail criteria: host load can
affect timing. The baseline must include the earlier reconstruction-buffer fix.
"""

import argparse
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys

import numpy as np

sys.dont_write_bytecode = True
from snowline_restart_energy import ROOT, read


def run(binary, output_dir, name, threads, steps):
    directory = output_dir/name
    directory.mkdir(parents=True, exist_ok=True)
    final_cycle = 20219108+steps
    args = [str(binary), "-r", str(ROOT/"tst/data_bug/iceline.01500.rst"),
            "-i", str(ROOT/"inputs/snowline/athinput.snowline_energy_restart"),
            "mesh/num_threads={}".format(threads), "time/nlim={}".format(final_cycle),
            "time/ncycle_out=1000000000", "time/dt_diagnostics=-1",
            "output3/dt=-1", "output4/dt=-1"]
    args += ["output{}/dcycle={}".format(i, final_cycle) for i in (1, 2, 5)]
    with (directory/"run.log").open("w") as log:
        completed = subprocess.run(args, cwd=directory, stdout=log,
                                   stderr=subprocess.STDOUT, timeout=300)
    text = (directory/"run.log").read_text()
    if completed.returncode or "FATAL ERROR" in text:
        raise RuntimeError(name+":\n"+text[-3000:])
    match = re.search(r"omp wtime used\s*=\s*([+\-\d.eE]+)", text)
    if not match:
        raise RuntimeError("Missing OpenMP wall-clock timing in "+name)
    fields = [read(sorted(directory.glob("iceline.out{}.*.athdf".format(i)))[-1])
              for i in (1, 2, 5)]
    elapsed = float(match[1])
    metrics = dict(name=name, threads=threads, steps=steps, wall_seconds=elapsed,
                   seconds_per_step=elapsed/steps, final_time=float(fields[0]["Time"]))
    print("{}: {:.5f} s/step ({} threads)".format(name, elapsed/steps, threads),
          flush=True)
    return dict(metrics=metrics, fields=fields)


def compare(reference, candidate):
    per_output = []
    finite = True
    for a, b in zip(reference["fields"], candidate["fields"]):
        worst = 0
        for variable in a["VariableNames"]:
            key = variable.decode() if isinstance(variable, bytes) else str(variable)
            finite = finite and bool(np.all(np.isfinite(b[key])))
            scale = max(float(np.max(np.abs(a[key]))), 1e-30)
            worst = max(worst, float(np.max(np.abs(b[key]-a[key])))/scale)
        per_output.append(worst)
    temperature_error = float(np.max(np.abs(candidate["fields"][1]["Tem"]
                                             /reference["fields"][1]["Tem"]-1)))
    return dict(max_normalized_primitive_error=per_output[0],
                max_normalized_diagnostic_error=per_output[1],
                max_normalized_conserved_error=per_output[2],
                max_relative_temperature_error=temperature_error,
                finite=finite,
                relative_time_error=abs(candidate["metrics"]["final_time"]
                                        /reference["metrics"]["final_time"]-1))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=128)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("steps must be positive")
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    old, new = args.baseline.resolve(), args.optimized.resolve()
    cases = [run(binary, directory, name, threads, args.steps)
             for name, binary, threads in (("before_4_a", old, 4),
                 ("after_4_a", new, 4), ("after_1", new, 1), ("after_2", new, 2),
                 ("after_4_b", new, 4), ("before_4_b", old, 4))]
    reference = cases[0]
    comparisons = {c["metrics"]["name"]: compare(reference, c) for c in cases[1:]}
    thread_comparisons = {c["metrics"]["name"]: compare(cases[1], c)
                          for c in cases[2:5]}
    before = statistics.mean(cases[i]["metrics"]["seconds_per_step"] for i in (0, 5))
    after = statistics.mean(cases[i]["metrics"]["seconds_per_step"] for i in (1, 4))
    checks = dict(
        all_fields_finite=all(c["finite"] for c in comparisons.values()),
        baseline_temperature_agrees=all(c["max_relative_temperature_error"] < 1e-10
                                        for c in comparisons.values()),
        baseline_physical_state_agrees=all(c["max_normalized_primitive_error"] < 1e-8
            and c["max_normalized_conserved_error"] < 1e-8 for c in comparisons.values()),
        same_timestep=all(c["relative_time_error"] < 1e-12 for c in comparisons.values()),
        thread_counts_agree=all(c["max_relative_temperature_error"] < 1e-12
            and c["max_normalized_conserved_error"] < 1e-10
            for c in thread_comparisons.values()))
    summary = dict(cases=[c["metrics"] for c in cases], comparisons=comparisons,
                   thread_comparisons=thread_comparisons,
                   before_seconds_per_step=before, after_seconds_per_step=after,
                   speedup=before/after, wall_time_reduction_percent=100*(1-after/before),
                   checks=checks)
    (directory/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)
    if not all(checks.values()):
        raise SystemExit("Performance regression changed the physical solution")


if __name__ == "__main__":
    main()
