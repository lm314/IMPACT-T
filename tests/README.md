# A1 test: pt2ptFields

`test_pt2ptfields.f90` checks the point-to-point space-charge field routine
(`pt2ptFields` in `src/Appl/BeamBunch.f90`) against an independent
reference implementation of the same pair formulas, at 1 and 4 MPI ranks.
It is not part of the CMake build (it links against the compiled
`ImpactTexe-mpi` object files directly), so build `build-mpi` first
(`-DUSE_MPI=ON -DUSE_FFTW=ON`), then:

```
cd ../build-mpi
find CMakeFiles/ImpactTexe-mpi.dir -name "*.o" | \
  grep -vE "Contrl/(main|AccSimulator|Input|Output)\.f90\.o" > /tmp/objlist.txt
cat /tmp/objlist.txt | xargs mpif90 -cpp -DUSE_MPI -fallow-argument-mismatch -I. \
  -o ../tests/test_pt2ptfields ../tests/test_pt2ptfields.f90 \
  <fftw3 and MPI shared libraries, as printed by
   CMakeFiles/ImpactTexe-mpi.dir/link.txt>

cd ../tests
mpirun -n 1 ./test_pt2ptfields
mpirun -n 4 ./test_pt2ptfields
```

(Build the object list to a file and pipe it through `xargs` rather than
`$(find ...)` inline on the command line -- under zsh, an unquoted
command substitution is passed as a single word, not one word per line,
so the linker sees one mangled "filename" instead of ~30 object files.)

For `USE_GPU=ON` (A2 onward), build `build-mpi-gpu` the same way (with
`-DUSE_GPU=ON`, nvfortran) and link `test_pt2ptfields`/`bench_pt2ptfields`
against *that* build's object list instead, on a node with a GPU, so the
`!$acc` directives in `pt2ptFields` actually offload.

Checks (see IMPACTT_GPU_PORT_PLAN.md, step A1):

- **A**: image on, matches a direct+self-image reference to ~1e-12.
- **B**: image off, matches a direct-only reference to ~1e-13 (this is
  the state after `zimage`, and the state whenever `Flagimg=0`).
- **C**: sanity check that the image flag actually changes the result.
- **D**: a single particle at rest at height z matches the analytic
  self-image field, e/(4 pi eps0 (2z)^2), to machine precision.

For A2's GPU-vs-CPU field-level check, set `IMPACTT_FIELD_DUMP=<path>`
(nproc=1 only) to also dump Check A's (image on) computed
(ex,ey,ez,bx,by,bz), one row per particle, to that file. Build the same
`test_pt2ptfields.f90` once against a `USE_GPU=ON` object set and run it
on a GPU node, and once against a CPU-only (`USE_GPU=OFF`) object set,
then diff the two dumps:

```
python3 compare_runs.py --diff-fields gpu_fields.txt cpu_fields.txt --rtol 1e-10
```

# A2 performance check: bench_pt2ptfields

`bench_pt2ptfields.f90` measures pair evaluations/second for
`pt2ptFields` at 1M particles, single rank, image charge on (2*N^2 pairs
per call -- the direct term over all pairs but the particle's own, plus
the image term over all pairs including the particle's own). Link it the
same way as `test_pt2ptfields` above, against a `USE_GPU=ON` object set
to measure one H100 or `USE_GPU=OFF` to measure one CPU core:

```
mpirun -n 1 ./bench_pt2ptfields
```

Prints the measured rate and, assuming it holds at scale, the projected
per-step and whole-run (2500-step) time at the plan's 6.24M-particle
target case.

# A2 GPU profiling: bench_pt2ptfields_prof

`bench_pt2ptfields_prof.f90` is a smaller-N (100k particles), one-rep
variant of `bench_pt2ptfields.f90`, for capturing an `ncu` (Nsight
Compute) profile of the `pt2ptFields` kernel without a full-metric-set
capture taking forever. Link it against a `USE_GPU=ON` object set (same
way as the other tests), then profile the timed call (the 2nd kernel
launch; the 1st is an untimed warm-up) with e.g.:

```
ncu --launch-skip 1 --launch-count 1 \
    --section SpeedOfLight --section SpeedOfLight_RooflineChart \
    --section Occupancy --section WarpStateStats \
    --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis \
    -f -o ncu_report mpirun -n 1 ./bench_pt2ptfields_prof
ncu -i ncu_report.ncu-rep --page details
```

See IMPACTT_GPU_PORT_PLAN.md's A2 Progress notes for what this found.

# Phase 0 test harness: compare_runs.py

`compare_runs.py` runs the same IMPACT-T input with two executables and
compares fort.18/24/25/26 (run-level check) and the `showtime_Timer`
timers printed to stdout (needs the MPI build, even at 1 rank -- the
serial build's timers always read 0). Used for:

- the `nvfortran` CPU-only build check against `gfortran` (0.2): looser
  tolerance (`--rtol`, default 1e-6), since different compilers reorder
  floating-point sums;
- CPU-vs-GPU checks from A2 onward: tight tolerance (`--rtol 1e-12`),
  since the plan expects near-bit-identical results there.

```
python3 compare_runs.py --ref /path/to/ImpactTexe-mpi \
    --cand /path/to/ImpactTexe-mpi-gpu \
    --input-dir /path/to/impact_t_files \
    --workdir /tmp/cmp --mpi --nproc 1 --rtol 1e-12
```

It also has a `--diff-fields REF CAND` mode (A2) that skips the run-level
check and just reports the max relative error between two
(ex,ey,ez,bx,by,bz) dump files, e.g. `test_pt2ptfields`'s
`IMPACTT_FIELD_DUMP` output -- see the A2 section above.
