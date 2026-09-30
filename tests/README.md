# A1 test: pt2ptFields

`test_pt2ptfields.f90` checks the point-to-point space-charge field routine
(`pt2ptFields` in `src/Appl/BeamBunch.f90`) against an independent
reference implementation of the same pair formulas, at 1 and 4 MPI ranks.
It is not part of the CMake build (it links against the compiled
`ImpactTexe-mpi` object files directly), so build `build-mpi` first
(`-DUSE_MPI=ON -DUSE_FFTW=ON`), then:

```
cd ../build-mpi
OBJS=$(find CMakeFiles/ImpactTexe-mpi.dir -name "*.o" | \
       grep -vE "Contrl/(main|AccSimulator|Input|Output)\.f90\.o")
mpif90 -cpp -DUSE_MPI -fallow-argument-mismatch -I. \
  -o ../tests/test_pt2ptfields ../tests/test_pt2ptfields.f90 $OBJS \
  <fftw3 and MPI shared libraries, as printed by
   CMakeFiles/ImpactTexe-mpi.dir/link.txt>

cd ../tests
mpirun -n 1 ./test_pt2ptfields
mpirun -n 4 ./test_pt2ptfields
```

Checks (see IMPACTT_GPU_PORT_PLAN.md, step A1):

- **A**: image on, matches a direct+self-image reference to ~1e-12.
- **B**: image off, matches a direct-only reference to ~1e-13 (this is
  the state after `zimage`, and the state whenever `Flagimg=0`).
- **C**: sanity check that the image flag actually changes the result.
- **D**: a single particle at rest at height z matches the analytic
  self-image field, e/(4 pi eps0 (2z)^2), to machine precision.

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

It also has a `compare_fields()` helper for the field-level check (dump
E/B at the particles for one step, under an env var) described in the
plan's 0.3 -- that dump doesn't exist yet; A2 should add it and wire up
`--field-dump-env`/`--field-dump-file` options here.
