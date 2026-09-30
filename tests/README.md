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
